"""5-way progression classification using the JEPA model.

Given a (prior, current) image pair and a target finding (e.g. "pneumonia"),
construct five candidate sentences — one per progression class
(new, worse, stable, improved, resolved). For each candidate, run the
predictor:

    ẑ_cur^k = predictor(LN(z_prior), text_encoder(prompt_k))

Score each by

    S_k = cos(ẑ_cur^k - LN(z_prior),  LN(z_cur) - LN(z_prior))

(the slide-deck inference rule). The class with the highest S_k is the
predicted progression. Compare to the ground-truth label from
``gold_progression_pairs.parquet``.

The prompt bank follows CheXTemporal's zero-shot protocol (Sec. 4.2 of
the paper): "the {finding} has improved/worsened/resolved", "the
{finding} is stable/new". You can override the templates with
``--prompts`` (one template per class, in the canonical order
``new worse stable improved resolved``).

Modes
-----
``--demo``
    Pick one row from gold and print its full 5-way scoring breakdown.
    Useful for sanity-checking that the predictor is responding to the
    text condition.

``--eval``
    Iterate over the whole gold parquet (or ``--limit`` rows) and
    report overall + per-class + per-finding accuracy.

Usage
-----
    # Demo on a single random gold row
    python progression_classify.py --demo

    # Demo on a specific row, specific finding
    python progression_classify.py --demo --idx 17

    # Full eval
    python progression_classify.py --eval

    # Eval on first 200 rows
    python progression_classify.py --eval --limit 200

    # Custom prompt templates
    python progression_classify.py --eval \\
        --prompts "new {finding}" "worsening {finding}" "stable {finding}" \\
                  "improving {finding}" "resolved {finding}"
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from dataset_combined import BASE_TRANSFORM, apply_augmentation, sample_augmentation
from dataset_combined_jepa import (
    DEFAULT_DATASET_DIR,
    DEFAULT_FINDINGS,
    _resolve_image_path,
)
from infer_jepa import IMAGE_ROOTS, jepa_metrics, load_jepa_model
from tempcxr.modules.jepa import TempCXRJEPA


# ============================================================
# CONSTANTS
# ============================================================
PROGRESSION_CLASSES = ["new", "worse", "stable", "improved", "resolved"]

# Default prompt bank — one template per class in PROGRESSION_CLASSES order.
# Phrasing mirrors the dynamic-sentence style the model was trained on.
DEFAULT_PROMPTS = [
    "new {finding}",
    "{finding} has worsened",
    "{finding} is stable",
    "{finding} has improved",
    "{finding} has resolved",
]

DEFAULT_GOLD_PARQUET = os.path.join(
    DEFAULT_DATASET_DIR, "gold_progression_pairs.parquet"
)


# ============================================================
# PROMPT BUILDING
# ============================================================
def build_prompts(finding: str, templates: List[str]) -> List[str]:
    """Format ``templates`` with ``{finding}`` substituted in."""
    f = finding.strip().lower()
    return [t.format(finding=f) for t in templates]


# ============================================================
# IMAGE LOADING (no augmentation, matches val pipeline)
# ============================================================
def load_image_tensor(dataset: str, rel_path: str) -> torch.Tensor:
    img = Image.open(_resolve_image_path(dataset, rel_path, IMAGE_ROOTS)).convert("RGB")
    img = BASE_TRANSFORM(img)
    params = sample_augmentation(train=False)
    return apply_augmentation(img, params)


# ============================================================
# GOLD PARQUET LOADING
# ============================================================
def _normalize_label(x) -> str:
    """Lowercase and strip the ground-truth progression label."""
    return str(x).strip().lower()


def _coalesce_columns(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first column name in ``candidates`` that exists in ``df``."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_gold_pairs(
    gold_parquet: str,
    findings_parquet: str,
    label_col: Optional[str] = None,
    finding_col: Optional[str] = None,
) -> pd.DataFrame:
    """Load gold_progression_pairs and ensure each row has image paths.

    The CheXTemporal gold parquet is keyed on
    ``(dataset, patient_id, study_id_curr, study_id_prev)`` plus a
    ``finding`` and a ``progression`` label. Image paths
    (``parent_image_curr`` / ``parent_image_prev``) may live in the gold
    parquet directly OR in ``silver_findings.parquet``; we join when
    needed.
    """
    print(f"[gold] loading {gold_parquet}")
    gold = pd.read_parquet(gold_parquet)
    print(f"[gold]   {len(gold)} rows, columns: {list(gold.columns)}")

    # ---- Identify label / finding columns ----
    label_col = label_col or _coalesce_columns(
        gold, ["progression", "progression_label", "label", "temporal_label"]
    )
    if label_col is None:
        raise ValueError(
            "Could not find a progression label column in gold parquet. "
            "Pass --label-col explicitly. Columns: " + str(list(gold.columns))
        )
    finding_col = finding_col or _coalesce_columns(
        gold, ["finding", "disease", "pathology", "label_name"]
    )
    if finding_col is None:
        raise ValueError(
            "Could not find a finding column in gold parquet. "
            "Pass --finding-col explicitly. Columns: " + str(list(gold.columns))
        )
    print(f"[gold]   label_col='{label_col}', finding_col='{finding_col}'")

    gold = gold.rename(columns={label_col: "progression", finding_col: "finding"})
    gold["progression"] = gold["progression"].apply(_normalize_label)
    gold = gold[gold["progression"].isin(PROGRESSION_CLASSES)].copy()
    print(f"[gold]   {len(gold)} rows after restricting to 5 canonical classes")

    # ---- Ensure image path columns ----
    if "parent_image_curr" not in gold.columns or "parent_image_prev" not in gold.columns:
        print(f"[gold] joining with {findings_parquet} for image paths")
        findings = pd.read_parquet(findings_parquet)
        keep = [
            "dataset", "patient_id", "study_id_curr", "study_id_prev",
            "parent_image_curr", "parent_image_prev",
        ]
        for c in ["dataset", "patient_id", "study_id_curr", "study_id_prev"]:
            findings[c] = findings[c].astype("string")
            gold[c] = gold[c].astype("string")
        findings = findings[keep].drop_duplicates(
            ["dataset", "patient_id", "study_id_curr", "study_id_prev"]
        )
        gold = gold.merge(
            findings,
            on=["dataset", "patient_id", "study_id_curr", "study_id_prev"],
            how="inner",
        )
        print(f"[gold]   {len(gold)} rows after image-path join")

    # Drop rows with empty image paths
    gold = gold[
        gold["parent_image_curr"].astype("string").str.strip().ne("")
        & gold["parent_image_prev"].astype("string").str.strip().ne("")
    ].reset_index(drop=True)
    print(f"[gold]   {len(gold)} rows with non-empty image paths")
    return gold


# ============================================================
# CORE SCORING
# ============================================================
@torch.no_grad()
def score_one_pair(
    model: TempCXRJEPA,
    prior_img: torch.Tensor,
    current_img: torch.Tensor,
    finding: str,
    templates: List[str],
    device: torch.device,
) -> Dict:
    """Run the 5-way progression scoring for ONE (prior, current, finding).

    Returns:
        prompts:   list[str]  — the 5 candidate sentences
        scores:    list[float] — slide-deck cosine score per class
        smooth_l1: list[float] — JEPA Smooth L1 per class (lower is closer)
        cos_patch_mean: list[float] — per-patch cos sim per class
        z_prior, z_cur: tensors for caller introspection
    """
    prompts = build_prompts(finding, templates)

    prior = prior_img.unsqueeze(0).to(device)
    current = current_img.unsqueeze(0).to(device)

    # We only need z_prior + z_cur once — both come from the image
    # encoders and don't depend on the text condition. So:
    #   1. Run forward once with prompt[0] just to obtain z_prior, z_cur,
    #      and the text-encoder outputs for ALL 5 prompts (batched).
    #   2. Run the predictor 5 times with the 5 different text outputs
    #      against the SAME z_prior.
    # That cuts the BioViL-T image forward pass from 5x to 1x.
    placeholder_pc = [""]  # prior/current text are unused at inference

    # Encode all 5 prompts in one text-encoder call.
    _, all_txt_local, all_token_mask = model.text_encoder.forward_contrastive(prompts)

    # Encode images once (online for prior, EMA for current — same
    # convention as training).
    _, prior_raw = model.image_encoder(prior)
    z_prior = F.layer_norm(prior_raw, (prior_raw.size(-1),))

    _, curr_raw = model.target_image_encoder(current)
    z_cur = F.layer_norm(curr_raw, (curr_raw.size(-1),)).detach()

    scores = []
    smooth_l1s = []
    cos_patch_means = []
    for k in range(len(prompts)):
        txt_k = all_txt_local[k:k + 1]  # (1, T, D)
        mask_k = all_token_mask[k:k + 1]
        pred_k = model.predictor(z_prior, txt_k, mask_k)  # (1, N, D)

        m = jepa_metrics(pred_k, z_cur, z_prior)
        scores.append(m["cos_delta"])
        smooth_l1s.append(m["smooth_l1"])
        cos_patch_means.append(m["cos_patch_mean"])

    return {
        "prompts": prompts,
        "scores": scores,
        "smooth_l1": smooth_l1s,
        "cos_patch_mean": cos_patch_means,
    }


# ============================================================
# DEMO MODE
# ============================================================
def run_demo(args, model, gold_df, device):
    if args.idx is None:
        rng = random.Random(args.seed)
        args.idx = rng.randrange(len(gold_df))
    if not (0 <= args.idx < len(gold_df)):
        raise IndexError(f"--idx {args.idx} out of range [0, {len(gold_df)})")

    row = gold_df.iloc[args.idx]
    finding = str(row["finding"])
    gt_label = _normalize_label(row["progression"])

    print(f"\n=== Gold sample {args.idx} of {len(gold_df)} ===")
    print(f"  dataset:       {row['dataset']}")
    print(f"  patient_id:    {row['patient_id']}")
    print(f"  study_id_prev: {row['study_id_prev']}")
    print(f"  study_id_curr: {row['study_id_curr']}")
    print(f"  finding:       {finding}")
    print(f"  ground-truth:  {gt_label}")

    prior = load_image_tensor(row["dataset"], row["parent_image_prev"])
    current = load_image_tensor(row["dataset"], row["parent_image_curr"])

    out = score_one_pair(
        model, prior, current, finding, args.prompts, device,
    )

    print("\n5-way progression scoring (slide-deck cosine; higher = better):")
    print(f"  {'class':<10} {'prompt':<40} {'cos_delta':>10} {'smoothL1':>10} {'cos_patch':>10}")
    best_idx = max(range(5), key=lambda k: out["scores"][k])
    for k, cls in enumerate(PROGRESSION_CLASSES):
        marker = "  <-- argmax" if k == best_idx else ""
        print(
            f"  {cls:<10} {out['prompts'][k][:38]:<40} "
            f"{out['scores'][k]:>10.4f} "
            f"{out['smooth_l1'][k]:>10.4f} "
            f"{out['cos_patch_mean'][k]:>10.4f}"
            f"{marker}"
        )

    pred_label = PROGRESSION_CLASSES[best_idx]
    correct = pred_label == gt_label
    print(f"\nPredicted: {pred_label}    Ground-truth: {gt_label}    "
          f"=> {'CORRECT' if correct else 'WRONG'}")


# ============================================================
# EVAL MODE
# ============================================================
def run_eval(args, model, gold_df, device):
    if args.limit is not None:
        gold_df = gold_df.head(args.limit).reset_index(drop=True)

    n_correct = 0
    n_seen = 0
    confusion = defaultdict(Counter)        # gt -> Counter(pred -> count)
    per_finding = defaultdict(lambda: [0, 0])  # finding -> [correct, total]
    skipped = 0

    print(f"\n[eval] running 5-way progression classification on {len(gold_df)} rows")
    for i in range(len(gold_df)):
        row = gold_df.iloc[i]
        finding = str(row["finding"])
        gt_label = _normalize_label(row["progression"])
        try:
            prior = load_image_tensor(row["dataset"], row["parent_image_prev"])
            current = load_image_tensor(row["dataset"], row["parent_image_curr"])
        except (FileNotFoundError, OSError) as e:
            skipped += 1
            if skipped <= 5:
                print(f"[eval] skipping row {i} (missing image: {e})")
            continue

        out = score_one_pair(
            model, prior, current, finding, args.prompts, device,
        )
        best_idx = max(range(5), key=lambda k: out["scores"][k])
        pred_label = PROGRESSION_CLASSES[best_idx]

        n_seen += 1
        n_correct += int(pred_label == gt_label)
        confusion[gt_label][pred_label] += 1
        per_finding[finding.lower()][1] += 1
        per_finding[finding.lower()][0] += int(pred_label == gt_label)

        if (i + 1) % max(1, len(gold_df) // 20) == 0:
            running = n_correct / max(1, n_seen)
            print(f"[eval]   {i + 1}/{len(gold_df)}  acc={running:.4f}  skipped={skipped}")

    print("\n=== Overall ===")
    if n_seen == 0:
        print("No samples evaluated.")
        return
    acc = n_correct / n_seen
    print(f"Accuracy: {n_correct}/{n_seen} = {acc:.4f}    (chance = 0.2)")
    if skipped:
        print(f"Skipped:  {skipped} rows due to missing images")

    print("\n=== Per-class accuracy (recall) ===")
    print(f"  {'gt class':<10} {'n':>6} {'acc':>8}")
    for cls in PROGRESSION_CLASSES:
        n = sum(confusion[cls].values())
        c = confusion[cls].get(cls, 0)
        a = c / n if n else float("nan")
        print(f"  {cls:<10} {n:>6} {a:>8.4f}")

    print("\n=== Confusion matrix (rows=gt, cols=pred) ===")
    header = "  " + " ".join(f"{c[:8]:>9}" for c in PROGRESSION_CLASSES)
    print(f"  {'':<10}{header}")
    for gt in PROGRESSION_CLASSES:
        cells = " ".join(f"{confusion[gt].get(p, 0):>9}" for p in PROGRESSION_CLASSES)
        print(f"  {gt:<10}{cells}")

    print("\n=== Per-finding accuracy ===")
    print(f"  {'finding':<26} {'n':>6} {'acc':>8}")
    for finding in sorted(per_finding):
        c, n = per_finding[finding]
        a = c / n if n else float("nan")
        print(f"  {finding:<26} {n:>6} {a:>8.4f}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.environ.get("JEPA_CKPT", "checkpoints_jepa/best.pt"),
    )
    parser.add_argument(
        "--gold-parquet",
        default=DEFAULT_GOLD_PARQUET,
        help=f"Path to gold_progression_pairs.parquet "
             f"(default: {DEFAULT_GOLD_PARQUET}).",
    )
    parser.add_argument(
        "--findings-parquet",
        default=DEFAULT_FINDINGS,
        help="Path to silver_findings.parquet (used to join image paths "
             "if the gold parquet does not have them).",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Override the gold-parquet column name for the progression "
             "label. Auto-detected by default.",
    )
    parser.add_argument(
        "--finding-col",
        default=None,
        help="Override the gold-parquet column name for the finding/disease. "
             "Auto-detected by default.",
    )
    parser.add_argument(
        "--prompts",
        nargs=5,
        default=DEFAULT_PROMPTS,
        help="Five prompt templates (one per class) in canonical order: "
             "new worse stable improved resolved. Each must contain "
             "'{finding}'.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Print 5-way scoring for one gold row.")
    mode.add_argument("--eval", action="store_true",
                      help="Compute overall + per-class + per-finding accuracy.")

    parser.add_argument("--idx", type=int, default=None,
                        help="(demo only) gold-row index. Default: random.")
    parser.add_argument("--seed", type=int, default=0,
                        help="(demo only) RNG seed for picking a random row.")
    parser.add_argument("--limit", type=int, default=None,
                        help="(eval only) Only evaluate the first N rows.")

    args = parser.parse_args()

    # Sanity: make sure prompt templates contain {finding}.
    for i, t in enumerate(args.prompts):
        if "{finding}" not in t:
            raise ValueError(
                f"Prompt {i} ({PROGRESSION_CLASSES[i]!r}) is missing "
                f"'{{finding}}' placeholder: {t!r}"
            )

    device = torch.device(args.device)
    model = load_jepa_model(args.ckpt, device)
    gold_df = load_gold_pairs(
        args.gold_parquet,
        args.findings_parquet,
        label_col=args.label_col,
        finding_col=args.finding_col,
    )
    if len(gold_df) == 0:
        raise RuntimeError("No usable gold rows after filtering.")

    if args.demo:
        run_demo(args, model, gold_df, device)
    else:
        run_eval(args, model, gold_df, device)


if __name__ == "__main__":
    main()
