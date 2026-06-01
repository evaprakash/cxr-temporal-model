"""5-way progression classification using the JEPA model.

Given a (prior, current) image pair and a target finding (e.g. "pneumonia"),
build a CLIP-style prompt bank of multiple phrasings per progression
class (improving / stable / worsening / new / resolved), feed each
phrasing through the predictor, and score how well each predicted
``ẑ_cur^k`` matches the actual ``z_cur``:

    ẑ_cur^k = predictor(LN(z_prior), text_encoder(prompt_k))

Two scoring rules are computed in parallel:

  * **Cosine** — slide-deck inference rule:
        S_cos = cos(ẑ_cur^k - LN(z_prior),  LN(z_cur) - LN(z_prior))
    Predicted class = argmax over classes of average cosine across that
    class's phrases. Direction-only (ignores magnitude).

  * **Smooth L1** — JEPA training-time loss:
        S_l1 = SmoothL1(ẑ_cur^k, LN(z_cur))
    Predicted class = argmin over classes of average L1 across that
    class's phrases. Includes both direction and magnitude.

The prompt bank lives in ``PROGRESSION_PHRASES`` and the template in
``PROMPT_TEMPLATE`` ("{} is {}" by default, e.g. "pneumonia is improving").

Ground-truth labels from ``gold_progression_pairs.parquet`` use the
CheXTemporal taxonomy (improved / worse / stable / new / resolved); the
script remaps "improved" -> "improving" and "worse" -> "worsening" via
``GT_TO_CLS`` so they match the present-tense phrasings. The reported
overall, per-class, and per-finding numbers are directly comparable to
Tables 4 and 5 of the CheXTemporal paper.

Modes
-----
``--demo``
    Print the per-class cosine and L1 scores for one gold row, plus
    both predictions and whether each matches the ground truth.

``--eval``
    Iterate over the whole gold parquet (or ``--limit`` rows) and
    report TWO independent classification results — one using cosine
    argmax, one using Smooth L1 argmin — each with overall accuracy,
    per-class accuracy (= recall), 5x5 confusion matrix, and per-finding
    accuracy.

Usage
-----
    python progression_classify.py --demo
    python progression_classify.py --demo --idx 17
    python progression_classify.py --eval
    python progression_classify.py --eval --limit 200
    python progression_classify.py --eval --prompt-template "{} appears {}"
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Prompt template uses positional substitution: "{disease} is {phrase}".
PROMPT_TEMPLATE = "{} is {}"

# Multi-phrase prompt bank — each class has multiple phrasings whose
# scores get averaged (CLIP-style prompt ensembling).
PROGRESSION_PHRASES = {
    "improving": [
        "better", "decreased", "decreasing", "improved",
        "improving", "reduced", "smaller",
    ],
    "stable": [
        "constant", "stable", "unchanged",
    ],
    "worsening": [
        "bigger", "developing", "enlarged", "enlarging", "greater",
        "growing", "increased", "increasing", "larger",
        "progressing", "progressive", "worse", "worsened", "worsening",
    ],
    "new": [
        "new", "newly developed", "newly appeared", "newly seen",
        "appeared", "emerged",
    ],
    "resolved": [
        "resolved", "resolving", "cleared", "disappeared",
        "no longer present", "no longer seen", "completely resolved",
    ],
}

# Display order for predictions / confusion matrices.
CLS_ORDER = ["improving", "stable", "worsening", "new", "resolved"]

# Map gold-parquet ground-truth labels (CheXTemporal taxonomy) to our
# class names. Gold uses past-tense / adjective forms ("improved",
# "worse"); we use present-tense forms that fit "{disease} is {phrase}".
GT_TO_CLS = {
    "improved": "improving",
    "worse": "worsening",
    "stable": "stable",
    "new": "new",
    "resolved": "resolved",
}

DEFAULT_GOLD_PARQUET = os.path.join(
    DEFAULT_DATASET_DIR, "gold_progression_pairs.parquet"
)
DATASETS = ["mimic", "chexpert", "rexgradient"]


# ============================================================
# PROMPT BUILDING
# ============================================================
def build_class_prompts(
    finding: str,
    template: str = PROMPT_TEMPLATE,
) -> Tuple[List[str], List[int]]:
    """Build all phrase-level prompts for one finding.

    Returns
    -------
    prompts
        Flat list of all phrase prompts across all classes.
    class_idx
        For each prompt, the index in ``CLS_ORDER`` of the class it
        belongs to.
    """
    f = finding.strip().lower()
    prompts: List[str] = []
    class_idx: List[int] = []
    for c_i, cls in enumerate(CLS_ORDER):
        for phrase in PROGRESSION_PHRASES[cls]:
            prompts.append(template.format(f, phrase))
            class_idx.append(c_i)
    return prompts, class_idx


# ============================================================
# GOLD IMAGE ROOT DISCOVERY
# ============================================================
def discover_gold_image_roots(parquet_dir: str) -> Dict[str, str]:
    """Look for ``final_gold_<dataset>_images/`` near the gold parquet.

    Search order (first hit per dataset wins):
      1. ``parquet_dir`` itself — when the gold images are co-located
         with the parquet (e.g. inside the CheXTemporal HF clone).
      2. ``parquet_dir``'s parent — when the gold images are siblings
         of the parquet's containing folder (e.g. ``~/jepa/`` holds both
         ``CheXTemporal/`` and ``final_gold_*_images/``).

    Returns a dict mapping ``dataset -> absolute_path``.
    """
    roots: Dict[str, str] = {}
    parent_dir = os.path.dirname(parquet_dir.rstrip("/"))
    search_bases = [parquet_dir, parent_dir]
    for d in DATASETS:
        for base in search_bases:
            cand = os.path.join(base, f"final_gold_{d}_images")
            if os.path.isdir(cand):
                roots[d] = cand
                break
    return roots


# ============================================================
# IMAGE LOADING (no augmentation, matches val pipeline)
# ============================================================
def _resolve_with_fallbacks(dataset: str, rel_path: str, roots: Dict[str, str]) -> Path:
    """Resolve ``rel_path`` under ``roots[dataset]`` with three fallbacks.

    1. The same prefix-stripping logic the silver pipeline uses
       (``_resolve_image_path``). Handles ``mimic/...``, ``chexpert/train/...``,
       ``rexgradient/deid_png/...`` style paths.
    2. The raw relative path joined onto the root, in case the gold
       parquet stores paths without a dataset prefix.
    3. Just the basename joined onto the root, in case
       ``final_gold_*_images/`` is a flat directory.

    Raises ``FileNotFoundError`` if none of the three resolutions hit
    an existing file.
    """
    rel_path = str(rel_path).strip()
    if dataset not in roots:
        raise FileNotFoundError(f"No image root configured for dataset {dataset!r}")

    tried = []
    try:
        p = _resolve_image_path(dataset, rel_path, roots)
        if p.exists():
            return p
        tried.append(str(p))
    except Exception:
        pass

    p = Path(roots[dataset]) / rel_path
    if p.exists():
        return p
    tried.append(str(p))

    p = Path(roots[dataset]) / os.path.basename(rel_path)
    if p.exists():
        return p
    tried.append(str(p))

    raise FileNotFoundError(
        f"Could not resolve gold image for {dataset}:{rel_path}. Tried: " + " | ".join(tried)
    )


def load_image_tensor(dataset: str, rel_path: str, roots: Dict[str, str]) -> torch.Tensor:
    img = Image.open(_resolve_with_fallbacks(dataset, rel_path, roots)).convert("RGB")
    img = BASE_TRANSFORM(img)
    params = sample_augmentation(train=False)
    return apply_augmentation(img, params)


# ============================================================
# GOLD PARQUET LOADING
# ============================================================
def _normalize_label(x) -> str:
    """Lowercase, strip, and map gold-taxonomy progression label to CLS_ORDER.

    The gold parquet uses ``improved`` / ``worse`` etc.; we re-key those
    to ``improving`` / ``worsening`` so they line up with CLS_ORDER and
    PROGRESSION_PHRASES. Anything not in GT_TO_CLS is returned as-is so
    out-of-vocabulary labels filter themselves out downstream.
    """
    s = str(x).strip().lower()
    return GT_TO_CLS.get(s, s)


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
        gold, ["finding", "disease", "disease_name", "pathology", "label_name"]
    )
    if finding_col is None:
        raise ValueError(
            "Could not find a finding column in gold parquet. "
            "Pass --finding-col explicitly. Columns: " + str(list(gold.columns))
        )
    print(f"[gold]   label_col='{label_col}', finding_col='{finding_col}'")

    gold = gold.rename(columns={label_col: "progression", finding_col: "finding"})
    gold["progression"] = gold["progression"].apply(_normalize_label)
    gold = gold[gold["progression"].isin(CLS_ORDER)].copy()
    print(f"[gold]   {len(gold)} rows after restricting to 5 canonical classes")

    # ---- Ensure image path columns ----
    img_curr_col = _coalesce_columns(
        gold,
        ["parent_image_curr", "img_path_curr", "image_curr", "image_path_curr",
         "image_curr_path", "current_image_path", "img_curr"],
    )
    img_prev_col = _coalesce_columns(
        gold,
        ["parent_image_prev", "img_path_prev", "image_prev", "image_path_prev",
         "image_prev_path", "prior_image_path", "img_prev"],
    )

    if img_curr_col and img_prev_col:
        if img_curr_col != "parent_image_curr" or img_prev_col != "parent_image_prev":
            print(f"[gold]   image cols: {img_prev_col} / {img_curr_col} "
                  f"(renaming to parent_image_prev / parent_image_curr)")
            gold = gold.rename(columns={
                img_prev_col: "parent_image_prev",
                img_curr_col: "parent_image_curr",
            })
    else:
        print(f"[gold] no image-path columns in gold parquet; "
              f"joining with {findings_parquet}")
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
def _encode_prompts(
    model: TempCXRJEPA,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, Tuple]] = None,
):
    """Encode all phrase prompts for one finding. Cached by (finding, template)."""
    cache_key = f"{finding}||{template}"
    if text_cache is not None and cache_key in text_cache:
        prompts, class_idx, txt_local, token_mask = text_cache[cache_key]
        return prompts, class_idx, txt_local.to(device), token_mask.to(device)

    prompts, class_idx = build_class_prompts(finding, template)
    _, txt_local, token_mask = model.text_encoder.forward_contrastive(prompts)
    if text_cache is not None:
        text_cache[cache_key] = (
            prompts, class_idx,
            txt_local.detach().cpu(), token_mask.detach().cpu(),
        )
    return prompts, class_idx, txt_local, token_mask


@torch.no_grad()
def score_one_pair(
    model: TempCXRJEPA,
    prior_img: torch.Tensor,
    current_img: torch.Tensor,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, Tuple]] = None,
) -> Dict:
    """5-way progression scoring for ONE (prior, current, finding).

    Runs each of the per-class phrase prompts through the predictor once
    (batched across phrases), computes per-phrase ``cos_delta`` and
    Smooth L1, then averages within each class. Returns both metrics so
    the caller can compare argmax(cos) vs argmin(L1) predictions.

    Returns
    -------
    prompts          : list[str]                — all phrase prompts
    class_idx        : list[int]                — phrase -> CLS_ORDER idx
    phrase_cos       : list[float]              — cos_delta per phrase
    phrase_l1        : list[float]              — Smooth L1 per phrase
    cos_class_scores : list[float] (len 5)      — mean cos per class
    l1_class_scores  : list[float] (len 5)      — mean L1 per class
    """
    prompts, class_idx, all_txt, all_mask = _encode_prompts(
        model, finding, template, device, text_cache,
    )
    n_phrases = len(prompts)

    prior = prior_img.unsqueeze(0).to(device)
    current = current_img.unsqueeze(0).to(device)

    # Encode images once (online for prior, EMA for current — matches training).
    _, prior_raw = model.image_encoder(prior)
    z_prior = F.layer_norm(prior_raw, (prior_raw.size(-1),))

    _, curr_raw = model.target_image_encoder(current)
    z_cur = F.layer_norm(curr_raw, (curr_raw.size(-1),)).detach()

    # Batch the predictor across all phrases by expanding prior to match.
    z_prior_b = z_prior.expand(n_phrases, -1, -1).contiguous()  # (n_phrases, N, D)
    preds = model.predictor(z_prior_b, all_txt, all_mask)        # (n_phrases, N, D)

    # Per-phrase metrics
    pred_f = preds.float()
    target_f = z_cur.float().expand_as(pred_f)
    prior_f = z_prior.float().expand_as(pred_f)

    smooth_l1 = F.smooth_l1_loss(
        pred_f, target_f, reduction="none"
    ).mean(dim=(1, 2))  # (n_phrases,)

    dpred = (pred_f - prior_f).flatten(start_dim=1)
    dtrue = (target_f - prior_f).flatten(start_dim=1)
    cos_delta = F.cosine_similarity(dpred, dtrue, dim=1)  # (n_phrases,)

    phrase_cos = cos_delta.cpu().tolist()
    phrase_l1 = smooth_l1.cpu().tolist()

    # Aggregate per class (mean of phrase scores)
    n_classes = len(CLS_ORDER)
    cos_sum = [0.0] * n_classes
    l1_sum = [0.0] * n_classes
    counts = [0] * n_classes
    for k in range(n_phrases):
        c = class_idx[k]
        cos_sum[c] += phrase_cos[k]
        l1_sum[c] += phrase_l1[k]
        counts[c] += 1
    cos_class_scores = [cos_sum[c] / max(1, counts[c]) for c in range(n_classes)]
    l1_class_scores = [l1_sum[c] / max(1, counts[c]) for c in range(n_classes)]

    return {
        "prompts": prompts,
        "class_idx": class_idx,
        "phrase_cos": phrase_cos,
        "phrase_l1": phrase_l1,
        "cos_class_scores": cos_class_scores,
        "l1_class_scores": l1_class_scores,
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

    prior = load_image_tensor(row["dataset"], row["parent_image_prev"], args.image_roots)
    current = load_image_tensor(row["dataset"], row["parent_image_curr"], args.image_roots)

    out = score_one_pair(model, prior, current, finding, args.prompt_template, device)

    n_phrases_per_class = [len(PROGRESSION_PHRASES[c]) for c in CLS_ORDER]
    cos_best = max(range(len(CLS_ORDER)), key=lambda k: out["cos_class_scores"][k])
    l1_best = min(range(len(CLS_ORDER)), key=lambda k: out["l1_class_scores"][k])

    print("\nPer-class scores (averaged across phrases of each class):")
    print(f"  {'class':<10} {'#phrases':>9} {'cos_score':>11} {'L1_score':>11}")
    for k, cls in enumerate(CLS_ORDER):
        cos_marker = "  <-- argmax cos" if k == cos_best else ""
        l1_marker = "  <-- argmin L1"  if k == l1_best  else ""
        print(
            f"  {cls:<10} {n_phrases_per_class[k]:>9} "
            f"{out['cos_class_scores'][k]:>11.4f} "
            f"{out['l1_class_scores'][k]:>11.4f}"
            f"{cos_marker}{l1_marker}"
        )

    cos_pred = CLS_ORDER[cos_best]
    l1_pred = CLS_ORDER[l1_best]
    print(f"\n  Cosine prediction:    {cos_pred:<10}  vs gt {gt_label:<10}  "
          f"=> {'CORRECT' if cos_pred == gt_label else 'WRONG'}")
    print(f"  Smooth L1 prediction: {l1_pred:<10}  vs gt {gt_label:<10}  "
          f"=> {'CORRECT' if l1_pred  == gt_label else 'WRONG'}")


# ============================================================
# EVAL MODE
# ============================================================
def _print_eval_summary(method_name: str, n_correct: int, n_seen: int,
                        confusion: Dict, per_finding: Dict,
                        classes: Optional[List[str]] = None):
    """Pretty-print the overall + per-class + confusion + per-finding block.

    ``classes`` (default ``CLS_ORDER``) controls which classes show up in
    the per-class accuracy and confusion-matrix tables. Pass a 3-element
    subset for benchmarks that only have improving / stable / worsening.
    """
    if classes is None:
        classes = CLS_ORDER
    chance = 1.0 / max(1, len(classes))

    print(f"\n{'=' * 60}")
    print(f"=== Results: {method_name}")
    print(f"{'=' * 60}")
    acc = n_correct / max(1, n_seen)
    print(f"Overall accuracy: {n_correct}/{n_seen} = {acc:.4f}    (chance = {chance:.3f})")

    print("\nPer-class accuracy (= per-class recall = paper's "
          "label-specific accuracy):")
    print(f"  {'gt class':<10} {'n':>6} {'acc':>8}")
    for cls in classes:
        n = sum(confusion[cls].values())
        c = confusion[cls].get(cls, 0)
        a = c / n if n else float("nan")
        print(f"  {cls:<10} {n:>6} {a:>8.4f}")

    print("\nConfusion matrix (rows=gt, cols=pred):")
    header = " ".join(f"{c[:9]:>9}" for c in classes)
    print(f"  {'':<10} {header}")
    for gt in classes:
        cells = " ".join(f"{confusion[gt].get(p, 0):>9}" for p in classes)
        print(f"  {gt:<10} {cells}")

    print("\nPer-finding accuracy:")
    print(f"  {'finding':<26} {'n':>6} {'acc':>8}")
    for finding in sorted(per_finding):
        c, n = per_finding[finding]
        a = c / n if n else float("nan")
        print(f"  {finding:<26} {n:>6} {a:>8.4f}")


def run_eval(args, model, gold_df, device):
    if args.limit is not None:
        gold_df = gold_df.head(args.limit).reset_index(drop=True)

    # Two parallel sets of accumulators — one per scoring method.
    n_correct_cos = 0
    n_correct_l1 = 0
    n_seen = 0
    conf_cos = defaultdict(Counter)
    conf_l1 = defaultdict(Counter)
    per_finding_cos = defaultdict(lambda: [0, 0])  # [correct, total]
    per_finding_l1 = defaultdict(lambda: [0, 0])
    skipped = 0

    text_cache: Dict[str, Tuple] = {}

    print(f"\n[eval] running 5-way progression classification on {len(gold_df)} rows")
    print(f"[eval] phrases per class: " + ", ".join(
        f"{c}={len(PROGRESSION_PHRASES[c])}" for c in CLS_ORDER
    ))
    print(f"[eval] reporting BOTH cosine-argmax and Smooth-L1-argmin predictions")

    for i in range(len(gold_df)):
        row = gold_df.iloc[i]
        finding = str(row["finding"])
        gt_label = _normalize_label(row["progression"])
        try:
            prior = load_image_tensor(row["dataset"], row["parent_image_prev"], args.image_roots)
            current = load_image_tensor(row["dataset"], row["parent_image_curr"], args.image_roots)
        except (FileNotFoundError, OSError) as e:
            skipped += 1
            if skipped <= 5:
                print(f"[eval] skipping row {i} (missing image: {e})")
            continue

        out = score_one_pair(
            model, prior, current, finding, args.prompt_template, device,
            text_cache=text_cache,
        )
        cos_idx = max(range(len(CLS_ORDER)), key=lambda k: out["cos_class_scores"][k])
        l1_idx  = min(range(len(CLS_ORDER)), key=lambda k: out["l1_class_scores"][k])
        cos_pred = CLS_ORDER[cos_idx]
        l1_pred  = CLS_ORDER[l1_idx]

        n_seen += 1
        finding_lc = finding.lower()

        n_correct_cos += int(cos_pred == gt_label)
        conf_cos[gt_label][cos_pred] += 1
        per_finding_cos[finding_lc][1] += 1
        per_finding_cos[finding_lc][0] += int(cos_pred == gt_label)

        n_correct_l1 += int(l1_pred == gt_label)
        conf_l1[gt_label][l1_pred] += 1
        per_finding_l1[finding_lc][1] += 1
        per_finding_l1[finding_lc][0] += int(l1_pred == gt_label)

        if (i + 1) % max(1, len(gold_df) // 20) == 0:
            acc_cos = n_correct_cos / max(1, n_seen)
            acc_l1 = n_correct_l1 / max(1, n_seen)
            print(f"[eval]   {i + 1}/{len(gold_df)}  "
                  f"cos_acc={acc_cos:.4f}  l1_acc={acc_l1:.4f}  "
                  f"skipped={skipped}")

    if n_seen == 0:
        print("No samples evaluated.")
        return

    if skipped:
        print(f"\nSkipped {skipped} rows due to missing images")

    _print_eval_summary("Cosine (argmax slide-deck cos(Δẑ, Δz_true))",
                        n_correct_cos, n_seen, conf_cos, per_finding_cos)
    _print_eval_summary("Smooth L1 (argmin ‖ẑ_cur − z_cur‖_smooth_l1)",
                        n_correct_l1, n_seen, conf_l1, per_finding_l1)


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
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Two-slot positional template: {disease} is filled in first, "
             "{progression-phrase} second. Default: '{} is {}'. The phrase "
             "bank itself is hardcoded in PROGRESSION_PHRASES; this only "
             "controls how (disease, phrase) get joined into a sentence.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override an image root for one dataset. Can repeat. "
             "Example: --image-root mimic=/data/final_gold_mimic_images. "
             "By default, uses final_gold_<dataset>_images/ next to the "
             "gold parquet if present, falling back to IMAGE_ROOTS.",
    )

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

    # Sanity check the prompt template: must accept two positional slots.
    try:
        _ = args.prompt_template.format("test_disease", "test_phrase")
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"--prompt-template must accept two positional {{}} slots "
            f"({{disease}}, {{phrase}}). Got {args.prompt_template!r}: {e}"
        )

    # Resolve image roots: start from silver IMAGE_ROOTS, prefer
    # final_gold_<dataset>_images/ next to the parquet, then apply
    # any --image-root overrides last.
    parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    auto_gold_roots = discover_gold_image_roots(parquet_dir)
    image_roots: Dict[str, str] = {**IMAGE_ROOTS, **auto_gold_roots}
    if auto_gold_roots:
        print(f"[gold] auto-detected gold image roots:")
        for d, p in auto_gold_roots.items():
            print(f"  {d}: {p}")
    for spec in args.image_root:
        if "=" not in spec:
            raise ValueError(f"--image-root expects DATASET=PATH, got: {spec!r}")
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(
                f"--image-root dataset must be one of {DATASETS}, got {d!r}"
            )
        image_roots[d] = p
        print(f"[gold] override: {d} -> {p}")
    args.image_roots = image_roots

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
