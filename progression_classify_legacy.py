"""Pre-projection-heads version of ``progression_classify.py`` (eval-only).

Use this script to fairly evaluate checkpoints trained on commits
``<= bc00c68`` (i.e. *before* ``618f374`` added the projection heads).
The current ``progression_classify.py`` cannot evaluate those
checkpoints because the new ``TempCXRJEPA`` class has
``proj_clip`` / ``proj_jepa`` / ``target_proj_jepa`` heads that are
not in the older checkpoints; loading with ``strict=False`` leaves
those heads at a random initialization, which then scrambles the
predictor's input and produces random-looking eval numbers (different
on every rerun, depending on the random init).

This script loads the pre-projheads checkpoint into ``TempCXRJEPALegacy``
(``tempcxr/modules/jepa_legacy.py``), which has the exact same key
set as the pre-projheads checkpoints, then runs the same 5-way
progression scoring (cosine + Smooth L1) as ``progression_classify.py``.

For checkpoints trained **after** ``618f374``, use the regular
``progression_classify.py``.

Usage
-----
    # 5-way eval on the whole gold parquet
    JEPA_CKPT=/path/to/pre-projheads/best.pt \\
        python progression_classify_legacy.py --eval \\
        --image-root mimic=/path/to/final_gold_mimic_images \\
        --image-root chexpert=/path/to/final_gold_chexpert_images \\
        --image-root rexgradient=/path/to/final_gold_rexgradient_images

    # Single-row demo
    JEPA_CKPT=/path/to/pre-projheads/best.pt \\
        python progression_classify_legacy.py --demo --idx 0 \\
        --image-root mimic=/path/to/final_gold_mimic_images \\
        --image-root chexpert=/path/to/final_gold_chexpert_images \\
        --image-root rexgradient=/path/to/final_gold_rexgradient_images
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dataset_combined_jepa import DEFAULT_DATASET_DIR, DEFAULT_FINDINGS
from infer_jepa import IMAGE_ROOTS
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    PROMPT_TEMPLATE,
    _normalize_label,
    _print_eval_summary,
    build_class_prompts,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER, PROGRESSION_PHRASES
from tempcxr.modules.jepa_legacy import TempCXRJEPALegacy


# ============================================================
# LOAD LEGACY MODEL
# ============================================================
def load_jepa_legacy_model(
    ckpt_path: str,
    device: torch.device,
) -> TempCXRJEPALegacy:
    """Build ``TempCXRJEPALegacy`` and load ``ckpt_path`` into it.

    Pre-projheads checkpoints should have zero missing keys; any
    missing keys are reported as a warning. Post-projheads
    checkpoints' projection-head weights will show up as
    ``unexpected`` and are silently ignored — but loading a
    post-projheads checkpoint into this class drops the projection
    heads entirely, which is almost certainly not what you want.
    """
    print(f"[infer-legacy] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise ValueError(
            f"{ckpt_path} does not look like a JEPA checkpoint "
            f"(missing 'model' key). Top-level keys: {list(ckpt.keys())}"
        )
    print(
        f"[infer-legacy]   epoch={ckpt.get('epoch', '?')}  "
        f"best_val_loss={ckpt.get('best_val_loss', '?')}"
    )

    model = TempCXRJEPALegacy()
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)

    # Pre-projheads checkpoints should produce 0 truly-missing keys.
    # BatchNorm's ``num_batches_tracked`` is a non-issue if it shows
    # up either side, so we filter it out before warning.
    def _is_bn_tracking(k: str) -> bool:
        return k.endswith("num_batches_tracked")

    nontrivial_missing = [k for k in missing if not _is_bn_tracking(k)]
    if nontrivial_missing:
        print(
            f"[infer-legacy] WARNING: {len(nontrivial_missing)} unexpected "
            f"MISSING keys (first 5: {nontrivial_missing[:5]}). This "
            f"usually means the checkpoint is NOT a pre-projheads "
            f"checkpoint, or the architecture has diverged. The eval "
            f"numbers below may be unreliable."
        )

    proj_unexpected = [
        k for k in unexpected
        if k.startswith(("proj_clip", "proj_jepa", "target_proj_jepa"))
    ]
    other_unexpected = [k for k in unexpected if k not in proj_unexpected]
    if proj_unexpected:
        print(
            f"[infer-legacy] WARNING: checkpoint contains "
            f"{len(proj_unexpected)} projection-head keys "
            f"(proj_clip / proj_jepa / target_proj_jepa). This looks "
            f"like a POST-projheads checkpoint and the projection heads "
            f"will be DROPPED. Use ``progression_classify.py`` (not the "
            f"legacy version) to evaluate post-projheads checkpoints."
        )
    if other_unexpected:
        print(
            f"[infer-legacy] WARNING: {len(other_unexpected)} other "
            f"unexpected keys in checkpoint "
            f"(first 5: {other_unexpected[:5]})"
        )

    model.to(device).eval()
    return model


# ============================================================
# TEXT ENCODING (local copy — same logic as progression_classify.py)
# ============================================================
@torch.no_grad()
def _encode_prompts(
    model: TempCXRJEPALegacy,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[Tuple[str, str], Tuple]] = None,
) -> Tuple[List[str], List[int], torch.Tensor, torch.Tensor]:
    """Build per-class prompts and run them through the text encoder.

    Cached by (finding, template) since the same finding shows up in
    many gold rows; one text-encoder forward per cached entry.
    """
    cache_key = (finding.strip().lower(), template)
    if text_cache is not None and cache_key in text_cache:
        return text_cache[cache_key]

    prompts, class_idx = build_class_prompts(finding, template=template)
    _, txt_local, token_mask = model.text_encoder.forward_contrastive(prompts)
    txt_local = txt_local.to(device)
    token_mask = token_mask.to(device)

    entry = (prompts, class_idx, txt_local, token_mask)
    if text_cache is not None:
        text_cache[cache_key] = entry
    return entry


# ============================================================
# 5-WAY SCORING (pre-projheads geometry: LN'd encoder output, no heads)
# ============================================================
@torch.no_grad()
def score_one_pair(
    model: TempCXRJEPALegacy,
    prior_img: torch.Tensor,
    current_img: torch.Tensor,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[Tuple[str, str], Tuple]] = None,
) -> Dict:
    """Pre-projheads 5-way scoring for ONE (prior, current, finding).

    Matches the pre-``618f374`` forward geometry exactly:

        z_prior    = LN(image_encoder(prior))
        z_cur      = LN(target_image_encoder(current))    (detached)
        preds[k]   = predictor(z_prior, prompt_k)

    No projection heads anywhere.

    Returns the same dict shape as
    ``progression_classify.score_one_pair`` so ``_print_eval_summary``
    can consume it.
    """
    prompts, class_idx, all_txt, all_mask = _encode_prompts(
        model, finding, template, device, text_cache,
    )
    n_phrases = len(prompts)

    prior = prior_img.unsqueeze(0).to(device)
    current = current_img.unsqueeze(0).to(device)

    # Online encoder + LN on prior (no proj head).
    _, prior_raw = model.image_encoder(prior)
    z_prior = F.layer_norm(prior_raw, (prior_raw.size(-1),))

    # Target (EMA) encoder + LN + detach on current (no proj head).
    _, curr_raw = model.target_image_encoder(current)
    z_cur = F.layer_norm(curr_raw, (curr_raw.size(-1),)).detach()

    # Predictor (batched across phrases).
    z_prior_b = z_prior.expand(n_phrases, -1, -1).contiguous()
    preds = model.predictor(z_prior_b, all_txt, all_mask)

    # Per-phrase metrics.
    pred_f = preds.float()
    target_f = z_cur.float().expand_as(pred_f)
    prior_f = z_prior.float().expand_as(pred_f)

    smooth_l1 = F.smooth_l1_loss(
        pred_f, target_f, reduction="none",
    ).mean(dim=(1, 2))

    dpred = (pred_f - prior_f).flatten(start_dim=1)
    dtrue = (target_f - prior_f).flatten(start_dim=1)
    cos_delta = F.cosine_similarity(dpred, dtrue, dim=1)

    phrase_cos = cos_delta.cpu().tolist()
    phrase_l1 = smooth_l1.cpu().tolist()

    # Aggregate per class (mean of phrase scores).
    n_classes = len(CLS_ORDER)
    cos_sum = [0.0] * n_classes
    l1_sum = [0.0] * n_classes
    counts = [0] * n_classes
    for k in range(n_phrases):
        c = class_idx[k]
        cos_sum[c] += phrase_cos[k]
        l1_sum[c] += phrase_l1[k]
        counts[c] += 1
    cos_class_scores = [
        cos_sum[c] / max(1, counts[c]) for c in range(n_classes)
    ]
    l1_class_scores = [
        l1_sum[c] / max(1, counts[c]) for c in range(n_classes)
    ]

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
        raise IndexError(
            f"--idx {args.idx} out of range [0, {len(gold_df)})"
        )

    row = gold_df.iloc[args.idx]
    finding = str(row["finding"])
    gt_label = _normalize_label(row["progression"])

    print(f"\n=== Gold sample {args.idx} of {len(gold_df)} (legacy eval) ===")
    print(f"  dataset:       {row['dataset']}")
    print(f"  patient_id:    {row['patient_id']}")
    print(f"  study_id_prev: {row['study_id_prev']}")
    print(f"  study_id_curr: {row['study_id_curr']}")
    print(f"  finding:       {finding}")
    print(f"  ground-truth:  {gt_label}")

    prior = load_image_tensor(
        row["dataset"], row["parent_image_prev"], args.image_roots,
    )
    current = load_image_tensor(
        row["dataset"], row["parent_image_curr"], args.image_roots,
    )

    out = score_one_pair(
        model, prior, current, finding, args.prompt_template, device,
    )

    n_phrases_per_class = [len(PROGRESSION_PHRASES[c]) for c in CLS_ORDER]
    cos_best = max(
        range(len(CLS_ORDER)),
        key=lambda k: out["cos_class_scores"][k],
    )
    l1_best = min(
        range(len(CLS_ORDER)),
        key=lambda k: out["l1_class_scores"][k],
    )

    print("\nPer-class scores (averaged across phrases of each class):")
    print(f"  {'class':<10} {'#phrases':>9} {'cos_score':>11} {'L1_score':>11}")
    for k, cls in enumerate(CLS_ORDER):
        cos_marker = "  <-- argmax cos" if k == cos_best else ""
        l1_marker = "  <-- argmin L1" if k == l1_best else ""
        print(
            f"  {cls:<10} {n_phrases_per_class[k]:>9} "
            f"{out['cos_class_scores'][k]:>11.4f} "
            f"{out['l1_class_scores'][k]:>11.4f}"
            f"{cos_marker}{l1_marker}"
        )

    cos_pred = CLS_ORDER[cos_best]
    l1_pred = CLS_ORDER[l1_best]
    print(
        f"\n  Cosine prediction:    {cos_pred:<10}  vs gt {gt_label:<10}  "
        f"=> {'CORRECT' if cos_pred == gt_label else 'WRONG'}"
    )
    print(
        f"  Smooth L1 prediction: {l1_pred:<10}  vs gt {gt_label:<10}  "
        f"=> {'CORRECT' if l1_pred == gt_label else 'WRONG'}"
    )


# ============================================================
# EVAL MODE
# ============================================================
def run_eval(args, model, gold_df, device):
    if args.limit is not None:
        gold_df = gold_df.head(args.limit).reset_index(drop=True)

    n_correct_cos = 0
    n_correct_l1 = 0
    n_seen = 0
    conf_cos = defaultdict(Counter)
    conf_l1 = defaultdict(Counter)
    per_finding_cos = defaultdict(lambda: [0, 0])
    per_finding_l1 = defaultdict(lambda: [0, 0])
    skipped = 0

    text_cache: Dict[Tuple[str, str], Tuple] = {}

    print(
        f"\n[eval-legacy] running 5-way progression classification on "
        f"{len(gold_df)} rows"
    )
    print(
        "[eval-legacy] phrases per class: "
        + ", ".join(
            f"{c}={len(PROGRESSION_PHRASES[c])}" for c in CLS_ORDER
        )
    )
    print(
        "[eval-legacy] reporting BOTH cosine-argmax and "
        "Smooth-L1-argmin predictions"
    )

    for i in range(len(gold_df)):
        row = gold_df.iloc[i]
        finding = str(row["finding"])
        gt_label = _normalize_label(row["progression"])
        try:
            prior = load_image_tensor(
                row["dataset"],
                row["parent_image_prev"],
                args.image_roots,
            )
            current = load_image_tensor(
                row["dataset"],
                row["parent_image_curr"],
                args.image_roots,
            )
        except (FileNotFoundError, OSError) as e:
            skipped += 1
            if skipped <= 5:
                print(f"[eval-legacy] skipping row {i} (missing image: {e})")
            continue

        out = score_one_pair(
            model, prior, current, finding,
            args.prompt_template, device,
            text_cache=text_cache,
        )
        cos_idx = max(
            range(len(CLS_ORDER)),
            key=lambda k: out["cos_class_scores"][k],
        )
        l1_idx = min(
            range(len(CLS_ORDER)),
            key=lambda k: out["l1_class_scores"][k],
        )
        cos_pred = CLS_ORDER[cos_idx]
        l1_pred = CLS_ORDER[l1_idx]

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
            print(
                f"[eval-legacy]   {i + 1}/{len(gold_df)}  "
                f"cos_acc={acc_cos:.4f}  l1_acc={acc_l1:.4f}  "
                f"skipped={skipped}"
            )

    if n_seen == 0:
        print("No samples evaluated.")
        return

    if skipped:
        print(f"\nSkipped {skipped} rows due to missing images")

    _print_eval_summary(
        "Cosine (argmax slide-deck cos(Δẑ, Δz_true))",
        n_correct_cos, n_seen, conf_cos, per_finding_cos,
    )
    _print_eval_summary(
        "Smooth L1 (argmin ‖ẑ_cur − z_cur‖_smooth_l1)",
        n_correct_l1, n_seen, conf_l1, per_finding_l1,
    )


# ============================================================
# MAIN — argparse identical to progression_classify.main()
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pre-projection-heads 5-way progression classification eval. "
            "Use this for checkpoints trained before commit 618f374 "
            "(no proj_clip / proj_jepa / target_proj_jepa heads)."
        ),
    )
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
        help="Path to silver_findings.parquet (used to join image "
             "paths if the gold parquet does not have them).",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Override the gold-parquet column name for the "
             "progression label. Auto-detected by default.",
    )
    parser.add_argument(
        "--finding-col",
        default=None,
        help="Override the gold-parquet column name for the finding/"
             "disease. Auto-detected by default.",
    )
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Two-slot positional template: {disease} is filled in "
             "first, {progression-phrase} second. Default: '{} is {}'.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override an image root for one dataset. Can repeat. "
             "Example: --image-root mimic=/data/final_gold_mimic_images.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--demo", action="store_true",
        help="Print 5-way scoring for one gold row.",
    )
    mode.add_argument(
        "--eval", action="store_true",
        help="Compute overall + per-class + per-finding accuracy.",
    )

    parser.add_argument(
        "--idx", type=int, default=None,
        help="(demo only) gold-row index. Default: random.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="(demo only) RNG seed for picking a random row.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="(eval only) Only evaluate the first N rows.",
    )

    args = parser.parse_args()

    # Sanity check the prompt template: must accept two positional slots.
    try:
        _ = args.prompt_template.format("test_disease", "test_phrase")
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"--prompt-template must accept two positional {{}} slots "
            f"({{disease}}, {{phrase}}). Got {args.prompt_template!r}: {e}"
        )

    # Resolve image roots (same logic as progression_classify.main).
    parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    auto_gold_roots = discover_gold_image_roots(parquet_dir)
    image_roots: Dict[str, str] = {**IMAGE_ROOTS, **auto_gold_roots}
    if auto_gold_roots:
        print("[gold] auto-detected gold image roots:")
        for d, p in auto_gold_roots.items():
            print(f"  {d}: {p}")
    for spec in args.image_root:
        if "=" not in spec:
            raise ValueError(
                f"--image-root expects DATASET=PATH, got: {spec!r}"
            )
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(
                f"--image-root dataset must be one of {DATASETS}, got {d!r}"
            )
        image_roots[d] = p
        print(f"[gold] override: {d} -> {p}")
    args.image_roots = image_roots

    device = torch.device(args.device)
    model = load_jepa_legacy_model(args.ckpt, device)
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
