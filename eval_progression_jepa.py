"""5-way progression classification via image-image cosine matching.

This eval mirrors the training-time JEPA invariant exactly:
``cos(ẑ_cur, z_cur)`` on the unit sphere. Per gold row
``(prior_image, current_image, finding, gt_progression)``:

  1. Encode the prior image with the **online** image encoder, giving
     unit-norm patch features ``z_prior``.
  2. Encode the current image with the **EMA target** image encoder,
     giving unit-norm patch features ``z_cur`` (detached).
  3. For each of the 5 progression classes in ``CLS_ORDER`` build a
     single canonical templated prompt::

         "{Finding} is {class}."

     (matching the templated training condition format). Encode all 5
     prompts through the text encoder.
  4. Batch the predictor across the 5 prompts (same ``z_prior``
     expanded to a batch of 5) to obtain five candidate predictions
     ``ẑ_cur^c`` — one per progression class.
  5. Score each candidate by ``mean over patches of cos(ẑ_cur^c, z_cur)``.
  6. **Predicted class = argmax_c**.

The class whose prompt produced the predicted current latent closest
to the real current latent wins. This is the image-image inference
rule, not the image-text rule — the model is now being asked at test
time exactly the question it was trained on: "given this prior and
this change description, can you predict the current latent?"

Do-nothing baseline
-------------------
For each pair we also report ``cos(z_prior, z_cur)``. If every
``ẑ_cur^c`` scores below this baseline, the predictor's delta was
worse than predicting "no change" on this pair; if some classes beat
it and others don't, the per-class margins above the baseline are the
real discriminative signal.

Usage
-----
    # Sanity-check one gold row (random or specific --idx)
    python eval_progression_jepa.py --demo
    python eval_progression_jepa.py --demo --idx 17

    # Full 5-way eval over the gold parquet
    python eval_progression_jepa.py --eval
    python eval_progression_jepa.py --eval --limit 200

    # Custom checkpoint and per-dataset image roots
    python eval_progression_jepa.py --eval \
        --ckpt checkpoints_jepa_dynamic/epoch_30.pt \
        --image-root mimic=/data/final_gold_mimic_images
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dataset_combined_jepa import DEFAULT_FINDINGS
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    _normalize_label,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER
from tempcxr.modules.jepa import TempCXRJEPA


# ============================================================
# CONSTANTS
# ============================================================
# Single canonical template per class — matches the templated training
# condition exactly (capitalized finding, lowercase class, trailing
# period). No synonym ensembling — the whole point of this eval is
# train/test consistency.
PROMPT_TEMPLATE = "{} is {}."


# ============================================================
# PROMPT BUILDING
# ============================================================
def build_class_prompts(
    finding: str,
    template: str = PROMPT_TEMPLATE,
    classes: Optional[List[str]] = None,
) -> List[str]:
    """One templated prompt per class in ``classes`` (default: full
    ``CLS_ORDER`` for 5-way gold; pass a 3-way subset for benchmarks
    like MS-CXR-T / CIG which lack ``new`` and ``resolved``).

    Capitalizes the first letter of ``finding`` to match the templated
    training condition format built by
    ``JEPACombinedDataset._build_templated_condition``.
    """
    cls_list = CLS_ORDER if classes is None else list(classes)
    if not finding:
        return [template.format(finding, cls) for cls in cls_list]
    f_cap = finding[:1].upper() + finding[1:]
    return [template.format(f_cap, cls) for cls in cls_list]


# ============================================================
# CORE SCORING (image-image cosine)
# ============================================================
@torch.no_grad()
def _encode_prompts(
    model: TempCXRJEPA,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, Tuple]] = None,
    classes: Optional[List[str]] = None,
):
    """Encode the per-class prompts for one finding. Cached by
    ``(finding, template, classes)`` so different class subsets don't
    collide in the cache.

    The cache stores CPU tensors so we don't blow up GPU memory across
    many distinct findings; on each call we move them to ``device``.
    """
    cls_list = CLS_ORDER if classes is None else list(classes)
    cache_key = f"{finding}||{template}||{','.join(cls_list)}"
    if text_cache is not None and cache_key in text_cache:
        prompts, txt_local, token_mask = text_cache[cache_key]
        return prompts, txt_local.to(device), token_mask.to(device)

    prompts = build_class_prompts(finding, template, classes=cls_list)
    _, txt_local, token_mask = model.text_encoder.forward_contrastive(prompts)
    if text_cache is not None:
        text_cache[cache_key] = (
            prompts,
            txt_local.detach().cpu(),
            token_mask.detach().cpu(),
        )
    return prompts, txt_local, token_mask


@torch.no_grad()
def score_one_pair(
    model: TempCXRJEPA,
    prior_img: torch.Tensor,
    current_img: torch.Tensor,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, Tuple]] = None,
    classes: Optional[List[str]] = None,
) -> Dict:
    """N-way image-image scoring for ONE (prior, current, finding) row.

    With ``classes=None`` (the default) this runs the full 5-way gold
    eval. Passing a 3-element subset
    (``["improving", "stable", "worsening"]``) gives the 3-way variant
    used by MS-CXR-T / CIG.

    Returns
    -------
    prompts          : list[str] (length n_classes) — per-class templated prompts
    cos_class_scores : list[float] (length n_classes) — mean per-patch
                       ``cos(ẑ_cur^c, z_cur)`` for each class ``c``
    pred_class       : int — argmax over the class scores (index into
                       the supplied ``classes``, not into ``CLS_ORDER``)
    cos_naive        : float — mean per-patch ``cos(z_prior, z_cur)``,
                       a do-nothing baseline
    """
    prompts, txt_local, token_mask = _encode_prompts(
        model, finding, template, device, text_cache, classes=classes,
    )
    n_prompts = len(prompts)

    prior = prior_img.unsqueeze(0).to(device)
    current = current_img.unsqueeze(0).to(device)

    # Encode images once. Encoders already L2-normalize their outputs
    # along the feature dim, so ``z_prior`` and ``z_cur`` live on the
    # unit sphere.
    _, z_prior = model.image_encoder(prior)               # (1, N, D)
    _, z_cur = model.target_image_encoder(current)        # (1, N, D)
    z_cur = z_cur.detach()

    # Batch the predictor across all prompts by broadcasting the same
    # z_prior. delta_z differs per prompt because txt_local does.
    z_prior_b = z_prior.expand(n_prompts, -1, -1).contiguous()
    preds = model.predictor(z_prior_b, txt_local, token_mask)

    pred_f = preds.float()
    target_f = z_cur.float().expand_as(pred_f)
    cos_per_patch = F.cosine_similarity(pred_f, target_f, dim=-1)  # (n_prompts, N)
    cos_class_scores = cos_per_patch.mean(dim=1).tolist()

    # Do-nothing baseline: would a "predict z_prior" predictor have
    # scored higher than any of these? Useful sanity check.
    cos_naive = F.cosine_similarity(
        z_prior.float(), z_cur.float(), dim=-1,
    ).mean().item()

    pred_class = max(range(n_prompts), key=lambda k: cos_class_scores[k])

    return {
        "prompts": prompts,
        "cos_class_scores": cos_class_scores,
        "pred_class": pred_class,
        "cos_naive": cos_naive,
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

    prior = load_image_tensor(
        row["dataset"], row["parent_image_prev"], args.image_roots,
    )
    current = load_image_tensor(
        row["dataset"], row["parent_image_curr"], args.image_roots,
    )

    out = score_one_pair(
        model, prior, current, finding, args.prompt_template, device,
    )
    best = out["pred_class"]
    pred_label = CLS_ORDER[best]
    naive = out["cos_naive"]

    print("\nPer-class cos(ẑ_cur^c, z_cur) (mean over patches):")
    print(f"  {'class':<10}  {'prompt':<40}  {'cos':>8}  {'Δ vs naive':>10}")
    for k, cls in enumerate(CLS_ORDER):
        marker = "  <-- argmax" if k == best else ""
        score = out["cos_class_scores"][k]
        delta = score - naive
        print(
            f"  {cls:<10}  {out['prompts'][k]:<40}  "
            f"{score:>+8.4f}  {delta:>+10.4f}{marker}"
        )
    print(
        f"\n  do-nothing baseline (cos(z_prior, z_cur)) = {naive:+.4f}"
    )
    print(
        f"\n  Prediction: {pred_label:<10}  vs gt {gt_label:<10}  "
        f"=> {'CORRECT' if pred_label == gt_label else 'WRONG'}"
    )


# ============================================================
# EVAL MODE
# ============================================================
def _compute_balanced_metrics(
    confusion: Dict,
    classes: List[str],
    n_correct: int,
) -> Dict:
    """Compute imbalance-corrected metrics from a confusion matrix.

    Follows the sklearn convention: when a class is never predicted
    (precision undefined) or has no ground-truth rows (recall undefined),
    that quantity is treated as 0 for the purpose of macro-averaging and
    F1. This means macro F1 penalizes never-predicting a class rather
    than silently dropping it from the average — important because
    "always predict the majority class" should score low here.

    Cohen's kappa uses the standard formula::

        kappa = (p_o - p_e) / (1 - p_e)

    where p_o is the observed agreement (= overall accuracy) and p_e is
    the agreement expected by chance given the marginal distributions of
    ground truth and predictions. kappa = 0 means "no better than a
    classifier that picks each class independently with the observed
    marginal frequency" — i.e. it directly punishes the
    "always-predict-majority" shortcut.
    """
    n_true = {gt: sum(confusion[gt].values()) for gt in classes}
    n_pred = {p: sum(confusion[gt].get(p, 0) for gt in classes) for p in classes}
    total = sum(n_true.values())

    per_class_recall: List[float] = []
    per_class_precision: List[float] = []
    per_class_f1: List[float] = []
    for cls in classes:
        tp = confusion[cls].get(cls, 0)
        rec = tp / n_true[cls] if n_true[cls] else 0.0
        prec = tp / n_pred[cls] if n_pred[cls] else 0.0
        if (prec + rec) > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        else:
            f1 = 0.0
        per_class_recall.append(rec)
        per_class_precision.append(prec)
        per_class_f1.append(f1)

    n = max(1, len(classes))
    macro_recall = sum(per_class_recall) / n
    macro_precision = sum(per_class_precision) / n
    macro_f1 = sum(per_class_f1) / n

    if total > 0:
        p_o = n_correct / total
        p_e = sum(
            (n_true[c] / total) * (n_pred[c] / total) for c in classes
        )
        kappa = (
            (p_o - p_e) / (1.0 - p_e)
            if abs(1.0 - p_e) > 1e-12
            else 0.0
        )
    else:
        kappa = float("nan")

    if total > 0:
        majority_class = max(classes, key=lambda c: n_true[c])
        majority_acc = n_true[majority_class] / total
    else:
        majority_class = ""
        majority_acc = float("nan")

    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "total": total,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "macro_recall": macro_recall,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "cohen_kappa": kappa,
        "majority_class": majority_class,
        "majority_acc": majority_acc,
    }


def _print_eval_summary(
    n_correct: int,
    n_seen: int,
    confusion: Dict,
    per_finding: Dict,
    cos_class_sums: List[float],
    naive_sum: float,
    n_above_naive: int,
    classes: Optional[List[str]] = None,
):
    if classes is None:
        classes = CLS_ORDER
    n_classes = len(classes)
    chance = 1.0 / max(1, n_classes)

    print(f"\n{'=' * 60}")
    print(f"=== Results: {n_classes}-way image-image cosine matching")
    print(f"{'=' * 60}")
    acc = n_correct / max(1, n_seen)
    print(
        f"Overall accuracy: {n_correct}/{n_seen} = {acc:.4f}    "
        f"(chance = {chance:.3f})"
    )

    print("\nPer-class accuracy (= per-class recall):")
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

    print(
        "\nMean cos(ẑ_cur^c, z_cur) per candidate class "
        "(averaged across all eval samples):"
    )
    print(f"  {'class':<10} {'mean_cos':>10}")
    for k, cls in enumerate(classes):
        s = cos_class_sums[k] / max(1, n_seen)
        print(f"  {cls:<10} {s:>10.4f}")
    naive_mean = naive_sum / max(1, n_seen)
    above_pct = 100.0 * n_above_naive / max(1, n_seen)
    print(f"  {'naive':<10} {naive_mean:>10.4f}  (cos(z_prior, z_cur))")
    print(
        f"\n  Pairs where argmax-c cos(ẑ_cur^c, z_cur) > "
        f"cos(z_prior, z_cur): {n_above_naive}/{n_seen} ({above_pct:.1f}%)"
    )
    print(
        "  (Higher = predictor's text-conditioned best guess beats "
        "the do-nothing baseline on more pairs.)"
    )

    print("\nPer-finding accuracy:")
    print(f"  {'finding':<26} {'n':>6} {'acc':>8}")
    for finding in sorted(per_finding):
        c, n = per_finding[finding]
        a = c / n if n else float("nan")
        print(f"  {finding:<26} {n:>6} {a:>8.4f}")

    # --------------------------------------------------------
    # Imbalance-corrected metrics (ADDED — does not replace any
    # of the per-class / confusion-matrix reporting above).
    #
    # Overall accuracy is misleading on class-imbalanced benchmarks
    # like MS-CXR-T (improving / stable / worsening ≈ 18 / 40 / 42 %):
    # a model that just always predicts "worsening" scores ~42% raw.
    # Use macro recall / macro F1 / Cohen's kappa for fair comparison.
    # --------------------------------------------------------
    m = _compute_balanced_metrics(confusion, classes, n_correct)

    print("\nBalanced (imbalance-corrected) metrics:")
    print(
        f"  macro recall       {m['macro_recall']:>8.4f}   "
        "(= mean per-class accuracy / balanced accuracy)"
    )
    print(f"  macro precision    {m['macro_precision']:>8.4f}")
    print(f"  macro F1           {m['macro_f1']:>8.4f}")
    print(
        f"  Cohen's kappa      {m['cohen_kappa']:>8.4f}   "
        "(0 = chance given marginals, 1 = perfect)"
    )
    if m["total"] > 0:
        print(
            f"  majority baseline  {m['majority_acc']:>8.4f}   "
            f"(always predict {m['majority_class']!r} "
            "on this label distribution)"
        )

    print("\nPer-class precision / recall / F1:")
    print(
        f"  {'class':<10} {'n_gt':>6} {'precision':>10} "
        f"{'recall':>8} {'F1':>8}"
    )
    for k, cls in enumerate(classes):
        print(
            f"  {cls:<10} {m['n_true'][cls]:>6} "
            f"{m['per_class_precision'][k]:>10.4f} "
            f"{m['per_class_recall'][k]:>8.4f} "
            f"{m['per_class_f1'][k]:>8.4f}"
        )

    print(
        "\nPredicted vs true class distribution "
        "(diagnoses imbalance / over-prediction bias):"
    )
    print(
        f"  {'class':<10} {'n_pred':>7} {'pred%':>7} "
        f"{'n_true':>7} {'true%':>7}"
    )
    total = m["total"]
    for cls in classes:
        npred = m["n_pred"][cls]
        ntrue = m["n_true"][cls]
        pred_pct = 100.0 * npred / total if total else float("nan")
        true_pct = 100.0 * ntrue / total if total else float("nan")
        print(
            f"  {cls:<10} {npred:>7} {pred_pct:>6.1f}% "
            f"{ntrue:>7} {true_pct:>6.1f}%"
        )


def run_eval(args, model, gold_df, device):
    if args.limit is not None:
        gold_df = gold_df.head(args.limit).reset_index(drop=True)

    n_correct = 0
    n_seen = 0
    n_above_naive = 0
    confusion: Dict[str, Counter] = defaultdict(Counter)
    per_finding: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    cos_class_sums = [0.0] * len(CLS_ORDER)
    naive_sum = 0.0
    skipped = 0

    text_cache: Dict[str, Tuple] = {}

    print(
        f"\n[eval] running 5-way image-image cosine matching on "
        f"{len(gold_df)} rows"
    )
    print(
        f"[eval] one prompt per class (template: {args.prompt_template!r})"
    )

    for i in range(len(gold_df)):
        row = gold_df.iloc[i]
        finding = str(row["finding"])
        gt_label = _normalize_label(row["progression"])
        try:
            prior = load_image_tensor(
                row["dataset"], row["parent_image_prev"], args.image_roots,
            )
            current = load_image_tensor(
                row["dataset"], row["parent_image_curr"], args.image_roots,
            )
        except (FileNotFoundError, OSError) as e:
            skipped += 1
            if skipped <= 5:
                print(f"[eval] skipping row {i} (missing image: {e})")
            continue

        out = score_one_pair(
            model, prior, current, finding, args.prompt_template, device,
            text_cache=text_cache,
        )
        pred_idx = out["pred_class"]
        pred_label = CLS_ORDER[pred_idx]

        n_seen += 1
        finding_lc = finding.lower()
        n_correct += int(pred_label == gt_label)
        confusion[gt_label][pred_label] += 1
        per_finding[finding_lc][1] += 1
        per_finding[finding_lc][0] += int(pred_label == gt_label)

        best_cos = out["cos_class_scores"][pred_idx]
        if best_cos > out["cos_naive"]:
            n_above_naive += 1

        for k in range(len(CLS_ORDER)):
            cos_class_sums[k] += out["cos_class_scores"][k]
        naive_sum += out["cos_naive"]

        if (i + 1) % max(1, len(gold_df) // 20) == 0:
            acc = n_correct / max(1, n_seen)
            print(
                f"[eval]   {i + 1}/{len(gold_df)}  "
                f"acc={acc:.4f}  skipped={skipped}"
            )

    if n_seen == 0:
        print("No samples evaluated.")
        return

    if skipped:
        print(f"\nSkipped {skipped} rows due to missing images")

    _print_eval_summary(
        n_correct=n_correct,
        n_seen=n_seen,
        confusion=confusion,
        per_finding=per_finding,
        cos_class_sums=cos_class_sums,
        naive_sum=naive_sum,
        n_above_naive=n_above_naive,
    )


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        default=os.environ.get(
            "JEPA_CKPT", "checkpoints_jepa_dynamic/best.pt"
        ),
        help="Path to a JEPA checkpoint "
             "(default: checkpoints_jepa_dynamic/best.pt — matches the "
             "current main training default).",
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
        help="Override the gold-parquet column name for the "
             "finding/disease. Auto-detected by default.",
    )
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Two-slot positional template: {finding} is filled in first, "
             "{class} second. Default: '{} is {}.' (matches the templated "
             "training condition format).",
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
             "Example: --image-root mimic=/data/final_gold_mimic_images. "
             "Defaults: final_gold_<dataset>_images/ next to the gold "
             "parquet if present, else IMAGE_ROOTS from infer_jepa.py.",
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

    # Sanity-check the prompt template: must accept two positional slots.
    try:
        _ = args.prompt_template.format("test_disease", "test_class")
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"--prompt-template must accept two positional {{}} slots "
            f"({{disease}}, {{class}}). Got {args.prompt_template!r}: {e}"
        )

    # Resolve image roots: start from silver IMAGE_ROOTS, prefer
    # final_gold_<dataset>_images/ next to the parquet, then apply
    # any --image-root overrides last.
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
