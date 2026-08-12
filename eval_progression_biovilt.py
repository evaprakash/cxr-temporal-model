#!/usr/bin/env python3
"""5-way CheXTemporal gold progression eval with official BioViL-T.

Companion to ``eval_progression_jepa_perpatch.py`` / ``eval_progression_jepa.py``:
same gold parquet, same image roots, same optional
``--drop-multi-progression`` filter, same reported metrics (overall /
per-class / confusion / per-finding / macro / kappa).

Scoring matches the CheXTemporal zero-shot prompt-bank protocol and
``biovilt_progression_pairs.py``:

  1. Encode ``(current, prior)`` with pretrained BioViL-T
     ``MultiImageModel`` → L2-normalized global image embedding.
  2. Build the 37-phrase bank ``"{finding} is {phrase}"`` over the 5
     progression classes.
  3. Cosine-similarity the image embedding against every phrase;
     predicted class = class of the top-1 phrase.

No JEPA checkpoint — this is the vanilla BioViL-T baseline on the same
gold rows your JEPA eval uses.

Usage
-----
    python eval_progression_biovilt.py --eval
    python eval_progression_biovilt.py --eval --drop-multi-progression
    python eval_progression_biovilt.py --demo --idx 0
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dataset_combined_jepa import DEFAULT_FINDINGS
from eval_progression_jepa import _compute_balanced_metrics
from infer_jepa import IMAGE_ROOTS
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    _normalize_label,
    discover_gold_image_roots,
    drop_multi_progression_labels,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER, PROGRESSION_PHRASES
from tempcxr.modules.image_encoder_jepa import BioViLTImageEncoderJEPA
from tempcxr.modules.text_encoder import BioViLTTextEncoder

# CheXTemporal / BioViL-T phrase-bank template (no trailing period).
PROMPT_TEMPLATE = "{} is {}"


class BioViLTPairModel:
    """Thin holder for official BioViL-T image + text encoders."""

    def __init__(self, device: torch.device):
        print("[biovilt] loading official BioViL-T image encoder …")
        self.image_encoder = BioViLTImageEncoderJEPA(mode="biovilt").to(device)
        print("[biovilt] loading BioViL-T text encoder …")
        self.text_encoder = BioViLTTextEncoder(mode="biovilt").to(device)
        self.image_encoder.eval()
        self.text_encoder.eval()
        self.device = device


def build_phrase_bank(
    finding: str,
    template: str = PROMPT_TEMPLATE,
) -> Tuple[List[str], List[str]]:
    """Return (prompts, phrase_to_class) for one finding."""
    f = finding.strip().lower()
    prompts: List[str] = []
    phrase_to_class: List[str] = []
    for cls in CLS_ORDER:
        for phrase in PROGRESSION_PHRASES[cls]:
            prompts.append(template.format(f, phrase))
            phrase_to_class.append(cls)
    return prompts, phrase_to_class


@torch.no_grad()
def score_one_pair(
    model: BioViLTPairModel,
    prior_img: torch.Tensor,
    current_img: torch.Tensor,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, Tuple]] = None,
) -> Dict:
    """Image–text cosine over the 37-phrase bank; top-1 phrase wins."""
    cache_key = f"{finding}||{template}"
    if text_cache is not None and cache_key in text_cache:
        prompts, phrase_to_class, txt_global = text_cache[cache_key]
        txt_global = txt_global.to(device)
    else:
        prompts, phrase_to_class = build_phrase_bank(finding, template)
        txt_global, _, _ = model.text_encoder.forward_contrastive(prompts)
        txt_global = F.normalize(txt_global.float(), dim=-1)
        if text_cache is not None:
            text_cache[cache_key] = (
                prompts,
                phrase_to_class,
                txt_global.detach().cpu(),
            )

    prior = prior_img.unsqueeze(0).to(device)
    current = current_img.unsqueeze(0).to(device)
    # BioViL-T MultiImageModel: (current, previous)
    img_global, _ = model.image_encoder(current, prior)
    img_emb = F.normalize(img_global.float(), dim=-1)  # (1, 128)

    sims = (img_emb @ txt_global.T).squeeze(0)  # (n_phrases,)
    best_idx = int(sims.argmax().item())
    pred_class_name = phrase_to_class[best_idx]
    pred_class = CLS_ORDER.index(pred_class_name)

    # Per-class score = max phrase cosine within that class (for logging).
    cos_class_scores: List[float] = []
    for cls in CLS_ORDER:
        idxs = [i for i, c in enumerate(phrase_to_class) if c == cls]
        cos_class_scores.append(float(sims[idxs].max().item()))

    return {
        "prompts": prompts,
        "phrase_to_class": phrase_to_class,
        "phrase_sims": sims.detach().cpu().tolist(),
        "cos_class_scores": cos_class_scores,
        "pred_class": pred_class,
        "best_phrase": prompts[best_idx],
        "best_sim": float(sims[best_idx].item()),
    }


def _print_eval_summary(
    n_correct: int,
    n_seen: int,
    confusion: Dict,
    per_finding: Dict,
    cos_class_sums: List[float],
    classes: Optional[List[str]] = None,
):
    if classes is None:
        classes = CLS_ORDER
    n_classes = len(classes)
    chance = 1.0 / max(1, n_classes)

    print(f"\n{'=' * 60}")
    print("=== Results: 5-way BioViL-T image–text phrase-bank matching")
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
        "\nMean max-phrase cosine per candidate class "
        "(averaged across all eval samples):"
    )
    print(f"  {'class':<10} {'mean_cos':>10}")
    for k, cls in enumerate(classes):
        s = cos_class_sums[k] / max(1, n_seen)
        print(f"  {cls:<10} {s:>10.4f}")

    print("\nPer-finding accuracy:")
    print(f"  {'finding':<26} {'n':>6} {'acc':>8}")
    for finding in sorted(per_finding):
        c, n = per_finding[finding]
        a = c / n if n else float("nan")
        print(f"  {finding:<26} {n:>6} {a:>8.4f}")

    m = _compute_balanced_metrics(confusion, classes, n_correct)
    print("\nBalanced (imbalance-corrected) metrics:")
    print(
        f"  macro recall       {m['macro_recall']:>8.4f}   "
        "(= mean per-class accuracy / balanced accuracy)"
    )
    print(f"  macro precision    {m['macro_precision']:>8.4f}")
    print(f"  macro F1           {m['macro_f1']:>8.4f}")
    print(f"  Cohen's kappa      {m['cohen_kappa']:>8.4f}")
    print(
        f"  majority baseline  {m['majority_acc']:>8.4f}   "
        f"(always predict {m['majority_class']!r})"
    )

    print("\nPrediction vs ground-truth class frequencies:")
    total = m["total"]
    print(f"  {'class':<10} {'n_pred':>7} {'%pred':>7} {'n_true':>7} {'%true':>7}")
    for cls in classes:
        npred = m["n_pred"][cls]
        ntrue = m["n_true"][cls]
        pred_pct = 100.0 * npred / total if total else float("nan")
        true_pct = 100.0 * ntrue / total if total else float("nan")
        print(
            f"  {cls:<10} {npred:>7} {pred_pct:>6.1f}% "
            f"{ntrue:>7} {true_pct:>6.1f}%"
        )


def run_demo(args, model, gold_df, device):
    if args.idx is None:
        rng = random.Random(args.seed)
        args.idx = rng.randrange(len(gold_df))
    row = gold_df.iloc[args.idx]
    finding = str(row["finding"])
    gt_label = _normalize_label(row["progression"])

    print(f"\n=== Gold sample {args.idx} of {len(gold_df)} ===")
    print(f"  dataset:       {row['dataset']}")
    print(f"  patient:       {row['patient_id']}")
    print(f"  studies:       {row['study_id_prev']} → {row['study_id_curr']}")
    print(f"  finding:       {finding}")
    print(f"  gt progression:{gt_label}")

    prior = load_image_tensor(
        row["dataset"], row["parent_image_prev"], args.image_roots,
    )
    current = load_image_tensor(
        row["dataset"], row["parent_image_curr"], args.image_roots,
    )
    out = score_one_pair(
        model, prior, current, finding, args.prompt_template, device,
    )
    pred_label = CLS_ORDER[out["pred_class"]]
    print(f"\n  predicted:     {pred_label}  "
          f"(best phrase: {out['best_phrase']!r}, "
          f"sim={out['best_sim']:+.4f})")
    print(f"  correct:       {pred_label == gt_label}")
    print("\nPer-class max phrase cosine:")
    for k, cls in enumerate(CLS_ORDER):
        mark = " <-- pred" if k == out["pred_class"] else ""
        gt = " <-- gt" if cls == gt_label else ""
        print(f"  {cls:<10}  {out['cos_class_scores'][k]:>+.4f}{mark}{gt}")


def run_eval(args, model, gold_df, device):
    if args.limit is not None:
        gold_df = gold_df.head(args.limit).reset_index(drop=True)

    n_correct = 0
    n_seen = 0
    confusion: Dict[str, Counter] = defaultdict(Counter)
    per_finding: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    cos_class_sums = [0.0] * len(CLS_ORDER)
    skipped = 0
    text_cache: Dict[str, Tuple] = {}

    print(
        f"\n[eval] BioViL-T 5-way phrase-bank scoring on {len(gold_df)} rows"
    )
    print(f"[eval] template: {args.prompt_template!r}")

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
        pred_label = CLS_ORDER[out["pred_class"]]

        n_seen += 1
        finding_lc = finding.lower()
        n_correct += int(pred_label == gt_label)
        confusion[gt_label][pred_label] += 1
        per_finding[finding_lc][1] += 1
        per_finding[finding_lc][0] += int(pred_label == gt_label)
        for k in range(len(CLS_ORDER)):
            cos_class_sums[k] += out["cos_class_scores"][k]

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
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="Path to silver_findings.parquet (join for image paths "
             "if needed).",
    )
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--finding-col", default=None)
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Two-slot positional template (default: '{} is {}').",
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
        help="Override image root. Example: "
             "--image-root mimic=/data/final_gold_mimic_images",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--eval", action="store_true")

    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--drop-multi-progression",
        action="store_true",
        help="Drop (pair, finding) groups with >1 distinct progression "
             "label (~17%% of groups / ~31%% of rows on CheXTemporal gold).",
    )

    args = parser.parse_args()
    try:
        _ = args.prompt_template.format("test_disease", "test_class")
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"--prompt-template must accept two positional {{}} slots. "
            f"Got {args.prompt_template!r}: {e}"
        )

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
    model = BioViLTPairModel(device)
    gold_df = load_gold_pairs(
        args.gold_parquet,
        args.findings_parquet,
        label_col=args.label_col,
        finding_col=args.finding_col,
    )
    if args.drop_multi_progression:
        gold_df = drop_multi_progression_labels(gold_df)
    if len(gold_df) == 0:
        raise RuntimeError("No usable gold rows after filtering.")

    if args.demo:
        run_demo(args, model, gold_df, device)
    else:
        run_eval(args, model, gold_df, device)


if __name__ == "__main__":
    main()
