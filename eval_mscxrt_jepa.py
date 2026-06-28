"""3-way progression classification on MS-CXR-T via image-image cosine.

The MS-CXR-T 3-way analog of ``eval_progression_jepa.py``. Both eval
scripts share the same scoring rule — ``cos(ẑ_cur^c, z_cur)`` on the
unit sphere, argmax over candidate progression classes — but they read
different label sources:

  * ``eval_progression_jepa.py``  — 5-way ``gold_progression_pairs.parquet``
  * ``eval_mscxrt_jepa.py`` (this) — 3-way MS-CXR-T CSV (improving /
                                     stable / worsening only)

Per CSV row ``(prior_image, current_image, finding, gt_progression)``:

  1. Encode the prior image with the **online** image encoder, giving
     unit-norm patch features ``z_prior``.
  2. Encode the current image with the **EMA target** image encoder,
     giving unit-norm patch features ``z_cur`` (detached).
  3. Build 3 templated prompts — one per valid class — using the same
     canonical format as the templated training condition:
     ``"{Finding} is {class}."``.
  4. Batch the predictor with the same ``z_prior`` and the 3 different
     text prompts to obtain three candidate ``ẑ_cur^c``.
  5. Score each by ``mean over patches of cos(ẑ_cur^c, z_cur)``.
  6. **Predicted class = argmax_c** over the 3 cosines.

Unlike the legacy ``eval_mscxrt.py`` (which scores predicted patches
against text prompts — image-text cosine), this script scores predicted
patches against the *actual* current latent — image-image cosine — so
the test-time question matches the JEPA training loss exactly.

Also reports the do-nothing baseline ``cos(z_prior, z_cur)`` and the
fraction of pairs where the argmax predicted class beats it.

Expected CSV schema (same as the legacy ``eval_mscxrt.py``):

    patient_id, study_id_prev, study_id_curr,
    img_path_prev, img_path_curr, disease_name, comparison

with absolute MIMIC-CXR-JPG paths in ``img_path_prev`` / ``img_path_curr``.

Usage
-----
    # Sanity-check one row
    python eval_mscxrt_jepa.py --demo
    python eval_mscxrt_jepa.py --demo --idx 17

    # Full 3-way eval over the CSV
    python eval_mscxrt_jepa.py --eval
    python eval_mscxrt_jepa.py --eval --limit 200

    # Custom checkpoint / CSV path
    python eval_mscxrt_jepa.py --eval \\
        --ckpt checkpoints_jepa_dynamic/epoch_30.pt \\
        --csv /path/to/mscxrt_labels_new.csv
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import torch

from csv_progression_eval import load_csv_pairs, load_image_tensor_abs
from eval_progression_jepa import (
    PROMPT_TEMPLATE,
    _print_eval_summary,
    score_one_pair,
)
from infer_jepa import load_jepa_model


BENCHMARK_NAME = "MS-CXR-T"
DEFAULT_CSV = os.environ.get("MSCXRT_CSV", "mscxrt_labels_new.csv")

# Same label mapping as ``eval_mscxrt.py``: present-tense canonicals
# plus a handful of synonyms so a CSV regenerated with slightly
# different vocab still works.
LABEL_MAP = {
    "improving": "improving",
    "improved":  "improving",
    "stable":    "stable",
    "no change": "stable",
    "no-change": "stable",
    "unchanged": "stable",
    "worsening": "worsening",
    "worsened":  "worsening",
}

# MS-CXR-T has no ``new`` or ``resolved`` cases — restrict to 3 classes.
VALID_CLASSES = ["improving", "stable", "worsening"]


# ============================================================
# DEMO MODE
# ============================================================
def run_demo(args, model, df, device):
    if args.idx is None:
        rng = random.Random(args.seed)
        args.idx = rng.randrange(len(df))
    if not (0 <= args.idx < len(df)):
        raise IndexError(f"--idx {args.idx} out of range [0, {len(df)})")

    row = df.iloc[args.idx]
    finding = str(row["finding"])
    gt_label = str(row["progression"])

    print(f"\n=== {BENCHMARK_NAME} sample {args.idx} of {len(df)} ===")
    print(f"  patient_id:    {row['patient_id']}")
    print(f"  study_id_prev: {row['study_id_prev']}")
    print(f"  study_id_curr: {row['study_id_curr']}")
    print(f"  finding:       {finding}")
    print(f"  ground-truth:  {gt_label}")

    prior = load_image_tensor_abs(row["img_path_prev"])
    current = load_image_tensor_abs(row["img_path_curr"])

    out = score_one_pair(
        model, prior, current, finding, args.prompt_template, device,
        classes=VALID_CLASSES,
    )
    best = out["pred_class"]
    pred_label = VALID_CLASSES[best]
    naive = out["cos_naive"]

    print("\nPer-class cos(ẑ_cur^c, z_cur) (mean over patches):")
    print(f"  {'class':<10}  {'prompt':<40}  {'cos':>8}  {'Δ vs naive':>10}")
    for k, cls in enumerate(VALID_CLASSES):
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
def run_eval(args, model, df, device):
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)

    n_correct = 0
    n_seen = 0
    n_above_naive = 0
    confusion: Dict[str, Counter] = defaultdict(Counter)
    per_finding: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    cos_class_sums = [0.0] * len(VALID_CLASSES)
    naive_sum = 0.0
    skipped = 0

    text_cache: Dict[str, Tuple] = {}

    print(
        f"\n[{BENCHMARK_NAME}] running {len(VALID_CLASSES)}-way image-image "
        f"cosine matching on {len(df)} rows"
    )
    print(f"[{BENCHMARK_NAME}] valid classes: {VALID_CLASSES}")
    print(
        f"[{BENCHMARK_NAME}] one prompt per class "
        f"(template: {args.prompt_template!r})"
    )

    for i in range(len(df)):
        row = df.iloc[i]
        finding = str(row["finding"])
        gt_label = str(row["progression"])
        try:
            prior = load_image_tensor_abs(row["img_path_prev"])
            current = load_image_tensor_abs(row["img_path_curr"])
        except (FileNotFoundError, OSError) as e:
            skipped += 1
            if skipped <= 5:
                print(
                    f"[{BENCHMARK_NAME}] skipping row {i} "
                    f"(image error: {e})"
                )
            continue

        out = score_one_pair(
            model, prior, current, finding, args.prompt_template, device,
            text_cache=text_cache,
            classes=VALID_CLASSES,
        )
        pred_idx = out["pred_class"]
        pred_label = VALID_CLASSES[pred_idx]

        n_seen += 1
        finding_lc = finding.lower()
        n_correct += int(pred_label == gt_label)
        confusion[gt_label][pred_label] += 1
        per_finding[finding_lc][1] += 1
        per_finding[finding_lc][0] += int(pred_label == gt_label)

        best_cos = out["cos_class_scores"][pred_idx]
        if best_cos > out["cos_naive"]:
            n_above_naive += 1

        for k in range(len(VALID_CLASSES)):
            cos_class_sums[k] += out["cos_class_scores"][k]
        naive_sum += out["cos_naive"]

        if (i + 1) % max(1, len(df) // 20) == 0:
            acc = n_correct / max(1, n_seen)
            print(
                f"[{BENCHMARK_NAME}]   {i + 1}/{len(df)}  "
                f"acc={acc:.4f}  skipped={skipped}"
            )

    if n_seen == 0:
        print("No samples evaluated.")
        return

    if skipped:
        print(
            f"\nSkipped {skipped} rows due to missing / unreadable images"
        )

    _print_eval_summary(
        n_correct=n_correct,
        n_seen=n_seen,
        confusion=confusion,
        per_finding=per_finding,
        cos_class_sums=cos_class_sums,
        naive_sum=naive_sum,
        n_above_naive=n_above_naive,
        classes=VALID_CLASSES,
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
        "--csv",
        default=DEFAULT_CSV,
        help=f"Path to the MS-CXR-T CSV (default: {DEFAULT_CSV}).",
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

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--demo", action="store_true",
        help="Print 3-way scoring for one CSV row.",
    )
    mode.add_argument(
        "--eval", action="store_true",
        help="Compute overall + per-class + per-finding accuracy.",
    )

    parser.add_argument(
        "--idx", type=int, default=None,
        help="(demo only) row index. Default: random.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="(demo only) RNG seed for picking a random row.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="(eval only) only evaluate the first N rows.",
    )

    args = parser.parse_args()

    try:
        _ = args.prompt_template.format("test_disease", "test_class")
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"--prompt-template must accept two positional {{}} slots "
            f"({{disease}}, {{class}}). Got {args.prompt_template!r}: {e}"
        )

    device = torch.device(args.device)
    model = load_jepa_model(args.ckpt, device)
    df = load_csv_pairs(args.csv, LABEL_MAP, VALID_CLASSES)
    if len(df) == 0:
        raise RuntimeError(
            f"No usable {BENCHMARK_NAME} rows after label filtering."
        )

    if args.demo:
        run_demo(args, model, df, device)
    else:
        run_eval(args, model, df, device)


if __name__ == "__main__":
    main()
