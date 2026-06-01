"""Shared eval helpers for CSV-format progression-classification benchmarks.

Used by:
  * ``eval_mscxrt.py`` — MS-CXR-T progression labels (3-way: improving /
    stable / worsening).
  * ``eval_cig.py``    — Chest ImaGenome temporal-comparison labels
    (3-way: improving / stable / worsening).

Both CSVs share the same column schema:

    patient_id, study_id_prev, study_id_curr,
    img_path_prev, img_path_curr, disease_name, comparison

with absolute MIMIC-CXR JPG paths and a free-form ``comparison`` label
that we map onto our internal CLS_ORDER via the dataset-specific
``LABEL_MAP``. Anything not covered by ``LABEL_MAP`` is dropped.

The 5-way phrase bank from ``progression_classify.py`` is reused
unchanged; we just restrict the argmax / argmin to the dataset's valid
class subset so the model is never penalized for predicting a class
the dataset doesn't label.
"""

import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image

from dataset_combined import (
    BASE_TRANSFORM,
    apply_augmentation,
    sample_augmentation,
)
from progression_classify import (
    CLS_ORDER,
    PROGRESSION_PHRASES,
    PROMPT_TEMPLATE,
    _print_eval_summary,
    score_one_pair,
)


# ============================================================
# CSV LOADING + LABEL MAPPING
# ============================================================
def normalize_csv_label(s, label_map: Dict[str, str]) -> Optional[str]:
    """Return the mapped class name, or ``None`` if the label is unknown."""
    return label_map.get(str(s).strip().lower())


def load_csv_pairs(
    csv_path: str,
    label_map: Dict[str, str],
    valid_classes: List[str],
) -> pd.DataFrame:
    """Load a 3-way progression CSV and normalize labels.

    Reads the CSV, lower-cases / strips ``comparison``, maps it through
    ``label_map``, drops rows whose mapped label isn't in
    ``valid_classes``, and renames a couple of columns to match the
    schema the runner expects (``finding``, ``progression``).

    Also prints a one-time value-counts table of the raw ``comparison``
    column so unmapped labels are visible.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    print(f"[csv] loading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[csv]   {len(df)} rows, columns: {list(df.columns)}")

    needed = {
        "patient_id", "study_id_prev", "study_id_curr",
        "img_path_prev", "img_path_curr", "disease_name", "comparison",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}. "
            f"Got: {list(df.columns)}"
        )

    raw_counts = (
        df["comparison"]
        .astype(str)
        .str.strip()
        .str.lower()
        .value_counts()
    )
    print("[csv]   raw 'comparison' value counts:")
    for label, count in raw_counts.items():
        mapped = label_map.get(label)
        kept = mapped in valid_classes
        marker = f"-> {mapped}" if kept else "DROPPED"
        print(f"           {label:<24} {count:>6}  {marker}")

    df["progression"] = df["comparison"].apply(
        lambda x: normalize_csv_label(x, label_map)
    )
    df = df[df["progression"].isin(valid_classes)].copy()
    df = df.rename(columns={"disease_name": "finding"})
    df = df.drop_duplicates(
        ["patient_id", "study_id_prev", "study_id_curr", "finding", "progression"]
    ).reset_index(drop=True)
    print(f"[csv]   {len(df)} rows after label mapping, filter, "
          f"and dedup")
    return df


# ============================================================
# IMAGE LOADING (absolute paths, with mild fallbacks)
# ============================================================
def load_image_tensor_abs(abs_path: str) -> torch.Tensor:
    """Open an image at an absolute path, transform to model input.

    The two CSV-based benchmarks ship absolute paths into MIMIC-CXR's
    ``mimic-cxr-jpg`` tree; on the cluster these resolve directly so we
    don't need any prefix-stripping logic.
    """
    abs_path = str(abs_path).strip()
    img = Image.open(abs_path).convert("RGB")
    img = BASE_TRANSFORM(img)
    params = sample_augmentation(train=False)
    return apply_augmentation(img, params)


# ============================================================
# DEMO MODE
# ============================================================
def run_csv_demo(
    args,
    model,
    df: pd.DataFrame,
    valid_classes: List[str],
    device: torch.device,
):
    """Print 5-way scoring breakdown for one CSV row, with prediction
    restricted to ``valid_classes``."""
    if args.idx is None:
        rng = random.Random(args.seed)
        args.idx = rng.randrange(len(df))
    if not (0 <= args.idx < len(df)):
        raise IndexError(f"--idx {args.idx} out of range [0, {len(df)})")

    row = df.iloc[args.idx]
    finding = str(row["finding"])
    gt_label = str(row["progression"])

    print(f"\n=== Sample {args.idx} of {len(df)} ===")
    print(f"  patient_id:    {row['patient_id']}")
    print(f"  study_id_prev: {row['study_id_prev']}")
    print(f"  study_id_curr: {row['study_id_curr']}")
    print(f"  finding:       {finding}")
    print(f"  ground-truth:  {gt_label}")

    prior = load_image_tensor_abs(row["img_path_prev"])
    current = load_image_tensor_abs(row["img_path_curr"])

    out = score_one_pair(model, prior, current, finding, args.prompt_template, device)

    valid_idx = [CLS_ORDER.index(c) for c in valid_classes]
    cos_best = max(valid_idx, key=lambda k: out["cos_class_scores"][k])
    l1_best = min(valid_idx, key=lambda k: out["l1_class_scores"][k])

    print("\nPer-class scores (averaged across phrases of each class; "
          "only valid classes are shown):")
    print(f"  {'class':<10} {'#phrases':>9} {'cos_score':>11} {'L1_score':>11}")
    for k_idx in valid_idx:
        cls = CLS_ORDER[k_idx]
        cos_marker = "  <-- argmax cos" if k_idx == cos_best else ""
        l1_marker = "  <-- argmin L1"  if k_idx == l1_best  else ""
        print(
            f"  {cls:<10} {len(PROGRESSION_PHRASES[cls]):>9} "
            f"{out['cos_class_scores'][k_idx]:>11.4f} "
            f"{out['l1_class_scores'][k_idx]:>11.4f}"
            f"{cos_marker}{l1_marker}"
        )

    cos_pred = CLS_ORDER[cos_best]
    l1_pred = CLS_ORDER[l1_best]
    print(f"\n  Cosine prediction:    {cos_pred:<10}  vs gt {gt_label:<10}  "
          f"=> {'CORRECT' if cos_pred == gt_label else 'WRONG'}")
    print(f"  Smooth L1 prediction: {l1_pred:<10}  vs gt {gt_label:<10}  "
          f"=> {'CORRECT' if l1_pred == gt_label else 'WRONG'}")


# ============================================================
# EVAL MODE
# ============================================================
def run_csv_eval(
    args,
    model,
    df: pd.DataFrame,
    valid_classes: List[str],
    device: torch.device,
    benchmark_name: str = "CSV",
):
    """Run 3-way (or N-way) progression classification over an entire CSV.

    Restricts argmax/argmin to ``valid_classes`` (so e.g. the model
    can't predict 'new' on MS-CXR-T which has no such class). Reports
    two parallel result blocks — one for cosine argmax, one for
    Smooth L1 argmin.
    """
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)

    n_correct_cos = n_correct_l1 = n_seen = 0
    conf_cos: Dict = defaultdict(Counter)
    conf_l1: Dict = defaultdict(Counter)
    per_finding_cos = defaultdict(lambda: [0, 0])
    per_finding_l1 = defaultdict(lambda: [0, 0])
    skipped = 0

    text_cache: Dict[str, Tuple] = {}
    valid_idx = [CLS_ORDER.index(c) for c in valid_classes]

    print(f"\n[{benchmark_name}] running {len(valid_classes)}-way progression "
          f"classification on {len(df)} rows")
    print(f"[{benchmark_name}] valid classes: {valid_classes}")
    print(f"[{benchmark_name}] phrases per class: " + ", ".join(
        f"{c}={len(PROGRESSION_PHRASES[c])}" for c in valid_classes
    ))
    print(f"[{benchmark_name}] reporting BOTH cosine-argmax and Smooth-L1-argmin")

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
                print(f"[{benchmark_name}] skipping row {i} (image error: {e})")
            continue

        out = score_one_pair(
            model, prior, current, finding, args.prompt_template, device,
            text_cache=text_cache,
        )
        cos_idx = max(valid_idx, key=lambda k: out["cos_class_scores"][k])
        l1_idx = min(valid_idx, key=lambda k: out["l1_class_scores"][k])
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

        if (i + 1) % max(1, len(df) // 20) == 0:
            acc_cos = n_correct_cos / max(1, n_seen)
            acc_l1 = n_correct_l1 / max(1, n_seen)
            print(f"[{benchmark_name}]   {i + 1}/{len(df)}  "
                  f"cos_acc={acc_cos:.4f}  l1_acc={acc_l1:.4f}  "
                  f"skipped={skipped}")

    if n_seen == 0:
        print("No samples evaluated.")
        return

    if skipped:
        print(f"\nSkipped {skipped} rows due to missing or unreadable images")

    _print_eval_summary(
        f"Cosine (argmax slide-deck cos(Δẑ, Δz_true))",
        n_correct_cos, n_seen, conf_cos, per_finding_cos,
        classes=valid_classes,
    )
    _print_eval_summary(
        f"Smooth L1 (argmin ‖ẑ_cur − z_cur‖_smooth_l1)",
        n_correct_l1, n_seen, conf_l1, per_finding_l1,
        classes=valid_classes,
    )


# ============================================================
# CLI BUILDER (shared between eval_mscxrt.py and eval_cig.py)
# ============================================================
def build_parser(default_csv: str, benchmark_name: str):
    """Build a CLI parser with the flags that both eval_mscxrt and
    eval_cig need."""
    import argparse
    p = argparse.ArgumentParser(description=f"{benchmark_name} progression eval")
    p.add_argument(
        "--ckpt",
        default=os.environ.get("JEPA_CKPT", "checkpoints_jepa/best.pt"),
    )
    p.add_argument("--csv", default=default_csv,
                   help=f"Path to the {benchmark_name} CSV "
                        f"(default: {default_csv}).")
    p.add_argument("--prompt-template", default=PROMPT_TEMPLATE,
                   help="Two-slot positional template; default '{} is {}'.")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true",
                      help="Print 3-way scoring for one CSV row.")
    mode.add_argument("--eval", action="store_true",
                      help="Compute overall + per-class + per-finding accuracy.")
    p.add_argument("--idx", type=int, default=None,
                   help="(demo only) row index. Default: random.")
    p.add_argument("--seed", type=int, default=0,
                   help="(demo only) RNG seed for picking a random row.")
    p.add_argument("--limit", type=int, default=None,
                   help="(eval only) only evaluate the first N rows.")
    return p
