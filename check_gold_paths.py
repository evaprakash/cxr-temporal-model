"""Diagnostic — verify gold images are reachable on this machine.

Loads ``gold_progression_pairs.parquet``, picks one sample row per
dataset (mimic / chexpert / rexgradient), and tries to resolve both the
prior and current image paths against several candidate image roots:

  1. ``IMAGE_ROOTS`` from ``infer_jepa.py`` (the ``all_data`` paths used
     during training).
  2. Any ``final_gold_*_images`` directory next to the gold parquet
     (auto-discovered).
  3. ``final_gold_<dataset>_images`` under explicitly common candidate
     parent directories.

For each (dataset, prev/curr) it prints WHICH candidate root the image
was found under (if any) and the full resolved path. Run this once
before kicking off ``progression_classify.py`` so you know whether to
override ``IMAGE_ROOTS`` for the gold eval.

Usage
-----
    python check_gold_paths.py
    python check_gold_paths.py --gold-parquet /path/to/gold_progression_pairs.parquet
    python check_gold_paths.py --extra-root /path/to/maybe_gold_images
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from dataset_combined_jepa import DEFAULT_DATASET_DIR, _resolve_image_path
from infer_jepa import IMAGE_ROOTS


DEFAULT_GOLD_PARQUET = os.path.join(DEFAULT_DATASET_DIR, "gold_progression_pairs.parquet")
DATASETS = ["mimic", "chexpert", "rexgradient"]


# ============================================================
# Candidate root discovery
# ============================================================
def discover_final_gold_roots(parquet_dir: str) -> Dict[str, str]:
    """Look for final_gold_<dataset>_images dirs near the parquet.

    Searches both ``parquet_dir`` (gold images co-located with the parquet)
    and ``parquet_dir``'s parent (gold images sit as siblings of the
    parquet's containing folder, e.g. ``~/jepa/final_gold_*_images/``).
    """
    roots: Dict[str, str] = {}
    parent_dir = os.path.dirname(parquet_dir.rstrip("/"))
    search_bases = [parquet_dir, parent_dir]
    for d in DATASETS:
        for base in search_bases:
            for name in [
                f"final_gold_{d}_images",
                f"gold_{d}_images",
                f"{d}_gold_images",
                f"final_{d}_images",
            ]:
                cand = os.path.join(base, name)
                if os.path.isdir(cand):
                    roots[d] = cand
                    break
            if d in roots:
                break
    return roots


def list_candidate_dirs(parquet_dir: str) -> List[str]:
    """Glob anything that looks like a gold images dir under parquet_dir or its parent."""
    patterns = ["final_gold_*", "gold_*_images", "*_gold_*"]
    parent_dir = os.path.dirname(parquet_dir.rstrip("/"))
    found = []
    for base in [parquet_dir, parent_dir]:
        for p in patterns:
            found.extend(sorted(glob.glob(os.path.join(base, p))))
    return sorted({f for f in found if os.path.isdir(f)})


# ============================================================
# Resolution attempts
# ============================================================
def try_resolve(dataset: str, rel_path: str, roots: Dict[str, str]) -> Tuple[bool, str]:
    """Try several path resolutions under ``roots[dataset]``.

    Strategies, in order:
      1. The same prefix-stripping logic the training pipeline uses
         (``_resolve_image_path``).
      2. The raw relative path joined onto the root.
      3. Just the basename joined onto the root.
    """
    if dataset not in roots:
        return False, ""
    rel_path = str(rel_path).strip()
    if not rel_path:
        return False, ""

    # 1. Prefix-stripping logic from the training pipeline
    try:
        p = _resolve_image_path(dataset, rel_path, roots)
        if p.exists():
            return True, str(p)
    except Exception:
        pass

    # 2. Raw rel path under the root
    p = Path(roots[dataset]) / rel_path
    if p.exists():
        return True, str(p)

    # 3. Just the basename
    p = Path(roots[dataset]) / os.path.basename(rel_path)
    if p.exists():
        return True, str(p)

    return False, ""


# ============================================================
# Column auto-detection (matches progression_classify.py)
# ============================================================
def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    return ""


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-parquet", default=DEFAULT_GOLD_PARQUET)
    parser.add_argument(
        "--extra-root",
        action="append",
        default=[],
        help="Additional candidate parent directory to search for "
             "final_gold_<dataset>_images/ subdirs. Can repeat.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.gold_parquet):
        raise FileNotFoundError(f"Gold parquet not found: {args.gold_parquet}")

    # ---------------------------------------------------------------
    # 1. Inspect the gold parquet
    # ---------------------------------------------------------------
    print(f"Loading {args.gold_parquet}")
    gold = pd.read_parquet(args.gold_parquet)
    print(f"  rows: {len(gold)}")
    print(f"  columns: {list(gold.columns)}")

    if "dataset" in gold.columns:
        print("  per-dataset row counts:")
        for d, n in gold["dataset"].value_counts().items():
            print(f"    {d}: {n}")
    else:
        print("  WARNING: no 'dataset' column; cannot stratify per-dataset checks.")
        return

    img_curr_col = find_col(
        gold,
        ["parent_image_curr", "img_path_curr", "image_curr", "image_path_curr",
         "image_curr_path", "current_image_path", "img_curr"],
    )
    img_prev_col = find_col(
        gold,
        ["parent_image_prev", "img_path_prev", "image_prev", "image_path_prev",
         "image_prev_path", "prior_image_path", "img_prev"],
    )

    has_image_cols = bool(img_curr_col and img_prev_col)
    if has_image_cols:
        print(f"\n  using image-path columns from gold: "
              f"{img_prev_col} / {img_curr_col}")
    else:
        print(f"\n  no obvious image-path columns in gold parquet; "
              f"progression_classify.py will join with silver_findings.parquet "
              f"to fetch them.")
        # Even without image cols, we can still report what dirs exist
        # under candidate roots.

    # ---------------------------------------------------------------
    # 2. Discover candidate gold-image roots
    # ---------------------------------------------------------------
    parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    print(f"\nLooking for final_gold_*_images dirs under: {parquet_dir}")
    listed = list_candidate_dirs(parquet_dir)
    if listed:
        for d in listed:
            print(f"  {d}")
    else:
        print("  (none found)")

    final_gold_roots = discover_final_gold_roots(parquet_dir)
    print(f"\nAuto-discovered final-gold roots: {final_gold_roots or '(none)'}")

    # Add any user-provided extra roots
    extra_root_dicts = []
    for parent in args.extra_root:
        d = discover_final_gold_roots(parent)
        if d:
            extra_root_dicts.append((f"--extra-root {parent}", d))

    candidate_roots: List[Tuple[str, Dict[str, str]]] = [
        ("IMAGE_ROOTS (all_data)", IMAGE_ROOTS),
    ]
    if final_gold_roots:
        candidate_roots.append(
            (f"final_gold_*_images/ next to parquet", final_gold_roots)
        )
    candidate_roots.extend(extra_root_dicts)

    # ---------------------------------------------------------------
    # 3. Per-dataset path-resolution test
    # ---------------------------------------------------------------
    if not has_image_cols:
        print("\nSkipping per-dataset resolution test (no image columns).")
        print("If progression_classify.py succeeds at joining with silver_findings, "
              "you can re-run this script after adding --image-curr-col / --image-prev-col.")
        return

    print("\n" + "=" * 60)
    print("Per-dataset path resolution check")
    print("=" * 60)

    for dataset in DATASETS:
        sub = gold[gold["dataset"] == dataset]
        print(f"\n--- {dataset} ({len(sub)} gold rows) ---")
        if len(sub) == 0:
            print("  (no rows for this dataset)")
            continue

        row = sub.iloc[0]
        prev = row[img_prev_col]
        curr = row[img_curr_col]
        print(f"  raw prev path: {prev}")
        print(f"  raw curr path: {curr}")

        for label, raw_path in [("prev", prev), ("curr", curr)]:
            print(f"  {label}:")
            found_anywhere = False
            for source_name, roots in candidate_roots:
                ok, resolved = try_resolve(dataset, raw_path, roots)
                if ok:
                    print(f"    [FOUND] in {source_name}")
                    print(f"            -> {resolved}")
                    found_anywhere = True
                    break  # first hit wins
            if not found_anywhere:
                print(f"    [NOT FOUND] in any candidate root")
                # Print the resolved candidates we tried so the user can debug
                for source_name, roots in candidate_roots:
                    if dataset in roots:
                        try:
                            cand = _resolve_image_path(dataset, raw_path, roots)
                            print(f"      tried: {cand}  (from {source_name})")
                        except Exception:
                            print(f"      tried (raw join): "
                                  f"{Path(roots[dataset]) / raw_path}  "
                                  f"(from {source_name})")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        "If [FOUND] consistently shows the same source for a dataset, "
        "use that as the IMAGE_ROOTS entry for that dataset when running "
        "progression_classify.py.\n"
        "If everything is [NOT FOUND], you'll need to download/sync the "
        "gold images. Check the CheXTemporal HF page for the correct "
        "asset names (likely tarballs named final_gold_<dataset>_images.tar)."
    )


if __name__ == "__main__":
    main()
