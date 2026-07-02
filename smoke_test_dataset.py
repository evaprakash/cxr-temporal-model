"""End-to-end smoke test for ``JEPACombinedDataset`` in both modes.

Builds the dataset twice — once with ``condition_mode="dynamic"`` and
once with ``condition_mode="templated"`` — and for each prints:

  * row counts (overall + per parent dataset),
  * the distribution of findings-per-study-pair,
  * the normalized progression-class distribution,
  * a handful of fully-formatted example rows showing
    ``patient_id``, ``study_id_prev``/``study_id_curr``, the raw vs
    resolved prior/current image paths, the per-finding metadata, and
    the ``condition_text`` the predictor would actually see.

The two modes share the same underlying study pairs, so the comparison
is apples-to-apples: only ``condition_text`` should differ between
matched rows.

By default this runs without ever calling ``__getitem__``, so it works
on any machine that has the silver parquets even if the parent image
corpora aren't synced locally. Pass ``--load-images`` to additionally
fetch a few examples through ``__getitem__`` and report the resulting
tensor shapes.

Usage
-----
    # Basic: read the silver parquets and print example rows for both modes.
    python smoke_test_dataset.py

    # Show more / fewer examples per mode.
    python smoke_test_dataset.py --num-examples 5

    # Restrict to a single split (matches the trainer's val pool).
    python smoke_test_dataset.py --split val

    # Also run __getitem__ on the first sample of each parent dataset.
    python smoke_test_dataset.py --load-images
"""

import argparse
import os
import random
import sys
from collections import Counter

from dataset_combined_jepa import (
    CONDITION_MODES,
    DEFAULT_DATASET_DIR,
    JEPACombinedDataset,
    _resolve_image_path,
)


# Bulk data lives under ``<repo>/all_data`` by default (typically a
# symlink to scratch storage). ``JEPA_IMAGE_ROOTS_DIR`` overrides the
# whole tree; the per-dataset ``*_ROOT`` env vars still win over that.
_DEFAULT_ROOTS_DIR = os.environ.get(
    "JEPA_IMAGE_ROOTS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_data"),
)
DEFAULT_IMAGE_ROOTS = {
    "mimic": os.environ.get(
        "MIMIC_ROOT", os.path.join(_DEFAULT_ROOTS_DIR, "mimic")
    ),
    "chexpert": os.environ.get(
        "CHEXPERT_ROOT", os.path.join(_DEFAULT_ROOTS_DIR, "chexpert", "train")
    ),
    "rexgradient": os.environ.get(
        "REXGRADIENT_ROOT",
        os.path.join(_DEFAULT_ROOTS_DIR, "rexgradient", "deid_png"),
    ),
}


# ============================================================
# PRETTY-PRINTING HELPERS
# ============================================================
def _short(s: object, n: int = 200) -> str:
    """Collapse whitespace and truncate ``s`` for log lines."""
    text = " ".join(str(s).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _resolve_safe(dataset: str, rel_path: str, roots) -> str:
    """Resolve ``rel_path`` onto ``roots`` without requiring the file to exist."""
    try:
        return str(_resolve_image_path(dataset, rel_path, roots))
    except Exception as exc:  # pragma: no cover — purely a print helper
        return f"<unresolved: {exc}>"


def _print_dataset_stats(ds: JEPACombinedDataset) -> None:
    df = ds.df
    print(f"  total pairs                    : {len(df)}")
    print("  per parent dataset             :")
    for name in ("mimic", "chexpert", "rexgradient"):
        sub = df[df["dataset"] == name]
        print(f"    {name:<12s} : {len(sub):>7d}")

    n_findings = df["finding"].apply(len)
    print(
        "  findings per study pair        : "
        f"min={int(n_findings.min())}, "
        f"max={int(n_findings.max())}, "
        f"mean={n_findings.mean():.2f}"
    )

    flat_prog = [c for row in df["progression_cls"] for c in row]
    prog_counts = Counter(flat_prog)
    total_prog = sum(prog_counts.values())
    if total_prog:
        print("  progression class distribution :")
        for cls in ("improving", "stable", "worsening", "new", "resolved"):
            count = prog_counts.get(cls, 0)
            pct = (100.0 * count / total_prog) if total_prog else 0.0
            print(f"    {cls:<10s} : {count:>7d}  ({pct:5.2f}%)")

    flat_findings = [f for row in df["finding"] for f in row]
    finding_counts = Counter(flat_findings)
    if finding_counts:
        print("  top findings                   :")
        for name, count in finding_counts.most_common(8):
            print(f"    {name:<28s} : {count:>7d}")


def _print_example(
    ds: JEPACombinedDataset,
    idx: int,
    image_roots,
    load_images: bool,
) -> None:
    row = ds.df.iloc[idx]
    sample = ds[idx] if load_images else None

    print(f"  [{ds.condition_mode}] sample idx={idx} of {len(ds)}")
    print(f"    dataset             : {row['dataset']}")
    print(f"    patient_id          : {row['patient_id']}")
    print(f"    study_id_prev       : {row['study_id_prev']}")
    print(f"    study_id_curr       : {row['study_id_curr']}")
    print(f"    parent_image_prev   : {row['parent_image_prev']}")
    print(
        "    resolved prev path  : "
        f"{_resolve_safe(row['dataset'], row['parent_image_prev'], image_roots)}"
    )
    print(f"    parent_image_curr   : {row['parent_image_curr']}")
    print(
        "    resolved curr path  : "
        f"{_resolve_safe(row['dataset'], row['parent_image_curr'], image_roots)}"
    )

    findings = list(row["finding"])
    progressions = list(row["progression_cls"])
    print(f"    n_findings          : {len(findings)}")
    for f, p in zip(findings, progressions):
        print(f"      - {f:<28s} {p}")

    print(
        "    prior_report (200c) : "
        f"{_short(row['prior_report'], 200)}"
    )
    print(
        "    current_report (200): "
        f"{_short(row['current_report'], 200)}"
    )
    print(
        "    dynamic_report (200): "
        f"{_short(row['dynamic_report'], 200)}"
    )

    # The condition_text comes from __getitem__ in this mode; the
    # dataset's mode determines whether it's the dynamic sentences
    # or the templated finding-progression string.
    if sample is None:
        # Build the condition without loading images so we can still
        # show what the predictor would receive.
        if ds.condition_mode == "dynamic":
            condition_text = row["dynamic_report"]
        else:
            condition_text = ds._build_templated_condition(row)
    else:
        condition_text = sample["condition_text"]
    print(
        "    condition_text (320): "
        f"{_short(condition_text, 320)}"
    )

    if sample is not None:
        print(
            "    tensors             : "
            f"prior_image={tuple(sample['prior_image'].shape)}  "
            f"current_image={tuple(sample['current_image'].shape)}"
        )


def _pick_example_indices(ds: JEPACombinedDataset, n: int, seed: int):
    """One sample per parent dataset if possible, then random fill to ``n``."""
    df = ds.df
    chosen = []
    rng = random.Random(seed)
    for name in ("mimic", "chexpert", "rexgradient"):
        idxs = df.index[df["dataset"] == name].tolist()
        if idxs:
            chosen.append(rng.choice(idxs))
    remaining = [i for i in range(len(df)) if i not in chosen]
    rng.shuffle(remaining)
    while len(chosen) < n and remaining:
        chosen.append(remaining.pop())
    return chosen[:n]


# ============================================================
# MAIN
# ============================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of example rows to print per condition mode (default: 3).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default=None,
        help="Restrict to one split; default uses all rows.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Used only when --split is set and the studies parquet has "
             "no native split column (default 0.1, matches the trainer).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the fallback stratified split (default 42, matches "
             "the trainer).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for picking the example indices (default 0). The "
             "same seed is used for both modes so the matched-row "
             "comparison is reproducible.",
    )
    parser.add_argument(
        "--mimic-root",
        default=DEFAULT_IMAGE_ROOTS["mimic"],
    )
    parser.add_argument(
        "--chexpert-root",
        default=DEFAULT_IMAGE_ROOTS["chexpert"],
    )
    parser.add_argument(
        "--rexgradient-root",
        default=DEFAULT_IMAGE_ROOTS["rexgradient"],
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Override the CheXTemporal parquet directory (default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--load-images",
        action="store_true",
        help="Also run __getitem__ on each example to confirm the image "
             "files resolve and the tensor shapes are correct. Off by "
             "default so the script works without the parent image corpora.",
    )
    args = parser.parse_args()

    image_roots = {
        "mimic": args.mimic_root,
        "chexpert": args.chexpert_root,
        "rexgradient": args.rexgradient_root,
    }
    findings_path = os.path.join(args.dataset_dir, "silver_findings.parquet")
    studies_path = os.path.join(args.dataset_dir, "silver_studies.parquet")
    sentences_path = os.path.join(args.dataset_dir, "silver_sentences.parquet")

    for label, path in (
        ("silver_findings", findings_path),
        ("silver_studies", studies_path),
        ("silver_sentences", sentences_path),
    ):
        if not os.path.exists(path):
            print(
                f"[smoke] missing {label} parquet at {path}\n"
                f"        override the directory via --dataset-dir "
                f"or the CHEXTEMPORAL_DIR env var.",
                file=sys.stderr,
            )
            return 1

    print(f"[smoke] silver parquets: {args.dataset_dir}")
    print(f"[smoke] image roots:     {image_roots}")
    print(f"[smoke] split:           {args.split or 'all'}")
    print(f"[smoke] num examples:    {args.num_examples}")
    print(f"[smoke] load images:     {args.load_images}\n")

    datasets = {}
    for mode in CONDITION_MODES:
        print(f"=== Building dataset (condition_mode={mode!r}) ===")
        datasets[mode] = JEPACombinedDataset(
            image_roots=image_roots,
            findings_path=findings_path,
            studies_path=studies_path,
            sentences_path=sentences_path,
            split=args.split,
            train=False,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            condition_mode=mode,
        )
        print()

    # The two modes share the same underlying rows, so the row counts
    # should match. Sanity-check that here.
    sizes = {m: len(ds) for m, ds in datasets.items()}
    print(f"[smoke] dataset sizes per mode: {sizes}")
    distinct_sizes = set(sizes.values())
    if len(distinct_sizes) != 1:
        print(
            "[smoke] WARNING: row counts differ across modes — both "
            "should see the same set of study pairs.",
            file=sys.stderr,
        )
    print()

    # Pre-pick indices once on the dynamic dataset; both modes are
    # row-aligned so the same indices map to the same study pair in
    # both, which is exactly what we want to compare.
    indices = _pick_example_indices(
        datasets["dynamic"], args.num_examples, seed=args.seed
    )

    for mode in CONDITION_MODES:
        ds = datasets[mode]
        print(f"=== Stats (condition_mode={mode!r}) ===")
        _print_dataset_stats(ds)
        print()

        print(f"=== Example rows (condition_mode={mode!r}) ===")
        for idx in indices:
            _print_example(ds, idx, image_roots, args.load_images)
            print()

    print("[smoke] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
