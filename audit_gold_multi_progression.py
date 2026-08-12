#!/usr/bin/env python3
"""Audit CheXTemporal gold for multi-label (pair, finding) groups.

Image-level progression classifiers predict one class per
``(prior, current, finding)``. Gold often keeps one row per lesion box,
so the same disease on one pair can carry different progression labels
(e.g. worse at one site, stable at another). Those rows share an
identical image-level prediction but conflicting GTs.

Usage
-----
    python audit_gold_multi_progression.py
    python audit_gold_multi_progression.py \\
        --gold-parquet /path/to/gold_progression_pairs.parquet

To evaluate only the unique-label subset with the per-patch JEPA rule::

    python eval_progression_jepa_perpatch.py --eval --drop-multi-progression \\
        --ckpt /path/to/best.pt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

# Lightweight audit — no torch / model imports. Mirrors the helpers in
# ``progression_classify.py`` so this runs in a bare pandas env.
PAIR_FINDING_KEY_COLS = [
    "dataset",
    "patient_id",
    "study_id_curr",
    "study_id_prev",
    "finding",
]

GT_TO_CLS = {
    "improved": "improving",
    "improving": "improving",
    "worse": "worsening",
    "worsening": "worsening",
    "stable": "stable",
    "new": "new",
    "resolved": "resolved",
}
CLS_ORDER = ["improving", "stable", "worsening", "new", "resolved"]

_DEFAULT_DIR = Path(__file__).resolve().parent / "CheXTemporal"
DEFAULT_GOLD_PARQUET = str(_DEFAULT_DIR / "gold_progression_pairs.parquet")
DEFAULT_FINDINGS = str(_DEFAULT_DIR / "silver_findings.parquet")


def _normalize_label(x) -> str:
    s = str(x).strip().lower()
    return GT_TO_CLS.get(s, s)


def _load_gold(gold_parquet: str, findings_parquet: str) -> pd.DataFrame:
    gold = pd.read_parquet(gold_parquet)
    finding_col = next(
        (
            c
            for c in ("finding", "disease", "disease_name", "pathology")
            if c in gold.columns
        ),
        None,
    )
    label_col = next(
        (
            c
            for c in ("progression", "progression_label", "label")
            if c in gold.columns
        ),
        None,
    )
    if finding_col is None or label_col is None:
        raise ValueError(
            f"Need finding + progression columns; got {list(gold.columns)}"
        )
    gold = gold.rename(
        columns={label_col: "progression", finding_col: "finding"}
    )
    gold["progression"] = gold["progression"].apply(_normalize_label)
    gold = gold[gold["progression"].isin(CLS_ORDER)].copy()

    has_imgs = (
        "parent_image_curr" in gold.columns
        or "img_path_curr" in gold.columns
    )
    if not has_imgs and os.path.isfile(findings_parquet):
        findings = pd.read_parquet(findings_parquet)
        for c in ("dataset", "patient_id", "study_id_curr", "study_id_prev"):
            if c in findings.columns and c in gold.columns:
                findings[c] = findings[c].astype("string")
                gold[c] = gold[c].astype("string")
        keep = [
            c
            for c in (
                "dataset",
                "patient_id",
                "study_id_curr",
                "study_id_prev",
                "parent_image_curr",
                "parent_image_prev",
            )
            if c in findings.columns
        ]
        gold = gold.merge(
            findings[keep].drop_duplicates(
                ["dataset", "patient_id", "study_id_curr", "study_id_prev"]
            ),
            on=["dataset", "patient_id", "study_id_curr", "study_id_prev"],
            how="inner",
        )
    return gold.reset_index(drop=True)


def audit_multi_progression_labels(gold: pd.DataFrame) -> pd.DataFrame:
    g = gold.copy()
    g["_finding_lc"] = g["finding"].astype(str).str.strip().str.lower()
    g["_prog_lc"] = g["progression"].astype(str).str.strip().str.lower()
    keys = [
        "dataset",
        "patient_id",
        "study_id_curr",
        "study_id_prev",
        "_finding_lc",
    ]
    rows = []
    for key, sub in g.groupby(keys, dropna=False):
        progs = tuple(sorted(sub["_prog_lc"].unique()))
        if len(progs) <= 1:
            continue
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        row["n_rows"] = len(sub)
        row["n_unique_prog"] = len(progs)
        row["progressions"] = progs
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gold-parquet",
        default=os.environ.get("GOLD_PARQUET", DEFAULT_GOLD_PARQUET),
    )
    p.add_argument(
        "--findings-parquet",
        default=os.environ.get("FINDINGS_PARQUET", DEFAULT_FINDINGS),
    )
    args = p.parse_args()

    gold = _load_gold(args.gold_parquet, args.findings_parquet)
    print(f"[gold] loaded {len(gold)} rows from {args.gold_parquet}")
    audit = audit_multi_progression_labels(gold)
    n_groups = (
        gold.assign(_f=gold["finding"].astype(str).str.strip().str.lower())
        .drop_duplicates(
            ["dataset", "patient_id", "study_id_curr", "study_id_prev", "_f"]
        )
        .shape[0]
    )
    n_multi = len(audit)
    n_multi_rows = int(audit["n_rows"].sum()) if n_multi else 0
    print(
        f"\n(pair, finding) groups: {n_groups}\n"
        f"  multi-label groups:   {n_multi} "
        f"({100 * n_multi / max(n_groups, 1):.1f}%)\n"
        f"  rows in those groups: {n_multi_rows}/{len(gold)} "
        f"({100 * n_multi_rows / max(len(gold), 1):.1f}%)\n"
        f"  unique-label rows:    {len(gold) - n_multi_rows}"
    )
    if n_multi:
        print("\nConflicting progression combos:")
        for combo, cnt in audit["progressions"].value_counts().items():
            print(f"  {cnt:>4}  {combo}")

    # Match eval filter keep count.
    g = gold.copy()
    g["_finding_lc"] = g["finding"].astype(str).str.strip().str.lower()
    g["_prog_lc"] = g["progression"].astype(str).str.strip().str.lower()
    keys = [
        "dataset",
        "patient_id",
        "study_id_curr",
        "study_id_prev",
        "_finding_lc",
    ]
    nuniq = g.groupby(keys, dropna=False)["_prog_lc"].transform("nunique")
    print(
        f"\n[--drop-multi-progression] would keep "
        f"{int((nuniq == 1).sum())}/{len(g)} rows"
    )


if __name__ == "__main__":
    main()
