#!/usr/bin/env python3
"""Silver laterality labels for laterality VL-JEPA.

For every silver finding row, map the MedGemma anatomy (silver_findings
``anatomy``) and the Qwen3-32B-AWQ anatomy (``qwen_outputs_*_long``) onto a
side with the same vocabulary as ``compare_qwen_to_silver.py``:

  only left-sided anatomy   -> left
  only right-sided anatomy  -> right
  both                      -> bilateral
  neither (midline / empty) -> none

``side`` is kept only where the two models agree on left, right or bilateral.
Findings without a side (cardiomegaly, enlarged cardiomediastinum) and
(pair, finding) keys that occur more than once in silver get no side.

The output holds silver IDs; it is git-ignored (*.parquet). Copy it next to
silver_findings.parquet on the cluster.

    python -m vljepa.build_laterality \\
        --silver ~/mask_silver/silver_release_v1/hf_upload/silver/silver_findings.parquet \\
        --qwen-root ~ --out /tmp/silver_laterality.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Optional, Set

import pandas as pd

SIDES = ("left", "right", "bilateral")
NO_SIDE_FINDINGS = {"cardiomegaly", "enlarged cardiomediastinum"}
QWEN_DIRS = {
    "chexpert": "qwen_outputs_chexpert_long",
    "mimic": "qwen_outputs_mimic_long",
    "rexgradient": "qwen_outputs_rexgradient_long",
}
KEY = ["dataset", "patient_id", "study_id_curr", "study_id_prev", "finding"]

VALID_ANATOMIES = {
    "left lung", "right lung", "cardiac silhouette", "mediastinum",
    "left lower lung", "right lower lung", "right hilar structures",
    "left hilar structures", "upper mediastinum",
    "left costophrenic angle", "right costophrenic angle",
    "left mid lung", "right mid lung", "aortic arch",
    "right upper lung", "left upper lung",
    "right hemidiaphragm", "right clavicle", "left clavicle",
    "left hemidiaphragm", "right apex", "trachea", "left apex",
    "carina", "svc", "right atrium", "cavoatrial junction",
    "abdomen", "spine",
}


def map_anatomy(text: str) -> list:
    """Free-text anatomy -> VALID_ANATOMIES (compare_qwen_to_silver.py)."""
    t = text.lower()
    left, right = "left" in t, "right" in t

    def sided(name):
        if left:
            return [f"left {name}"]
        if right:
            return [f"right {name}"]
        return [f"left {name}", f"right {name}"]

    if re.search(r"apex|apical", t):
        return sided("apex")
    if re.search(r"upper|suprahilar", t):
        return sided("upper lung")
    if re.search(r"mid", t):
        return sided("mid lung")
    if re.search(r"lower|base|basilar|bibasilar|bases", t):
        return sided("lower lung")
    if "hilar" in t or "hilum" in t or "perihilar" in t:
        return sided("hilar structures")
    if "mediast" in t or "paratracheal" in t or "subcarinal" in t:
        return ["mediastinum"]
    if "cardiac" in t or "heart" in t:
        return ["cardiac silhouette"]
    if "aorta" in t:
        return ["aortic arch"]
    if "svc" in t or "cavoatrial" in t:
        return ["svc"]
    if "trachea" in t:
        return ["trachea"]
    if "pleura" in t or "pleural" in t or "costophrenic" in t:
        return sided("costophrenic angle")
    if "diaphragm" in t:
        return sided("hemidiaphragm")
    if "spine" in t or "vertebra" in t:
        return ["spine"]
    if "clavicle" in t:
        return sided("clavicle")
    if any(x in t for x in ["abdomen", "bowel", "stomach", "liver"]):
        return ["abdomen"]
    return ["left lung", "right lung"]


def anatomy_set(raw) -> Set[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    out = set()
    for part in str(raw).split(","):
        part = part.strip().lower()
        if part:
            out.update([part] if part in VALID_ANATOMIES else map_anatomy(part))
    return out


def side_of(raw) -> Optional[str]:
    tokens = anatomy_set(raw)
    left = any(t.startswith("left ") for t in tokens)
    right = any(t.startswith("right ") for t in tokens)
    if left and right:
        return "bilateral"
    if left:
        return "left"
    if right:
        return "right"
    return None


def load_qwen_anatomy(root: str) -> dict:
    """(dataset, '{patient}_{study}_{token}') -> raw Qwen anatomy string."""
    out = {}
    for ds, sub in QWEN_DIRS.items():
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            print(f"[laterality] WARNING: no Qwen outputs for {ds} at {d}")
            continue
        names = [f for f in os.listdir(d) if f.endswith("_findings.json")]
        bad = 0
        for f in names:
            try:
                with open(os.path.join(d, f)) as fh:
                    out[(ds, f[: -len("_findings.json")])] = json.load(fh).get("anatomy", "")
            except (OSError, ValueError):
                bad += 1
        print(f"[laterality] {ds}: {len(names):,} Qwen findings files ({bad} unreadable)")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--silver", required=True, help="silver_findings.parquet")
    p.add_argument("--qwen-root", required=True, help="folder holding qwen_outputs_*_long")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_parquet(args.silver, columns=[
        "dataset", "patient_id", "study_id_curr", "study_id_prev",
        "finding_token", "finding", "anatomy"])
    for c in ("patient_id", "study_id_curr", "study_id_prev"):
        df[c] = df[c].astype("string")
    df["finding"] = df["finding"].astype(str).str.strip().str.lower()
    qwen = load_qwen_anatomy(args.qwen_root)

    df["side_medgemma"] = df["anatomy"].map(side_of)
    qkeys = [(d, f"{p}_{s}_{t}") for d, p, s, t in
             zip(df["dataset"], df["patient_id"], df["study_id_curr"], df["finding_token"])]
    df["has_qwen"] = [k in qwen for k in qkeys]
    df["side_qwen"] = [side_of(qwen[k]) if k in qwen else None for k in qkeys]
    agree = df["has_qwen"] & df["side_medgemma"].notna() & (df["side_medgemma"] == df["side_qwen"])
    df["side"] = df["side_medgemma"].where(agree)
    df.loc[df["finding"].isin(NO_SIDE_FINDINGS), "side"] = None
    dup = df.duplicated(KEY, keep=False)
    df.loc[dup, "side"] = None

    n = len(df)
    print(f"\n[laterality] {n:,} silver rows; Qwen output for {int(df['has_qwen'].sum()):,}; "
          f"{int(dup.sum())} rows in duplicated (pair, finding) keys get no side")
    both = df[df["has_qwen"]]
    same = (both["side_medgemma"].fillna("none") == both["side_qwen"].fillna("none")).mean()
    print(f"[laterality] MedGemma vs Qwen side agreement (incl. none): {same:.1%}")
    c = Counter(df["side"].fillna("none"))
    print("[laterality] final side: " + ", ".join(f"{k} {v:,} ({v / n:.1%})" for k, v in c.most_common()))
    print("\n[laterality] sided share per finding:")
    for f, g in df.groupby("finding"):
        s = Counter(g["side"].dropna())
        print(f"  {f:28s} n={len(g):7,}  sided {g['side'].notna().mean():6.1%}  "
              + "  ".join(f"{k} {s.get(k, 0):,}" for k in SIDES))

    out = df.loc[df["side"].notna(), KEY + ["side", "side_medgemma", "side_qwen"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\n[laterality] wrote {len(out):,} sided rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
