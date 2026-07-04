"""Text-free image-image change-map grounding (Approach B) for BioViL-T.

Motivation
----------
This is the *text-free* grounding companion to
``biovilt_heatmap_progression_pairs.py`` (which does text-conditioned
grounding — heatmap = ``cos(patches, "{finding} is {progression}")``).
Here we build the heatmap from the image pair alone: how much did each
14x14 patch representation change between the previous and current
visit?

Recipe
------
1. ``patches_curr = image_encoder(current=curr, previous=prev)`` — the
   BioViL-T ``MultiImageModel`` produces patches spatially aligned to
   the *current* image, informed by the prior.
2. ``patches_prev = image_encoder(current=prev, previous=curr)`` —
   swap the roles so we get patches spatially aligned to the *prev*
   image, informed by the current. Both output grids are 14x14 and
   both are L2-normalized by the encoder.
3. Per-patch change score  ``change[p] = 1 - cos(patches_curr[p],
   patches_prev[p])``  in [0, 2]. Assumes rough anatomical alignment
   at patch position ``p`` between the two visits (standard CXR pair
   assumption).
4. Bilinearly upsample the 14x14 change grid to the 448x448 model
   crop.
5. Ground-truth mask: union of *all* the pair's finding bboxes
   (``prior_bboxes`` on the prev side, ``current_bboxes`` on the
   curr side). One mask per side, same change map for both sides,
   matching the ``side=prev/curr`` output convention of the existing
   heatmap scripts.
6. Score per side with BioViL CNR and Pointing Game on the model 448x448
   grid; aggregate overall, per dataset, and per progression class.

Because step 1-4 use only images (no text), this evaluation is fully
text-free for BioViL-T. It is meant to be compared against
``jepa_change_map_pairs.py``, which builds a symmetric change map
from JEPA's predictor delta.

Model
-----
``TempCXR(mode="biovilt")`` — no checkpoint, uses whatever weights the
BioViL-T image encoder / text encoder classes ship with — same as
``biovilt_heatmap_progression_pairs.py`` and ``eval_disease_biovilt.py``.

Usage
-----
    # Every pair in the gold parquet, PNGs + CSV + summary
    python biovilt_change_map_pairs.py

    # Metrics only, skip PNG rendering
    python biovilt_change_map_pairs.py --no-render

    # Restrict to one dataset
    python biovilt_change_map_pairs.py --datasets mimic

    # Pick N pairs (one per progression class when possible) for a demo
    python biovilt_change_map_pairs.py --num-examples 10

The stdout summary and per-pair CSV are drop-in comparable with
``jepa_change_map_pairs.py``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# The BioViL-T model file imports ``losses`` from the project root; we
# never call the training losses here, so stub the module out if it
# isn't importable before we pull TempCXR in. Same shim
# ``eval_disease_biovilt.py`` uses.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
try:
    import losses  # noqa: F401
except ModuleNotFoundError:
    import types as _types

    _stub = _types.ModuleType("losses")
    _stub.global_contrastive_loss = None
    _stub.local_contrastive_loss = None
    _stub.mlm_loss = None
    sys.modules["losses"] = _stub

# Shared bbox / mask / CNR / PG / rendering / schema helpers live in
# ``jepa_heatmap_progression_pairs``. Importing them here means both
# grounding scripts share exactly the same downstream pipeline; the
# only thing that differs between the two is how the change map is
# produced (this file: image role-swap;
# jepa_change_map_pairs.py: predictor delta).
from dataset_combined_jepa import DEFAULT_DATASET_DIR  # noqa: E402
from infer_jepa import IMAGE_ROOTS  # noqa: E402
from jepa_heatmap_progression_pairs import (  # noqa: E402
    _parse_bboxes,
    boxes_mask_in_model_space,
    compute_cnr,
    compute_pointing_game,
    load_image,
    normalize_gold_bboxes_schema,
    render_heatmap_png,
)
from progression_classify import (  # noqa: E402
    DATASETS,
    _resolve_with_fallbacks,
    discover_gold_image_roots,
)
from tempcxr_biovilt.modules.tempcxr_model import TempCXR  # noqa: E402


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_GOLD_PARQUET = os.path.join(DEFAULT_DATASET_DIR, "gold_bboxes.parquet")

INPUT_SIZE = 448  # matches BASE_TRANSFORM
CLS_ORDER = ["improving", "stable", "worsening", "new", "resolved"]


# ============================================================
# MODEL
# ============================================================
def load_model() -> TempCXR:
    print("🔧 Initializing TempCXR with mode='biovilt' (no checkpoint).")
    model = TempCXR(mode="biovilt").to(DEVICE)
    model.eval()
    print("✅ Model loaded.")
    return model


@torch.no_grad()
def compute_change_map(model: TempCXR,
                       curr_t: torch.Tensor,
                       prev_t: torch.Tensor,
                       out_size: int = INPUT_SIZE) -> np.ndarray:
    """Text-free change map for one image pair.

    Returns a (out_size, out_size) ``float32`` numpy array containing
    ``1 - cos(patches_curr[p], patches_prev[p])`` per patch, bilinearly
    upsampled from the encoder's 14x14 grid. Both patch grids are unit-
    norm, so ``sum(a*b)`` is the per-patch cosine similarity.
    """
    _, patches_curr = model.image_encoder(
        curr_t.unsqueeze(0), prev_t.unsqueeze(0))
    _, patches_prev = model.image_encoder(
        prev_t.unsqueeze(0), curr_t.unsqueeze(0))
    patches_curr = patches_curr.squeeze(0)  # (L, D), L=196, D=128
    patches_prev = patches_prev.squeeze(0)

    cos = (patches_curr * patches_prev).sum(dim=-1)  # (L,)
    change = 1.0 - cos  # in [0, 2]

    L = change.shape[0]
    side = int(round(L ** 0.5))
    assert side * side == L, f"Patch count {L} is not a perfect square"

    grid = change.view(1, 1, side, side)
    up = F.interpolate(grid, size=(out_size, out_size),
                       mode="bilinear", align_corners=False)
    return up.squeeze().cpu().float().numpy()


# ============================================================
# PAIR AGGREGATION (union of all findings' bboxes per side)
# ============================================================
def group_by_pair(df: pd.DataFrame) -> list[dict]:
    """Collapse the gold parquet into one dict per (dataset, patient_id,
    study_id_prev, study_id_curr) pair. Each dict carries the *union*
    of all findings' ``prior_bboxes`` / ``current_bboxes`` for the pair
    plus the set of progression labels present in the pair (used to
    build per-progression summaries) and the finding names (kept for
    reference in the CSV output).
    """
    key_cols = ["dataset", "patient_id", "study_id_prev", "study_id_curr"]
    records: list[dict] = []
    for keys, sub in df.groupby(key_cols, sort=False):
        prev_boxes: list = []
        curr_boxes: list = []
        for _, row in sub.iterrows():
            prev_boxes.extend(_parse_bboxes(row.get("prior_bboxes")))
            curr_boxes.extend(_parse_bboxes(row.get("current_bboxes")))
        progressions = sorted({str(p).lower() for p in sub["progression"]})
        findings = sorted({str(f) for f in sub["finding"]})
        records.append({
            "dataset": keys[0],
            "patient_id": keys[1],
            "study_id_prev": keys[2],
            "study_id_curr": keys[3],
            "parent_image_prev": sub.iloc[0]["parent_image_prev"],
            "parent_image_curr": sub.iloc[0]["parent_image_curr"],
            "prev_boxes": prev_boxes,
            "curr_boxes": curr_boxes,
            "progressions": progressions,
            "findings": findings,
            "n_rows": int(len(sub)),
        })
    return records


def pick_pair_examples(records: list[dict], n: int, seed: int) -> list[dict]:
    """Pick n pairs covering progression classes evenly.

    A pair contributes to every progression class it contains, so we
    cycle over classes and pop pairs while unassigned pairs remain.
    """
    rng = random.Random(seed)
    by_cls: dict[str, list[int]] = {}
    for i, rec in enumerate(records):
        for prog in rec["progressions"]:
            by_cls.setdefault(prog, []).append(i)
    for ids in by_cls.values():
        rng.shuffle(ids)

    ordered = [c for c in CLS_ORDER if c in by_cls] + \
        [c for c in by_cls if c not in CLS_ORDER]

    chosen: list[int] = []
    seen: set[int] = set()
    while len(chosen) < n and any(by_cls[c] for c in ordered):
        for c in ordered:
            while by_cls[c] and by_cls[c][-1] in seen:
                by_cls[c].pop()
            if not by_cls[c]:
                continue
            idx = by_cls[c].pop()
            if idx in seen:
                continue
            seen.add(idx)
            chosen.append(idx)
            if len(chosen) >= n:
                break
    return [records[i] for i in chosen]


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold-parquet", default=DEFAULT_GOLD_PARQUET,
        help=f"Path to gold_bboxes.parquet (default: {DEFAULT_GOLD_PARQUET}).",
    )
    parser.add_argument(
        "--out-dir", default="change_maps_biovilt",
        help="Folder to save PNGs (if --no-render is not set) + cnr.csv.",
    )
    parser.add_argument(
        "--image-root", action="append", default=[], metavar="DATASET=PATH",
        help="Override an image root for one dataset. Repeatable.",
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Optional dataset filter (e.g. mimic chexpert).",
    )
    parser.add_argument(
        "--num-examples", type=int, default=None,
        help="If set, only process N pairs (spread across progression "
             "classes). Default: every pair in the parquet.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="Heatmap opacity (0..1) for PNG rendering.")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip writing PNGs; only compute metrics.")
    parser.add_argument("--cnr-csv", default=None,
                        help="Where to write the per-(pair, side) CSV "
                             "(default: <out-dir>/cnr.csv).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    auto_gold_roots = discover_gold_image_roots(parquet_dir)
    image_roots: dict[str, str] = {**IMAGE_ROOTS, **auto_gold_roots}
    if auto_gold_roots:
        print("[gold] auto-detected gold image roots:")
        for d, p in auto_gold_roots.items():
            print(f"  {d}: {p}")
    for spec in args.image_root:
        if "=" not in spec:
            raise ValueError(f"--image-root expects DATASET=PATH, got {spec!r}")
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(f"--image-root dataset must be in {DATASETS}")
        image_roots[d] = p
        print(f"[gold] override: {d} -> {p}")

    df = pd.read_parquet(args.gold_parquet)
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)].reset_index(drop=True)
    print(f"📄 Loaded {len(df)} rows from {args.gold_parquet}")
    df = normalize_gold_bboxes_schema(df)
    print(f"📄 {len(df)} rows after schema normalization + label filter")

    def _resolve(dataset, rel):
        try:
            return _resolve_with_fallbacks(dataset, rel, image_roots)
        except FileNotFoundError:
            return None

    df["_prev_resolved"] = df.apply(
        lambda r: _resolve(r["dataset"], r["parent_image_prev"]), axis=1)
    df["_curr_resolved"] = df.apply(
        lambda r: _resolve(r["dataset"], r["parent_image_curr"]), axis=1)
    df = df[df["_prev_resolved"].notna() & df["_curr_resolved"].notna()]
    df = df.reset_index(drop=True)
    print(f"📄 {len(df)} rows have both images present locally")
    if len(df) == 0:
        raise SystemExit(
            "No usable examples — check --gold-parquet, --image-root, and "
            "the presence of final_gold_<dataset>_images/ next to the parquet."
        )

    pairs = group_by_pair(df)
    print(f"🎯 Collapsed to {len(pairs)} unique image pairs")

    # Attach resolved paths (first row's resolution is fine — every
    # row for a given pair points to the same two images).
    path_by_pair: dict[tuple, tuple[Path, Path]] = {}
    for _, row in df.iterrows():
        k = (row["dataset"], row["patient_id"],
             row["study_id_prev"], row["study_id_curr"])
        if k not in path_by_pair:
            path_by_pair[k] = (row["_prev_resolved"], row["_curr_resolved"])
    for rec in pairs:
        k = (rec["dataset"], rec["patient_id"],
             rec["study_id_prev"], rec["study_id_curr"])
        rec["_prev_path"], rec["_curr_path"] = path_by_pair[k]

    if args.num_examples is not None:
        pairs = pick_pair_examples(pairs, n=args.num_examples, seed=args.seed)
        print(f"🎯 Down-sampled to {len(pairs)} pairs (--num-examples)")

    model = load_model()

    records: list[dict] = []
    for i, rec in enumerate(pairs):
        prev_t, prev_disp, prev_orig = load_image(rec["_prev_path"])
        curr_t, curr_disp, curr_orig = load_image(rec["_curr_path"])

        change_map = compute_change_map(model, curr_t, prev_t, out_size=INPUT_SIZE)

        # Shared color range across the two rendered sides so PNGs are
        # visually comparable.
        vmin = float(change_map.min())
        vmax = float(change_map.max())

        prev_boxes = rec["prev_boxes"]
        curr_boxes = rec["curr_boxes"]

        prev_mask = boxes_mask_in_model_space(
            prev_boxes, prev_orig, prev_disp.size)
        curr_mask = boxes_mask_in_model_space(
            curr_boxes, curr_orig, curr_disp.size)
        prev_cnr = compute_cnr(change_map, prev_mask) if prev_boxes else None
        curr_cnr = compute_cnr(change_map, curr_mask) if curr_boxes else None
        prev_pg = compute_pointing_game(change_map, prev_mask) if prev_boxes else None
        curr_pg = compute_pointing_game(change_map, curr_mask) if curr_boxes else None

        # Represent each side as one row (mirrors the JEPA heatmap
        # script's CSV layout), but note ``disease_name`` and
        # ``comparison`` are aggregates here since the change map is
        # per-pair, not per-finding.
        prog_str = "|".join(rec["progressions"])
        find_str = "|".join(rec["findings"])
        meta_common = {
            "dataset": rec["dataset"],
            "patient_id": rec["patient_id"],
            "study_id_prev": rec["study_id_prev"],
            "study_id_curr": rec["study_id_curr"],
            "disease_name": find_str,
            "comparison": prog_str,
            "n_rows_in_pair": rec["n_rows"],
        }
        records.append({**meta_common, "side": "prev",
                        "n_boxes": len(prev_boxes),
                        "cnr": prev_cnr,
                        "pointing_game": prev_pg})
        records.append({**meta_common, "side": "curr",
                        "n_boxes": len(curr_boxes),
                        "cnr": curr_cnr,
                        "pointing_game": curr_pg})

        p_drawn = c_drawn = 0
        if not args.no_render:
            base = (
                f"{i:04d}_{rec['dataset']}_{rec['patient_id']}_"
                f"{rec['study_id_prev']}_{rec['study_id_curr']}_"
                f"{prog_str.replace(' ', '-') or 'none'}"
            )
            prev_out = out_dir / f"{base}__prev.png"
            curr_out = out_dir / f"{base}__curr.png"
            p_drawn = render_heatmap_png(
                prev_disp, change_map, prev_boxes, prev_orig,
                vmin=vmin, vmax=vmax, alpha=args.alpha, out_path=prev_out,
            )
            c_drawn = render_heatmap_png(
                curr_disp, change_map, curr_boxes, curr_orig,
                vmin=vmin, vmax=vmax, alpha=args.alpha, out_path=curr_out,
            )

        if (i + 1) % 25 == 0 or (i + 1) == len(pairs):
            def _fmt(v):
                return f"{v:.3f}" if v is not None else "  - "

            def _fmt_pg(v):
                return str(v) if v is not None else "-"

            print(
                f"  [{i+1}/{len(pairs)}] ds={rec['dataset']:11s} "
                f"progs={prog_str[:22]:22s} "
                f"prev_box={len(prev_boxes)} curr_box={len(curr_boxes)} "
                f"CNR(prev)={_fmt(prev_cnr)} CNR(curr)={_fmt(curr_cnr)} "
                f"PG(prev)={_fmt_pg(prev_pg)} PG(curr)={_fmt_pg(curr_pg)}"
                + (f"  drew prev={p_drawn} curr={c_drawn}"
                   if not args.no_render else "")
            )

    _report(records, out_dir=out_dir, cnr_csv=args.cnr_csv, tag="BioViL-T",
            rendered=not args.no_render)


def _report(records: list[dict], *, out_dir: Path,
            cnr_csv: str | None, tag: str, rendered: bool) -> None:
    df = pd.DataFrame(records)
    csv_path = Path(cnr_csv) if cnr_csv else (out_dir / "cnr.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\n📊 Wrote per-(pair, side) CNR table to {csv_path}")

    valid = df.dropna(subset=["cnr"])
    print("\n" + "=" * 70)
    print(f"{tag} CHANGE-MAP CNR SUMMARY (union bboxes per side, "
          "model 448x448 space)")
    print("=" * 70)
    print(f"  scored sides:   {len(valid)} / {len(df)}")
    if len(valid):
        print(f"  overall CNR:    mean={valid['cnr'].mean():.4f}  "
              f"median={valid['cnr'].median():.4f}  "
              f"std={valid['cnr'].std():.4f}")
        print("\n  per dataset:")
        for ds, sub in valid.groupby("dataset"):
            print(f"    {ds:12s}  n={len(sub):4d}  "
                  f"mean={sub['cnr'].mean():.4f}  "
                  f"median={sub['cnr'].median():.4f}  "
                  f"std={sub['cnr'].std():.4f}")
        print("\n  per side (prev vs curr):")
        for side, sub in valid.groupby("side"):
            print(f"    {side:4s}  n={len(sub):4d}  "
                  f"mean={sub['cnr'].mean():.4f}  "
                  f"median={sub['cnr'].median():.4f}")
        # Pair-level breakdown by progression: a pair belongs to every
        # class it contains, so exploded rows may double-count pairs
        # that mix classes — this is intended (matches how the
        # progression label appears in ``comparison`` as a joined
        # string).
        exploded = valid.assign(
            comparison=valid["comparison"].str.split("|")).explode("comparison")
        print("\n  per progression label (pair belongs to every class it "
              "contains):")
        for lbl, sub in exploded.groupby("comparison"):
            print(f"    {lbl:10s}  n={len(sub):4d}  "
                  f"mean={sub['cnr'].mean():.4f}  "
                  f"median={sub['cnr'].median():.4f}")

    valid_pg = df.dropna(subset=["pointing_game"])
    print("\n" + "=" * 70)
    print(f"{tag} CHANGE-MAP POINTING GAME SUMMARY")
    print("=" * 70)
    print(f"  scored sides:   {len(valid_pg)} / {len(df)}")
    if len(valid_pg):
        print(f"  overall PG:     acc={valid_pg['pointing_game'].mean():.4f}  "
              f"hits={int(valid_pg['pointing_game'].sum())}/{len(valid_pg)}")
        print("\n  per dataset:")
        for ds, sub in valid_pg.groupby("dataset"):
            print(f"    {ds:12s}  n={len(sub):4d}  "
                  f"acc={sub['pointing_game'].mean():.4f}  "
                  f"hits={int(sub['pointing_game'].sum())}/{len(sub)}")
        print("\n  per side (prev vs curr):")
        for side, sub in valid_pg.groupby("side"):
            print(f"    {side:4s}  n={len(sub):4d}  "
                  f"acc={sub['pointing_game'].mean():.4f}  "
                  f"hits={int(sub['pointing_game'].sum())}/{len(sub)}")
        exploded_pg = valid_pg.assign(
            comparison=valid_pg["comparison"].str.split("|")).explode(
                "comparison")
        print("\n  per progression label:")
        for lbl, sub in exploded_pg.groupby("comparison"):
            print(f"    {lbl:10s}  n={len(sub):4d}  "
                  f"acc={sub['pointing_game'].mean():.4f}  "
                  f"hits={int(sub['pointing_game'].sum())}/{len(sub)}")

    if rendered:
        print(f"\n✅ DONE — PNGs written to {out_dir.resolve()}")
    else:
        print("\n✅ DONE — metrics only (no PNGs written, --no-render set)")


if __name__ == "__main__":
    main()
