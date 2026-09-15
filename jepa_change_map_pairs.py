"""Image-image change-map grounding (Approach B) for TempCXRJEPA.

Motivation
----------
The image-image change-map companion to
``biovilt_change_map_pairs.py``. Both scripts ground bounding boxes
with a change map whose *evaluation output* is purely image-space
(no text embeddings in the scoring). They differ only in how the
change map is produced, matching each model's native temporal
mechanism:

  * BioViL-T (``biovilt_change_map_pairs.py``): role-swap the pair-
    conditioned patch encoder, take ``1 - cos(patches_curr, patches_prev)``.
    Fully text-free at both input and output.

  * JEPA (this file): run the JEPA predictor conditioned on each of
    the pair's ``(finding, progression)`` tuples from
    ``gold_bboxes.parquet``, average the predicted current-patch
    tensors across findings, and take the per-patch L2 norm of
    ``ẑ_avg - z_prior``. Text is used as an *input* to the
    predictor (JEPA's predictor requires it) but the change map
    itself is image-space and the evaluation is image-only.

Recipe
------
1. ``_, z_prior = model.image_encoder(prior)``  → ``(1, N, D)``. Matches
   JEPA training-time usage.
2. Collect the pair's ``(finding, progression)`` tuples from the
   gold parquet.
3. For each tuple, build prompt ``"{finding} is {progression}"``, encode
   with ``model.text_encoder.forward_contrastive`` (same as the disease
   / grounding eval scripts).
4. Batch-run the predictor: ``ẑ = predictor(z_prior_expanded,
   txt_local, token_mask)`` → ``(K, N, D)``.
5. Vector-average across tuples: ``ẑ_avg = ẑ.mean(dim=0)``  → ``(N, D)``.
   Then per-patch change = ``‖ẑ_avg[p] - z_prior[p]‖_2``  → ``(N,)``.
6. Bilinearly upsample the 14x14 change grid to 448x448.
7. Ground-truth mask: union of *all* the pair's finding bboxes
   (``prior_bboxes`` on the prev side, ``current_bboxes`` on the
   curr side). One mask per side, same change map for both sides.
8. Score per side with BioViL CNR, Pointing Game, mIoU, and pixel
   AUROC on the model 448x448 grid.

Note on asymmetry with BioViL-T
--------------------------------
JEPA's predictor cannot run without text conditioning, so its change
map necessarily takes ``(finding, progression)`` as input. We pass the
pair's oracle tuples (the ones with bboxes in the gold parquet) so
the predictor is being asked "given what actually changed, where do
you predict the delta?" — which is the fair regime for JEPA's
temporal specialization. BioViL-T's change map uses no text at all.
This is the intentional design trade-off (see the summary at the top
of ``biovilt_change_map_pairs.py``).

Usage
-----
    # Every pair in the gold parquet, PNGs + CSV + summary
    python jepa_change_map_pairs.py --ckpt path/to/best.pt

    # Metrics only, skip PNG rendering
    python jepa_change_map_pairs.py --ckpt path/to/best.pt --no-render

    # Restrict to one dataset
    python jepa_change_map_pairs.py --ckpt path/to/best.pt --datasets mimic

    # Pick N pairs (one per progression class when possible) for a demo
    python jepa_change_map_pairs.py --ckpt path/to/best.pt --num-examples 10

The stdout summary and per-pair CSV are drop-in comparable with
``biovilt_change_map_pairs.py``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from biovilt_change_map_pairs import (
    _report,
    group_by_pair,
    pick_pair_examples,
)
from dataset_combined_jepa import DEFAULT_DATASET_DIR
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from jepa_heatmap_progression_pairs import (
    boxes_mask_in_model_space,
    compute_change_map_side_metrics,
    compute_cnr,
    compute_pointing_game,
    load_image,
    normalize_gold_bboxes_schema,
    render_heatmap_png,
)
from progression_classify import (
    DATASETS,
    _resolve_with_fallbacks,
    discover_gold_image_roots,
)
from tempcxr.modules.jepa import TempCXRJEPA


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_GOLD_PARQUET = os.path.join(DEFAULT_DATASET_DIR, "gold_bboxes.parquet")
DEFAULT_CKPT = os.environ.get("JEPA_CKPT", "checkpoints_jepa_dynamic/best.pt")

INPUT_SIZE = 448
PROMPT_TEMPLATE = "{} is {}"
CLS_ORDER = ["improving", "stable", "worsening", "new", "resolved"]


# ============================================================
# MODEL
# ============================================================
def load_model(ckpt_path: str) -> TempCXRJEPA:
    print(f"🔧 Loading TempCXRJEPA checkpoint: {ckpt_path}")
    device = torch.device(DEVICE)
    model = load_jepa_model(ckpt_path, device)
    model.eval()
    print("✅ Model loaded.")
    return model


def _capitalize_finding(f: str) -> str:
    """Match the training-time capitalization convention used by
    ``eval_disease_jepa.build_disease_prompts``.
    """
    if not f:
        return f
    return f[:1].upper() + f[1:]


@torch.no_grad()
def compute_change_map(model: TempCXRJEPA,
                       prior_img: torch.Tensor,
                       finding_progression_pairs: list[tuple[str, str]],
                       out_size: int = INPUT_SIZE) -> np.ndarray | None:
    """Predictor-delta change map for one image pair.

    * ``prior_img``: model tensor ``(3, H, W)`` for the prev image
      (already on device).
    * ``finding_progression_pairs``: list of ``(finding, progression)``
      tuples from the gold parquet for this pair. Empty list ⇒ returns
      ``None`` (no way to condition the predictor without text).

    Returns a ``(out_size, out_size)`` ``float32`` numpy array of
    ``‖ẑ_avg[p] - z_prior[p]‖_2``, bilinearly upsampled from the
    14x14 patch grid.
    """
    if not finding_progression_pairs:
        return None

    prompts = [PROMPT_TEMPLATE.format(_capitalize_finding(f), p.lower())
               for f, p in finding_progression_pairs]
    _, txt_local, token_mask = model.text_encoder.forward_contrastive(prompts)
    k = len(prompts)

    device = prior_img.device
    prior_b = prior_img.unsqueeze(0).to(device)
    _, z_prior = model.image_encoder(prior_b)  # (1, N, D)

    z_prior_b = z_prior.expand(k, -1, -1).contiguous()
    preds = model.predictor(z_prior_b, txt_local, token_mask)  # (K, N, D)

    # Vector-average across findings first, then per-patch L2 norm of
    # the delta. This is the "average direction of predicted change"
    # rather than the "average magnitude of predicted change".
    z_hat_avg = preds.mean(dim=0)          # (N, D)
    delta = z_hat_avg - z_prior.squeeze(0)  # (N, D)
    change = delta.float().norm(dim=-1)    # (N,)

    L = change.shape[0]
    side = int(round(L ** 0.5))
    assert side * side == L, f"Patch count {L} is not a perfect square"

    grid = change.view(1, 1, side, side)
    up = F.interpolate(grid, size=(out_size, out_size),
                       mode="bilinear", align_corners=False)
    return up.squeeze().cpu().float().numpy()


# Paper JEPA vs unfrozen supervised on the same 1,333 gold sides
# (change_maps_region / job 486113). Loc-loss watch targets are
# energy + mIoU; pixel AUROC is already tied; CNR already leads.
SUPERVISED_GOLD_MAPS = {
    "cnr": 0.5176,
    "energy_in_box": 0.2133,
    "iou_eqarea": 0.1628,
    "pixel_auroc": 0.6778,
    "pointing_game": 0.3121,
}
PAPER_JEPA_GOLD_MAPS = {
    "cnr": 0.5564,
    "energy_in_box": 0.1729,
    "iou_eqarea": 0.1348,
    "pixel_auroc": 0.6730,
    "pointing_game": 0.1725,
}


def summarize_change_map_records(records: list[dict]) -> dict | None:
    """Mean grounding scores over sides that have a valid CNR."""
    if not records:
        return None
    df = pd.DataFrame(records)
    valid = df.dropna(subset=["cnr"])
    if len(valid) == 0:
        return None

    def _mean(col: str):
        if col not in valid.columns:
            return None
        s = valid[col].dropna()
        return float(s.mean()) if len(s) else None

    return {
        "n_sides": int(len(valid)),
        "n_total": int(len(df)),
        "cnr": _mean("cnr"),
        "energy_in_box": _mean("energy_in_box"),
        "iou_eqarea": _mean("iou_eqarea"),
        "pixel_auroc": _mean("pixel_auroc"),
        "pointing_game": _mean("pointing_game"),
        "box_area": _mean("box_area"),
    }


def prepare_gold_change_map_pairs(
    gold_parquet: str,
    image_roots: dict[str, str],
) -> list[dict]:
    """Load ``gold_bboxes.parquet`` once and collapse to scored pairs."""
    df = pd.read_parquet(gold_parquet)
    print(f"[gold-maps] loaded {len(df)} rows from {gold_parquet}")
    df = normalize_gold_bboxes_schema(df)
    print(f"[gold-maps] {len(df)} rows after schema + label filter")

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
    print(f"[gold-maps] {len(df)} rows have both images present")

    pairs = group_by_pair(df)
    path_by_pair: dict[tuple, tuple] = {}
    fp_by_pair: dict[tuple, list[tuple[str, str]]] = {}
    for _, row in df.iterrows():
        k = (row["dataset"], row["patient_id"],
             row["study_id_prev"], row["study_id_curr"])
        if k not in path_by_pair:
            path_by_pair[k] = (row["_prev_resolved"], row["_curr_resolved"])
        fp_by_pair.setdefault(k, []).append(
            (str(row["finding"]), str(row["progression"]).lower()))

    for rec in pairs:
        k = (rec["dataset"], rec["patient_id"],
             rec["study_id_prev"], rec["study_id_curr"])
        rec["_prev_path"], rec["_curr_path"] = path_by_pair[k]
        seen: set[tuple[str, str]] = set()
        rec["_fp_tuples"] = []
        for tup in fp_by_pair[k]:
            if tup in seen:
                continue
            seen.add(tup)
            rec["_fp_tuples"].append(tup)

    print(f"[gold-maps] {len(pairs)} unique pairs (predictor-delta + union box)")
    print(
        "[gold-maps] watch vs supervised: "
        f"energy {PAPER_JEPA_GOLD_MAPS['energy_in_box']:.3f}→"
        f"{SUPERVISED_GOLD_MAPS['energy_in_box']:.3f}  "
        f"mIoU {PAPER_JEPA_GOLD_MAPS['iou_eqarea']:.3f}→"
        f"{SUPERVISED_GOLD_MAPS['iou_eqarea']:.3f}  "
        f"(pixel AUROC already {PAPER_JEPA_GOLD_MAPS['pixel_auroc']:.3f} "
        f"vs {SUPERVISED_GOLD_MAPS['pixel_auroc']:.3f}; "
        f"CNR already leads {PAPER_JEPA_GOLD_MAPS['cnr']:.3f} "
        f"vs {SUPERVISED_GOLD_MAPS['cnr']:.3f})"
    )
    return pairs


@torch.no_grad()
def eval_gold_change_maps(model, pairs, epoch, device=None) -> dict | None:
    """In-training predictor-delta grounding (same recipe as this script)."""
    import sys
    from tqdm import tqdm

    model.eval()
    records: list[dict] = []
    skipped = 0
    pbar = tqdm(
        range(len(pairs)),
        desc=f"gold change-maps ep{epoch}",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for i in pbar:
        rec = pairs[i]
        try:
            prev_t, prev_disp, prev_orig = load_image(rec["_prev_path"])
            curr_t, curr_disp, curr_orig = load_image(rec["_curr_path"])
        except (FileNotFoundError, OSError):
            skipped += 1
            continue
        if device is not None:
            prev_t = prev_t.to(device)

        change_map = compute_change_map(
            model, prev_t, rec["_fp_tuples"], out_size=INPUT_SIZE)
        if change_map is None:
            skipped += 1
            continue

        prev_boxes = rec["prev_boxes"]
        curr_boxes = rec["curr_boxes"]
        prev_mask = boxes_mask_in_model_space(
            prev_boxes, prev_orig, prev_disp.size)
        curr_mask = boxes_mask_in_model_space(
            curr_boxes, curr_orig, curr_disp.size)
        empty = np.zeros_like(change_map, dtype=bool)
        prev_m = compute_change_map_side_metrics(
            change_map, prev_mask if prev_boxes else empty)
        curr_m = compute_change_map_side_metrics(
            change_map, curr_mask if curr_boxes else empty)

        prog_str = "|".join(rec["progressions"])
        find_str = "|".join(rec["findings"])
        meta = {
            "dataset": rec["dataset"],
            "comparison": prog_str,
            "disease_name": find_str,
        }
        records.append({**meta, "side": "prev", **prev_m})
        records.append({**meta, "side": "curr", **curr_m})

        summary = summarize_change_map_records(records)
        if summary is not None:
            pbar.set_postfix(
                skipped=skipped,
                energy=(
                    f"{summary['energy_in_box']:.3f}"
                    if summary["energy_in_box"] is not None else "-"
                ),
                mIoU=(
                    f"{summary['iou_eqarea']:.3f}"
                    if summary["iou_eqarea"] is not None else "-"
                ),
            )
    pbar.close()
    if skipped:
        print(f"[gold-maps] skipped pairs: {skipped}")
    summary = summarize_change_map_records(records)
    if summary is None:
        print("[gold-maps] no scored sides")
        return None

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "-"

    print(
        f"[gold-maps] ep{epoch}  sides={summary['n_sides']}/{summary['n_total']}  "
        f"energy={_fmt(summary['energy_in_box'])} "
        f"(paper {_fmt(PAPER_JEPA_GOLD_MAPS['energy_in_box'])} / "
        f"sup {_fmt(SUPERVISED_GOLD_MAPS['energy_in_box'])})  "
        f"mIoU={_fmt(summary['iou_eqarea'])} "
        f"(paper {_fmt(PAPER_JEPA_GOLD_MAPS['iou_eqarea'])} / "
        f"sup {_fmt(SUPERVISED_GOLD_MAPS['iou_eqarea'])})  "
        f"pixAUROC={_fmt(summary['pixel_auroc'])} "
        f"(sup {_fmt(SUPERVISED_GOLD_MAPS['pixel_auroc'])})  "
        f"CNR={_fmt(summary['cnr'])}  "
        f"PG={_fmt(summary['pointing_game'])}"
    )
    return summary


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", default=DEFAULT_CKPT,
        help=f"Path to a JEPA checkpoint (default: {DEFAULT_CKPT}, or "
             "$JEPA_CKPT).",
    )
    parser.add_argument(
        "--gold-parquet", default=DEFAULT_GOLD_PARQUET,
        help=f"Path to gold_bboxes.parquet (default: {DEFAULT_GOLD_PARQUET}).",
    )
    parser.add_argument(
        "--out-dir", default="change_maps_jepa",
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

    # Same-image-per-pair resolution + attach per-pair (finding,
    # progression) tuples for predictor conditioning.
    path_by_pair: dict[tuple, tuple[Path, Path]] = {}
    fp_by_pair: dict[tuple, list[tuple[str, str]]] = {}
    for _, row in df.iterrows():
        k = (row["dataset"], row["patient_id"],
             row["study_id_prev"], row["study_id_curr"])
        if k not in path_by_pair:
            path_by_pair[k] = (row["_prev_resolved"], row["_curr_resolved"])
        fp_by_pair.setdefault(k, []).append(
            (str(row["finding"]), str(row["progression"]).lower()))

    for rec in pairs:
        k = (rec["dataset"], rec["patient_id"],
             rec["study_id_prev"], rec["study_id_curr"])
        rec["_prev_path"], rec["_curr_path"] = path_by_pair[k]
        # De-duplicate while preserving order.
        seen: set[tuple[str, str]] = set()
        rec["_fp_tuples"] = []
        for tup in fp_by_pair[k]:
            if tup in seen:
                continue
            seen.add(tup)
            rec["_fp_tuples"].append(tup)

    if args.num_examples is not None:
        pairs = pick_pair_examples(pairs, n=args.num_examples, seed=args.seed)
        print(f"🎯 Down-sampled to {len(pairs)} pairs (--num-examples)")

    model = load_model(args.ckpt)

    records: list[dict] = []
    for i, rec in enumerate(pairs):
        prev_t, prev_disp, prev_orig = load_image(rec["_prev_path"])
        curr_t, curr_disp, curr_orig = load_image(rec["_curr_path"])

        change_map = compute_change_map(
            model, prev_t, rec["_fp_tuples"], out_size=INPUT_SIZE)
        if change_map is None:
            # No (finding, progression) tuples for this pair — should
            # never happen after the label filter but skip defensively.
            continue

        vmin = float(change_map.min())
        vmax = float(change_map.max())

        prev_boxes = rec["prev_boxes"]
        curr_boxes = rec["curr_boxes"]

        prev_mask = boxes_mask_in_model_space(
            prev_boxes, prev_orig, prev_disp.size)
        curr_mask = boxes_mask_in_model_space(
            curr_boxes, curr_orig, curr_disp.size)
        prev_m = (
            compute_change_map_side_metrics(change_map, prev_mask)
            if prev_boxes else compute_change_map_side_metrics(
                change_map, np.zeros_like(change_map, dtype=bool))
        )
        curr_m = (
            compute_change_map_side_metrics(change_map, curr_mask)
            if curr_boxes else compute_change_map_side_metrics(
                change_map, np.zeros_like(change_map, dtype=bool))
        )
        prev_cnr, prev_pg = prev_m["cnr"], prev_m["pointing_game"]
        curr_cnr, curr_pg = curr_m["cnr"], curr_m["pointing_game"]

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
            "n_predictor_tuples": len(rec["_fp_tuples"]),
        }
        records.append({**meta_common, "side": "prev",
                        "n_boxes": len(prev_boxes), **prev_m})
        records.append({**meta_common, "side": "curr",
                        "n_boxes": len(curr_boxes), **curr_m})

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
                f"K={len(rec['_fp_tuples']):2d} "
                f"progs={prog_str[:22]:22s} "
                f"prev_box={len(prev_boxes)} curr_box={len(curr_boxes)} "
                f"CNR(prev)={_fmt(prev_cnr)} CNR(curr)={_fmt(curr_cnr)} "
                f"PG(prev)={_fmt_pg(prev_pg)} PG(curr)={_fmt_pg(curr_pg)}"
                + (f"  drew prev={p_drawn} curr={c_drawn}"
                   if not args.no_render else "")
            )

    # Attach ckpt to every record so the CSV records provenance, then
    # reuse the shared reporter for aggregation printouts.
    for r in records:
        r["ckpt"] = args.ckpt
    _report(records, out_dir=out_dir, cnr_csv=args.cnr_csv,
            tag="JEPA (predictor-delta)", rendered=not args.no_render)


if __name__ == "__main__":
    main()
