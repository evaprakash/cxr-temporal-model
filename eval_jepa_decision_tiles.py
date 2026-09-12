#!/usr/bin/env python3
"""Where paper JEPA's 5-way argmax actually comes from (per-tile votes).

Paper score is ``mean_n cos(ẑ^c[n], z_cur[n])``. This script does **not**
change that. It writes the tiles that moved the winner:

    m[n] = cos(ẑ^ŷ[n], z_cur[n]) − max_{c≠ŷ} cos(ẑ^c[n], z_cur[n])

Positive ``m`` = tiles that caused the pick. Then, when gold boxes exist
on the **current** film, it reports how much of that positive mass sits
inside the finding box vs the rest of the chest.

Also prints ordinary perpatch set-match so you can confirm the same
~0.452 paper number.

    python eval_jepa_decision_tiles.py --eval \\
        --ckpt checkpoints_jepa_dynamic_cbw99999/epoch_5.pt

    # Cluster: sbatch eval_jepa_decision_tiles.sh
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from dataset_combined_jepa import DEFAULT_DATASET_DIR, DEFAULT_FINDINGS
from eval_progression_jepa import PROMPT_TEMPLATE, _encode_prompts
from gold_jepa_diagnostics import (
    five_forecast_offdiag_cos,
    print_jepa_score_diagnostics,
)
from gold_progression_setmatch import (
    PAIR_FINDING_KEYS,
    format_running_setmatch,
    group_gold_by_pair_finding,
    print_setmatch_report,
    topk_set_match,
)
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from jepa_heatmap_progression_pairs import (
    INPUT_SIZE,
    _parse_bboxes,
    boxes_mask_in_model_space,
    compute_cnr,
    compute_pointing_game,
    load_image,
    normalize_gold_bboxes_schema,
    render_heatmap_png,
)
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    _resolve_with_fallbacks,
    discover_gold_image_roots,
    load_gold_pairs,
)
from progression_phrases import CLS_ORDER

DEFAULT_BBOX_PARQUET = os.path.join(DEFAULT_DATASET_DIR, "gold_bboxes.parquet")
DEFAULT_CKPT = os.environ.get(
    "JEPA_CKPT", "checkpoints_jepa_dynamic_cbw99999/epoch_5.pt"
)
DEFAULT_OUT_DIR = "decision_tiles_jepa_paper"


def _upsample_tiles(tiles: torch.Tensor, out_size: int = INPUT_SIZE) -> np.ndarray:
    """``(N,)`` patch grid → ``(out_size, out_size)`` bilinear map."""
    n = int(tiles.numel())
    side = int(round(math.sqrt(n)))
    if side * side != n:
        raise ValueError(f"patch count {n} is not a square")
    grid = tiles.float().view(1, 1, side, side)
    up = F.interpolate(
        grid, size=(out_size, out_size), mode="bilinear", align_corners=False,
    )
    return up.squeeze().cpu().numpy()


def _tile_coverage(mask_hw: np.ndarray, n_tiles: int) -> np.ndarray:
    """448×448 bool mask → ``(N,)`` float coverage on the patch grid."""
    side = int(round(math.sqrt(n_tiles)))
    t = torch.from_numpy(mask_hw.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.avg_pool2d(t, kernel_size=INPUT_SIZE // side)
    return pooled.view(-1).numpy()


def _pair_key(row) -> Tuple:
    return tuple(row[k] for k in PAIR_FINDING_KEYS)


def _load_bbox_lookup(bbox_parquet: str) -> Dict[Tuple, list]:
    """``PAIR_FINDING_KEYS`` → union of current-side gold boxes."""
    if not os.path.isfile(bbox_parquet):
        print(f"[tiles] no bbox parquet at {bbox_parquet}; spatial stats skipped")
        return {}
    df = pd.read_parquet(bbox_parquet)
    df = normalize_gold_bboxes_schema(df)
    df["finding_lc"] = df["finding"].astype(str).str.strip().str.lower()
    lookup: Dict[Tuple, list] = {}
    for key, sub in df.groupby(PAIR_FINDING_KEYS, dropna=False, sort=False):
        key_t = key if isinstance(key, tuple) else (key,)
        boxes: list = []
        for raw in sub["current_bboxes"]:
            boxes.extend(_parse_bboxes(raw))
        lookup[key_t] = boxes
    print(f"[tiles] bbox lookup: {len(lookup)} (pair, finding) keys")
    return lookup


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _print_bucket(title: str, rows: List[dict], field: str) -> None:
    vals = [r[field] for r in rows if r.get(field) is not None
            and not (isinstance(r[field], float) and math.isnan(r[field]))]
    print(f"  {title:<28} n={len(vals):4d}  {field}={_mean(vals):.4f}")


@torch.no_grad()
def score_decision_tiles(model, prior_t, current_t, finding, device, text_cache):
    """Return paper 5-way scores plus per-tile cos / margin / vote."""
    prompts, txt_local, token_mask = _encode_prompts(
        model, finding, PROMPT_TEMPLATE, device, text_cache,
    )
    n_prompts = len(prompts)
    prior = prior_t.unsqueeze(0).to(device)
    current = current_t.unsqueeze(0).to(device)
    _, z_prior = model.image_encoder(prior)
    _, z_cur = model.target_image_encoder(current)
    z_cur = z_cur.detach()
    z_prior_b = z_prior.expand(n_prompts, -1, -1).contiguous()
    preds = model.predictor(z_prior_b, txt_local, token_mask)
    pred_f = F.normalize(preds.float(), dim=-1)
    target_f = F.normalize(z_cur.float(), dim=-1)
    cos_tiles = (pred_f * target_f).sum(dim=-1)  # (C, N)
    scores = cos_tiles.mean(dim=1)
    pred_i = int(scores.argmax().item())
    second = scores.clone()
    second[pred_i] = -1e9
    second_i = int(second.argmax().item())
    # Same runner-up class on every tile (the class the mean lost to).
    margin = cos_tiles[pred_i] - cos_tiles[second_i]
    others = cos_tiles.clone()
    others[pred_i] = -1e9
    margin_vs_best_other = cos_tiles[pred_i] - others.max(dim=0).values
    tile_vote = cos_tiles.argmax(dim=0)
    return {
        "scores": [float(x) for x in scores.tolist()],
        "pred_i": pred_i,
        "second_i": second_i,
        "cos_tiles": cos_tiles.cpu(),
        "margin": margin.cpu(),
        "margin_vs_best_other": margin_vs_best_other.cpu(),
        "tile_vote": tile_vote.cpu(),
        "zhat_offdiag": five_forecast_offdiag_cos(pred_f),
        "prompts": prompts,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--gold-parquet", default=DEFAULT_GOLD_PARQUET)
    parser.add_argument("--bbox-parquet", default=DEFAULT_BBOX_PARQUET)
    parser.add_argument("--findings-parquet", default=DEFAULT_FINDINGS)
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="CSV + optional PNG overlays.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-root", action="append", default=[], metavar="DATASET=PATH",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--num-examples", type=int, default=20,
        help="How many current-film margin PNGs to write (0 = none).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip PNGs; still write CSV + print tables.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    auto_gold_roots = discover_gold_image_roots(parquet_dir)
    image_roots: Dict[str, str] = {**IMAGE_ROOTS, **auto_gold_roots}
    if auto_gold_roots:
        print("[gold] auto-detected gold image roots:")
        for d, p in auto_gold_roots.items():
            print(f"  {d}: {p}")
    for spec in args.image_root:
        if "=" not in spec:
            raise ValueError(f"--image-root expects DATASET=PATH, got {spec!r}")
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}")
        image_roots[d] = p

    gold_df = load_gold_pairs(args.gold_parquet, args.findings_parquet)
    groups = group_gold_by_pair_finding(gold_df)
    if args.limit is not None:
        groups = groups.head(args.limit).reset_index(drop=True)
        print(f"[tiles] --limit → {len(groups)} groups")
    bbox_lookup = _load_bbox_lookup(args.bbox_parquet)

    device = torch.device(args.device)
    model = load_jepa_model(args.ckpt, device)
    model.eval()

    text_cache: Dict = {}
    results = []
    diag_rows = []
    tile_rows: List[dict] = []
    skipped = 0

    print(
        f"\n[tiles] decision maps on {len(groups)} groups "
        f"(paper perpatch 5-way; margin = ŷ vs runner-up)"
    )
    for i in range(len(groups)):
        row = groups.iloc[i]
        finding = str(row["finding"])
        gt_labels = list(row["gt_labels"])
        try:
            prior_path = _resolve_with_fallbacks(
                row["dataset"], row["parent_image_prev"], image_roots,
            )
            curr_path = _resolve_with_fallbacks(
                row["dataset"], row["parent_image_curr"], image_roots,
            )
            prior_t, _, _ = load_image(prior_path)
            curr_t, curr_disp, curr_orig = load_image(curr_path)
        except (FileNotFoundError, OSError) as e:
            skipped += 1
            if skipped <= 5:
                print(f"[tiles] skip {i}: {e}")
            continue

        out = score_decision_tiles(
            model, prior_t, curr_t, finding, device, text_cache,
        )
        sm = topk_set_match(
            out["scores"], gt_labels, CLS_ORDER, finding=finding,
        )
        results.append(sm)
        diag_rows.append(
            {
                "scores": out["scores"],
                "gt_labels": gt_labels,
                "zhat_offdiag": out["zhat_offdiag"],
            }
        )

        pred = CLS_ORDER[out["pred_i"]]
        second = CLS_ORDER[out["second_i"]]
        margin = out["margin"].numpy()
        margin_bo = out["margin_vs_best_other"].numpy()
        votes = out["tile_vote"].numpy()
        n = margin.size
        pos = np.clip(margin, 0.0, None)
        pos_sum = float(pos.sum())
        agree = float((votes == out["pred_i"]).mean())

        boxes = bbox_lookup.get(_pair_key(row), [])
        rec = {
            "dataset": row["dataset"],
            "patient_id": row["patient_id"],
            "study_id_prev": row["study_id_prev"],
            "study_id_curr": row["study_id_curr"],
            "finding": finding,
            "gt": "|".join(gt_labels),
            "gt0": gt_labels[0] if len(gt_labels) == 1 else "",
            "is_single": int(len(gt_labels) == 1),
            "pred": pred,
            "second": second,
            "correct": int(len(gt_labels) == 1 and pred == gt_labels[0]),
            "score_pred": out["scores"][out["pred_i"]],
            "score_second": out["scores"][out["second_i"]],
            "mean_margin": float(margin.mean()),
            "frac_tiles_agree": agree,
            "frac_tiles_pos": float((margin > 0).mean()),
            "frac_tiles_pos_strict": float((margin_bo > 0).mean()),
            "has_box": int(bool(boxes)),
            "box_area": float("nan"),
            "pos_mass_in_box": float("nan"),
            "mean_margin_in": float("nan"),
            "mean_margin_out": float("nan"),
            "cnr_margin": float("nan"),
            "pg_margin": float("nan"),
            "parent_image_prev": row["parent_image_prev"],
            "parent_image_curr": row["parent_image_curr"],
        }

        if boxes:
            mask = boxes_mask_in_model_space(
                boxes, curr_orig, curr_disp.size, INPUT_SIZE,
            )
            if mask.any():
                cov = _tile_coverage(mask, n)
                w = cov
                rec["box_area"] = float(w.mean())
                w_out = np.clip(1.0 - w, 0.0, None)
                if pos_sum > 1e-12 and float(w.sum()) > 1e-12:
                    rec["pos_mass_in_box"] = float((pos * w).sum() / pos_sum)
                if float(w.sum()) > 1e-12:
                    rec["mean_margin_in"] = float((margin * w).sum() / w.sum())
                if float(w_out.sum()) > 1e-12:
                    rec["mean_margin_out"] = float(
                        (margin * w_out).sum() / w_out.sum()
                    )
                hm = _upsample_tiles(torch.from_numpy(margin))
                cnr = compute_cnr(hm, mask)
                pg = compute_pointing_game(hm, mask)
                if cnr is not None:
                    rec["cnr_margin"] = float(cnr)
                if pg is not None:
                    rec["pg_margin"] = float(pg)
                rec["_margin"] = margin
                rec["_boxes"] = boxes
                rec["_curr_disp"] = curr_disp
                rec["_curr_orig"] = curr_orig
                rec["_mask"] = mask

        tile_rows.append(rec)
        if (i + 1) % max(1, len(groups) // 20) == 0:
            print(
                f"[tiles]   {i + 1}/{len(groups)}  skipped={skipped}  "
                f"{format_running_setmatch(results)}"
            )

    if skipped:
        print(f"\n[tiles] skipped missing images: {skipped}")
    if not results:
        print("[tiles] nothing evaluated")
        return

    print_setmatch_report(results, "jepa", ", pooling=perpatch, decision-tiles")
    print_jepa_score_diagnostics(diag_rows)

    singles = [r for r in tile_rows if r["is_single"]]
    boxed = [r for r in tile_rows if r["has_box"]]
    print()
    print("------------------------------------------------------------")
    print("(E) Decision tiles — what made the argmax")
    print("------------------------------------------------------------")
    print(f"  n_groups                 {len(tile_rows)}")
    print(f"  n_single                 {len(singles)}")
    print(f"  n_with_current_box       {len(boxed)}")
    print(f"  mean frac tiles agree    {_mean([r['frac_tiles_agree'] for r in tile_rows]):.4f}"
          f"   (tile-argmax == mean-argmax)")
    print(f"  mean frac tiles pos      {_mean([r['frac_tiles_pos'] for r in tile_rows]):.4f}"
          f"   (m[n] > 0 vs runner-up class)")
    print()
    print("  Positive-margin mass inside current finding box "
          "(1 = decision is on the finding; ~box_area = uniform):")
    _print_bucket("box area (chance)", boxed, "box_area")
    _print_bucket("all boxed", boxed, "pos_mass_in_box")
    _print_bucket("single boxed", [r for r in boxed if r["is_single"]],
                  "pos_mass_in_box")
    _print_bucket("single correct", [r for r in boxed if r["correct"] == 1],
                  "pos_mass_in_box")
    _print_bucket("single wrong", [r for r in boxed if r["is_single"] and r["correct"] == 0],
                  "pos_mass_in_box")
    print()
    print("  mean margin in vs out (boxed singles):")
    sb = [r for r in boxed if r["is_single"]]
    print(f"    in-box   {_mean([r['mean_margin_in'] for r in sb]):.4f}")
    print(f"    out-box  {_mean([r['mean_margin_out'] for r in sb]):.4f}")
    print(f"    CNR      {_mean([r['cnr_margin'] for r in sb]):.4f}")
    print(f"    PG       {_mean([r['pg_margin'] for r in sb]):.4f}")
    print()
    print("  pos_mass_in_box by GT class (single + boxed):")
    for cls in CLS_ORDER:
        _print_bucket(
            cls,
            [r for r in boxed if r["gt0"] == cls],
            "pos_mass_in_box",
        )
    print()
    print("  tile-agree by GT class (all singles):")
    for cls in CLS_ORDER:
        _print_bucket(
            cls,
            [r for r in singles if r["gt0"] == cls],
            "frac_tiles_agree",
        )
    print()
    print("  Read: if pos_mass_in_box ≈ finding area, the 5-way is holistic.")
    print("        if pos_mass_in_box >> area on correct improving/worsening,")
    print("        the decision already sits on the finding.")
    print("        resolved with low mean margin everywhere is a dead ẑ.")

    csv_path = out_dir / "decision_tiles.csv"
    drop = {"_margin", "_boxes", "_curr_disp", "_curr_orig", "_mask"}
    pd.DataFrame(
        [{k: v for k, v in r.items() if k not in drop} for r in tile_rows]
    ).to_csv(csv_path, index=False)
    print(f"\n[tiles] wrote {csv_path}")

    if args.no_render or args.num_examples <= 0:
        print("[tiles] done (no PNGs).")
        return

    rng = np.random.RandomState(args.seed)
    candidates = [r for r in tile_rows if "_margin" in r]
    if not candidates:
        print("[tiles] no boxed rows to render")
        return
    # Prefer a mix of classes and correct/wrong.
    picked: List[dict] = []
    by = {}
    for r in candidates:
        by.setdefault((r["gt0"] or "multi", r["correct"]), []).append(r)
    keys = list(by.keys())
    rng.shuffle(keys)
    while len(picked) < args.num_examples and any(by.values()):
        for k in keys:
            if not by[k]:
                continue
            idx = int(rng.randint(0, len(by[k])))
            picked.append(by[k].pop(idx))
            if len(picked) >= args.num_examples:
                break
    png_dir = out_dir / "png"
    png_dir.mkdir(exist_ok=True)
    n_ok = 0
    for j, r in enumerate(picked):
        hm = _upsample_tiles(torch.from_numpy(r["_margin"]))
        vmax = float(max(abs(hm.min()), abs(hm.max()), 1e-6))
        fname = (
            f"{j:02d}_{r['dataset']}_{r['finding']}_"
            f"gt-{r['gt'].replace('|', '+')}_pred-{r['pred']}_"
            f"{'ok' if r['correct'] else 'bad'}.png"
        )
        render_heatmap_png(
            r["_curr_disp"], hm, r["_boxes"], r["_curr_orig"],
            vmin=-vmax, vmax=vmax, alpha=args.alpha,
            out_path=png_dir / fname,
        )
        n_ok += 1
    print(f"[tiles] wrote {n_ok} PNGs under {png_dir}")
    print("[tiles] done.")


if __name__ == "__main__":
    main()
