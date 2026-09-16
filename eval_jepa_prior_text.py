#!/usr/bin/env python3
"""Prior + text JEPA-only eval on CheXTemporal gold boxes.

The predictor never sees the current film. Supervised pair-CE cannot run
these tests (no ``ẑ``). Current is loaded only as a *target* for section D.

  A. Class-conditional residual on THAT finding's prior boxes
     ``map^c = ‖ẑ("{Finding} is {class}.") − z_prior‖``
     Score energy / mIoU / AUROC / CNR vs the gold prior boxes (not the
     union of every finding on the pair).
     Headline: in-box residual of worsening vs stable on the SAME prior.

  B. Text ablations (same prior, same boxes)
     * finding-only   ``"{Finding}."``
     * wrong-finding  ``"{Other} is worsening."``
     If finding-only ≈ worsening, the class word is unused.
     If wrong-finding ≈ right worsening, the finding word is unused.

  C. Film-level (no boxes; includes edema / empty-prior-box rows)
     ``mean_p ‖ẑ^c − z_prior‖``, off-diag ``cos(ẑ^i, ẑ^j)``.

  D. Current as target only (not prior-only; JEPA-only vs supervised)
     Is ``ẑ^gt`` closer to ``z_cur`` than do-nothing ``z_prior``?
     Single-label 5-way argmax sanity.

Default ckpt is the paper 0.452 run:
``checkpoints_jepa_dynamic_cbw99999/epoch_5.pt``.

Usage
-----
    python eval_jepa_prior_text.py --eval
    python eval_jepa_prior_text.py --eval --limit 40
    sbatch eval_jepa_prior_text.sh
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset_combined_jepa import DEFAULT_DATASET_DIR
from eval_progression_jepa import PROMPT_TEMPLATE, build_class_prompts
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from jepa_heatmap_progression_pairs import (
    INPUT_SIZE,
    _parse_bboxes,
    boxes_mask_in_model_space,
    compute_change_map_side_metrics,
    load_image,
    normalize_gold_bboxes_schema,
)
from progression_classify import (
    DATASETS,
    _resolve_with_fallbacks,
    discover_gold_image_roots,
)
from progression_phrases import CLS_ORDER

DEFAULT_JEPA_CKPT = "checkpoints_jepa_dynamic_cbw99999/epoch_5.pt"
DEFAULT_GOLD_BBOXES = os.path.join(DEFAULT_DATASET_DIR, "gold_bboxes.parquet")
N_CLS = len(CLS_ORDER)
I_STABLE = CLS_ORDER.index("stable")
I_WORSE = CLS_ORDER.index("worsening")

GOLD_FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged cardiomediastinum",
    "lung lesion",
    "lung opacity",
    "pleural effusion",
    "pleural other",
    "pneumonia",
    "pneumothorax",
]

BOX_METRIC_KEYS = (
    "energy_in_box",
    "iou_eqarea",
    "pixel_auroc",
    "cnr",
    "mean_in",
    "mean_out",
    "box_area",
    "pointing_game",
)


def _other_finding(finding: str) -> str:
    """Deterministic wrong finding (not ``hash()``, which is salted)."""
    f = finding.strip().lower()
    others = [x for x in GOLD_FINDINGS if x != f]
    if not others:
        return "pneumothorax"
    return others[sum(ord(c) for c in f) % len(others)]


def _capitalize(s: str) -> str:
    if not s:
        return s
    return s[:1].upper() + s[1:]


def _nanmean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x == x]
    return sum(vals) / len(vals) if vals else float("nan")


def _mean_patch_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    an = F.normalize(a.float(), dim=-1)
    bn = F.normalize(b.float(), dim=-1)
    return (an * bn).sum(dim=-1).mean(dim=-1)


def _film_energy(zhat: torch.Tensor, z_prior: torch.Tensor) -> torch.Tensor:
    return (zhat.float() - z_prior.float()).norm(dim=-1).mean(dim=-1)


def residual_map(zhat: torch.Tensor, z_prior: torch.Tensor,
                 out_size: int = INPUT_SIZE) -> np.ndarray:
    """``zhat, z_prior`` are ``(N, D)``. Returns ``(out_size, out_size)``."""
    change = (zhat.float() - z_prior.float()).norm(dim=-1)
    n = int(change.numel())
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(f"Patch count {n} is not a perfect square")
    grid = change.view(1, 1, side, side)
    up = F.interpolate(
        grid, size=(out_size, out_size), mode="bilinear", align_corners=False,
    )
    return up.squeeze().cpu().float().numpy()


def _encode_prior_text_prompts(
    model,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Dict[str, Tuple],
):
    """Five class templates + finding-only + wrong-finding worsening."""
    other = _other_finding(finding)
    cache_key = f"{finding}||{template}||{other}"
    if cache_key in text_cache:
        prompts, txt_local, token_mask = text_cache[cache_key]
        return other, prompts, txt_local.to(device), token_mask.to(device)

    f_cap = _capitalize(finding)
    o_cap = _capitalize(other)
    prompts = build_class_prompts(finding, template)
    prompts = list(prompts) + [
        f"{f_cap}.",
        template.format(o_cap, "worsening"),
    ]
    _, txt_local, token_mask = model.text_encoder.forward_contrastive(prompts)
    text_cache[cache_key] = (
        prompts,
        txt_local.detach().cpu(),
        token_mask.detach().cpu(),
    )
    return other, prompts, txt_local.to(device), token_mask.to(device)


def _fmt(v: float, nd: int = 4) -> str:
    if v != v:
        return "   nan"
    return f"{v:.{nd}f}"


def _print_box_block(title: str, rows: List[dict], key: str = "energy_in_box") -> None:
    print(f"  {title}  n={len(rows)}")
    if not rows:
        return
    for cls in CLS_ORDER:
        vals = [r[f"{cls}_{key}"] for r in rows]
        print(f"    {cls:<12} {_fmt(_nanmean(vals))}")
    worse = [r[f"worsening_{key}"] for r in rows]
    stable = [r[f"stable_{key}"] for r in rows]
    improve = [r[f"improving_{key}"] for r in rows]
    d_ws = [w - s for w, s in zip(worse, stable) if w == w and s == s]
    d_is = [i - s for i, s in zip(improve, stable) if i == i and s == s]
    frac_ws = _nanmean([1.0 if d > 0 else 0.0 for d in d_ws])
    print(f"    worsening−stable     {_fmt(_nanmean(d_ws))}   "
          f"(frac worse>stable={_fmt(frac_ws)})")
    print(f"    improving−stable     {_fmt(_nanmean(d_is))}")
    fo = [r["finding_only_" + key] for r in rows]
    wf = [r["wrong_finding_" + key] for r in rows]
    print(f"    finding-only         {_fmt(_nanmean(fo))}")
    print(f"    wrong-finding worse  {_fmt(_nanmean(wf))}")
    print(f"    worse−finding-only   {_fmt(_nanmean([w - f for w, f in zip(worse, fo) if w == w and f == f]))}")
    print(f"    worse−wrong-finding  {_fmt(_nanmean([w - x for w, x in zip(worse, wf) if w == w and x == x]))}")


@torch.no_grad()
def run_eval(args, df: pd.DataFrame, image_roots: Dict[str, str],
             device: torch.device) -> None:
    model = load_jepa_model(args.jepa_ckpt, device)
    model.eval()
    text_cache: Dict[str, Tuple] = {}

    rows_out: List[dict] = []
    boxed: List[dict] = []
    skipped = 0

    pbar = tqdm(
        range(len(df)),
        desc="prior+text",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for i in pbar:
        row = df.iloc[i]
        dataset = str(row["dataset"])
        try:
            prev_path = _resolve_with_fallbacks(
                dataset, row["parent_image_prev"], image_roots,
            )
            curr_path = _resolve_with_fallbacks(
                dataset, row["parent_image_curr"], image_roots,
            )
        except FileNotFoundError:
            skipped += 1
            continue

        try:
            prev_t, prev_disp, prev_orig = load_image(prev_path)
            curr_t, _, _ = load_image(curr_path)
        except (FileNotFoundError, OSError):
            skipped += 1
            continue

        finding = str(row["finding"]).strip().lower()
        gt = str(row["progression"]).strip().lower()
        if gt not in CLS_ORDER:
            skipped += 1
            continue
        gt_idx = CLS_ORDER.index(gt)

        other, prompts, txt_local, token_mask = _encode_prior_text_prompts(
            model, finding, args.template, device, text_cache,
        )
        n_txt = txt_local.shape[0]
        prev_t = prev_t.to(device)
        curr_t = curr_t.to(device)
        _, z_prior = model.image_encoder(prev_t.unsqueeze(0))
        z_prior_b = z_prior.expand(n_txt, -1, -1).contiguous()
        zhats = model.predictor(z_prior_b, txt_local, token_mask)
        z_p = z_prior.squeeze(0)

        film_e = _film_energy(zhats[:N_CLS], z_prior.expand(N_CLS, -1, -1))
        zhat_n = F.normalize(zhats[:N_CLS].float(), dim=-1)
        sim = torch.einsum("cnd,knd->ck", zhat_n, zhat_n) / zhat_n.shape[1]
        off = (sim.sum() - sim.trace()) / (N_CLS * (N_CLS - 1))

        rec: dict = {
            "dataset": dataset,
            "patient_id": row.get("patient_id", ""),
            "study_id_prev": row.get("study_id_prev", ""),
            "study_id_curr": row.get("study_id_curr", ""),
            "finding": finding,
            "gt": gt,
            "wrong_finding": other,
            "has_prior_box": 0,
            "zhat_offdiag_cos": float(off.item()),
            "cos_zhat_stable_worsening": float(sim[I_STABLE, I_WORSE].item()),
            "film_energy_finding_only": float(
                _film_energy(zhats[N_CLS:N_CLS + 1], z_prior).item()
            ),
            "film_energy_wrong_finding": float(
                _film_energy(zhats[N_CLS + 1:N_CLS + 2], z_prior).item()
            ),
        }
        for c_i, cls in enumerate(CLS_ORDER):
            rec[f"film_energy_{cls}"] = float(film_e[c_i].item())
            for mk in BOX_METRIC_KEYS:
                rec[f"{cls}_{mk}"] = float("nan")
        for tag in ("finding_only", "wrong_finding"):
            for mk in BOX_METRIC_KEYS:
                rec[f"{tag}_{mk}"] = float("nan")

        prev_boxes = _parse_bboxes(row.get("prior_bboxes"))
        if prev_boxes:
            mask = boxes_mask_in_model_space(
                prev_boxes, prev_orig, prev_disp.size,
            )
            if mask.any():
                rec["has_prior_box"] = 1
                maps = {
                    cls: residual_map(zhats[c_i], z_p)
                    for c_i, cls in enumerate(CLS_ORDER)
                }
                maps["finding_only"] = residual_map(zhats[N_CLS], z_p)
                maps["wrong_finding"] = residual_map(zhats[N_CLS + 1], z_p)
                for tag, hm in maps.items():
                    mets = compute_change_map_side_metrics(hm, mask)
                    for mk in BOX_METRIC_KEYS:
                        v = mets.get(mk)
                        rec[f"{tag}_{mk}"] = (
                            float(v) if v is not None else float("nan")
                        )
                boxed.append(rec)

        if not args.skip_current:
            _, z_cur = model.target_image_encoder(curr_t.unsqueeze(0))
            z_cur = z_cur.detach()
            cos_to_cur = _mean_patch_cos(
                zhats[:N_CLS], z_cur.expand(N_CLS, -1, -1),
            )
            naive_cos = float(_mean_patch_cos(z_prior, z_cur).item())
            d_pred = 1.0 - float(cos_to_cur[gt_idx].item())
            d_naive = 1.0 - naive_cos
            rec["d_pred"] = d_pred
            rec["d_naive"] = d_naive
            rec["pred_beats_naive"] = int(d_pred < d_naive)
            rec["pred_class"] = CLS_ORDER[int(cos_to_cur.argmax().item())]
            rec["single_5way_ok"] = int(rec["pred_class"] == gt)
        else:
            rec["d_pred"] = float("nan")
            rec["d_naive"] = float("nan")
            rec["pred_beats_naive"] = float("nan")
            rec["pred_class"] = ""
            rec["single_5way_ok"] = float("nan")

        rows_out.append(rec)
        if boxed:
            last = boxed[-1]
            pbar.set_postfix(
                skipped=skipped,
                boxed=len(boxed),
                d_ws=(
                    f"{last['worsening_energy_in_box'] - last['stable_energy_in_box']:+.3f}"
                    if last["has_prior_box"]
                    and last["worsening_energy_in_box"] == last["worsening_energy_in_box"]
                    else "-"
                ),
            )

    pbar.close()
    n = len(rows_out)
    print()
    print(f"[prior-text] ckpt     = {args.jepa_ckpt}")
    print(f"[prior-text] template = {args.template!r}")
    print(f"[prior-text] rows     = {n}  skipped={skipped}  "
          f"with_prior_box={len(boxed)}")
    print()

    print("=" * 64)
    print("A. Class-conditional residual vs THAT finding's prior boxes")
    print("   map = ‖ẑ(class) − z_prior‖   (current film not used)")
    print("=" * 64)
    _print_box_block("all rows with a prior box", boxed, "energy_in_box")
    print()
    print("  same rows, mIoU (hottest |box| pixels):")
    _print_box_block("mIoU", boxed, "iou_eqarea")
    print()
    print("  same rows, pixel AUROC:")
    _print_box_block("pixel AUROC", boxed, "pixel_auroc")
    print()
    print("  mean_in (raw residual inside the box; high = change there):")
    _print_box_block("mean_in", boxed, "mean_in")
    print()

    by_gt: Dict[str, List[dict]] = {c: [] for c in CLS_ORDER}
    for r in boxed:
        by_gt[r["gt"]].append(r)
    print("  energy_in_box stratified by gold class of this finding:")
    for cls in CLS_ORDER:
        _print_box_block(f"GT={cls}", by_gt[cls], "energy_in_box")
        print()

    print("=" * 64)
    print("B. How to read A")
    print("=" * 64)
    print("  worsening energy >> stable energy     class word places change")
    print("  finding-only ≈ worsening              class word unused")
    print("  wrong-finding ≈ right worsening       finding word unused")
    print("  GT=stable still has worse>stable      counterfactual knob (good)")
    print("  GT=new usually has no prior box       those rows are in C only")
    print()

    print("=" * 64)
    print("C. Film-level  mean_p ‖ẑ^c − z_prior‖  (no boxes, no current)")
    print("=" * 64)
    print(f"  n={n}")
    for cls in CLS_ORDER:
        print(f"  {cls:<12} {_fmt(_nanmean([r[f'film_energy_{cls}'] for r in rows_out]))}")
    print(f"  finding-only {_fmt(_nanmean([r['film_energy_finding_only'] for r in rows_out]))}")
    print(f"  wrong-finding {_fmt(_nanmean([r['film_energy_wrong_finding'] for r in rows_out]))}")
    print(f"  off-diag cos(ẑ^i, ẑ^j)     "
          f"{_fmt(_nanmean([r['zhat_offdiag_cos'] for r in rows_out]))}")
    print(f"  cos(ẑ^stable, ẑ^worsening) "
          f"{_fmt(_nanmean([r['cos_zhat_stable_worsening'] for r in rows_out]))}")
    print("  (~1 = five forecasts are the same film)")
    print()

    if not args.skip_current and n:
        print("=" * 64)
        print("D. Current as target only (JEPA-only, not prior-only)")
        print("=" * 64)
        print(f"  1-cos(ẑ^gt, z_cur)     {_fmt(_nanmean([r['d_pred'] for r in rows_out]))}")
        print(f"  1-cos(z_prior, z_cur)  {_fmt(_nanmean([r['d_naive'] for r in rows_out]))}   "
              f"(do-nothing)")
        print(f"  pred beats naive       {_fmt(_nanmean([float(r['pred_beats_naive']) for r in rows_out]))}")
        accs = [float(r["single_5way_ok"]) for r in rows_out]
        print(f"  5-way argmax acc       {_fmt(_nanmean(accs))}   "
              f"(row-level; not set-match combined)")
        print()

    if args.csv:
        parent = os.path.dirname(os.path.abspath(args.csv))
        if parent:
            os.makedirs(parent, exist_ok=True)
        pd.DataFrame(rows_out).to_csv(args.csv, index=False)
        print(f"[prior-text] wrote {args.csv}  ({len(rows_out)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval", action="store_true", required=True)
    parser.add_argument(
        "--jepa-ckpt",
        default=os.environ.get("JEPA_CKPT", DEFAULT_JEPA_CKPT),
    )
    parser.add_argument("--gold-parquet", default=DEFAULT_GOLD_BBOXES)
    parser.add_argument("--template", default=PROMPT_TEMPLATE)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-root", action="append", default=[], metavar="DATASET=PATH",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-current",
        action="store_true",
        help="Skip section D (do not encode the current film).",
    )
    parser.add_argument(
        "--csv",
        default="logs_jepa_prior_text/jepa_prior_text.csv",
    )
    args = parser.parse_args()

    parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    auto_gold_roots = discover_gold_image_roots(parquet_dir)
    image_roots: Dict[str, str] = {**IMAGE_ROOTS, **auto_gold_roots}
    for spec in args.image_root:
        if "=" not in spec:
            raise ValueError(f"--image-root expects DATASET=PATH, got {spec!r}")
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {d!r}")
        image_roots[d] = p

    print(f"[prior-text] loading {args.gold_parquet}")
    df = pd.read_parquet(args.gold_parquet)
    df = normalize_gold_bboxes_schema(df)
    if args.limit is not None:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"[prior-text] --limit → {len(df)} rows")
    else:
        print(f"[prior-text] {len(df)} gold-bbox rows")

    run_eval(args, df, image_roots, torch.device(args.device))


if __name__ == "__main__":
    main()
