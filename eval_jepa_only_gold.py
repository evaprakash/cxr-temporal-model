#!/usr/bin/env python3
"""JEPA-only tests on CheXTemporal gold (supervised cannot run these).

The predictor never sees the current film. ``z_cur`` is only a *target*
for (1) and (2). Supervised pair-CE has no ``ẑ`` and cannot forecast.

  1. Next-film match — is ``ẑ`` closer to ``z_cur`` than do-nothing
     ``z_prior``?  ``1 − mean_p cos``.
  2. Change-vector alignment — slide-deck
     ``cos(ẑ − z_prior, z_cur − z_prior)``.
  3. Counterfactual ``ẑ`` — do the five class sentences move the
     forecast? Mean off-diagonal ``cos(ẑ^i, ẑ^j)`` (no ``z_cur``).
  4. Blind change energy — ``mean_p ‖ẑ^c − z_prior‖`` per class
     (no current film in the score).

Condition text is the same 5-way gold template
``"{Finding} is {class}."``. For (1)/(2) the true gold class(es) pick
which ``ẑ`` is the forecast. Multi-label groups average over GT classes.

Default ckpt is the paper 0.452 run:
``checkpoints_jepa_dynamic_cbw99999/epoch_5.pt``.

Usage
-----
    python eval_jepa_only_gold.py --eval
    python eval_jepa_only_gold.py --eval --limit 40
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from eval_progression_jepa import PROMPT_TEMPLATE, _encode_prompts
from gold_progression_setmatch import group_gold_by_pair_finding
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER
from dataset_combined_jepa import DEFAULT_FINDINGS

DEFAULT_JEPA_CKPT = "checkpoints_jepa_dynamic_cbw99999/epoch_5.pt"
N_CLS = len(CLS_ORDER)


def _mean_patch_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a, b`` are ``(..., N, D)``; returns mean-over-patches cosine."""
    an = F.normalize(a.float(), dim=-1)
    bn = F.normalize(b.float(), dim=-1)
    return (an * bn).sum(dim=-1).mean(dim=-1)


def _change_align(zhat: torch.Tensor, z_cur: torch.Tensor, z_prior: torch.Tensor) -> float:
    dpred = (zhat.float() - z_prior.float()).flatten()
    dtrue = (z_cur.float() - z_prior.float()).flatten()
    if dpred.norm() < 1e-8 or dtrue.norm() < 1e-8:
        return float("nan")
    return float(F.cosine_similarity(dpred, dtrue, dim=0).item())


def _energy(zhat: torch.Tensor, z_prior: torch.Tensor) -> torch.Tensor:
    return (zhat.float() - z_prior.float()).norm(dim=-1).mean(dim=-1)


def _nanmean(xs: Sequence[float]) -> float:
    vals = [x for x in xs if x == x]
    return sum(vals) / len(vals) if vals else float("nan")


@torch.no_grad()
def run_eval(args, groups, image_roots, device: torch.device) -> None:
    model = load_jepa_model(args.jepa_ckpt, device)
    text_cache: Dict[str, Tuple] = {}

    next_pred: List[float] = []
    next_naive: List[float] = []
    next_win: List[float] = []
    l1_pred: List[float] = []
    l1_naive: List[float] = []
    align: List[float] = []
    offdiag_zhat: List[float] = []
    stable_vs_worse: List[float] = []
    energy_by_cls: Dict[str, List[float]] = {c: [] for c in CLS_ORDER}
    energy_gt: List[float] = []
    energy_stable: List[float] = []
    single_5way_ok: List[float] = []
    skipped = 0
    n_single = 0
    n_multi = 0
    rows_out: List[Dict] = []

    i_st = CLS_ORDER.index("stable")
    i_wo = CLS_ORDER.index("worsening")

    for i, row in enumerate(groups):
        try:
            prior = load_image_tensor(
                row["dataset"], row["parent_image_prev"], image_roots,
            )
            current = load_image_tensor(
                row["dataset"], row["parent_image_curr"], image_roots,
            )
        except (FileNotFoundError, OSError):
            skipped += 1
            continue

        finding = str(row["finding"]).strip().lower()
        gt_labels = list(row["gt_labels"])
        gt_idx = [CLS_ORDER.index(c) for c in gt_labels if c in CLS_ORDER]
        if not gt_idx:
            skipped += 1
            continue

        prompts, txt_local, token_mask = _encode_prompts(
            model, finding, args.template, device, text_cache,
        )
        prior_b = prior.unsqueeze(0).to(device)
        current_b = current.unsqueeze(0).to(device)
        _, z_prior = model.image_encoder(prior_b)
        _, z_cur = model.target_image_encoder(current_b, prior_b)
        z_cur = z_cur.detach()
        z_prior_b = z_prior.expand(N_CLS, -1, -1).contiguous()
        zhats = model.predictor(z_prior_b, txt_local, token_mask)

        cos_to_cur = _mean_patch_cos(zhats, z_cur.expand_as(zhats))
        naive_cos = float(_mean_patch_cos(z_prior, z_cur).item())
        d_naive = 1.0 - naive_cos
        d_preds = [1.0 - float(cos_to_cur[j].item()) for j in gt_idx]
        d_pred = sum(d_preds) / len(d_preds)
        next_pred.append(d_pred)
        next_naive.append(d_naive)
        next_win.append(1.0 if d_pred < d_naive else 0.0)

        l1_p = F.smooth_l1_loss(
            zhats[gt_idx], z_cur.expand(len(gt_idx), -1, -1), reduction="none",
        ).mean().item()
        l1_n = F.smooth_l1_loss(z_prior, z_cur).item()
        l1_pred.append(l1_p)
        l1_naive.append(l1_n)

        aligns = [
            _change_align(zhats[j], z_cur, z_prior) for j in gt_idx
        ]
        align_i = _nanmean(aligns)
        align.append(align_i)

        zhat_n = F.normalize(zhats.float(), dim=-1)
        sim = torch.einsum("cnd,knd->ck", zhat_n, zhat_n) / zhat_n.shape[1]
        off = (sim.sum() - sim.trace()) / (N_CLS * (N_CLS - 1))
        offdiag_zhat.append(float(off.item()))
        stable_vs_worse.append(float(sim[i_st, i_wo].item()))

        e = _energy(zhats, z_prior.expand_as(zhats))
        for c_i, cls in enumerate(CLS_ORDER):
            energy_by_cls[cls].append(float(e[c_i].item()))
        e_gt = float(e[gt_idx].mean().item())
        energy_gt.append(e_gt)
        energy_stable.append(float(e[i_st].item()))

        is_multi = bool(row["is_multi"])
        if is_multi:
            n_multi += 1
            five_ok = float("nan")
        else:
            n_single += 1
            pred_c = int(cos_to_cur.argmax().item())
            five_ok = 1.0 if pred_c == gt_idx[0] else 0.0
            single_5way_ok.append(five_ok)

        rows_out.append(
            {
                "dataset": row["dataset"],
                "finding": finding,
                "gt": "|".join(gt_labels),
                "is_multi": int(is_multi),
                "d_pred": d_pred,
                "d_naive": d_naive,
                "pred_beats_naive": int(d_pred < d_naive),
                "smooth_l1_pred": l1_p,
                "smooth_l1_naive": l1_n,
                "cos_delta": align_i,
                "zhat_offdiag_cos": float(off.item()),
                "cos_zhat_stable_worsening": float(sim[i_st, i_wo].item()),
                "energy_gt": e_gt,
                "energy_stable": float(e[i_st].item()),
                "single_5way_ok": five_ok,
            }
        )

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"[jepa-only] {i + 1}/{len(groups)}  skipped={skipped}",
                flush=True,
            )

    n = len(next_pred)
    print()
    print(f"[jepa-only] ckpt     = {args.jepa_ckpt}")
    print(f"[jepa-only] groups   = {n}  skipped={skipped}  "
          f"single={n_single} multi={n_multi}")
    print(f"[jepa-only] template = {args.template!r}")
    print()
    print("=== 1. Next-film match (forecast vs do-nothing; current is target only) ===")
    print(f"  1-cos(ẑ^gt, z_cur)     {_nanmean(next_pred):.4f}")
    print(f"  1-cos(z_prior, z_cur)  {_nanmean(next_naive):.4f}   (do-nothing)")
    print(f"  pred beats naive       {_nanmean(next_win):.4f}   "
          f"({int(sum(next_win))}/{n})")
    print(f"  Smooth L1 ẑ            {_nanmean(l1_pred):.4f}")
    print(f"  Smooth L1 z_prior      {_nanmean(l1_naive):.4f}")
    print()
    print("=== 2. Change-vector alignment  cos(ẑ−z_prior, z_cur−z_prior) ===")
    print(f"  mean cos(Δẑ, Δz_true)  {_nanmean(align):.4f}   "
          f"(+1 = right direction of change)")
    print()
    print("=== 3. Counterfactual ẑ (no z_cur) — do class sentences move the forecast? ===")
    print(f"  mean off-diag cos(ẑ^i, ẑ^j)     {_nanmean(offdiag_zhat):.4f}")
    print(f"  cos(ẑ^stable, ẑ^worsening)      {_nanmean(stable_vs_worse):.4f}")
    print("  (~1 = all five forecasts are the same film)")
    print()
    print("=== 4. Blind change energy  mean_p ‖ẑ^c − z_prior‖  (no current film) ===")
    for cls in CLS_ORDER:
        print(f"  {cls:<12} {_nanmean(energy_by_cls[cls]):.4f}")
    print(f"  energy(ẑ^gt)           {_nanmean(energy_gt):.4f}")
    print(f"  energy(ẑ^stable)       {_nanmean(energy_stable):.4f}")
    print()
    if single_5way_ok:
        print("=== Sanity: usual 5-way (not JEPA-only; single-label argmax) ===")
        print(f"  single acc             {_nanmean(single_5way_ok):.4f}   "
              f"({int(sum(single_5way_ok))}/{len(single_5way_ok)})")
        print()

    if args.csv:
        parent = os.path.dirname(os.path.abspath(args.csv))
        if parent:
            os.makedirs(parent, exist_ok=True)
        keys = list(rows_out[0].keys()) if rows_out else []
        with open(args.csv, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in rows_out:
                f.write(
                    ",".join(
                        "" if (isinstance(r[k], float) and r[k] != r[k])
                        else f"{r[k]:.6f}" if isinstance(r[k], float)
                        else str(r[k])
                        for k in keys
                    )
                    + "\n"
                )
        print(f"[jepa-only] wrote {args.csv}")


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
    parser.add_argument("--gold-parquet", default=DEFAULT_GOLD_PARQUET)
    parser.add_argument("--findings-parquet", default=DEFAULT_FINDINGS)
    parser.add_argument("--template", default=PROMPT_TEMPLATE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-root", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--csv",
        default="logs_jepa_only_gold/jepa_only_gold.csv",
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

    gold_df = load_gold_pairs(args.gold_parquet, args.findings_parquet)
    groups = group_gold_by_pair_finding(gold_df)
    if args.limit is not None:
        groups = groups.head(args.limit).reset_index(drop=True)
        print(f"[jepa-only] --limit → {len(groups)} groups")

    run_eval(args, groups.to_dict("records"), image_roots, torch.device(args.device))


if __name__ == "__main__":
    main()
