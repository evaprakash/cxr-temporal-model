#!/usr/bin/env python3
"""Feature std on CheXTemporal gold (inference, no training).

Patch check (Chong): for a ``(1, N, D)`` L2-normalized grid,
``std_over_patches`` = std over N tiles, then mean over D.
Also mean off-diagonal tile cosine. ``std * sqrt(D)`` is for 128 vs 768.

Global check (across films): one 128-d vector per gold pair. Std over
the 738 films (mean over D), plus mean pairwise cosine between films.
That asks whether *studies* look different, not whether *regions* do.

Models (all BioViL-T 128-d, L2-normed):

  * ``biovilt``    — official weights: single-image and pair ``(curr, prior)``
  * ``supervised`` — unfrozen pair-CE image encoder (single + pair)
  * ``jepa``       — ``z_cur`` / ``z_prior`` encoders and predictor ``ẑ``
                     (finding name as condition; global ``ẑ`` = mean-pool)

Usage
-----
    python eval_gold_feature_std.py --eval

    python eval_gold_feature_std.py --eval --limit 40
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List

import torch
import torch.nn.functional as F

import tempcxr.modules.image_encoder_jepa as image_encoder_jepa
from dataset_combined_jepa import DEFAULT_FINDINGS
from eval_progression_biovilt import BioViLTPairModel, load_supervised_encoders
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from losses_jepa import patch_token_feature_stats
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)

image_encoder_jepa.DEBUG = False

DEFAULT_JEPA_CKPT = "checkpoints_jepa_dynamic_cbw99999/epoch_5.pt"
DEFAULT_SUP_CKPT = "checkpoints_supervised_progression_unfrozen/epoch_5.pt"


def _stats_row(name: str, patches: torch.Tensor) -> Dict[str, float]:
    s = patch_token_feature_stats(patches)
    std = float(s["std_over_patches"])
    off = float(s["mean_offdiag_cos"])
    d = int(patches.shape[-1])
    return {
        "name": name,
        "n": 1,
        "d": d,
        "std_over_patches": std,
        "std_x_sqrt_d": std * math.sqrt(d),
        "mean_offdiag_cos": off,
        "n_patches": int(patches.shape[1]),
    }


def _add(acc: Dict[str, Dict[str, float]], row: Dict[str, float]) -> None:
    slot = acc.setdefault(
        row["name"],
        {
            "n": 0.0,
            "d": float(row["d"]),
            "std_over_patches": 0.0,
            "std_x_sqrt_d": 0.0,
            "mean_offdiag_cos": 0.0,
            "n_patches": float(row["n_patches"]),
        },
    )
    slot["n"] += 1.0
    for k in ("std_over_patches", "std_x_sqrt_d", "mean_offdiag_cos"):
        slot[k] += row[k]


def _mean_table(acc: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    out = []
    for name, slot in acc.items():
        n = max(slot["n"], 1.0)
        out.append(
            {
                "name": name,
                "n": int(slot["n"]),
                "d": int(slot["d"]),
                "n_patches": int(slot["n_patches"]),
                "std_over_patches": slot["std_over_patches"] / n,
                "std_x_sqrt_d": slot["std_x_sqrt_d"] / n,
                "mean_offdiag_cos": slot["mean_offdiag_cos"] / n,
            }
        )
    return out


def _print_patch_table(rows: List[Dict[str, float]]) -> None:
    print()
    print("=== Patch tokens (std over 196 tiles on one film; then mean over films) ===")
    print(
        f"{'representation':<40} {'n':>5} {'D':>4} "
        f"{'std_patches':>12} {'std*sqrt(D)':>12} {'offdiag_cos':>12}"
    )
    print("-" * 90)
    for r in rows:
        print(
            f"{r['name']:<40} {r['n']:>5d} {r['d']:>4d} "
            f"{r['std_over_patches']:>12.4f} {r['std_x_sqrt_d']:>12.4f} "
            f"{r['mean_offdiag_cos']:>12.4f}"
        )
    print()
    print(
        "std_patches: do regions of one chest differ? "
        "Collapse: std → 0 and offdiag_cos → 1."
    )


def _global_set_stats(name: str, vecs: List[torch.Tensor]) -> Dict[str, float]:
    """Std / pairwise cosine of one global vector per gold pair."""
    z = torch.stack([v.detach().float().reshape(-1) for v in vecs], dim=0)
    z = F.normalize(z, dim=-1, eps=1e-8)
    n, d = z.shape
    std = float(z.std(dim=0, unbiased=False).mean())
    if n < 2:
        off = float("nan")
    else:
        sim = z @ z.T
        off = float((sim.sum() - sim.trace()) / (n * (n - 1)))
    return {
        "name": name,
        "n": n,
        "d": d,
        "std_over_set": std,
        "std_x_sqrt_d": std * math.sqrt(d),
        "mean_offdiag_cos": off,
    }


def _print_global_table(rows: List[Dict[str, float]]) -> None:
    print()
    print("=== Global vectors (one 128-d code per film; std / cosine across films) ===")
    print(
        f"{'representation':<40} {'n':>5} {'D':>4} "
        f"{'std_over_set':>12} {'std*sqrt(D)':>12} {'offdiag_cos':>12}"
    )
    print("-" * 90)
    for r in rows:
        print(
            f"{r['name']:<40} {r['n']:>5d} {r['d']:>4d} "
            f"{r['std_over_set']:>12.4f} {r['std_x_sqrt_d']:>12.4f} "
            f"{r['mean_offdiag_cos']:>12.4f}"
        )
    print()
    print(
        "std_over_set: do different studies look different? "
        "offdiag_cos → 1 means every film's global code is the same direction."
    )


@torch.no_grad()
def run_eval(args, pairs, image_roots, device: torch.device) -> None:
    biovilt = BioViLTPairModel(device)
    biovilt.image_encoder.eval()

    supervised = None
    if args.supervised_ckpt:
        supervised = load_supervised_encoders(args.supervised_ckpt, device)

    jepa = None
    if args.jepa_ckpt:
        jepa = load_jepa_model(args.jepa_ckpt, device)

    acc: Dict[str, Dict[str, float]] = {}
    globals_acc: Dict[str, List[torch.Tensor]] = {}
    skipped = 0
    text_cache: Dict[str, tuple] = {}

    def _keep_global(name: str, g: torch.Tensor) -> None:
        globals_acc.setdefault(name, []).append(g.detach().float().cpu().reshape(-1))

    for i, row in enumerate(pairs):
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
        prior_b = prior.unsqueeze(0).to(device)
        current_b = current.unsqueeze(0).to(device)

        bio_g_cur, bio_cur = biovilt.image_encoder(current_b)
        bio_g_pri, bio_pri = biovilt.image_encoder(prior_b)
        bio_g_pair, bio_pair = biovilt.image_encoder(current_b, prior_b)
        _add(acc, _stats_row("biovilt/z_cur (single, official)", bio_cur))
        _add(acc, _stats_row("biovilt/z_prior (single, official)", bio_pri))
        _add(acc, _stats_row("biovilt/z_cur (pair, official)", bio_pair))
        _keep_global("biovilt/global_cur (single, official)", bio_g_cur)
        _keep_global("biovilt/global_prior (single, official)", bio_g_pri)
        _keep_global("biovilt/global_cur (pair, official)", bio_g_pair)

        if supervised is not None:
            sup_g_cur, sup_cur = supervised.image_encoder(current_b)
            sup_g_pair, sup_pair = supervised.image_encoder(current_b, prior_b)
            _add(acc, _stats_row("supervised/z_cur (single)", sup_cur))
            _add(acc, _stats_row("supervised/z_cur (pair)", sup_pair))
            _keep_global("supervised/global_cur (single)", sup_g_cur)
            _keep_global("supervised/global_cur (pair)", sup_g_pair)

        if jepa is not None:
            g_prior, z_prior = jepa.image_encoder(prior_b)
            g_cur, z_cur = jepa.target_image_encoder(current_b)
            _add(acc, _stats_row("jepa/z_cur (EMA encoder)", z_cur))
            _add(acc, _stats_row("jepa/z_prior (online encoder)", z_prior))
            _keep_global("jepa/global_cur (EMA encoder)", g_cur)
            _keep_global("jepa/global_prior (online encoder)", g_prior)
            finding = str(row.get("finding") or "finding").strip().lower()
            if finding not in text_cache:
                _, loc, mask = jepa.text_encoder.forward_contrastive([finding])
                text_cache[finding] = (loc.cpu(), mask.cpu())
            loc, mask = text_cache[finding]
            zhat = jepa.predictor(z_prior, loc.to(device), mask.to(device))
            _add(acc, _stats_row("jepa/zhat (predictor, finding)", zhat))
            zhat_g = F.normalize(zhat.float().mean(dim=1), dim=-1)
            _keep_global("jepa/global_zhat (mean-pool ẑ)", zhat_g)

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"[featstd] {i + 1}/{len(pairs)}  skipped={skipped}",
                flush=True,
            )

    patch_rows = _mean_table(acc)
    global_rows = [
        _global_set_stats(name, vecs) for name, vecs in globals_acc.items()
    ]

    print(f"\n[featstd] unique gold pairs={len(pairs)} skipped={skipped}")
    if args.jepa_ckpt:
        print(f"[featstd] jepa ckpt        = {args.jepa_ckpt}")
    if args.supervised_ckpt:
        print(f"[featstd] supervised ckpt  = {args.supervised_ckpt}")
    _print_patch_table(patch_rows)
    _print_global_table(global_rows)

    if args.csv:
        parent = os.path.dirname(os.path.abspath(args.csv))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.csv, "w") as f:
            f.write(
                "kind,name,n,d,n_patches,std,std_x_sqrt_d,mean_offdiag_cos\n"
            )
            for r in patch_rows:
                f.write(
                    f"patch,{r['name']},{r['n']},{r['d']},{r['n_patches']},"
                    f"{r['std_over_patches']:.6f},{r['std_x_sqrt_d']:.6f},"
                    f"{r['mean_offdiag_cos']:.6f}\n"
                )
            for r in global_rows:
                f.write(
                    f"global,{r['name']},{r['n']},{r['d']},1,"
                    f"{r['std_over_set']:.6f},{r['std_x_sqrt_d']:.6f},"
                    f"{r['mean_offdiag_cos']:.6f}\n"
                )
        print(f"[featstd] wrote {args.csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval", action="store_true", required=True)
    parser.add_argument("--jepa-ckpt", default=os.environ.get("JEPA_CKPT", DEFAULT_JEPA_CKPT))
    parser.add_argument(
        "--supervised-ckpt",
        default=os.environ.get("SUPERVISED_CKPT", DEFAULT_SUP_CKPT),
    )
    parser.add_argument("--gold-parquet", default=DEFAULT_GOLD_PARQUET)
    parser.add_argument("--findings-parquet", default=DEFAULT_FINDINGS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-root", action="append", default=[], metavar="DATASET=PATH")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--csv",
        default="logs_gold_feature_std/gold_feature_std.csv",
    )
    parser.add_argument("--skip-jepa", action="store_true")
    parser.add_argument("--skip-supervised", action="store_true")
    args = parser.parse_args()
    if args.skip_jepa:
        args.jepa_ckpt = ""
    if args.skip_supervised:
        args.supervised_ckpt = ""

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
    pair_cols = ["dataset", "parent_image_prev", "parent_image_curr"]
    extra = [c for c in ("finding",) if c in gold_df.columns]
    pairs_df = (
        gold_df[pair_cols + extra]
        .drop_duplicates(pair_cols, keep="first")
        .reset_index(drop=True)
    )
    if args.limit is not None:
        pairs_df = pairs_df.head(args.limit).reset_index(drop=True)
        print(f"[featstd] --limit → {len(pairs_df)} unique pairs")
    else:
        print(
            f"[featstd] {len(gold_df)} gold rows → "
            f"{len(pairs_df)} unique image pairs"
        )

    run_eval(args, pairs_df.to_dict("records"), image_roots, torch.device(args.device))


if __name__ == "__main__":
    main()
