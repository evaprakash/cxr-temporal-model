#!/usr/bin/env python3
"""Patch-token feature std on CheXTemporal gold (inference, no training).

Same Chong check as training: for a ``(B, N, D)`` L2-normalized grid,
``std_over_patches`` = std over N, then mean over D (and the batch),
plus mean off-diagonal cosine. Also reports ``std * sqrt(D)`` so 128-d
unit vectors are comparable if you later add 768-d Rad-DINO.

Models (all BioViL-T 128-d, L2-normed patches):

  * ``biovilt``    — official weights, no finetune
  * ``supervised`` — unfrozen pair-CE image encoder
  * ``jepa``       — trained JEPA: encoder ``z_cur`` / ``z_prior`` and
                     predictor ``ẑ`` (finding name as condition)

Current-image patches are the fair encoder comparison. JEPA ``ẑ`` is
extra (predictor, needs a finding). Supervised is also scored as a
pair encoder ``(current, prior)`` — that is how it is trained.

Usage
-----
    python eval_gold_feature_std.py --eval

    python eval_gold_feature_std.py --eval \\
        --jepa-ckpt checkpoints_jepa_dynamic_cbw99999/epoch_5.pt \\
        --supervised-ckpt checkpoints_supervised_progression_unfrozen/epoch_5.pt

    python eval_gold_feature_std.py --eval --limit 40
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List

import torch

import tempcxr.modules.image_encoder_jepa as image_encoder_jepa
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
from dataset_combined_jepa import DEFAULT_FINDINGS

image_encoder_jepa.DEBUG = False

DEFAULT_JEPA_CKPT = "checkpoints_jepa_dynamic_cbw99999/epoch_5.pt"
DEFAULT_SUP_CKPT = "checkpoints_supervised_progression_unfrozen/epoch_5.pt"


def _stats_row(name: str, patches: torch.Tensor) -> Dict[str, float]:
    s = patch_token_feature_stats(patches)
    std = float(s["std_over_patches"])
    off = float(s["mean_offdiag_cos"])
    batch_std = float(s["std_over_batch"])
    d = int(patches.shape[-1])
    return {
        "name": name,
        "n": 1,
        "d": d,
        "std_over_patches": std,
        "std_x_sqrt_d": std * math.sqrt(d),
        "std_over_batch": batch_std,
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
            "std_over_batch": 0.0,
            "mean_offdiag_cos": 0.0,
            "n_patches": float(row["n_patches"]),
        },
    )
    slot["n"] += 1.0
    for k in (
        "std_over_patches",
        "std_x_sqrt_d",
        "std_over_batch",
        "mean_offdiag_cos",
    ):
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
                "std_over_batch": slot["std_over_batch"] / n,
                "mean_offdiag_cos": slot["mean_offdiag_cos"] / n,
            }
        )
    return out


def _print_table(rows: List[Dict[str, float]]) -> None:
    print()
    print(
        f"{'representation':<36} {'n':>5} {'D':>4} "
        f"{'std_patches':>12} {'std*sqrt(D)':>12} "
        f"{'offdiag_cos':>12} {'std_batch':>10}"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r['name']:<36} {r['n']:>5d} {r['d']:>4d} "
            f"{r['std_over_patches']:>12.4f} {r['std_x_sqrt_d']:>12.4f} "
            f"{r['mean_offdiag_cos']:>12.4f} {r['std_over_batch']:>10.4f}"
        )
    print()
    print(
        "std_patches = mean over dims of std over N patches (Chong check). "
        "Collapse on the unit sphere: std → 0 and offdiag_cos → 1."
    )
    print(
        "std*sqrt(D) rescales so a 128-d unit grid (~0.06–0.08) is "
        "comparable to a 768-d unit grid."
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
    skipped = 0
    text_cache: Dict[str, tuple] = {}

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

        _, bio_cur = biovilt.image_encoder(current_b)
        _, bio_pri = biovilt.image_encoder(prior_b)
        _add(acc, _stats_row("biovilt/z_cur (single, official)", bio_cur))
        _add(acc, _stats_row("biovilt/z_prior (single, official)", bio_pri))

        if supervised is not None:
            _, sup_cur = supervised.image_encoder(current_b)
            _, sup_pair = supervised.image_encoder(current_b, prior_b)
            _add(acc, _stats_row("supervised/z_cur (single)", sup_cur))
            _add(acc, _stats_row("supervised/z_cur (pair)", sup_pair))

        if jepa is not None:
            _, z_prior = jepa.image_encoder(prior_b)
            _, z_cur = jepa.target_image_encoder(current_b)
            _add(acc, _stats_row("jepa/z_cur (EMA encoder)", z_cur))
            _add(acc, _stats_row("jepa/z_prior (online encoder)", z_prior))
            finding = str(row.get("finding") or "finding").strip().lower()
            if finding not in text_cache:
                _, loc, mask = jepa.text_encoder.forward_contrastive([finding])
                text_cache[finding] = (loc.cpu(), mask.cpu())
            loc, mask = text_cache[finding]
            zhat = jepa.predictor(
                z_prior, loc.to(device), mask.to(device),
            )
            _add(acc, _stats_row("jepa/zhat (predictor, finding)", zhat))

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"[featstd] {i + 1}/{len(pairs)}  skipped={skipped}",
                flush=True,
            )

    rows = _mean_table(acc)
    print(f"\n[featstd] unique gold pairs={len(pairs)} skipped={skipped}")
    if args.jepa_ckpt:
        print(f"[featstd] jepa ckpt        = {args.jepa_ckpt}")
    if args.supervised_ckpt:
        print(f"[featstd] supervised ckpt  = {args.supervised_ckpt}")
    _print_table(rows)

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w") as f:
            f.write(
                "name,n,d,n_patches,std_over_patches,std_x_sqrt_d,"
                "mean_offdiag_cos,std_over_batch\n"
            )
            for r in rows:
                f.write(
                    f"{r['name']},{r['n']},{r['d']},{r['n_patches']},"
                    f"{r['std_over_patches']:.6f},{r['std_x_sqrt_d']:.6f},"
                    f"{r['mean_offdiag_cos']:.6f},{r['std_over_batch']:.6f}\n"
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
    parser.add_argument(
        "--skip-jepa",
        action="store_true",
        help="Only score BioViL-T ± supervised.",
    )
    parser.add_argument(
        "--skip-supervised",
        action="store_true",
    )
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

    pairs = pairs_df.to_dict("records")
    device = torch.device(args.device)
    run_eval(args, pairs, image_roots, device)


if __name__ == "__main__":
    main()
