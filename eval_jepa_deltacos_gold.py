#!/usr/bin/env python3
"""Paper JEPA gold: film cosine vs change-vector cosine (one forward).

Loads the paper checkpoint the same way as the 0.452 run:

  * ``load_jepa_model`` — no frozen text, no finding-query, no extra head
  * ``z_prior`` = online encoder(prior)  (single image)
  * ``z_cur``   = EMA encoder(current, prior)  (same pair-mode target
    the 0.452 gold script used; not a training-time joint revert)
  * five ``ẑ^c`` from ``"{finding} is {class}."``

Then scores the **same** five forecasts two ways:

  film     ``mean_p cos(ẑ^c, z_cur)``          ← 0.452 rule
  deltacos ``cos(ẑ^c − z_prior, z_cur − z_prior)``

Gold is already one ``(pair, finding)`` group per finding. The five
templates are for **that** finding only. ``Δz = z_cur − z_prior`` is
still the **whole film**, so another labeled (or unlabeled) disease can
dominate the arrow. This script prints deltacos again on pairs that
have only one gold finding vs pairs that have several, so you can see
whether that confound is eating the number.

Usage
-----
    python eval_jepa_deltacos_gold.py --eval

    python eval_jepa_deltacos_gold.py --eval \\
        --jepa-ckpt checkpoints_jepa_dynamic_cbw99999/epoch_5.pt \\
        --limit 40
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from dataset_combined_jepa import DEFAULT_FINDINGS
from eval_progression_jepa import PROMPT_TEMPLATE, _encode_prompts
from gold_jepa_diagnostics import (
    deltacos_class_scores,
    five_forecast_offdiag_cos,
    print_jepa_score_diagnostics,
)
from gold_progression_setmatch import (
    format_running_setmatch,
    group_gold_by_pair_finding,
    print_setmatch_report,
    topk_set_match,
)
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER

DEFAULT_JEPA_CKPT = "checkpoints_jepa_dynamic_cbw99999/epoch_5.pt"
_PAIR_KEYS = ("dataset", "patient_id", "study_id_curr", "study_id_prev")


def _pair_key(row) -> Tuple:
    return tuple(row[k] for k in _PAIR_KEYS)


def _film_class_scores(zhats: torch.Tensor, z_cur: torch.Tensor) -> List[float]:
    pred = F.normalize(zhats.float(), dim=-1)
    target = F.normalize(z_cur.float(), dim=-1)
    return F.cosine_similarity(pred, target.expand_as(pred), dim=-1).mean(dim=1).tolist()


def _print_block(title: str, results, diag_rows) -> None:
    print()
    print("#" * 60)
    print(f"# {title}")
    print("#" * 60)
    if not results:
        print("  no groups")
        return
    print_setmatch_report(results, "jepa", f", {title}")
    print_jepa_score_diagnostics(diag_rows)


@torch.no_grad()
def run_eval(args, groups, image_roots, device) -> None:
    print()
    print("------------------------------------------------------------")
    print("Change-vector rescoring (paper ckpt, no findq / freeze / joint train)")
    print("------------------------------------------------------------")
    print("  Gold is (pair, finding). Templates are '{finding} is {class}.'")
    print("  ẑ is finding-conditioned. Δz = z_cur − z_prior is the whole film.")
    print("  Another disease on the same pair can point Δz the wrong way.")
    print("  Single-finding-pair slice = only one gold finding on that pair")
    print("  (other unlabeled change can still sit in Δz).")
    print()

    model = load_jepa_model(args.jepa_ckpt, device)
    text_cache: Dict[str, Tuple] = {}

    n_findings = groups.groupby(list(_PAIR_KEYS), dropna=False)["finding"].transform(
        "nunique"
    )
    groups = groups.copy()
    groups["n_findings_on_pair"] = n_findings.astype(int)
    n_single_f = int((groups["n_findings_on_pair"] == 1).sum())
    n_multi_f = int((groups["n_findings_on_pair"] > 1).sum())
    print(
        f"[deltacos] {len(groups)} groups | "
        f"{n_single_f} on single-finding pairs | "
        f"{n_multi_f} on multi-finding pairs"
    )
    print(
        f"[deltacos] z_cur = EMA encoder(current, prior)  "
        f"(same as the 0.452 gold script; --single-image-current to drop prior)"
    )

    film_res, film_diag = [], []
    delta_res, delta_diag = [], []
    delta_single_f, delta_single_diag = [], []
    delta_multi_f, delta_multi_diag = [], []
    skipped = 0

    for i in range(len(groups)):
        row = groups.iloc[i]
        finding = str(row["finding"])
        gt_labels = list(row["gt_labels"])
        try:
            prior = load_image_tensor(
                row["dataset"], row["parent_image_prev"], image_roots,
            )
            current = load_image_tensor(
                row["dataset"], row["parent_image_curr"], image_roots,
            )
        except (FileNotFoundError, OSError) as e:
            skipped += 1
            if skipped <= 5:
                print(f"[deltacos] skipping group {i} (missing image: {e})")
            continue

        prompts, txt_local, token_mask = _encode_prompts(
            model, finding, args.template, device, text_cache,
        )
        prior_b = prior.unsqueeze(0).to(device)
        current_b = current.unsqueeze(0).to(device)
        _, z_prior = model.image_encoder(prior_b)
        if args.single_image_current:
            _, z_cur = model.target_image_encoder(current_b)
        else:
            _, z_cur = model.target_image_encoder(current_b, prior_b)
        z_cur = z_cur.detach()
        z_prior_b = z_prior.expand(len(prompts), -1, -1).contiguous()
        zhats = model.predictor(z_prior_b, txt_local, token_mask)
        off = five_forecast_offdiag_cos(zhats.float())

        film_scores = _film_class_scores(zhats, z_cur)
        delta_scores = deltacos_class_scores(zhats, z_cur, z_prior)

        film_sm = topk_set_match(film_scores, gt_labels, CLS_ORDER, finding=finding)
        delta_sm = topk_set_match(delta_scores, gt_labels, CLS_ORDER, finding=finding)
        film_res.append(film_sm)
        delta_res.append(delta_sm)
        film_row = {
            "scores": film_scores, "gt_labels": gt_labels, "zhat_offdiag": off,
        }
        delta_row = {
            "scores": delta_scores, "gt_labels": gt_labels, "zhat_offdiag": off,
        }
        film_diag.append(film_row)
        delta_diag.append(delta_row)
        if int(row["n_findings_on_pair"]) == 1:
            delta_single_f.append(delta_sm)
            delta_single_diag.append(delta_row)
        else:
            delta_multi_f.append(delta_sm)
            delta_multi_diag.append(delta_row)

        if (i + 1) % max(1, len(groups) // 20) == 0:
            print(
                f"[deltacos] {i + 1}/{len(groups)}  skipped={skipped}  "
                f"film {format_running_setmatch(film_res)}  |  "
                f"delta {format_running_setmatch(delta_res)}"
            )

    if skipped:
        print(f"\n[deltacos] skipped missing images: {skipped}")
    if not film_res:
        print("[deltacos] no groups evaluated")
        return

    _print_block("film cosine (0.452 rule)  mean_p cos(ẑ, z_cur)", film_res, film_diag)
    _print_block(
        "deltacos  cos(ẑ−z_prior, z_cur−z_prior)  all groups",
        delta_res, delta_diag,
    )
    _print_block(
        "deltacos  single-finding pairs only (cleaner Δz)",
        delta_single_f, delta_single_diag,
    )
    _print_block(
        "deltacos  multi-finding pairs (Δz mixes diseases)",
        delta_multi_f, delta_multi_diag,
    )
    print()
    print("[deltacos] done. Compare film vs deltacos combined / kappa / win vs stable.")
    print("[deltacos] If multi-finding deltacos << single-finding, other diseases own Δz.")


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
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--single-image-current",
        action="store_true",
        help="EMA encode current alone (paper train target). Default "
             "keeps pair-mode z_cur so film cosine matches the 0.452 script.",
    )
    args = parser.parse_args()

    parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    auto_gold_roots = discover_gold_image_roots(parquet_dir)
    image_roots: Dict[str, str] = {**IMAGE_ROOTS, **auto_gold_roots}
    if auto_gold_roots:
        print("[gold] auto-detected gold image roots:")
        for d, p in auto_gold_roots.items():
            print(f"  {d}: {p}")
    for spec in args.image_root:
        if "=" not in spec:
            raise SystemExit(f"--image-root expects DATASET=PATH, got {spec!r}")
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise SystemExit(f"dataset must be one of {DATASETS}, got {d!r}")
        image_roots[d] = p
        print(f"[gold] override: {d} -> {p}")

    gold_df = load_gold_pairs(args.gold_parquet, args.findings_parquet)
    groups = group_gold_by_pair_finding(gold_df)
    if args.limit is not None:
        groups = groups.head(args.limit).reset_index(drop=True)
        print(f"[deltacos] --limit → {len(groups)} groups")

    run_eval(args, groups, image_roots, torch.device(args.device))


if __name__ == "__main__":
    main()
