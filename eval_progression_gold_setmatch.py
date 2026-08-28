#!/usr/bin/env python3
"""CheXTemporal gold eval with top-|GT| set-match for multi-progression.

For each ``(pair, finding)`` group (one forward pass):

  * |GT| = 1 → ordinary argmax accuracy (same as usual single-label eval)
  * |GT| > 1 → rank 5 class cosines, take top-k (k=|GT|), report
    recall / precision / Jaccard against the GT set

Prints:
  (A) multi-progression summary (top-|GT| recall / precision / Jaccard)
  (B) combined overall = mean of (single 0/1, multi Jaccard)
  (C) single-label full breakdown (per-progression, confusion, per-disease,
      macro / kappa) — same tables as regular gold eval
  (D) multi-label full breakdown (per-progression retrieval, per-disease
      set-match, |GT| sizes)

Backends
--------
``jepa``     — JEPA image–image cosine (default **per-patch**; use
               ``--pooling global`` for global-pool checkpoints, or
               ``--pooling head`` for the [ẑ; z_cur; finding] head)
``biovilt``  — official BioViL-T image–text phrase-bank (max phrase
               cosine per class)

Usage
-----
    python eval_progression_gold_setmatch.py --backend jepa --eval \\
        --ckpt checkpoints_jepa_dynamic_cbw99999/best.pt

    python eval_progression_gold_setmatch.py --backend jepa --eval \\
        --ckpt checkpoints_jepa_dynamic_cbw99999_proghead/epoch_5.pt \\
        --pooling head

    python eval_progression_gold_setmatch.py --backend biovilt --eval

    # Quick smoke
    python eval_progression_gold_setmatch.py --backend jepa --eval \\
        --ckpt /path/to/best.pt --limit 50
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from dataset_combined_jepa import DEFAULT_FINDINGS
from eval_progression_biovilt import (
    BioViLTPairModel,
    score_one_pair as biovilt_score_one_pair,
)
from eval_progression_jepa import (
    PROMPT_TEMPLATE as JEPA_PROMPT_TEMPLATE,
    _encode_prompts,
)
from gold_progression_setmatch import (
    format_running_setmatch,
    group_gold_by_pair_finding,
    print_setmatch_report,
    topk_set_match,
)
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from losses_jepa import global_pool_normalize
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER
from tempcxr.modules.jepa import TempCXRJEPA

# BioViL-T CheXTemporal phrase template (no trailing period).
BIOVILT_PROMPT_TEMPLATE = "{} is {}"


@torch.no_grad()
def jepa_score_one_pair(
    model: TempCXRJEPA,
    prior_img: torch.Tensor,
    current_img: torch.Tensor,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, Tuple]] = None,
    pooling: str = "perpatch",
) -> Dict:
    """Return per-class scores (cosine or progression-head logits)."""
    if pooling == "head":
        key = finding.strip().lower()
        if text_cache is not None and key in text_cache:
            txt_global, txt_local, token_mask = text_cache[key]
            txt_global = txt_global.to(device)
            txt_local = txt_local.to(device)
            token_mask = token_mask.to(device)
        else:
            txt_global, txt_local, token_mask = (
                model.text_encoder.forward_contrastive([key])
            )
            if text_cache is not None:
                text_cache[key] = (
                    txt_global.detach().cpu(),
                    txt_local.detach().cpu(),
                    token_mask.detach().cpu(),
                )
        prior = prior_img.unsqueeze(0).to(device)
        current = current_img.unsqueeze(0).to(device)
        _, z_prior = model.image_encoder(prior)
        _, z_cur = model.target_image_encoder(current)
        zhat = model.predictor(z_prior, txt_local, token_mask)
        logits = model.progression_logits(zhat, z_cur.detach(), txt_global)
        scores = logits[0].float().tolist()
        pred_class = int(max(range(len(scores)), key=lambda k: scores[k]))
        return {
            "cos_class_scores": scores,
            "pred_class": pred_class,
            "prompts": [key],
        }

    prompts, txt_local, token_mask = _encode_prompts(
        model, finding, template, device, text_cache,
    )
    n_prompts = len(prompts)
    prior = prior_img.unsqueeze(0).to(device)
    current = current_img.unsqueeze(0).to(device)

    _, z_prior = model.image_encoder(prior)
    _, z_cur = model.target_image_encoder(current)
    z_cur = z_cur.detach()

    z_prior_b = z_prior.expand(n_prompts, -1, -1).contiguous()
    preds = model.predictor(z_prior_b, txt_local, token_mask)
    pred_f = preds.float()
    target_f = z_cur.float()

    if pooling == "global":
        pred_g = global_pool_normalize(pred_f)
        target_g = global_pool_normalize(target_f)
        cos_class_scores = F.cosine_similarity(
            pred_g, target_g.expand_as(pred_g), dim=-1,
        ).tolist()
    else:
        cos_per_patch = F.cosine_similarity(
            pred_f, target_f.expand_as(pred_f), dim=-1,
        )
        cos_class_scores = cos_per_patch.mean(dim=1).tolist()

    pred_class = max(range(n_prompts), key=lambda k: cos_class_scores[k])
    return {
        "cos_class_scores": cos_class_scores,
        "pred_class": pred_class,
        "prompts": prompts,
    }


def run_eval(args, score_fn, groups, device):
    results = []
    skipped = 0
    text_cache: Dict[str, Tuple] = {}

    print(
        f"\n[eval] set-match on {len(groups)} (pair, finding) groups "
        f"(backend={args.backend})"
    )
    print(
        "[eval] running: combined = mean of group scores "
        "(single = argmax accuracy, multi = top-|GT| Jaccard; not F1)"
    )

    for i in range(len(groups)):
        row = groups.iloc[i]
        finding = str(row["finding"])
        gt_labels = list(row["gt_labels"])
        try:
            prior = load_image_tensor(
                row["dataset"], row["parent_image_prev"], args.image_roots,
            )
            current = load_image_tensor(
                row["dataset"], row["parent_image_curr"], args.image_roots,
            )
        except (FileNotFoundError, OSError) as e:
            skipped += 1
            if skipped <= 5:
                print(f"[eval] skipping group {i} (missing image: {e})")
            continue

        out = score_fn(
            prior, current, finding, text_cache,
        )
        sm = topk_set_match(
            out["cos_class_scores"],
            gt_labels,
            CLS_ORDER,
            finding=finding,
        )
        results.append(sm)

        if (i + 1) % max(1, len(groups) // 20) == 0:
            print(
                f"[eval]   {i + 1}/{len(groups)}  skipped={skipped}  "
                f"{format_running_setmatch(results)}"
            )

    if skipped:
        print(f"\nSkipped {skipped} groups due to missing images")
    if not results:
        print("No groups evaluated.")
        return

    note = ""
    if args.backend == "jepa":
        note = f", pooling={args.pooling}"
    print_setmatch_report(results, args.backend, note)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["jepa", "biovilt"],
        help="Scoring backend.",
    )
    parser.add_argument(
        "--ckpt",
        default=os.environ.get(
            "JEPA_CKPT",
            "checkpoints_jepa_dynamic_cbw99999/best.pt",
        ),
        help="JEPA checkpoint (required for --backend jepa).",
    )
    parser.add_argument(
        "--pooling",
        default="perpatch",
        choices=["perpatch", "global", "head"],
        help="JEPA similarity rule (ignored for biovilt). "
             "``head`` = Linear([pool(ẑ); pool(z_cur); finding]).",
    )
    parser.add_argument(
        "--gold-parquet",
        default=DEFAULT_GOLD_PARQUET,
    )
    parser.add_argument(
        "--findings-parquet",
        default=DEFAULT_FINDINGS,
    )
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--finding-col", default=None)
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only first N (pair, finding) groups after grouping.",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--eval", action="store_true")

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
            raise ValueError(f"--image-root expects DATASET=PATH, got {spec!r}")
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {d!r}")
        image_roots[d] = p
        print(f"[gold] override: {d} -> {p}")
    args.image_roots = image_roots

    gold_df = load_gold_pairs(
        args.gold_parquet,
        args.findings_parquet,
        label_col=args.label_col,
        finding_col=args.finding_col,
    )
    groups = group_gold_by_pair_finding(gold_df)
    if args.limit is not None:
        groups = groups.head(args.limit).reset_index(drop=True)
        print(f"[eval] --limit → {len(groups)} groups")

    device = torch.device(args.device)

    if args.backend == "jepa":
        model = load_jepa_model(args.ckpt, device)
        template = JEPA_PROMPT_TEMPLATE

        def score_fn(prior, current, finding, text_cache):
            return jepa_score_one_pair(
                model, prior, current, finding, template, device,
                text_cache=text_cache,
                pooling=args.pooling,
            )
    else:
        model = BioViLTPairModel(device)
        template = BIOVILT_PROMPT_TEMPLATE

        def score_fn(prior, current, finding, text_cache):
            return biovilt_score_one_pair(
                model, prior, current, finding, template, device,
                text_cache=text_cache,
            )

    run_eval(args, score_fn, groups, device)


if __name__ == "__main__":
    main()
