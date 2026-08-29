#!/usr/bin/env python3
"""Ensemble JEPA cosine scores and supervised head logits on gold set-match.

No training. For each ``(pair, finding)`` group:

  1. JEPA 5-way scores (default: per-patch mean cosine, same as the 0.452 run)
  2. Supervised 5-way logits (unfrozen BioViL-T pair + finding head)
  3. Combine into one 5-vector, then the usual set-match
     (single = argmax, multi = top-|GT| Jaccard)

Score scales differ (cosine vs logits), so the default mix is softmax
on each side then a weighted average of probabilities. JEPA softmax
uses temperature 0.1 to match training cosine CE; supervised uses 1.0.

Usage
-----
    python eval_ensemble_gold_setmatch.py --eval

    python eval_ensemble_gold_setmatch.py --eval \\
        --jepa-ckpt checkpoints_jepa_dynamic_cbw99999/epoch_5.pt \\
        --supervised-ckpt checkpoints_supervised_progression_unfrozen/epoch_5.pt

    # Smoke
    python eval_ensemble_gold_setmatch.py --eval --limit 50
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset_combined_jepa import DEFAULT_FINDINGS
from eval_progression_biovilt import BioViLTPairModel
from eval_progression_gold_setmatch import (
    JEPA_PROMPT_TEMPLATE,
    jepa_score_one_pair,
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
from tempcxr.modules.jepa import TempCXRJEPA

DEFAULT_JEPA_CKPT = "checkpoints_jepa_dynamic_cbw99999/epoch_5.pt"
DEFAULT_SUP_CKPT = (
    "checkpoints_supervised_progression_unfrozen/epoch_5.pt"
)


class SupervisedHead(nn.Module):
    def __init__(self, in_dim: int = 256, n_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


class SupervisedProgressionModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, head: SupervisedHead):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.head = head

    def forward(self, priors, currents, findings) -> torch.Tensor:
        img_global, _ = self.image_encoder(currents, priors)
        img_emb = F.normalize(img_global.float(), dim=-1)
        txt, _, _ = self.text_encoder.forward_contrastive(list(findings))
        txt = F.normalize(txt.float(), dim=-1)
        return self.head(torch.cat([img_emb, txt], dim=-1))


def load_supervised_model(ckpt_path: str, device: torch.device):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"supervised checkpoint not found: {ckpt_path}"
        )
    print(f"[ensemble] loading supervised {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "finding_vocab" in ckpt:
        raise ValueError(
            f"{ckpt_path} is a finding-ID checkpoint, not the text-encoder head."
        )
    if "head" not in ckpt:
        raise ValueError(f"{ckpt_path} has no 'head' (need unfrozen supervised).")
    holder = BioViLTPairModel(device)
    head = SupervisedHead(in_dim=int(ckpt.get("in_dim", 256)))
    head.load_state_dict(ckpt["head"])
    model = SupervisedProgressionModel(
        holder.image_encoder, holder.text_encoder, head,
    ).to(device)
    if "image_encoder" in ckpt:
        model.image_encoder.load_state_dict(ckpt["image_encoder"])
        print("[ensemble] loaded trained supervised image encoder")
    if "text_encoder" in ckpt:
        model.text_encoder.load_state_dict(ckpt["text_encoder"])
        print("[ensemble] loaded trained supervised text encoder")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def combine_scores(
    jepa_scores: Sequence[float],
    sup_scores: Sequence[float],
    combine: str,
    jepa_weight: float,
    sup_weight: float,
    jepa_temp: float,
    sup_temp: float,
) -> List[float]:
    j = torch.tensor(list(jepa_scores), dtype=torch.float32)
    s = torch.tensor(list(sup_scores), dtype=torch.float32)
    wj, ws = float(jepa_weight), float(sup_weight)
    if wj < 0 or ws < 0 or (wj + ws) <= 0:
        raise ValueError("weights must be non-negative and not both zero")
    if combine == "softmax":
        pj = F.softmax(j / jepa_temp, dim=0)
        ps = F.softmax(s / sup_temp, dim=0)
        mix = (wj * pj + ws * ps) / (wj + ws)
    elif combine == "zscore":
        def _z(x: torch.Tensor) -> torch.Tensor:
            return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)

        mix = (wj * _z(j) + ws * _z(s)) / (wj + ws)
    else:
        raise ValueError(f"unknown --combine {combine!r}")
    return mix.tolist()


@torch.no_grad()
def supervised_logits(
    model: SupervisedProgressionModel,
    prior: torch.Tensor,
    current: torch.Tensor,
    finding: str,
    device: torch.device,
) -> List[float]:
    logits = model(
        prior.unsqueeze(0).to(device),
        current.unsqueeze(0).to(device),
        [finding],
    )
    return logits.squeeze(0).float().tolist()


def run_eval(args, groups, image_roots, device):
    jepa: TempCXRJEPA = load_jepa_model(args.jepa_ckpt, device)
    sup = load_supervised_model(args.supervised_ckpt, device)

    print(
        f"\n[eval] ensemble set-match on {len(groups)} groups "
        f"(jepa pooling={args.pooling}, combine={args.combine}, "
        f"w_jepa={args.jepa_weight}, w_sup={args.supervised_weight}, "
        f"jepa_temp={args.jepa_temp}, sup_temp={args.supervised_temp})"
    )
    print(
        "[eval] running: combined = mean of group scores "
        "(single = argmax accuracy, multi = top-|GT| Jaccard; not F1)"
    )

    results = []
    skipped = 0
    jepa_cache: Dict[str, Tuple] = {}
    n_agree = 0

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
                print(f"[eval] skipping group {i} (missing image: {e})")
            continue

        jepa_out = jepa_score_one_pair(
            jepa, prior, current, finding, JEPA_PROMPT_TEMPLATE, device,
            text_cache=jepa_cache,
            pooling=args.pooling,
        )
        sup_scores = supervised_logits(sup, prior, current, finding, device)
        mixed = combine_scores(
            jepa_out["cos_class_scores"],
            sup_scores,
            combine=args.combine,
            jepa_weight=args.jepa_weight,
            sup_weight=args.supervised_weight,
            jepa_temp=args.jepa_temp,
            sup_temp=args.supervised_temp,
        )
        pred_j = int(
            max(range(len(CLS_ORDER)), key=lambda k: jepa_out["cos_class_scores"][k])
        )
        pred_s = int(max(range(len(CLS_ORDER)), key=lambda k: sup_scores[k]))
        if pred_j == pred_s:
            n_agree += 1

        sm = topk_set_match(mixed, gt_labels, CLS_ORDER, finding=finding)
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

    print(
        f"[ensemble] single-argmax agreement JEPA vs supervised: "
        f"{n_agree}/{len(results)} = {n_agree / len(results):.4f}"
    )
    note = (
        f", {args.combine} w_jepa={args.jepa_weight} "
        f"w_sup={args.supervised_weight} pooling={args.pooling}"
    )
    print_setmatch_report(results, "ensemble", note)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--eval", action="store_true", required=True)
    parser.add_argument(
        "--jepa-ckpt",
        default=os.environ.get("JEPA_CKPT", DEFAULT_JEPA_CKPT),
    )
    parser.add_argument(
        "--supervised-ckpt",
        default=os.environ.get("SUPERVISED_CKPT", DEFAULT_SUP_CKPT),
    )
    parser.add_argument(
        "--pooling",
        default="perpatch",
        choices=["perpatch", "global", "head"],
        help="JEPA score rule. Use perpatch for the 0.452 checkpoint.",
    )
    parser.add_argument(
        "--combine",
        default="softmax",
        choices=["softmax", "zscore"],
        help="How to mix the two 5-vectors (default: softmax then average).",
    )
    parser.add_argument("--jepa-weight", type=float, default=0.5)
    parser.add_argument("--supervised-weight", type=float, default=0.5)
    parser.add_argument(
        "--jepa-temp",
        type=float,
        default=0.1,
        help="Softmax temperature for JEPA scores (ignored for zscore).",
    )
    parser.add_argument(
        "--supervised-temp",
        type=float,
        default=1.0,
        help="Softmax temperature for supervised logits (ignored for zscore).",
    )
    parser.add_argument("--gold-parquet", default=DEFAULT_GOLD_PARQUET)
    parser.add_argument("--findings-parquet", default=DEFAULT_FINDINGS)
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
    parser.add_argument("--limit", type=int, default=None)

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
    print(f"[ensemble] jepa ckpt        = {args.jepa_ckpt}")
    print(f"[ensemble] supervised ckpt  = {args.supervised_ckpt}")
    run_eval(args, groups, image_roots, device)


if __name__ == "__main__":
    main()
