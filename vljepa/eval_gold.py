"""CheXTemporal gold 5-way / set-match for VL-JEPA.

For each gold ``(pair, finding)``:

  1. Encode prior+current with BioViL-T (pair).
  2. Query = ``What is the progression of {finding}?`` (Llama tok+embed)
  3. Predict Ŝ.
  4. Score cosine(Ŝ, BioViL-T Y-encoder(``{Finding} is {class}.``)) for all 5 classes.
  5. Argmax / top-|GT| set-match (same tables as the JEPA trainer).
"""

from __future__ import annotations

import argparse
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

from typing import Dict, Optional

import torch
from tqdm import tqdm

from . import _root  # noqa: F401

from gold_progression_setmatch import (
    format_running_setmatch,
    group_gold_by_pair_finding,
    print_setmatch_report,
    summarize_setmatch,
    topk_set_match,
)
from progression_classify import (
    DEFAULT_GOLD_PARQUET,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER
from dataset_combined_jepa import DEFAULT_FINDINGS

from .model import VLJEPA
from .prompts import QUERY_TEMPLATE, TARGET_TEMPLATE


@torch.no_grad()
def eval_gold_setmatch(
    model: VLJEPA,
    groups,
    image_roots: Dict[str, str],
    epoch: int,
    device: torch.device,
) -> Optional[dict]:
    model.eval()
    results = []
    skipped = 0
    text_cache: Dict[str, torch.Tensor] = {}
    pbar = tqdm(
        range(len(groups)),
        desc=f"vljepa gold ep{epoch}",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for i in pbar:
        row = groups.iloc[i]
        try:
            prior = load_image_tensor(
                row["dataset"], row["parent_image_prev"], image_roots,
            ).to(device)
            current = load_image_tensor(
                row["dataset"], row["parent_image_curr"], image_roots,
            ).to(device)
        except (FileNotFoundError, OSError):
            skipped += 1
            pbar.set_postfix(
                skipped=skipped,
                metrics=format_running_setmatch(results),
            )
            continue
        scores = model.score_classes(
            prior.unsqueeze(0),
            current.unsqueeze(0),
            str(row["finding"]),
            text_cache=text_cache,
        ).detach().float().cpu().tolist()
        results.append(
            topk_set_match(
                scores,
                list(row["gt_labels"]),
                CLS_ORDER,
                finding=str(row["finding"]),
            )
        )
        pbar.set_postfix(
            skipped=skipped,
            metrics=format_running_setmatch(results),
        )
    pbar.close()
    if skipped:
        print(f"[vljepa gold] skipped missing images: {skipped}")
    if not results:
        print("[vljepa gold] set-match: no groups evaluated")
        return None
    print_setmatch_report(
        results,
        "vljepa",
        f", query={QUERY_TEMPLATE!r}, target={TARGET_TEMPLATE!r}, "
        f"epoch={epoch}",
    )
    return summarize_setmatch(results)


def load_gold_groups(gold_parquet: Optional[str] = None):
    gold = load_gold_pairs(
        gold_parquet or DEFAULT_GOLD_PARQUET,
        DEFAULT_FINDINGS,
    )
    return group_gold_by_pair_finding(gold)


def load_vljepa_from_ckpt(ckpt_path: str, device: torch.device) -> VLJEPA:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("vljepa_cfg", {})
    model = VLJEPA(
        image_mode="biovilt_no_pretrained",
        smoke=False,
        n_llama_layers=int(cfg.get("n_llama_layers", 8)),
        llama_name=cfg.get("llama_name", "meta-llama/Llama-3.2-1B"),
        llama_local=cfg.get("llama_local"),
        freeze_image_encoder=bool(cfg.get("freeze_image_encoder", True)),
        freeze_text_encoder=bool(cfg.get("freeze_text_encoder", False)),
        gradient_checkpointing=False,
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[vljepa gold] missing keys: {missing[:8]}...")
    if unexpected:
        print(f"[vljepa gold] unexpected keys: {unexpected[:8]}...")
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vljepa_from_ckpt(args.ckpt, device)
    groups = load_gold_groups()
    if args.limit:
        groups = groups.iloc[: args.limit].reset_index(drop=True)
    gold_parquet_dir = os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))
    _image_roots_dir = os.environ.get(
        "JEPA_IMAGE_ROOTS_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "all_data"),
    )
    roots = {
        "mimic": os.path.join(_image_roots_dir, "mimic"),
        "chexpert": os.path.join(_image_roots_dir, "chexpert", "train"),
        "rexgradient": os.path.join(_image_roots_dir, "rexgradient", "deid_png"),
        **discover_gold_image_roots(gold_parquet_dir),
    }
    eval_gold_setmatch(model, groups, roots, epoch=-1, device=device)


if __name__ == "__main__":
    main()
