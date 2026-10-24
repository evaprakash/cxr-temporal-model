#!/usr/bin/env python3
"""CPU smoke on one gold pair: forward pass, then the gold set-match metrics.

Run on ccw-gpu-1 after a CPU srun, from the repo on /nvme:

    CUDA_VISIBLE_DEVICES="" python -m vljepa.smoke_ccw_cpu
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

import torch

from .cluster_paths import gold_image_dirs, image_roots, inventory
from .eval_gold import eval_gold_setmatch, load_gold_groups
from .model import VLJEPA
from progression_classify import DEFAULT_GOLD_PARQUET, discover_gold_image_roots


def _print(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    _print("[smoke] cpu forward + one gold group")
    bad = []
    for name, path, ok, detail in inventory(require_sample=False):
        mark = "OK  " if ok else "FAIL"
        extra = f"  ({detail})" if detail else ""
        _print(f"[smoke] [{mark}] {name:32s} {path}{extra}")
        if not ok:
            bad.append(name)
    if bad:
        _print("[smoke] missing: " + ", ".join(bad))
        return 1

    device = torch.device("cpu")
    model = VLJEPA(
        image_mode="biovilt",
        smoke=False,
        gradient_checkpointing=False,
        freeze_image_encoder=True,
        freeze_text_encoder=True,
    ).to(device).eval()
    text_trainable = sum(p.requires_grad for p in model.text_encoder.parameters())
    image_trainable = sum(p.requires_grad for p in model.image_encoder.parameters())
    _print(f"[smoke] text lora = {model.text_lora_path}")
    _print(f"[smoke] trainable text params = {text_trainable}")
    _print(f"[smoke] trainable image params = {image_trainable}")
    if text_trainable or image_trainable:
        _print("[smoke] text and image encoders must stay frozen")
        return 1

    groups = load_gold_groups().iloc[:1].reset_index(drop=True)
    gold_dir = os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))
    roots = {
        **image_roots(),
        **discover_gold_image_roots(gold_dir),
        **gold_image_dirs(),
    }
    summary = eval_gold_setmatch(model, groups, roots, epoch=0, device=device)
    if not summary:
        _print("[smoke] gold metrics produced no rows (image missing?)")
        return 1
    _print(f"[smoke] gold summary keys = {sorted(summary)}")
    _print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
