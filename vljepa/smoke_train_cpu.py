#!/usr/bin/env python3
"""CPU smoke of the five-guess trainer.

One synthetic example and one optimizer step per epoch, for five epochs.
Uses the tiny Llama predictor and stub image/text encoders, then writes
``epoch_1.pt`` ... ``epoch_5.pt``.

    python -m vljepa.smoke_train_cpu
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

import torch
from torch.optim import AdamW

from .model import N_CLS, five_guess_loss
from .prompts import class_target_texts, query_text
from .smoke_test import _build_stub_model
from progression_phrases import CLS_ORDER

EPOCHS = 5
TEMPERATURE = 0.07
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
CKPT_DIR = os.environ.get(
    "VLJEPA_SMOKE_CKPT_DIR",
    os.path.join(_ROOT, "checkpoints_vljepa_fiveguess_smoke"),
)

# One finding, one gold class per epoch. Epoch 1 is improving, then stable,
# worsening, new, resolved.
_EXAMPLES = (
    ("edema", "improving"),
    ("atelectasis", "stable"),
    ("cardiomegaly", "worsening"),
    ("consolidation", "new"),
    ("pleural effusion", "resolved"),
)


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _reject_single_guess() -> None:
    pred = torch.randn(2, 128)
    targets = torch.randn(2, N_CLS, 128)
    labels = torch.zeros(2, dtype=torch.long)
    try:
        five_guess_loss(pred, targets, labels, temperature=TEMPERATURE)
    except ValueError:
        _print("[smoke-train] single-guess pred rejected")
        return
    raise SystemExit("[smoke-train] single-guess pred was accepted")


def main() -> int:
    torch.manual_seed(0)
    _print("=" * 72)
    _print("VL-JEPA CPU train smoke (five guesses, 1 example x 5 epochs)")
    _print(f"  ckpt dir: {CKPT_DIR}")
    _print("=" * 72)
    _reject_single_guess()

    model = _build_stub_model()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("[smoke-train] no trainable parameters")
    head = model.predictor.out_proj.weight
    if head.shape[0] != N_CLS * model.predictor.out_dim:
        raise SystemExit(
            f"[smoke-train] out_proj {tuple(head.shape)} is not "
            f"{N_CLS} guesses x {model.predictor.out_dim}"
        )
    optimizer = AdamW(params, lr=2e-5, weight_decay=0.01)
    os.makedirs(CKPT_DIR, exist_ok=True)

    saved = []
    for epoch in range(1, EPOCHS + 1):
        finding, cls_name = _EXAMPLES[epoch - 1]
        cls_idx = CLS_ORDER.index(cls_name)
        prior = torch.randn(1, 3, 64, 64)
        current = torch.randn(1, 3, 64, 64)
        query = query_text(finding)
        targets = class_target_texts(finding)
        labels = torch.tensor([cls_idx], dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        out = model(prior, current, [query], target_texts=targets)
        if tuple(out["pred"].shape) != (1, N_CLS, model.predictor.out_dim):
            raise SystemExit(
                f"[smoke-train] pred shape {tuple(out['pred'].shape)}, "
                f"expected (1, {N_CLS}, {model.predictor.out_dim})"
            )
        loss, logits = five_guess_loss(
            out["pred"], out["target_global"], labels, temperature=TEMPERATURE,
        )
        if not torch.isfinite(loss):
            raise SystemExit(f"[smoke-train] non-finite loss at epoch {epoch}")
        loss.backward()
        if model.predictor.out_proj.weight.grad is None:
            raise SystemExit("[smoke-train] guess head got no gradient")
        optimizer.step()
        pred_cls = int(logits.argmax(dim=-1).item())
        path = os.path.join(CKPT_DIR, f"epoch_{epoch}.pt")
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": float("inf"),
            "vljepa_cfg": {
                "loss": "five_guess",
                "n_guesses": N_CLS,
                "freeze_image_encoder": True,
                "freeze_text_encoder": True,
                "temperature": TEMPERATURE,
                "smoke": True,
            },
        }
        torch.save(ckpt, path)
        saved.append(path)
        _print(
            f"[smoke-train] epoch {epoch}: loss={float(loss.item()):.4f} "
            f"gold={cls_name} pred={CLS_ORDER[pred_cls]} saved {path}"
        )

    last = torch.load(saved[-1], map_location="cpu")
    if last.get("vljepa_cfg", {}).get("loss") != "five_guess":
        raise SystemExit("[smoke-train] reloaded checkpoint is not five-guess")
    if int(last["vljepa_cfg"].get("n_guesses", 0)) != N_CLS:
        raise SystemExit("[smoke-train] reloaded checkpoint has the wrong guess count")
    reloaded = _build_stub_model()
    reloaded.load_state_dict(last["model"], strict=True)
    missing = [p for p in saved if not os.path.isfile(p)]
    if missing or len(saved) != EPOCHS:
        raise SystemExit(f"[smoke-train] expected {EPOCHS} checkpoints, missing {missing}")
    _print("")
    _print(f"SMOKE TRAIN PASSED  ({EPOCHS} checkpoints in {CKPT_DIR})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
