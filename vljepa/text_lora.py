"""Frozen LoRA add-on for the BioViL-T text encoder.

The base CXR-BERT stays as released. Rank-8 query and value matrices
from ``vljepa/lora/biovilt_text_epoch38.pt`` are loaded and then frozen
with the rest of the text tower.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LORA_CKPT = os.path.join(_HERE, "lora", "biovilt_text_epoch38.pt")
RANK = 8
ALPHA = 16.0


class LoRALinear(nn.Module):
    """Frozen pretrained linear plus a low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = float(alpha) / float(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (x @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + update * self.scale


def inject_lora(model: nn.Module, rank: int = RANK, alpha: float = ALPHA) -> int:
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and (name.endswith(".query") or name.endswith(".value"))
    ]
    for name in names:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        base = getattr(parent, parts[-1])
        setattr(parent, parts[-1], LoRALinear(base, rank, alpha))
    return len(names)


def lora_ckpt_path() -> str:
    return os.environ.get("VLJEPA_TEXT_LORA", DEFAULT_LORA_CKPT)


def apply_text_lora(text_encoder: nn.Module, ckpt_path: str | None = None) -> str:
    """Install the epoch-38 adapter on ``text_encoder.model`` and freeze it."""
    path = ckpt_path or lora_ckpt_path()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"BioViL-T text LoRA weights not found: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    weights = ckpt["lora"] if isinstance(ckpt, dict) and "lora" in ckpt else ckpt
    n_layers = inject_lora(
        text_encoder.model,
        rank=int(ckpt.get("rank", RANK)) if isinstance(ckpt, dict) else RANK,
        alpha=float(ckpt.get("alpha", ALPHA)) if isinstance(ckpt, dict) else ALPHA,
    )
    missing, unexpected = text_encoder.model.load_state_dict(weights, strict=False)
    bad_missing = [key for key in missing if "lora_" in key]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"LoRA load mismatch missing={bad_missing[:4]} unexpected={unexpected[:4]}"
        )
    if n_layers != 24:
        raise RuntimeError(f"expected 24 LoRA linears, installed {n_layers}")
    for param in text_encoder.parameters():
        param.requires_grad_(False)
    text_encoder.eval()
    return path
