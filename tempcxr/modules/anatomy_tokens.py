"""Learnable anatomy region tokens with BioViL-T text warm-start.

Each token is a 128-D vector on the same unit sphere as JEPA patch
features. At construction they are random; ``init_from_text_encoder``
overwrites them with BioViL-T's projected global embedding of the
region name phrase (e.g. ``"cardiac silhouette"``), then training
continues to update the tokens via the anatomy contrastive loss.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

class AnatomyTokenBank(nn.Module):
    """``(K, D)`` learnable region tokens + attention pooling helpers."""

    def __init__(
        self,
        region_names: Sequence[str],
        d_model: int = 128,
        attn_temperature: float = 0.1,
    ):
        super().__init__()
        names = list(region_names)
        if not names:
            raise ValueError("region_names must be non-empty")
        self.region_names: List[str] = names
        self.d_model = d_model
        self.attn_temperature = attn_temperature
        # Random init; overwritten by BioViL-T text globals before train.
        self.tokens = nn.Parameter(
            F.normalize(torch.randn(len(names), d_model), dim=-1)
        )

    @property
    def num_regions(self) -> int:
        return len(self.region_names)

    @torch.no_grad()
    def init_from_text_encoder(self, text_encoder: nn.Module) -> None:
        """Warm-start tokens from BioViL-T text globals of region names.

        Runs ``text_encoder.forward_contrastive`` once on the region
        phrases and copies the L2-normalized projected CLS embeddings
        into ``self.tokens``. The text encoder is not kept as part of
        this path afterward — tokens are free parameters.
        """
        device = self.tokens.device
        was_training = text_encoder.training
        text_encoder.eval()
        txt_global, _, _ = text_encoder.forward_contrastive(self.region_names)
        txt_global = F.normalize(txt_global.detach(), dim=-1)
        if txt_global.shape != self.tokens.shape:
            raise RuntimeError(
                f"text globals {tuple(txt_global.shape)} != "
                f"tokens {tuple(self.tokens.shape)}"
            )
        self.tokens.copy_(txt_global.to(device=device, dtype=self.tokens.dtype))
        if was_training:
            text_encoder.train()

    def attend_pool(
        self,
        patches: torch.Tensor,  # (N, D) or (B, N, D)
        region_id: int,
    ) -> torch.Tensor:
        """Softmax-attention pool of patches with token ``region_id`` as query."""
        token = self.tokens[region_id]  # (D,)
        token = F.normalize(token, dim=-1)
        single = patches.ndim == 2
        if single:
            patches = patches.unsqueeze(0)
        patches = F.normalize(patches, dim=-1)
        # (B, N)
        logits = torch.einsum("bnd,d->bn", patches, token) / self.attn_temperature
        attn = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bn,bnd->bd", attn, patches)
        pooled = F.normalize(pooled, dim=-1)
        return pooled.squeeze(0) if single else pooled
