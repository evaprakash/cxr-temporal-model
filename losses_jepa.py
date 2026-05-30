"""I-JEPA-style losses for the temporal CXR task.

This module is intentionally narrow: it only contains losses that are NEW
relative to ``losses.py``. The contrastive (GLoRIA) losses are reused
unchanged via ``from losses import local_contrastive_loss`` in callers.

Design choice: the LayerNorm-no-params on the JEPA target is applied
*inside the model's forward* (mirroring I-JEPA's ``forward_target``),
not here. By the time the loss is called, the target is already
LN-normalized; this function is therefore a thin wrapper around
``F.smooth_l1_loss``.
"""

import torch
import torch.nn.functional as F


# =========================================================
# JEPA SMOOTH L1 LOSS
# =========================================================
def jepa_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    I-JEPA-style Smooth L1 (Huber) loss between predicted and target patches.

    pred   : (B, N, D) predictor output (with gradient).
    target : (B, N, D) LayerNorm-normalized target encoder output
             (already detached / under stop-gradient).

    Smooth L1 is MSE for small residuals and L1 in the tails; this is the
    same loss used in the official I-JEPA reference implementation:
        https://github.com/facebookresearch/ijepa
    """
    return F.smooth_l1_loss(pred, target)


# =========================================================
# (Optional) helper: feature-dim LayerNorm on JEPA targets
# =========================================================
def jepa_target_layernorm(target: torch.Tensor) -> torch.Tensor:
    """
    Feature-dim LayerNorm with no learnable parameters.

    Mirrors I-JEPA's ``F.layer_norm(h, (h.size(-1),))`` step. Stabilizes
    the target distribution as the EMA target encoder drifts. The model's
    forward applies this internally; exposing it here lets callers
    re-derive the same target geometry outside the model if needed.
    """
    return F.layer_norm(target, (target.size(-1),))
