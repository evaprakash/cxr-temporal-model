"""JEPA-side losses for the unit-sphere temporal CXR model.

After the unit-sphere refactor, both ``pred`` (the predictor's
``ẑ_cur``) and ``target`` (the EMA encoder's ``z_cur``) are L2-normalized
along the feature dim inside the model's forward pass. The natural loss
in that geometry is cosine — a directional loss that is automatically
scale-invariant — so this module exposes:

  * ``jepa_cosine_loss`` — per-patch ``1 - cos(ẑ, z_cur)`` (legacy /
    optional full-grid term; anatomy runs typically set ``W_JEPA=0``).
  * ``anatomy_masked_pool_jepa_loss`` — dual-mask anatomy JEPA: for each
    of 22 fixed CXAS anatomies, soft-pool ``ẑ`` with the **prior**
    anatomy mask and ``z_cur`` with the **current** anatomy mask, then
    ``1 - cos(u, v)``. Mean over anatomies, then over active pairs.
  * ``progression_classification_loss`` — 5-way CE. With anatomy weights,
    logit = mean over 22 organs of ``cos(u_a^c, v_a)`` (prior masks on
    ``ẑ^c``, current masks on ``z_cur``). Without weights, falls back to
    mean per-patch cosine (original rule).
  * ``change_localization_loss`` (legacy / other branches).

The contrastive (GLoRIA) losses live in ``losses.py`` and are reused
unchanged; they re-L2-normalize their inputs internally, so passing
already-unit-norm patches is a no-op.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def global_pool_normalize(
    patches: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean-pool over the patch axis and L2-normalize.

    patches : ``(..., N, D)`` — last two dims are patches × features.
    Returns ``(..., D)`` unit vectors.
    """
    pooled = patches.mean(dim=-2)
    return F.normalize(pooled, dim=-1, eps=eps)


def anatomy_dual_mask_cosine(
    pred_patches: torch.Tensor,
    target_patches: torch.Tensor,
    pred_patch_weights: torch.Tensor,
    target_patch_weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-anatomy cosine after dual-mask soft pooling.

    pred_patches         : ``(B, N, D)`` or ``(B, C, N, D)``
    target_patches       : ``(B, N, D)``
    pred_patch_weights   : ``(B, A, N)`` — prior-grid masks (pool ``ẑ``)
    target_patch_weights : ``(B, A, N)`` — current-grid masks (pool ``z_cur``)

    Returns
    -------
    ``(B, A)`` if pred is ``(B, N, D)``, or ``(B, C, A)`` if pred is
    ``(B, C, N, D)``.
    """
    pred = F.normalize(pred_patches, dim=-1, eps=eps)
    target = F.normalize(target_patches, dim=-1, eps=eps)
    w_pred = pred_patch_weights.to(device=pred.device, dtype=pred.dtype).clamp(min=0.0)
    w_tgt = target_patch_weights.to(device=pred.device, dtype=pred.dtype).clamp(min=0.0)

    w_pred_n = w_pred / w_pred.sum(dim=-1, keepdim=True).clamp(min=eps)
    w_tgt_n = w_tgt / w_tgt.sum(dim=-1, keepdim=True).clamp(min=eps)

    # v: (B, A, D) — current anatomy summaries
    v = torch.einsum("ban,bnd->bad", w_tgt_n, target)
    v = F.normalize(v, dim=-1, eps=eps)

    if pred.ndim == 3:
        # u: (B, A, D)
        u = torch.einsum("ban,bnd->bad", w_pred_n, pred)
        u = F.normalize(u, dim=-1, eps=eps)
        return (u * v).sum(dim=-1)  # (B, A)

    if pred.ndim == 4:
        # u: (B, C, A, D)
        u = torch.einsum("ban,bcnd->bcad", w_pred_n, pred)
        u = F.normalize(u, dim=-1, eps=eps)
        return (u * v.unsqueeze(1)).sum(dim=-1)  # (B, C, A)

    raise ValueError(
        f"pred_patches must be (B,N,D) or (B,C,N,D); got {tuple(pred.shape)}"
    )


# =========================================================
# JEPA COSINE LOSS (PER-PATCH — LEGACY / OPTIONAL FULL-GRID)
# =========================================================
def jepa_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-patch cosine distance between predicted and target patches.

    pred   : (B, N, D) predictor output (L2-norm, with gradient).
    target : (B, N, D) EMA target encoder output (L2-norm, detached).

    Returns the mean of ``1 - cos(pred, target)`` across batch and
    patches. Anatomy runs typically leave this unused (``W_JEPA=0``).
    """
    pred = F.normalize(pred, dim=-1, eps=eps)
    target = F.normalize(target, dim=-1, eps=eps)
    cos_per_patch = (pred * target).sum(dim=-1)  # (B, N)
    return (1.0 - cos_per_patch).mean()


# =========================================================
# 4TH LOSS — PROGRESSION CLASSIFICATION (IMAGE–IMAGE 5-WAY CE)
# =========================================================
def progression_classification_loss(
    pred_progression_patches: torch.Tensor,  # (B, C, N, D)
    current_patches_target: torch.Tensor,    # (B, N, D), detached
    silver_labels: torch.Tensor,             # (B,) long, values in [0, C)
    temperature: float = 0.1,
    eps: float = 1e-8,
    class_weights: Optional[torch.Tensor] = None,
    pred_patch_weights: Optional[torch.Tensor] = None,   # (B, A, N) prior
    target_patch_weights: Optional[torch.Tensor] = None, # (B, A, N) curr
) -> torch.Tensor:
    """5-way image-image cross-entropy on candidate latents.

    Without anatomy weights (default)::

        logit[b, c] = mean_p cos(ẑ_cur^c[b, p], z_cur[b, p])

    With dual anatomy masks (prior on ``ẑ^c``, current on ``z_cur``)::

        logit[b, c] = mean_a cos(u_a^c, v_a)

    where ``u_a^c`` / ``v_a`` are soft-mask-pooled organ summaries.
    """
    if pred_progression_patches.numel() == 0:
        return pred_progression_patches.new_zeros(())

    pred = F.normalize(pred_progression_patches, dim=-1, eps=eps)
    target = F.normalize(current_patches_target, dim=-1, eps=eps)

    if pred_patch_weights is not None and target_patch_weights is not None:
        cos_per_anat = anatomy_dual_mask_cosine(
            pred, target, pred_patch_weights, target_patch_weights, eps=eps,
        )  # (B, C, A)
        logits = cos_per_anat.mean(dim=-1)  # (B, C)
    else:
        cos_per_patch = (pred * target.unsqueeze(1)).sum(dim=-1)  # (B, C, N)
        logits = cos_per_patch.mean(dim=-1)  # (B, C)

    logits = logits / temperature
    return F.cross_entropy(logits, silver_labels, weight=class_weights)


# =========================================================
# ANATOMY DUAL-MASK POOLED JEPA (22 FIXED CXAS)
# =========================================================
def anatomy_masked_pool_jepa_loss(
    pred_patches: torch.Tensor,          # (B, N, D) dynamic-conditioned ẑ_cur
    target_patches: torch.Tensor,        # (B, N, D) z_cur (stop-grad)
    pred_patch_weights: torch.Tensor,    # (B, A, N) prior-image anatomy masks
    target_patch_weights: torch.Tensor,  # (B, A, N) current-image anatomy masks
    active: torch.Tensor,                # (B,) bool — True → contribute
    eps: float = 1e-8,
) -> torch.Tensor:
    """JEPA cosine on per-anatomy soft-mask-pooled region summaries.

    For each active sample and anatomy ``a``::

        u_a = normalize( Σ_n w^prior_{a,n}  ẑ_n / Σ_n w^prior_{a,n} )
        v_a = normalize( Σ_n w^curr_{a,n}   z_n / Σ_n w^curr_{a,n} )
        L_a = 1 - cos(u_a, v_a)

    Mean over the A anatomies, then over active batch rows. Inactive
    rows (missing / incomplete 22-mask inventory on prior or current)
    are omitted. Returns 0 when no row is active.
    """
    if pred_patches.numel() == 0 or not bool(active.any()):
        return pred_patches.new_zeros(())

    w_pred = pred_patch_weights.to(
        device=pred_patches.device, dtype=pred_patches.dtype
    )
    w_tgt = target_patch_weights.to(
        device=pred_patches.device, dtype=pred_patches.dtype
    )

    if w_pred.ndim != 3 or w_tgt.ndim != 3:
        raise ValueError(
            f"anatomy weights must be (B, A, N); got "
            f"pred={tuple(w_pred.shape)} tgt={tuple(w_tgt.shape)}"
        )
    if w_pred.shape[0] != pred_patches.shape[0] or w_pred.shape[2] != pred_patches.shape[1]:
        raise ValueError(
            f"pred_patch_weights shape {tuple(w_pred.shape)} incompatible "
            f"with pred_patches {tuple(pred_patches.shape)}"
        )
    if w_tgt.shape[0] != target_patches.shape[0] or w_tgt.shape[2] != target_patches.shape[1]:
        raise ValueError(
            f"target_patch_weights shape {tuple(w_tgt.shape)} incompatible "
            f"with target_patches {tuple(target_patches.shape)}"
        )
    if w_pred.shape[1] != w_tgt.shape[1]:
        raise ValueError(
            f"anatomy count mismatch: pred A={w_pred.shape[1]} tgt A={w_tgt.shape[1]}"
        )

    active = active.to(device=pred_patches.device).bool()
    cos = anatomy_dual_mask_cosine(
        pred_patches[active],
        target_patches[active],
        w_pred[active],
        w_tgt[active],
        eps=eps,
    )  # (B', A)
    return (1.0 - cos).mean()


# =========================================================
# CHANGE LOCALIZATION (PRIOR FINDING MASK, ADD-ON)
# =========================================================
def change_localization_loss(
    pred_patches: torch.Tensor,    # (B, N, D) dynamic-conditioned ẑ_cur
    prior_patches: torch.Tensor,   # (B, N, D) z_prior (same prior grid)
    patch_weights: torch.Tensor,   # (B, N) prior-image soft finding mask
    active: torch.Tensor,          # (B,) bool — True → contribute
    eps: float = 1e-8,
) -> torch.Tensor:
    """Concentrate predictor change energy inside the prior finding mask.

    For each active sample::

        s_n   = 1 - cos(ẑ_n, z_prior_n)          # per-patch change map
        s_in  = Σ_n w_n s_n / Σ_n w_n            # soft float pool inside
        s_out = Σ_n (1-w_n) s_n / Σ_n (1-w_n)    # soft float pool outside
        L     = -(s_in - s_out)

    ``w`` is the downsampled float finding-mask coverage on the **prior**
    image (ẑ / z_prior live on the prior patch grid). Inactive rows are
    omitted from the mean. Returns 0 when no row is active.
    """
    if pred_patches.numel() == 0 or not bool(active.any()):
        return pred_patches.new_zeros(())

    pred = F.normalize(pred_patches, dim=-1, eps=eps)
    prior = F.normalize(prior_patches, dim=-1, eps=eps)
    s = (1.0 - (pred * prior).sum(dim=-1)).clamp(min=0.0)  # (B, N)

    w = patch_weights.to(device=s.device, dtype=s.dtype).clamp(0.0, 1.0)
    if w.shape != s.shape:
        raise ValueError(
            f"patch_weights shape {tuple(w.shape)} != change map {tuple(s.shape)}"
        )

    active = active.to(device=s.device).bool()
    s = s[active]
    w = w[active]
    w_out = (1.0 - w).clamp(min=0.0)

    w_sum = w.sum(dim=-1).clamp(min=eps)
    w_out_sum = w_out.sum(dim=-1).clamp(min=eps)
    s_in = (w * s).sum(dim=-1) / w_sum
    s_out = (w_out * s).sum(dim=-1) / w_out_sum
    return (-(s_in - s_out)).mean()
