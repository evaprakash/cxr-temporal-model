"""JEPA-side losses for the unit-sphere temporal CXR model.

After the unit-sphere refactor, both ``pred`` (the predictor's
``ẑ_cur``) and ``target`` (the EMA encoder's ``z_cur``) are L2-normalized
along the feature dim inside the model's forward pass. The natural loss
in that geometry is cosine — a directional loss that is automatically
scale-invariant — so this module exposes:

  * ``jepa_cosine_loss`` for the main JEPA invariant:
    global-pool ``ẑ_cur`` and ``z_cur`` over patches, re-normalize, then
    ``1 - cos(ẑ_global, z_global)``.
  * ``progression_classification_loss`` for the 4th loss: a 5-way CE on
    image-image cosine *logits*, computed from N candidate ``ẑ_cur^c``
    (one per progression class). Same global-pool cosine as JEPA, so
    train rule = test rule. Supports optional per-class weights (Cui et
    al. 2019 "Class-Balanced Loss Based on Effective Number of Samples")
    so the minority silver classes (``resolved`` ≈ 1 % of silver) get a
    proportionally larger gradient than the majority ``stable`` class.
  * ``anatomy_masked_pool_jepa_loss`` (optional / off on main): for each
    of 22 fixed CXAS anatomies, soft-pool ``ẑ`` with the prior anatomy
    mask and ``z_cur`` with the current anatomy mask, then take
    ``1 - cos(u, v)``. Kept for ablations; not used when global-pool
    JEPA is the main objective.
  * ``change_localization_loss`` (legacy / other branches): concentrate
    prior-grid change energy inside a finding mask.

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


# =========================================================
# JEPA COSINE LOSS (GLOBAL POOL)
# =========================================================
def jepa_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Global-pool cosine distance between predicted and target maps.

    pred   : (B, N, D) predictor output (L2-norm, with gradient).
    target : (B, N, D) EMA target encoder output (L2-norm, detached).

    Averages patches → one vector per sample, re-normalizes, then returns
    the mean of ``1 - cos(pred_global, target_global)`` over the batch.
    This avoids per-patch index matching (registration noise across
    prior/current grids).
    """
    pred = F.normalize(pred, dim=-1, eps=eps)
    target = F.normalize(target, dim=-1, eps=eps)
    pred_g = global_pool_normalize(pred, eps=eps)
    target_g = global_pool_normalize(target, eps=eps)
    cos = (pred_g * target_g).sum(dim=-1)  # (B,)
    return (1.0 - cos).mean()


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
) -> torch.Tensor:
    """5-way image-image cross-entropy on global-pooled candidate latents.

    For each pair ``b`` and progression class ``c``:

        logit[b, c] = cos(pool(ẑ_cur^c[b]), pool(z_cur[b]))

    where ``pool`` is mean over patches then L2-normalize, and
    ``ẑ_cur^c[b]`` is the predictor's output when conditioned on the
    class-c prompt ``"{prog_finding[b]} is {class[c]}."`` (computed
    upstream by ``TempCXRJEPA.forward`` running the predictor C times per
    pair). The standard cross-entropy is then applied to
    ``logits / temperature`` against the silver progression label.

    Same aggregation as ``jepa_cosine_loss`` so the classification
    objective matches the JEPA regression objective and the image-image
    eval rule in ``eval_progression_jepa.py``.

    Parameters
    ----------
    pred_progression_patches
        ``(B, C, N, D)``. The predictor's ``ẑ_cur^c`` for each pair and
        candidate class. Already L2-normalized by the predictor's final
        renormalization; we re-normalize defensively.
    current_patches_target
        ``(B, N, D)``. The EMA target encoder's ``z_cur``, detached
        (stop-grad).
    silver_labels
        ``(B,)`` integers in ``[0, C)``. The silver-derived progression
        class index for the per-pair ``prog_finding``.
    temperature
        Softmax temperature. Cosine logits live in ``[-1, 1]``, so
        ``temperature=0.1`` gives an effective ``[-10, 10]`` logit range
        — peaky enough to be discriminative without saturating.
    eps
        L2-normalization numeric stability epsilon.
    class_weights
        Optional ``(C,)`` float tensor of per-class weights forwarded to
        ``F.cross_entropy(..., weight=class_weights)``. Intended for
        class-balanced re-weighting of the CE — e.g. the effective-
        number-of-samples scheme (Cui et al. 2019) that up-weights rare
        silver classes (``resolved`` at 1 % of silver would otherwise
        contribute negligible gradient). When ``None`` (default) the
        loss reduces to standard unweighted CE.

    Returns
    -------
    Scalar CE loss tensor on the same device as ``pred_progression_patches``.
    Returns 0 if the batch carries no candidates (degenerate edge case
    that shouldn't fire in practice but lets the trainer keep a single
    code path).
    """
    if pred_progression_patches.numel() == 0:
        return pred_progression_patches.new_zeros(())

    pred = F.normalize(pred_progression_patches, dim=-1, eps=eps)
    target = F.normalize(current_patches_target, dim=-1, eps=eps)
    # pred   : (B, C, N, D) → pool N → (B, C, D)
    # target : (B, N, D)    → pool N → (B, D)
    pred_g = global_pool_normalize(pred, eps=eps)
    target_g = global_pool_normalize(target, eps=eps)
    logits = (pred_g * target_g.unsqueeze(1)).sum(dim=-1)  # (B, C)
    logits = logits / temperature

    return F.cross_entropy(logits, silver_labels, weight=class_weights)


# =========================================================
# ANATOMY DUAL-MASK POOLED JEPA (22 FIXED CXAS, ADD-ON)
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
    are omitted. Returns 0 when no row is active. Full-grid / global-pool
    ``jepa_cosine_loss`` is separate.
    """
    if pred_patches.numel() == 0 or not bool(active.any()):
        return pred_patches.new_zeros(())

    w_pred = pred_patch_weights.to(
        device=pred_patches.device, dtype=pred_patches.dtype
    ).clamp(min=0.0)
    w_tgt = target_patch_weights.to(
        device=pred_patches.device, dtype=pred_patches.dtype
    ).clamp(min=0.0)

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
    pred = pred_patches[active]          # (B', N, D)
    target = target_patches[active]
    w_pred = w_pred[active]              # (B', A, N)
    w_tgt = w_tgt[active]

    # Broadcast patches over anatomy: (B', A, N, D)
    a = w_pred.shape[1]
    pred_e = pred.unsqueeze(1).expand(-1, a, -1, -1)
    tgt_e = target.unsqueeze(1).expand(-1, a, -1, -1)

    w_pred_sum = w_pred.sum(dim=-1, keepdim=True).clamp(min=eps)
    w_tgt_sum = w_tgt.sum(dim=-1, keepdim=True).clamp(min=eps)
    w_pred_n = w_pred / w_pred_sum
    w_tgt_n = w_tgt / w_tgt_sum

    u = (pred_e * w_pred_n.unsqueeze(-1)).sum(dim=2)  # (B', A, D)
    v = (tgt_e * w_tgt_n.unsqueeze(-1)).sum(dim=2)
    u = F.normalize(u, dim=-1, eps=eps)
    v = F.normalize(v, dim=-1, eps=eps)
    return (1.0 - (u * v).sum(dim=-1)).mean()


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
    Full-grid ``jepa_cosine_loss`` is separate — this does **not** match
    ẑ to z_cur appearance.
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
