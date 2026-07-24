"""JEPA-side losses for the unit-sphere temporal CXR model.

After the unit-sphere refactor, both ``pred`` (the predictor's
``ẑ_cur``) and ``target`` (the EMA encoder's ``z_cur``) are L2-normalized
along the feature dim inside the model's forward pass. The natural loss
in that geometry is cosine — a directional loss that is automatically
scale-invariant — so this module exposes:

  * ``jepa_cosine_loss`` for the main JEPA invariant:
    ``1 - cos(ẑ_cur, z_cur)`` averaged over patches.
  * ``progression_classification_loss`` for the 4th loss: a 5-way CE on
    image-image cosine *logits*, computed from N candidate ``ẑ_cur^c``
    (one per progression class). This is the train-time analog of the
    image-image eval rule in ``eval_progression_jepa.py`` —
    "best match is determined through cos(ẑ_cur, z_cur)" — so the
    training objective matches what's actually being measured at test
    time. Supports optional per-class weights (Cui et al. 2019
    "Class-Balanced Loss Based on Effective Number of Samples") so the
    minority silver classes (``resolved`` ≈ 1 % of silver) get a
    proportionally larger gradient than the majority ``stable`` class.
  * ``change_localization_directional_loss`` for optional grounding:
    (1) **where** — soft-pool the prior-grid change map
    ``s = 1 - cos(ẑ, z_prior)`` inside vs outside the prior finding mask
    and maximize ``s_in - s_out``; (2) **which way** — soft-pool region
    summaries and push ẑ toward the progression-appropriate side of
    ``(z_prior, z_cur)``. Stable is excluded by the trainer. Full-grid
    JEPA is unchanged.

The contrastive (GLoRIA) losses live in ``losses.py`` and are reused
unchanged; they re-L2-normalize their inputs internally, so passing
already-unit-norm patches is a no-op.
"""

from typing import Optional

import torch
import torch.nn.functional as F


# =========================================================
# JEPA COSINE LOSS
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
    patches. Inputs are expected to already be L2-normalized along the
    feature dim by the model; we still re-normalize defensively so the
    loss is well-defined even if a caller forgets.
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
) -> torch.Tensor:
    """5-way image-image cross-entropy on the predictor's candidate latents.

    For each pair ``b`` and progression class ``c``:

        logit[b, c] = mean over patches of cos(ẑ_cur^c[b], z_cur[b])

    where ``ẑ_cur^c[b]`` is the predictor's output when conditioned on the
    class-c prompt ``"{prog_finding[b]} is {class[c]}."`` (computed
    upstream by ``TempCXRJEPA.forward`` running the predictor C times per
    pair). The standard cross-entropy is then applied to
    ``logits / temperature`` against the silver progression label.

    The aggregation is mean-over-patches because the JEPA loss is
    per-patch cosine averaged over patches — this keeps the
    classification objective consistent with the regression objective:
    both are scored on the same per-patch quantity, just rolled up two
    different ways (regression = average; classification = pick the
    argmax candidate's average).

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
    # Broadcast target over the candidate-class dim:
    #   pred   : (B, C, N, D)
    #   target : (B, 1, N, D)
    cos_per_patch = (pred * target.unsqueeze(1)).sum(dim=-1)  # (B, C, N)
    logits = cos_per_patch.mean(dim=-1)                        # (B, C)
    logits = logits / temperature

    return F.cross_entropy(logits, silver_labels, weight=class_weights)


# =========================================================
# CHANGE LOCALIZATION — WHERE + WHICH WAY (ADD-ON)
# =========================================================
# CLS_ORDER indices: improving=0, stable=1, worsening=2, new=3, resolved=4
_CLS_IMPROVING = 0
_CLS_WORSENING = 2
_CLS_NEW = 3
_CLS_RESOLVED = 4


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


def _soft_pool_patches(
    patches: torch.Tensor,  # (B, N, D)
    weights: torch.Tensor,  # (B, N)
    eps: float,
) -> torch.Tensor:
    """L2-normalized soft-mask pooled patch summary ``(B, D)``."""
    w = weights.to(device=patches.device, dtype=patches.dtype).clamp(min=0.0)
    w_sum = w.sum(dim=-1, keepdim=True).clamp(min=eps)
    w_n = w / w_sum
    u = (patches * w_n.unsqueeze(-1)).sum(dim=1)
    return F.normalize(u, dim=-1, eps=eps)


def change_direction_loss(
    pred_patches: torch.Tensor,       # (B, N, D) ẑ
    prior_patches: torch.Tensor,      # (B, N, D) z_prior
    current_patches: torch.Tensor,    # (B, N, D) z_cur (stop-grad)
    prior_weights: torch.Tensor,      # (B, N)
    curr_weights: torch.Tensor,       # (B, N)
    prior_active: torch.Tensor,       # (B,)
    curr_active: torch.Tensor,        # (B,)
    prog_cls_idx: torch.Tensor,       # (B,)
    active: torch.Tensor,             # (B,) class+mask gate from trainer
    eps: float = 1e-8,
) -> torch.Tensor:
    """Push ẑ toward the progression-appropriate side of (z_prior, z_cur).

    Soft-pool finding-region summaries (prior mask for ẑ / z_prior when
    available, else current mask; current mask for z_cur when available,
    else prior mask). Per active sample:

      * worsening / resolved: ``L = -(cos(û, u_cur) - cos(û, u_prior))``
      * improving:            ``L = -(cos(û, u_prior) - cos(û, u_cur))``
      * new:                  ``L = 1 - cos(û, u_cur)``

    Returns 0 when no row is active.
    """
    if pred_patches.numel() == 0 or not bool(active.any()):
        return pred_patches.new_zeros(())

    device = pred_patches.device
    active = active.to(device=device).bool()
    prior_active = prior_active.to(device=device).bool()
    curr_active = curr_active.to(device=device).bool()

    # Need at least one mask to define a finding region.
    region_ok = prior_active | curr_active
    active = active & region_ok
    if not bool(active.any()):
        return pred_patches.new_zeros(())

    pred = F.normalize(pred_patches, dim=-1, eps=eps)
    prior = F.normalize(prior_patches, dim=-1, eps=eps)
    current = F.normalize(current_patches, dim=-1, eps=eps)
    w_prior = prior_weights.to(device=device, dtype=pred.dtype).clamp(0.0, 1.0)
    w_curr = curr_weights.to(device=device, dtype=pred.dtype).clamp(0.0, 1.0)

    # Per-row pool weight for ẑ / z_prior: prefer prior mask, else curr.
    use_prior_for_hat = prior_active & active
    w_hat = torch.where(
        use_prior_for_hat.unsqueeze(-1),
        w_prior,
        w_curr,
    )
    # z_cur pool weight: prefer curr mask, else prior.
    use_curr_for_cur = curr_active & active
    w_cur = torch.where(
        use_curr_for_cur.unsqueeze(-1),
        w_curr,
        w_prior,
    )

    pred_a = pred[active]
    prior_a = prior[active]
    current_a = current[active]
    w_hat_a = w_hat[active]
    w_cur_a = w_cur[active]
    cls_a = prog_cls_idx.to(device=device)[active]

    u_hat = _soft_pool_patches(pred_a, w_hat_a, eps)
    u_pri = _soft_pool_patches(prior_a, w_hat_a, eps)
    u_cur = _soft_pool_patches(current_a, w_cur_a, eps)

    c_cur = (u_hat * u_cur).sum(dim=-1)
    c_pri = (u_hat * u_pri).sum(dim=-1)

    # Default: unused classes contribute 0 (should not appear if trainer gates).
    per = pred_a.new_zeros(pred_a.shape[0])

    is_worsening = cls_a == _CLS_WORSENING
    is_resolved = cls_a == _CLS_RESOLVED
    is_improving = cls_a == _CLS_IMPROVING
    is_new = cls_a == _CLS_NEW

    # Move toward current away from prior.
    toward_cur = -(c_cur - c_pri)
    # Move toward prior away from current (improving "which way").
    toward_pri = -(c_pri - c_cur)
    match_cur = 1.0 - c_cur

    per = torch.where(is_worsening | is_resolved, toward_cur, per)
    per = torch.where(is_improving, toward_pri, per)
    per = torch.where(is_new, match_cur, per)

    return per.mean()


def change_localization_directional_loss(
    pred_patches: torch.Tensor,
    prior_patches: torch.Tensor,
    current_patches: torch.Tensor,
    prior_weights: torch.Tensor,
    curr_weights: torch.Tensor,
    prior_active: torch.Tensor,
    curr_active: torch.Tensor,
    prog_cls_idx: torch.Tensor,
    active: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Where (change map in mask) + which-way (region direction) sum.

    ``active`` is the trainer's class gate (non-stable). Where further
    requires a prior mask; which-way requires prior or current mask.
    Either term may be 0 if its active set is empty; the returned scalar
    is their sum (so the shared ``W_CHANGE_LOC`` scales both together).
    """
    where_active = active.bool() & prior_active.bool()
    where = change_localization_loss(
        pred_patches,
        prior_patches,
        prior_weights,
        where_active,
        eps=eps,
    )
    which = change_direction_loss(
        pred_patches,
        prior_patches,
        current_patches,
        prior_weights,
        curr_weights,
        prior_active,
        curr_active,
        prog_cls_idx,
        active,
        eps=eps,
    )
    return where + which
