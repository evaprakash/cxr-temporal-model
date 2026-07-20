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
    Optional per-sample ``patch_weights`` (B, N) replace the uniform
    mean with a masked/soft mean when a filtered segmentation mask is
    available for the progression finding; missing masks use all-ones
    weights (identical to the legacy global mean).

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
    patch_weights: Optional[torch.Tensor] = None,  # (B, N) or None
) -> torch.Tensor:
    """5-way image-image cross-entropy on the predictor's candidate latents.

    For each pair ``b`` and progression class ``c``:

        logit[b, c] = weighted mean over patches of cos(ẑ_cur^c[b], z_cur[b])

    where ``ẑ_cur^c[b]`` is the predictor's output when conditioned on the
    class-c prompt ``"{prog_finding[b]} is {class[c]}."`` (computed
    upstream by ``TempCXRJEPA.forward`` running the predictor C times per
    pair). The standard cross-entropy is then applied to
    ``logits / temperature`` against the silver progression label.

    With ``patch_weights is None`` (or all-ones weights) this is the
    legacy uniform mean-over-patches. When a filtered finding mask is
    available, ``patch_weights[b]`` is the soft 14×14 coverage of that
    region and **replaces** the global mean for that sample (no add-on
    term). Rows whose weights sum to ~0 fall back to uniform.

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
    patch_weights
        Optional ``(B, N)`` non-negative weights. Normalized per row to
        sum to 1 before the weighted mean. ``None`` → uniform mean.

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

    if patch_weights is None:
        logits = cos_per_patch.mean(dim=-1)  # (B, C)
    else:
        w = patch_weights.to(
            device=cos_per_patch.device, dtype=cos_per_patch.dtype
        ).clamp(min=0.0)
        if w.ndim != 2 or w.shape[0] != cos_per_patch.shape[0] or w.shape[1] != cos_per_patch.shape[2]:
            raise ValueError(
                f"patch_weights shape {tuple(w.shape)} incompatible with "
                f"cos_per_patch {tuple(cos_per_patch.shape)} "
                f"(expected (B, N)=({cos_per_patch.shape[0]}, "
                f"{cos_per_patch.shape[2]}))"
            )
        w_sum = w.sum(dim=-1, keepdim=True)  # (B, 1)
        uniform = torch.full_like(w, 1.0 / w.shape[-1])
        w_norm = torch.where(w_sum > eps, w / w_sum.clamp(min=eps), uniform)
        logits = (cos_per_patch * w_norm.unsqueeze(1)).sum(dim=-1)  # (B, C)

    logits = logits / temperature

    return F.cross_entropy(logits, silver_labels, weight=class_weights)
