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
  * ``disease_multilabel_loss`` for the finding-level add-on: multi-label
    BCE on image–text cosine logits between a patch bank (prior or
    predicted-current) and K finding-name embeddings. All findings
    present on a pair are positives at once (no random single-finding
    sampling). Optional per-finding Cui weights fight rare-disease
    collapse. No progression-template predictor pass — uses the same
    ``ẑ_cur`` as report local contrastive (dynamic-conditioned).

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
# DISEASE MULTI-LABEL LOSS (IMAGE–TEXT, NO PROG PREDICTOR)
# =========================================================
def disease_multilabel_loss(
    img_patches: torch.Tensor,          # (B, N, D)
    finding_txt_global: torch.Tensor,   # (K, D)
    multi_hot: torch.Tensor,            # (B, K) float {0,1}
    temperature: float = 0.1,
    eps: float = 1e-8,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-label BCE on image–text cosine logits over findings.

    For each pair ``b`` and finding ``f``::

        logit[b, f] = mean_n cos(img_patches[b, n], finding_txt[f])

    then ``binary_cross_entropy_with_logits(logit / τ, y[b, f])`` where
    ``y`` is the multi-hot of all findings listed on that silver pair.
    Samples with no in-vocab findings are skipped (no all-negative
    batches). Optional ``class_weights`` (K,) scale each finding's
    contribution (Cui et al. disease CBW).

    ``img_patches`` is typically ``prior_patches`` or dynamic-conditioned
    ``pred_current_patches`` — not the progression-class ``ẑ_cur^c``.
    """
    if img_patches.numel() == 0 or finding_txt_global.numel() == 0:
        return img_patches.new_zeros(())

    img = F.normalize(img_patches, dim=-1, eps=eps)            # (B, N, D)
    txt = F.normalize(finding_txt_global, dim=-1, eps=eps)     # (K, D)
    # (B, K, N) → mean over patches → (B, K)
    logits = torch.einsum("bnd,kd->bkn", img, txt).mean(dim=-1)
    logits = logits / temperature

    valid = multi_hot.sum(dim=-1) > 0
    if not bool(valid.any()):
        return img_patches.new_zeros(())

    logits = logits[valid]
    targets = multi_hot[valid]

    if class_weights is None:
        weight = None
    else:
        weight = class_weights.to(dtype=logits.dtype, device=logits.device)
        weight = weight.unsqueeze(0).expand_as(targets)

    return F.binary_cross_entropy_with_logits(
        logits, targets, weight=weight, reduction="mean"
    )
