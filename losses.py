import torch
import torch.nn.functional as F
import torch.nn as nn


# =========================================================
# GLOBAL CONTRASTIVE LOSS (InfoNCE)
# =========================================================
def global_contrastive_loss(
    img_emb: torch.Tensor,
    txt_emb: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Standard symmetric InfoNCE loss.

    img_emb : (B, D)
    txt_emb : (B, D)
    """

    logits = img_emb @ txt_emb.T
    logits = logits / temperature

    labels = torch.arange(
        img_emb.size(0),
        device=img_emb.device,
    )

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return 0.5 * (loss_i2t + loss_t2i)


# =========================================================
# LOCAL CONTRASTIVE LOSS (BioViL-style)
# =========================================================
def local_contrastive_loss(
    img_patches: torch.Tensor,   # (B, N, D)
    txt_tokens: torch.Tensor,    # (B, T, D)
    token_mask: torch.Tensor,    # (B, T)  True for valid tokens
    temperature: float = 10.0,   # corresponds to temp3
    eps: float = 1e-8,
    temp1: float = 4.0,          # attention temperature
    temp2: float = 5.0,          # word aggregation temperature
):
    """
    Exact GLoRIA-style weighted local contrastive loss.

    Implements:
    - Soft attention over patches
    - Cosine similarity between token and weighted patch
    - Log-sum-exp aggregation over tokens
    - Symmetric InfoNCE over batch
    """

    B, N, D = img_patches.shape
    _, T, _ = txt_tokens.shape

    # --------------------------------------------------
    # 1️⃣ Normalize features (cosine similarity)
    # --------------------------------------------------
    img_patches = F.normalize(img_patches, dim=-1)
    txt_tokens = F.normalize(txt_tokens, dim=-1)

    # --------------------------------------------------
    # 2️⃣ Cross-batch token–patch similarity
    # Shape: (B_text, B_image, T, N)
    # --------------------------------------------------
    sim = torch.einsum("btd,knd->bktn", txt_tokens, img_patches)

    # --------------------------------------------------
    # 3️⃣ Soft attention over patches (GLoRIA weighting)
    # First softmax over tokens (as in original code)
    # Then temperature scaling + second softmax
    # --------------------------------------------------
    attn = F.softmax(sim, dim=-1)          # over patches
    attn = attn * temp1
    attn = F.softmax(attn, dim=-1)         # second softmax

    # --------------------------------------------------
    # 4️⃣ Weighted patch representation per token
    # weighted_context[b,k,t,d] =
    #    sum_n attn[b,k,t,n] * img_patch[k,n,d]
    # --------------------------------------------------
    weighted_context = torch.einsum(
        "bktn,knd->bktd", attn, img_patches
    )

    # --------------------------------------------------
    # 5️⃣ Cosine similarity between tokens and weighted context
    # (since normalized, dot product = cosine)
    # Shape: (B_text, B_image, T)
    # --------------------------------------------------
    token_sim = (txt_tokens.unsqueeze(1) * weighted_context).sum(dim=-1)

    # --------------------------------------------------
    # 6️⃣ Mask padding tokens
    # --------------------------------------------------
    token_mask = token_mask.unsqueeze(1)  # (B_text,1,T)
    token_sim = token_sim.masked_fill(~token_mask, 0.0)

    # --------------------------------------------------
    # 7️⃣ Log-sum-exp aggregation over tokens (GLoRIA)
    # Implements:
    # log( sum_t exp(temp2 * cosine) )
    # --------------------------------------------------
    token_sim = torch.exp(token_sim * temp2)
    token_sim = token_sim.sum(dim=-1) + eps
    sim_matrix = torch.log(token_sim)

    # --------------------------------------------------
    # 8️⃣ Final temperature scaling (temp3)
    # --------------------------------------------------
    sim_matrix = sim_matrix * temperature  # (B,B)

    # --------------------------------------------------
    # 9️⃣ Symmetric InfoNCE
    # --------------------------------------------------
    labels = torch.arange(B, device=sim_matrix.device)

    loss_i2t = F.cross_entropy(sim_matrix, labels)
    loss_t2i = F.cross_entropy(sim_matrix.transpose(0, 1), labels)

    return (loss_i2t + loss_t2i) / 2

# =========================================================
# PROGRESSION CLASSIFICATION LOSS (4th JEPA loss)
# =========================================================
def progression_classification_loss(
    pred_patches: torch.Tensor,         # (B, N, D)
    class_prompts_local: torch.Tensor,  # (P, T, D)
    class_prompts_mask: torch.Tensor,   # (P, T)  True for valid tokens
    pair_idx_per_finding: torch.Tensor, # (F_total,)  int64
    silver_per_finding: torch.Tensor,   # (F_total,)  int64, in [0, n_classes)
    batch_size: int,
    n_classes: int = 5,
    temperature: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-finding 5-way softmax cross-entropy on prompt alignment.

    For each finding ``j`` in pair ``i`` (silver class ``c_{i,j}^*``):

      1. Build five templated prompts (one per class) — *not* done here;
         the caller passes their encoded tokens via ``class_prompts_local``
         in a flat, finding-major order::

             [pair_0_finding_0_cls_0, …, pair_0_finding_0_cls_{n-1},
              pair_0_finding_1_cls_0, …, pair_i_finding_j_cls_k, …]

         so ``class_prompts_local`` has shape ``(F_total * n_classes, T, D)``.
      2. Mean-pool ``pred_patches[i]`` over patches and the prompt tokens
         over their valid tokens. Take cosine similarity. This is a single
         scalar per (pair, finding, class) triple.
      3. Reshape to ``(F_total, n_classes)`` and softmax-CE against
         ``silver_per_finding``.
      4. Average CE across findings within each pair (so multi-finding
         pairs don't outweigh single-finding pairs), then average across
         pairs in the batch.

    Returns a scalar tensor on the same device as ``pred_patches``.
    Returns 0 if the batch carries no findings (e.g. degenerate batch).

    Notes
    -----
    Scoring is mean-pooled cosine rather than the GLoRIA soft-aligned
    score used by ``local_contrastive_loss``. Cosine in ``[-1, 1]`` is
    well-behaved for a 5-way softmax: temperature ``τ=0.1`` gives a
    logit range of ``[-10, 10]`` which is peaky but not pathological,
    and avoids the wide-dynamic-range issues you'd get from the
    log-sum-exp scoring (which is calibrated for batch-contrastive
    InfoNCE, not per-pair classification).
    """
    if class_prompts_local.numel() == 0:
        return pred_patches.new_zeros(())

    F_total = pair_idx_per_finding.size(0)
    if F_total == 0:
        return pred_patches.new_zeros(())

    P = class_prompts_local.size(0)
    assert P == F_total * n_classes, (
        f"Expected {F_total * n_classes} prompts (={F_total} findings x "
        f"{n_classes} classes), got {P}"
    )

    # 1. Mean-pool the predicted current patches per pair -> (B, D).
    img_pool = pred_patches.mean(dim=1)

    # 2. Mask-aware mean-pool of text tokens per prompt -> (P, D).
    mask = class_prompts_mask.float()
    mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    text_pool = (
        class_prompts_local * mask.unsqueeze(-1)
    ).sum(dim=1) / mask_sum

    # 3. Cosine similarity per prompt against its pair's pooled patch
    #    embedding. Normalize both, then dot.
    img_norm = F.normalize(img_pool, dim=-1, eps=eps)    # (B, D)
    text_norm = F.normalize(text_pool, dim=-1, eps=eps)  # (P, D)

    pair_idx_per_prompt = pair_idx_per_finding.repeat_interleave(n_classes)
    img_per_prompt = img_norm[pair_idx_per_prompt]       # (P, D)
    cos_per_prompt = (img_per_prompt * text_norm).sum(dim=-1)  # (P,)

    # 4. Reshape to (F_total, n_classes) and per-finding softmax-CE.
    logits = cos_per_prompt.view(F_total, n_classes) / temperature
    ce_per_finding = F.cross_entropy(
        logits, silver_per_finding, reduction="none"
    )  # (F_total,)

    # 5. Aggregate: mean across findings within a pair, then mean across
    #    pairs. The per-pair mean is the key step that prevents a pair
    #    with N findings from contributing N times the gradient of a pair
    #    with 1 finding.
    per_pair_sum = torch.zeros(
        batch_size, device=pred_patches.device, dtype=ce_per_finding.dtype
    )
    per_pair_count = torch.zeros(
        batch_size, device=pred_patches.device, dtype=ce_per_finding.dtype
    )
    per_pair_sum.scatter_add_(0, pair_idx_per_finding, ce_per_finding)
    per_pair_count.scatter_add_(
        0, pair_idx_per_finding, torch.ones_like(ce_per_finding)
    )

    valid = per_pair_count > 0
    if not valid.any():
        return pred_patches.new_zeros(())
    per_pair_mean = per_pair_sum[valid] / per_pair_count[valid]
    return per_pair_mean.mean()


# =========================================================
# MLM LOSS (CROSS ENTROPY)
# =========================================================
def mlm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Standard MLM loss with ignore_index = -100

    logits : (B, T, vocab)
    labels : (B, T)
    """

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    return loss_fn(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )

