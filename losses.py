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

