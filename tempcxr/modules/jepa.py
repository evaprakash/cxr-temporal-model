"""TempCXR-JEPA model: forward orchestration only.

The ``forward`` returns a dict of representations and lets the training
script compute losses externally (using ``losses.local_contrastive_loss``
and ``losses_jepa.jepa_smooth_l1_loss``).

Architecture
------------

    Prior CXR  ──►  E (online)         ─┬──►  proj_clip ──►  prior_clip ──────────►  CLIP loss (vs prior_text)
                                        │
                                        └──►  proj_jepa  ──►  LN  ──►  prior_jepa  ──┐
    Condition  ──►  text_encoder       ─────────────────────────►  τ_cond           ├──►  Predictor  ──►  ẑ_cur ──┬──►  JEPA loss
                                                                                                                  │
                                                                                                                  └──►  CLIP loss (vs current_text)
    Current CXR──►  E_target (EMA, SG)  ──►  proj_jepa (shared, SG)  ──►  LN  ──►  current_target_jepa  ─────────►  JEPA loss (target)

The "condition" text is built upstream by the dataset and can be either
the joined dynamic sentences (``condition_mode="dynamic"``) or the
per-finding templated string ``"{finding} is {progression}"``
(``condition_mode="templated"``). The model treats it as an opaque
string either way.

Why two projection heads
------------------------

The two losses want different feature geometries: the contrastive loss
lives on the unit sphere (cosine), the JEPA Smooth L1 lives in a bounded
Euclidean space. Without separate heads, the *same* patch tensor would
have to satisfy both geometries simultaneously and the gradients of the
two losses would directly contend over a single shared representation.

``proj_clip`` and ``proj_jepa`` give each loss its own per-loss
projection of the (shared) trunk features. The trunk still trains from
both losses, but each loss only ever sees its own projected view, so the
two loss types don't interfere with each other in the loss-side
representation.

The JEPA target reuses the SAME ``proj_jepa`` (under stop-gradient) as
the prior side, rather than going through a separate EMA-tracked copy.
This is the BYOL/SimSiam-style "shared projector + stop-grad on the
target" recipe: it removes the projection-head EMA drift that was
breaking the predictor's "do-nothing" prior, while the EMA target image
encoder still provides the collapse-prevention asymmetry between the
two sides.  Concretely: for a stable pair where the prior and current
images encode to similar features, applying the same ``proj_jepa`` to
both sides yields ``proj_jepa(z_prior) ≈ proj_jepa(z_cur)``, so the
predictor's identity output ``Δz = 0`` is approximately the right
answer (its magnitude calibration on stable cases comes back for free).

The predictor's output ``ẑ_cur`` still feeds the current-side CLIP loss
directly (no extra projection): the local contrastive loss L2-normalizes
both image and text features internally, so any scale from the
predictor's output is absorbed by that normalization. This keeps the
predictor under explicit text-conditional supervision — the only way the
contrastive loss can match different ``ẑ_cur`` against different
current-side reports is if ``ẑ_cur`` actually varies with the predictor's
text condition.

Both sides of the JEPA path apply a feature-dim LayerNorm (no learnable
parameters) on top of ``proj_jepa``'s output, so the predictor's
delta-prediction starts in the same geometry as the JEPA target (no
scale gap in the Smooth L1 loss).

Losses (computed by the caller):
    - JEPA Smooth L1: ẑ_cur  ↔ stop-grad LN(proj_jepa(z_cur))
    - GLoRIA local contrastive: prior_clip ↔ τ_prior
    - GLoRIA local contrastive: ẑ_cur     ↔ τ_current
"""

import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# Make project root visible so ../../losses.py and losses_jepa.py are
# importable when `tempcxr.modules.jepa` is imported as a package.
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from .image_encoder_jepa import BioViLTImageEncoderJEPA
from .text_encoder import BioViLTTextEncoder

from losses import local_contrastive_loss
from losses_jepa import jepa_smooth_l1_loss


# =========================================================
# EMA HELPERS
# =========================================================
EMA_START = 0.996
EMA_END = 1.0


@torch.no_grad()
def _build_target_encoder(online_encoder: nn.Module) -> nn.Module:
    """Frozen deepcopy of the online encoder (initial weights = identity)."""
    target = copy.deepcopy(online_encoder)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()
    return target


def _make_projection_head(d_in: int, d_out: int, d_hidden: int = None) -> nn.Sequential:
    """Two-layer MLP projection head (Linear → GELU → Linear).

    Used to give each downstream loss its own learnable view of the
    encoder features so the JEPA Smooth L1 path and the CLIP local
    contrastive path don't have to share a single geometry. Matches the
    standard SSL projector design (BYOL / SimSiam / SimCLR), just with a
    smaller hidden dim that's appropriate for our d_model=128 setup.
    """
    if d_hidden is None:
        d_hidden = d_in
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )


@torch.no_grad()
def _update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    """In-place: target ← m*target + (1-m)*online; copy buffers verbatim."""
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(momentum).add_(p_o.data, alpha=1.0 - momentum)
    for b_t, b_o in zip(target.buffers(), online.buffers()):
        b_t.data.copy_(b_o.data)


def make_momentum_scheduler(
    m_start: float = EMA_START,
    m_end: float = EMA_END,
    total_iters: int = 1,
):
    """Linear ramp m_start → m_end over ``total_iters`` steps (I-JEPA style).

    Usage:
        sched = make_momentum_scheduler(0.996, 1.0, total_iters=ipe*epochs)
        ...
        optimizer.step()
        model.update_ema(momentum=next(sched))
    """
    total_iters = max(int(total_iters), 1)
    return (
        m_start + i * (m_end - m_start) / total_iters
        for i in range(total_iters + 1)
    )


# =========================================================
# I-JEPA TEMPORAL PREDICTOR
# =========================================================
class IJEPATemporalPredictor(nn.Module):
    """Small transformer that predicts the *delta* between prior and current.

    Inputs to the transformer (concatenated, in this order):
      - learnable query tokens, one per output patch position (carrying the
        positional + type embedding for "predict ẑ_cur here")
      - prior patch tokens with their own positional + type embedding
      - text condition tokens with a type embedding (no positional)

    Output: the predicted current patches reconstructed as
    ``ẑ_cur = prior_patches + Δz``, where ``Δz`` is the first
    ``num_patches`` positions of the transformer stack (raw, post the
    trailing LayerNorm; no L2 normalization).

    This is the "Predict Δz" variant from the slide deck. The "do
    nothing" baseline becomes free (Δz = 0 reconstructs ẑ_cur = z_prior),
    so the predictor only spends capacity on the *change* induced by the
    text condition. Because Δz is computed against the same prior, this
    also matches the slide-deck inference rule
        S_k = cos(ẑ_cur^k - z_prior, z_cur - z_prior)
    where the predictor's output is exactly the quantity being scored.
    """

    def __init__(
        self,
        num_patches: int = 196,
        d_model: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.d_model = d_model

        self.current_queries = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        self.prior_pos = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        self.query_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.prior_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.text_type = nn.Parameter(torch.zeros(1, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=int(mlp_ratio * d_model),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, prior_patches, text_tokens, text_mask):
        B, N, D = prior_patches.shape

        assert N == self.num_patches
        assert D == self.d_model

        prior_tokens = prior_patches + self.prior_pos + self.prior_type
        text_tokens = text_tokens + self.text_type

        query_tokens = self.current_queries.expand(B, -1, -1)
        query_tokens = query_tokens + self.query_pos + self.query_type

        x = torch.cat([query_tokens, prior_tokens, text_tokens], dim=1)

        query_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        prior_mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        text_padding_mask = ~text_mask.bool()

        key_padding_mask = torch.cat(
            [query_mask, prior_mask, text_padding_mask],
            dim=1,
        )

        x = self.blocks(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # Delta-prediction: the first N positions are Δz, and ẑ_cur is
        # reconstructed by adding back the original (un-positionally-
        # embedded) prior patches.
        delta_z = x[:, :N, :]
        return prior_patches + delta_z


# =========================================================
# TEMPCXR-JEPA MODEL (FORWARD ORCHESTRATION ONLY)
# =========================================================
class TempCXRJEPA(nn.Module):
    """Forward-pass orchestration for the JEPA-style temporal CXR setup.

    Holds:
      - ``image_encoder``        : online BioViL-T (raw, no L2) — trained.
      - ``target_image_encoder`` : EMA copy of ``image_encoder`` — frozen.
      - ``text_encoder``         : BioViL-T text encoder — trained.
      - ``proj_clip``            : projection head feeding the prior-side
                                   CLIP local contrastive loss — trained.
      - ``proj_jepa``            : projection head for the JEPA Smooth L1
                                   path. Applied to BOTH the prior side
                                   (with gradient) and the target side
                                   (under stop-gradient) — trained.
      - ``predictor``            : IJEPATemporalPredictor — trained.

    Separate ``proj_clip`` and ``proj_jepa`` heads decouple the two loss
    geometries: the prior-side contrastive loss reads from ``proj_clip``'s
    output (which the loss L2-normalizes onto the unit sphere), while the
    JEPA Smooth L1 loss reads from ``proj_jepa``'s output (which is
    LayerNorm-normalized so prior and target sit on the same Euclidean
    scale). This means the two losses don't have to share a single
    representation and their gradients can't directly contend over the
    same patch tensor.

    The JEPA target uses the SAME ``proj_jepa`` (under
    ``torch.no_grad``) as the prior side rather than a separate
    EMA-tracked ``target_proj_jepa``. The EMA target image encoder still
    provides the asymmetry needed to prevent representation collapse,
    but applying the same projection head on both sides preserves the
    predictor's "do-nothing" prior — ``Δz = 0`` lands close to the
    target for stable pairs, so the magnitude calibration that the
    Smooth L1 metric depends on survives.

    The current-side CLIP loss reads directly from the predictor's
    output ``ẑ_cur`` (no extra projection): the contrastive loss
    L2-normalizes inputs internally, and routing the current-side CLIP
    loss through ``ẑ_cur`` is what gives the predictor a direct
    text-conditional supervision signal.

    Returns a dict of representations; losses live in ``losses.py`` and
    ``losses_jepa.py``.
    """

    def __init__(
        self,
        mode: str = "biovilt",
        checkpoint_path: str = None,
        num_patches: int = 196,
        d_model: int = 128,
        predictor_depth: int = 6,
        predictor_heads: int = 4,
    ):
        super().__init__()

        self.image_encoder = BioViLTImageEncoderJEPA(
            mode=mode,
            checkpoint_path=checkpoint_path,
        )
        self.target_image_encoder = _build_target_encoder(self.image_encoder)

        self.text_encoder = BioViLTTextEncoder(
            mode=mode,
            checkpoint_path=checkpoint_path,
        )

        # Per-loss projection heads. The encoder, the predictor's input,
        # and the text encoder all run at d_model=128, so the projections
        # are square (128 → 128). proj_clip's output dim must equal the
        # text-token dim because the contrastive loss compares the two
        # via cosine similarity, and proj_jepa's output dim must equal
        # the predictor's d_model because the predictor consumes it
        # directly as token features.
        d_enc = self.image_encoder.embed_dim
        self.proj_clip = _make_projection_head(d_enc, d_model)
        self.proj_jepa = _make_projection_head(d_enc, d_model)

        self.predictor = IJEPATemporalPredictor(
            num_patches=num_patches,
            d_model=d_model,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )

    # --------------------------------------------------
    # FORWARD (NO LOSSES)
    # --------------------------------------------------
    def forward(
        self,
        prior_imgs: torch.Tensor,
        current_imgs: torch.Tensor,
        prior_reports,
        current_reports,
        condition_texts,
    ):
        """
        prior_imgs       : (B, 3, H, W)
        current_imgs     : (B, 3, H, W)
        prior_reports    : list[str]
        current_reports  : list[str]
        condition_texts  : list[str] — the predictor's text condition.
                           Source content depends on the dataset's
                           ``condition_mode``: either the joined dynamic
                           sentences (``"dynamic"``) or the templated
                           per-finding ``"{finding} is {progression}"``
                           string (``"templated"``). The model treats
                           it as an opaque string either way.

        Returns a dict containing:
          - prior_clip               (B, N, D)  proj_clip(online_prior),
                                                prior-side CLIP-loss input
          - prior_jepa               (B, N, D)  LN(proj_jepa(online_prior)),
                                                predictor input
          - current_target_jepa      (B, N, D)  LN(proj_jepa(target_current)),
                                                under torch.no_grad +
                                                detach — JEPA-loss target
          - pred_current_patches     (B, N, D)  predictor output ẑ_cur,
                                                JEPA-loss prediction +
                                                current-side CLIP-loss input
          - prior_txt_local          (B, T, D)
          - prior_token_mask         (B, T)
          - current_txt_local        (B, T, D)
          - current_token_mask       (B, T)
          - condition_txt_local      (B, T, D)
          - condition_token_mask     (B, T)
        """

        # ---- Online encoder on prior (gradients flow) ----
        # We only need the online encoder on the prior side: the prior
        # CLIP loss reads from proj_clip(prior_raw), the predictor reads
        # from LN(proj_jepa(prior_raw)), and the current-side CLIP loss
        # reads from the predictor output directly (so no current-side
        # online encoder call is needed). The current-image branch is
        # encoded by the EMA target encoder below for the JEPA target.
        _, prior_raw = self.image_encoder(prior_imgs)

        # ---- CLIP projection on prior ----
        # No LayerNorm here; the local contrastive loss L2-normalizes its
        # inputs internally, so any output scale from proj_clip is fine.
        prior_clip = self.proj_clip(prior_raw)

        # ---- JEPA projection on prior, then LayerNorm ----
        # Match the LayerNorm scale of the target so the predictor's
        # delta-prediction starts in the same geometry as the JEPA target.
        # Without LN, proj_jepa's output has whatever scale it learned
        # while the LN'd target has scale ~sqrt(D), and the loss would be
        # dominated by the scale gap rather than the directional residual.
        prior_jepa = self.proj_jepa(prior_raw)
        prior_jepa = F.layer_norm(prior_jepa, (prior_jepa.size(-1),))

        # ---- Text encoder on all three reports (one batched call) ----
        # Concatenate the three text lists, run the encoder once, then
        # split. This costs the same memory as three calls but does only
        # one CXR-BERT forward pass.
        B = prior_imgs.size(0)
        all_reports = (
            list(prior_reports) + list(current_reports) + list(condition_texts)
        )
        _, all_txt_local, all_token_mask = (
            self.text_encoder.forward_contrastive(all_reports)
        )
        prior_txt_local, current_txt_local, condition_txt_local = (
            all_txt_local[:B],
            all_txt_local[B:2 * B],
            all_txt_local[2 * B:],
        )
        prior_token_mask, current_token_mask, condition_token_mask = (
            all_token_mask[:B],
            all_token_mask[B:2 * B],
            all_token_mask[2 * B:],
        )

        # ---- Target encoder + shared JEPA projection on current image ----
        # The EMA target image encoder is the only EMA-coupled module on
        # the target path: the JEPA projection head ``proj_jepa`` is
        # shared with the prior side and run here under ``torch.no_grad``
        # (BYOL/SimSiam-style stop-grad). Sharing the projection head
        # preserves the predictor's "do-nothing" prior — for stable pairs,
        # ``proj_jepa(prior) ≈ proj_jepa(current)`` so ``Δz = 0`` is
        # approximately correct. I-JEPA's feature-dim LayerNorm on the
        # target still stabilizes the target distribution as the EMA
        # encoder drifts.
        with torch.no_grad():
            _, current_raw_target = self.target_image_encoder(current_imgs)
            current_target_jepa = self.proj_jepa(current_raw_target)
            current_target_jepa = F.layer_norm(
                current_target_jepa,
                (current_target_jepa.size(-1),),
            )
        current_target_jepa = current_target_jepa.detach()

        # ---- Predictor: ẑ_cur from prior_jepa + condition text ----
        pred_current_patches = self.predictor(
            prior_jepa,
            condition_txt_local,
            condition_token_mask,
        )

        return {
            "prior_clip": prior_clip,
            "prior_jepa": prior_jepa,
            "current_target_jepa": current_target_jepa,
            "pred_current_patches": pred_current_patches,
            "prior_txt_local": prior_txt_local,
            "prior_token_mask": prior_token_mask,
            "current_txt_local": current_txt_local,
            "current_token_mask": current_token_mask,
            "condition_txt_local": condition_txt_local,
            "condition_token_mask": condition_token_mask,
        }

    # --------------------------------------------------
    # EMA UPDATE (call after optimizer.step())
    # --------------------------------------------------
    @torch.no_grad()
    def update_ema(self, momentum: float):
        """EMA the image encoder only.

        ``proj_jepa`` is shared between the prior side (with gradient)
        and the target side (under ``torch.no_grad``); there is no
        separate EMA copy of the projection head. This is the
        BYOL/SimSiam-style shared-projector recipe — the EMA target
        image encoder provides the asymmetry needed to prevent
        representation collapse, and the shared projection preserves
        the predictor's "do-nothing" prior on stable pairs.
        """
        _update_ema(self.image_encoder, self.target_image_encoder, momentum)


# =========================================================
# SELF-TEST
# =========================================================
if __name__ == "__main__":
    print("\nRunning TempCXRJEPA forward + external loss test\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TempCXRJEPA().to(device)
    model.train()

    # --------------------------------------------------
    # Dummy inputs
    # --------------------------------------------------
    B = 2
    prior_imgs = torch.randn(B, 3, 448, 448, device=device)
    current_imgs = torch.randn(B, 3, 448, 448, device=device)

    prior_reports = [
        "No acute cardiopulmonary process. Lungs clear.",
        "Stable cardiomegaly. No effusion.",
    ]
    current_reports = [
        "Increased right pleural effusion.",
        "Left lower lobe pneumonia is improving.",
    ]
    condition_texts = [
        "right pleural effusion is new and increased",
        "pneumonia is getting better",
    ]

    out = model(
        prior_imgs,
        current_imgs,
        prior_reports,
        current_reports,
        condition_texts,
    )

    # --------------------------------------------------
    # Losses (external). The prior-side CLIP loss reads from
    # proj_clip's output; the JEPA loss reads from proj_jepa (shared
    # between prior and target sides, target under stop-grad); the
    # current-side CLIP loss reads from the predictor output ẑ_cur so
    # the predictor stays under direct text-conditional supervision.
    # --------------------------------------------------
    jepa_loss = jepa_smooth_l1_loss(
        out["pred_current_patches"],
        out["current_target_jepa"],
    )

    prior_loss = local_contrastive_loss(
        out["prior_clip"],
        out["prior_txt_local"],
        out["prior_token_mask"],
    )

    pred_loss = local_contrastive_loss(
        out["pred_current_patches"],
        out["current_txt_local"],
        out["current_token_mask"],
    )

    total = jepa_loss + 0.1 * prior_loss + 0.1 * pred_loss

    # --------------------------------------------------
    # Print
    # --------------------------------------------------
    print("prior_clip:           ", tuple(out["prior_clip"].shape))
    print("prior_jepa:           ", tuple(out["prior_jepa"].shape))
    print("current_target_jepa:  ", tuple(out["current_target_jepa"].shape))
    print("pred_current_patches: ", tuple(out["pred_current_patches"].shape))
    print()
    print("JEPA Smooth L1:        ", jepa_loss.item())
    print("CLIP (prior, z_prior): ", prior_loss.item())
    print("CLIP (current, ẑ_cur): ", pred_loss.item())
    print("Total:                 ", total.item())

    total.backward()
    print("\nBackward pass successful.")

    # --------------------------------------------------
    # EMA update (demo)
    # --------------------------------------------------
    sched = make_momentum_scheduler(EMA_START, EMA_END, total_iters=1)
    m = next(sched)
    model.update_ema(momentum=m)
    print(f"\nEMA update applied (momentum={m:.4f}).")
