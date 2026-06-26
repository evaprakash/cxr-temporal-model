"""TempCXR-JEPA model: forward orchestration only.

The ``forward`` returns a dict of representations and lets the training
script compute losses externally (using ``losses.local_contrastive_loss``
and ``losses_jepa.jepa_cosine_loss``).

Architecture (unit-sphere variant):

    Prior CXR  ─►  E (online, trained, L2-norm)  ──►  z_prior  ──┐
                                                                  ├──►  Predictor  ──►  ẑ_cur (L2-norm)
    Condition  ─►  text_encoder (L2-norm)        ──►  τ_cond  ──┘
    Current CXR─►  E_target (EMA, stop-grad, L2) ──►  z_cur

Every encoder output lives on the unit sphere (L2-norm along the feature
dim is applied inside the image and text encoders). The predictor takes
unit-norm prior patches + unit-norm text tokens, computes a small Δz at
the residual stream's natural scale, and renormalizes the sum to the
sphere::

    ẑ_cur = normalize(prior_patches + Δz)

That keeps "do nothing" (Δz = 0) as the free default — a near-zero delta
gives ẑ_cur ≈ prior on the sphere — while letting every loss read the
same unit-norm features without further scaling.

The "condition" text is built upstream by the dataset and can be either
the joined dynamic sentences (``condition_mode="dynamic"``, the default)
or the per-finding templated string ``"{finding} is {progression}"``
(``condition_mode="templated"``). The model treats it as an opaque
string either way.

Losses (computed by the caller):
    - JEPA cosine                   : 1 − cos(ẑ_cur, stop-grad z_cur)
    - GLoRIA local contrastive      : z_prior ↔ τ_prior
    - GLoRIA local contrastive      : ẑ_cur ↔ τ_current
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
from losses_jepa import jepa_cosine_loss


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
    """Small transformer that predicts the *delta* between prior and current
    on the unit sphere.

    Inputs to the transformer (concatenated, in this order):
      - learnable query tokens, one per output patch position (carrying the
        positional + type embedding for "predict ẑ_cur here")
      - prior patch tokens (unit-norm from the image encoder) with their
        own positional + type embedding
      - text condition tokens (unit-norm from the text encoder) with a
        type embedding (no positional)

    Output: the predicted current patches reconstructed as
    ``ẑ_cur = normalize(prior_patches + Δz)``, where ``Δz`` is the first
    ``num_patches`` positions of the transformer stack (no final
    LayerNorm — the transformer's internal per-block LayerNorms already
    stabilize the residual stream, so ``delta_z`` comes out at the
    transformer's natural small scale and stays comparable to the
    unit-norm prior).

    This is the "Predict Δz on the sphere" variant. The "do nothing"
    baseline is essentially free: Δz ≈ 0 gives ẑ_cur ≈ z_prior after the
    final L2-norm, so the predictor only spends capacity on the *change*
    induced by the text condition. Because Δz is computed against the
    same prior, this also matches the slide-deck inference rule
        S_k = cos(ẑ_cur^k - z_prior, z_cur - z_prior)
    where the predictor's residual (pre-renormalization) is exactly the
    quantity being scored.
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

        # Delta-prediction on the unit sphere: the first N positions are
        # Δz at the transformer's natural residual scale. Reconstruct
        # ẑ_cur by adding back the unit-norm prior_patches and
        # L2-renormalizing to put the result back on the sphere.
        delta_z = x[:, :N, :]
        pred = prior_patches + delta_z
        return F.normalize(pred, dim=-1)


# =========================================================
# TEMPCXR-JEPA MODEL (FORWARD ORCHESTRATION ONLY)
# =========================================================
class TempCXRJEPA(nn.Module):
    """Forward-pass orchestration for the JEPA-style temporal CXR setup.

    Holds:
      - ``image_encoder``        : online BioViL-T (raw, no L2) — trained.
      - ``target_image_encoder`` : EMA copy of ``image_encoder`` — frozen.
      - ``text_encoder``         : BioViL-T text encoder — trained.
      - ``predictor``            : IJEPATemporalPredictor — trained.

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
          - prior_patches            (B, N, D)  online encoder, unit-norm,
                                                with grad
          - current_patches_target   (B, N, D)  EMA target encoder,
                                                unit-norm, detached
                                                (stop-gradient)
          - pred_current_patches     (B, N, D)  predictor output ẑ_cur,
                                                unit-norm
          - prior_txt_local          (B, T, D)  unit-norm
          - prior_token_mask         (B, T)
          - current_txt_local        (B, T, D)  unit-norm
          - current_token_mask       (B, T)
          - condition_txt_local      (B, T, D)  unit-norm
          - condition_token_mask     (B, T)
        """

        # ---- Online encoder on prior (gradients flow) ----
        # The image encoder already L2-normalizes its outputs, so
        # ``prior_patches`` arrives on the unit sphere.
        _, prior_patches = self.image_encoder(prior_imgs)

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

        # ---- Target encoder on current image: stop-gradient ----
        # The target encoder is a frozen EMA copy of the online encoder
        # and inherits the same L2-norm at its output, so the target
        # also lives on the unit sphere — no extra normalization needed
        # here. Detach to harden the stop-gradient.
        with torch.no_grad():
            _, current_patches_target = self.target_image_encoder(current_imgs)
        current_patches_target = current_patches_target.detach()

        # ---- Predictor: ẑ_cur from prior + condition text ----
        pred_current_patches = self.predictor(
            prior_patches,
            condition_txt_local,
            condition_token_mask,
        )

        return {
            "prior_patches": prior_patches,
            "current_patches_target": current_patches_target,
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
    # Losses (external)
    # --------------------------------------------------
    jepa_loss = jepa_cosine_loss(
        out["pred_current_patches"],
        out["current_patches_target"],
    )

    prior_loss = local_contrastive_loss(
        out["prior_patches"],
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
    print("prior_patches:", tuple(out["prior_patches"].shape))
    print("pred_current_patches:", tuple(out["pred_current_patches"].shape))
    print("current_patches_target:", tuple(out["current_patches_target"].shape))
    print()
    print("JEPA cosine:", jepa_loss.item())
    print("Report (z_prior):", prior_loss.item())
    print("Report (ẑ_cur):", pred_loss.item())
    print("Total:", total.item())

    total.backward()
    print("\nBackward pass successful.")

    # --------------------------------------------------
    # EMA update (demo)
    # --------------------------------------------------
    sched = make_momentum_scheduler(EMA_START, EMA_END, total_iters=1)
    m = next(sched)
    model.update_ema(momentum=m)
    print(f"\nEMA update applied (momentum={m:.4f}).")
