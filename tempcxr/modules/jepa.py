"""TempCXR-JEPA model: forward orchestration only.

The ``forward`` returns a dict of representations and lets the training
script compute losses externally (using ``losses.local_contrastive_loss``,
``losses.progression_classification_loss``, and
``losses_jepa.jepa_smooth_l1_loss``).

Architecture (matches the slide-deck diagram):

    Prior CXR  ─►  E (online, trained)         ──►  LN(z_prior) ──┐
                                                                   ├──►  Predictor  ──►  ẑ_cur
    Condition  ─►  text_encoder                ──►  τ_cond     ──┘
    Current CXR─►  E_target (EMA, stop-grad)   ──►  LN(z_cur)

    [progression loss only]
    "{finding} is {cls}." prompts ──► text_encoder ──► τ_class
    ẑ_cur  ↔  τ_class            ── per-finding 5-way softmax-CE

The "condition" text is built upstream by the dataset and can be either
the joined dynamic sentences (``condition_mode="dynamic"``, the default)
or the per-finding templated string ``"{finding} is {progression}"``
(``condition_mode="templated"``). The model treats it as an opaque
string either way.

Both the prior and the target are LayerNorm-normalized over the feature
dim, so the predictor's delta-prediction starts in the same geometry
as the JEPA target (no scale gap in the Smooth L1 loss).

Losses (computed by the caller):
    - JEPA Smooth L1                : ẑ_cur ↔ stop-grad LN(z_cur)
    - GLoRIA local contrastive      : LN(z_prior) ↔ τ_prior
    - GLoRIA local contrastive      : ẑ_cur ↔ τ_current
    - Progression classification    : per-finding 5-way softmax-CE on
                                      cos(ẑ_cur, τ_{finding-is-class})
                                      using silver progression labels
                                      (see ``losses.progression_classification_loss``).
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

from losses import local_contrastive_loss, progression_classification_loss
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
        class_prompts=None,
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
        class_prompts    : Optional[list[str]] — flat list of templated
                           ``"{Finding} is {class}."`` prompts for the
                           progression-classification loss, in finding-
                           major order (see
                           ``losses.progression_classification_loss``).
                           When provided, the text encoder runs once on
                           the prompts inside this forward so DDP can
                           see the gradient path. Pass ``None`` (or an
                           empty list) to skip — the prog-loss caller
                           is expected to short-circuit in that case.

        Returns a dict containing:
          - prior_patches            (B, N, D)  online encoder, with grad
          - current_patches_target   (B, N, D)  target encoder, LN-normed,
                                                detached (stop-gradient)
          - pred_current_patches     (B, N, D)  predictor output ẑ_cur
          - prior_txt_local          (B, T, D)
          - prior_token_mask         (B, T)
          - current_txt_local        (B, T, D)
          - current_token_mask       (B, T)
          - condition_txt_local      (B, T, D)
          - condition_token_mask     (B, T)
          - class_prompts_local      (P, T_p, D) or None
          - class_prompts_mask       (P, T_p)     or None
        """

        # ---- Online encoder on prior (gradients flow) ----
        _, prior_patches = self.image_encoder(prior_imgs)

        # Match the LayerNorm scale of the target so the predictor's
        # delta-prediction starts in the same geometry as the JEPA target.
        # Without this, prior_patches has BioViL-T's raw scale (~1) while
        # the LN'd target has scale ~sqrt(D), and the loss is dominated by
        # the scale gap rather than the directional residual.
        prior_patches = F.layer_norm(
            prior_patches,
            (prior_patches.size(-1),),
        )

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

        # ---- Optional class-prompt encoding (progression loss) ----
        # Kept as a separate text-encoder call rather than concatenated
        # with the reports above so the report batch padding stays
        # short. Class prompts are ~5 tokens each; reports are 100s.
        # Concatenating would pad every report to the report length and
        # blow up memory.
        if class_prompts is not None and len(class_prompts) > 0:
            _, class_prompts_local, class_prompts_mask = (
                self.text_encoder.forward_contrastive(list(class_prompts))
            )
        else:
            class_prompts_local = None
            class_prompts_mask = None

        # ---- Target encoder on current image: stop-gradient + LN target ----
        # I-JEPA's recipe: feature-dim LayerNorm on target stabilizes the
        # target distribution as the EMA encoder drifts.
        with torch.no_grad():
            _, current_patches_target = self.target_image_encoder(current_imgs)
            current_patches_target = F.layer_norm(
                current_patches_target,
                (current_patches_target.size(-1),),
            )
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
            "class_prompts_local": class_prompts_local,
            "class_prompts_mask": class_prompts_mask,
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

    # Per-pair (finding, silver-class) lists for the progression loss.
    pair_findings = [
        ["pleural effusion"],
        ["pneumonia", "atelectasis"],
    ]
    pair_silver_cls = [
        [2],         # pleural effusion → worsening (CLS_ORDER idx 2)
        [0, 1],      # pneumonia → improving (0); atelectasis → stable (1)
    ]

    # Flatten into the finding-major prompt order the loss expects.
    n_classes = 5
    cls_names = ["improving", "stable", "worsening", "new", "resolved"]
    class_prompts = []
    pair_idx_per_finding = []
    silver_per_finding = []
    for pair_i, (fs, scs) in enumerate(zip(pair_findings, pair_silver_cls)):
        for finding, silver in zip(fs, scs):
            for cls in cls_names:
                class_prompts.append(
                    f"{finding[:1].upper()}{finding[1:]} is {cls}."
                )
            pair_idx_per_finding.append(pair_i)
            silver_per_finding.append(silver)

    out = model(
        prior_imgs,
        current_imgs,
        prior_reports,
        current_reports,
        condition_texts,
        class_prompts=class_prompts,
    )

    # --------------------------------------------------
    # Losses (external)
    # --------------------------------------------------
    jepa_loss = jepa_smooth_l1_loss(
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

    prog_loss = progression_classification_loss(
        pred_patches=out["pred_current_patches"],
        class_prompts_local=out["class_prompts_local"],
        class_prompts_mask=out["class_prompts_mask"],
        pair_idx_per_finding=torch.tensor(
            pair_idx_per_finding, device=device, dtype=torch.long
        ),
        silver_per_finding=torch.tensor(
            silver_per_finding, device=device, dtype=torch.long
        ),
        batch_size=B,
        n_classes=n_classes,
    )

    total = jepa_loss + 0.1 * prior_loss + 0.1 * pred_loss + 0.1 * prog_loss

    # --------------------------------------------------
    # Print
    # --------------------------------------------------
    print("prior_patches:", tuple(out["prior_patches"].shape))
    print("pred_current_patches:", tuple(out["pred_current_patches"].shape))
    print("current_patches_target:", tuple(out["current_patches_target"].shape))
    print("class_prompts_local:", tuple(out["class_prompts_local"].shape))
    print()
    print("JEPA Smooth L1:", jepa_loss.item())
    print("Report (z_prior):", prior_loss.item())
    print("Report (ẑ_cur):", pred_loss.item())
    print("Progression CE:", prog_loss.item())
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
