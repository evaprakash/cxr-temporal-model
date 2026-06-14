"""Pre-projection-heads ``TempCXRJEPA`` model class (read-only — eval use).

This is the model architecture as of the commit just *before* ``618f374``
("Add separate proj_clip / proj_jepa heads so the two losses don't
interfere"). It is preserved here only so we can fairly evaluate
pre-``618f374`` checkpoints without checking out an old commit:
``progression_classify_legacy.py`` is the matching eval driver.

Differences from the current ``TempCXRJEPA`` in ``jepa.py``:

  * **No** ``proj_clip`` / ``proj_jepa`` / ``target_proj_jepa`` heads.
  * The predictor reads from ``LN(image_encoder(prior))`` directly.
  * The JEPA target is ``LN(target_image_encoder(current))`` directly.
  * ``update_ema`` only updates the image encoder.
  * ``forward`` returns ``prior_patches`` / ``current_patches_target``
    (the old key names) rather than ``prior_clip`` /
    ``current_target_jepa``.

Do **not** use this class for training — it's only here so we can load
older checkpoints into the matching architecture for evaluation.

The predictor itself (``IJEPATemporalPredictor``) hasn't changed between
the two eras, so we reuse it from ``tempcxr.modules.jepa``. Same for
the EMA helpers.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make project root visible so ../../losses.py etc. are importable when
# this module is imported as a package.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from .image_encoder_jepa import BioViLTImageEncoderJEPA
from .text_encoder import BioViLTTextEncoder
from .jepa import (
    IJEPATemporalPredictor,
    _build_target_encoder,
    _update_ema,
)


class TempCXRJEPALegacy(nn.Module):
    """Pre-projection-heads ``TempCXRJEPA`` (eval-only).

    Holds the same components as the current ``TempCXRJEPA`` minus the
    three projection heads:

      - ``image_encoder``        : online BioViL-T — was trained.
      - ``target_image_encoder`` : EMA copy of ``image_encoder``.
      - ``text_encoder``         : BioViL-T text encoder — was trained.
      - ``predictor``            : ``IJEPATemporalPredictor`` — was trained.

    The state-dict key set of this class exactly matches the keys
    written by ``resume_train_jepa.py`` on any commit ``<= bc00c68``
    (the commit immediately before ``618f374``). Loading such a
    checkpoint with ``strict=False`` should report zero missing keys.
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
    # FORWARD (NO LOSSES) — matches pre-``618f374`` shape
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
        Returns a dict containing:
          - prior_patches            (B, N, D)  LN(image_encoder(prior))
          - current_patches_target   (B, N, D)  LN(target_image_encoder(current))
                                                — detached
          - pred_current_patches     (B, N, D)  predictor output ẑ_cur
          - prior_txt_local          (B, T, D)
          - prior_token_mask         (B, T)
          - current_txt_local        (B, T, D)
          - current_token_mask       (B, T)
          - condition_txt_local      (B, T, D)
          - condition_token_mask     (B, T)
        """
        # Online encoder on prior; feature-dim LayerNorm so the
        # predictor's delta-prediction starts in the same scale as
        # the LN'd JEPA target.
        _, prior_patches = self.image_encoder(prior_imgs)
        prior_patches = F.layer_norm(
            prior_patches,
            (prior_patches.size(-1),),
        )

        # Text encoder on all three reports in one batched call.
        B = prior_imgs.size(0)
        all_reports = (
            list(prior_reports)
            + list(current_reports)
            + list(condition_texts)
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

        # Target (EMA) encoder on current image; LN; stop-grad.
        with torch.no_grad():
            _, current_patches_target = self.target_image_encoder(current_imgs)
            current_patches_target = F.layer_norm(
                current_patches_target,
                (current_patches_target.size(-1),),
            )
        current_patches_target = current_patches_target.detach()

        # Predictor: ẑ_cur from prior + condition text.
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
    # EMA UPDATE — only the image encoder, no projections
    # --------------------------------------------------
    @torch.no_grad()
    def update_ema(self, momentum: float):
        _update_ema(self.image_encoder, self.target_image_encoder, momentum)
