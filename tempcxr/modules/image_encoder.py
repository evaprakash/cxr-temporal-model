# image_encoder.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# Make hi-ml multimodal visible
# ------------------------------------------------------------------
HI_ML_SRC = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "hi-ml",
        "hi-ml-multimodal",
        "src",
    )
)
sys.path.insert(0, HI_ML_SRC)
#sys.path.insert(0, os.path.abspath("hi-ml/hi-ml-multimodal/src"))

from health_multimodal.image.model.model import MultiImageModel
from health_multimodal.image.model.types import ImageEncoderType
from health_multimodal.image.model.pretrained import (
    _download_biovil_image_model_weights,
    _download_biovil_t_image_model_weights,
)

DEBUG = True


class BioViLTImageEncoder(nn.Module):
    """
    Image encoder supporting three modes:

    mode="biovil":
        - BioViL-T architecture
        - CNN initialized from BioViL
        - temporal transformer randomly initialized
        - prev_imgs optional (EXACT previous behavior)

    mode="biovilt":
        - Fully pretrained BioViL-T (official)

    mode="biovilt_finetuned":
        - BioViL-T initialized from a user-trained checkpoint
    """

    def __init__(
        self,
        mode: str = "biovil",
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        assert mode in {"biovil", "biovilt", "biovilt_finetuned"}
        self.mode = mode
        self.embed_dim = 128

        # --------------------------------------------------
        # Always build the SAME architecture
        # --------------------------------------------------
        self.model = MultiImageModel(
            img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
            joint_feature_size=128,
            pretrained_model_path=None,  # weights loaded manually
        )

        # --------------------------------------------------
        # Load weights
        # --------------------------------------------------
        if mode == "biovil":
            # EXACT previous setup
            ckpt = _download_biovil_image_model_weights()
            state = torch.load(ckpt, map_location="cpu")

            # CNN ONLY
            self.model.encoder.encoder.load_state_dict(state, strict=False)

            if DEBUG:
                print("[ImageEncoder] Mode = BioViL (CNN init only)")

        elif mode == "biovilt":
            ckpt = _download_biovil_t_image_model_weights()
            state = torch.load(ckpt, map_location="cpu")

            self.model.load_state_dict(state, strict=True)

            if DEBUG:
                print("[ImageEncoder] Mode = BioViL-T (official pretrained)")

        else:  # biovilt_finetuned
            assert checkpoint_path is not None, \
                "checkpoint_path required for biovilt_finetuned"

            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state, strict=True)

            if DEBUG:
                print(
                    f"[ImageEncoder] Mode = BioViL-T (finetuned): {checkpoint_path}"
                )

    def forward(self, curr_imgs, prev_imgs=None):
        """
        curr_imgs : Tensor (B,3,H,W)
        prev_imgs : optional Tensor (B,3,H,W)

        Same forward behavior for all modes.
        """

        out = self.model(
            current_image=curr_imgs,
            previous_image=prev_imgs,
        )

        # ---- global embedding ----
        img_emb = F.normalize(
            out.projected_global_embedding, dim=-1
        )  # (B,128)

        # ---- patch embeddings ----
        feat = out.projected_patch_embeddings  # (B,128,H',W')
        B, C, H, W = feat.shape
        patch_emb = F.normalize(
            feat.flatten(2).transpose(1, 2), dim=-1
        )  # (B,L,128)

        if DEBUG:
            print(
                f"[Image:{self.mode}] global:",
                img_emb.shape,
                "mean norm:",
                img_emb.norm(dim=-1).mean().item(),
            )
            print(f"[Image:{self.mode}] patches:", patch_emb.shape)

        return img_emb, patch_emb


# ------------------------------------------------------------------
# SELF-TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n🔍 Running image encoder sanity checks\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 1
    curr_imgs = torch.randn(B, 3, 448, 448).to(device)
    prev_imgs = torch.randn(B, 3, 448, 448).to(device)

    # ---- BioViL (previous behavior) ----
    print("\n--- BioViL (CNN init only) ---")
    enc_biovil = BioViLTImageEncoder(mode="biovil").to(device)
    enc_biovil.eval()
    with torch.no_grad():
        enc_biovil(curr_imgs, prev_imgs)

    # ---- BioViL-T (official) ----
    print("\n--- BioViL-T (official pretrained) ---")
    enc_biovilt = BioViLTImageEncoder(mode="biovilt").to(device)
    enc_biovilt.eval()
    with torch.no_grad():
        enc_biovilt(curr_imgs, prev_imgs)

    print("\n✅ Sanity checks passed")

