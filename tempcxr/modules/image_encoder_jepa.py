"""Unit-sphere image encoder for temporal CXR JEPA training.

Wraps the BioViL-T ``MultiImageModel`` backbone and L2-normalizes both
patch and global outputs along the feature dim so every downstream
consumer (predictor, JEPA cosine loss, GLoRIA contrastive loss,
progression loss) receives unit-norm vectors.

Why unit-sphere instead of LayerNorm:
    The previous I-JEPA-style recipe kept raw magnitudes and applied
    LayerNorm-no-params on the target side at loss time. With multiple
    multimodal heads (JEPA, GLoRIA, progression) reading from the same
    encoder, the cleanest geometry is a shared unit sphere — every
    consumer becomes scale-invariant by construction, and the JEPA loss
    can be a plain cosine without any extra normalization step.

Constructor signature: ``mode in {"biovil", "biovilt", "biovilt_finetuned"}``,
optional ``checkpoint_path`` for the finetuned mode. ``forward`` takes
``(curr_imgs, prev_imgs=None)`` and returns ``(global, patches)``, both
L2-normalized along the last dim.
"""

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

from health_multimodal.image.model.model import MultiImageModel
from health_multimodal.image.model.types import ImageEncoderType
from health_multimodal.image.model.pretrained import (
    _download_biovil_image_model_weights,
    _download_biovil_t_image_model_weights,
)

DEBUG = True


class BioViLTImageEncoderJEPA(nn.Module):
    """BioViL-T image encoder with L2-normalized patch + global outputs.

    Both outputs are projected to 128-d by BioViL-T's built-in
    ``joint_feature_size=128`` head, then L2-normalized along the
    feature dim before being returned. Consumers (predictor, JEPA
    cosine loss, GLoRIA, progression loss) all read from these
    unit-norm features directly — no extra projection heads anywhere.

    Modes (same as the non-JEPA variant):

    mode="biovil":
        - BioViL-T architecture
        - CNN initialized from BioViL
        - Temporal transformer randomly initialized

    mode="biovilt":
        - Fully pretrained BioViL-T (official)

    mode="biovilt_finetuned":
        - BioViL-T initialized from a user-trained checkpoint
    """

    def __init__(
        self,
        mode: str = "biovilt",
        checkpoint_path: str | None = None,
    ):
        super().__init__()
        assert mode in {"biovil", "biovilt", "biovilt_finetuned"}
        self.mode = mode
        self.embed_dim = 128

        self.model = MultiImageModel(
            img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
            joint_feature_size=128,
            pretrained_model_path=None,
        )

        if mode == "biovil":
            ckpt = _download_biovil_image_model_weights()
            state = torch.load(ckpt, map_location="cpu")
            self.model.encoder.encoder.load_state_dict(state, strict=False)
            if DEBUG:
                print("[ImageEncoderJEPA] Mode = BioViL (CNN init only)")

        elif mode == "biovilt":
            ckpt = _download_biovil_t_image_model_weights()
            state = torch.load(ckpt, map_location="cpu")
            self.model.load_state_dict(state, strict=True)
            if DEBUG:
                print("[ImageEncoderJEPA] Mode = BioViL-T (official pretrained)")

        else:  # biovilt_finetuned
            assert checkpoint_path is not None, \
                "checkpoint_path required for biovilt_finetuned"
            state = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state, strict=True)
            if DEBUG:
                print(
                    f"[ImageEncoderJEPA] Mode = BioViL-T (finetuned): {checkpoint_path}"
                )

    def forward(self, curr_imgs, prev_imgs=None):
        """Returns (global, patches) — both L2-normalized along feature dim.

        curr_imgs : Tensor (B,3,H,W)
        prev_imgs : optional Tensor (B,3,H,W)
        """
        out = self.model(
            current_image=curr_imgs,
            previous_image=prev_imgs,
        )

        # ---- global embedding (B, 128), L2-normalized ----
        img_emb = out.projected_global_embedding
        img_emb = F.normalize(img_emb, dim=-1)

        # ---- patch embeddings (B, L, 128), L2-normalized ----
        feat = out.projected_patch_embeddings  # (B,128,H',W')
        patch_emb = feat.flatten(2).transpose(1, 2)
        patch_emb = F.normalize(patch_emb, dim=-1)

        if DEBUG:
            print(
                f"[ImageJEPA:{self.mode}] global:",
                tuple(img_emb.shape),
                "mean L2 norm:",
                img_emb.norm(dim=-1).mean().item(),
            )
            print(f"[ImageJEPA:{self.mode}] patches:", tuple(patch_emb.shape))

        return img_emb, patch_emb


# ------------------------------------------------------------------
# SELF-TEST
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\nRunning JEPA image encoder sanity checks\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 1
    curr_imgs = torch.randn(B, 3, 448, 448).to(device)
    prev_imgs = torch.randn(B, 3, 448, 448).to(device)

    print("\n--- BioViL-T (official pretrained), JEPA-style outputs ---")
    enc = BioViLTImageEncoderJEPA(mode="biovilt").to(device)
    enc.eval()
    with torch.no_grad():
        g, p = enc(curr_imgs, prev_imgs)
    print("global shape:", tuple(g.shape), "mean norm:", g.norm(dim=-1).mean().item())
    print("patches shape:", tuple(p.shape), "mean norm:", p.norm(dim=-1).mean().item())

    print("\nSanity checks passed")
