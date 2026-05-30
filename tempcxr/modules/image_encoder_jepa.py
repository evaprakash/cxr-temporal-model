"""I-JEPA-style image encoder.

Differences from `image_encoder.BioViLTImageEncoder`:

  * Patch and global outputs are returned RAW (no `F.normalize`).
    I-JEPA's recipe avoids L2 normalization on the JEPA path because it
    discards magnitude information that may carry signal. Scale stability
    is achieved via LayerNorm-no-params on the target side at loss time
    (see `jepa.py`), not via projecting to the unit sphere.
  * Loss-side L2 normalization for downstream contrastive heads (e.g.
    `local_contrastive_loss` in `jepa.py`) is handled by those losses
    themselves; they re-normalize their inputs internally, so consumers
    that need unit vectors are unaffected.

Otherwise this is a drop-in replacement for `BioViLTImageEncoder` with
the same constructor signature, the same `MultiImageModel` backbone,
and the same `(global, patches)` return contract.
"""

import os
import sys
import torch
import torch.nn as nn

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
    """BioViL-T image encoder with raw (un-normalized) patch + global outputs.

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
        """Returns (global, patches) — both RAW (no L2 normalization).

        curr_imgs : Tensor (B,3,H,W)
        prev_imgs : optional Tensor (B,3,H,W)
        """
        out = self.model(
            current_image=curr_imgs,
            previous_image=prev_imgs,
        )

        # ---- raw global embedding (B, 128) ----
        img_emb = out.projected_global_embedding

        # ---- raw patch embeddings (B, L, 128) ----
        feat = out.projected_patch_embeddings  # (B,128,H',W')
        patch_emb = feat.flatten(2).transpose(1, 2)

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
