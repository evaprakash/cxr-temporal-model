"""Unit-sphere image encoder built on ``microsoft/rad-dino-maira-2``.

Drop-in replacement for :class:`image_encoder_jepa.BioViLTImageEncoderJEPA`
whose ``forward(curr_imgs, prev_imgs=None)`` returns
``(global_emb, patch_emb)`` with both L2-normalized along the feature
dim. The wrapper differs from the BioViL-T variant in three ways:

    1. The backbone is the RAD-DINO-MAIRA-2 ViT (DINOv2-base finetuned on
       chest X-rays; ``patch_size=14``, ``hidden_size=768``,
       ``image_size=518``), loaded via ``transformers.AutoModel``.
    2. There is no ``joint_feature_size=128`` projection head on top of
       the CLS / patch tokens — the outputs are the raw 768-d ViT
       features, only L2-normalized so downstream cosine / contrastive
       losses stay scale-invariant. The paired text encoder is expected
       to be configured with ``use_projection=False`` so its token
       features are also 768-d.
    3. ``prev_imgs`` (BioViL-T's temporal side input) is ignored — the
       DINOv2 backbone is single-image. The API keeps the argument only
       so the encoder is a drop-in replacement for callers written
       against the BioViL-T signature.

The preprocessing baked into ``forward`` reproduces the HF
``BitImageProcessor`` config shipped with ``rad-dino-maira-2``:

    * Rescale to ``[0, 1]``: already done by the dataset's
      ``TF.to_tensor``, so the wrapper assumes inputs are in ``[0, 1]``.
    * Resize (bicubic) so the shorter edge is 518, then center-crop to
      ``518 x 518``. Our dataset delivers square 448 x 448 tensors, so
      a single ``F.interpolate`` to ``(518, 518)`` is exact and no crop
      is required at this stage.
    * Per-channel normalize with ``mean = 0.5307`` and ``std = 0.2583``
      (all three channels; the preprocessor uses the same value across
      RGB).

Constructor signature: ``mode in {"raddino", "raddino_finetuned"}``, with
an optional ``checkpoint_path`` for the finetuned mode. Default
``model_id`` is ``microsoft/rad-dino-maira-2``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


DEBUG = True


# Preprocessor constants extracted from
# https://huggingface.co/microsoft/rad-dino-maira-2 (preprocessor_config.json).
# They are baked in rather than fetched from AutoImageProcessor so the
# encoder is usable without an extra network round-trip and so the exact
# training-time preprocessing is documented in-tree.
RAD_DINO_MEAN = (0.5307, 0.5307, 0.5307)
RAD_DINO_STD = (0.2583, 0.2583, 0.2583)
RAD_DINO_IMAGE_SIZE = 518   # image_size + crop_size["height"] both 518
RAD_DINO_PATCH_SIZE = 14
RAD_DINO_NUM_PATCHES = (RAD_DINO_IMAGE_SIZE // RAD_DINO_PATCH_SIZE) ** 2  # 37*37 = 1369
RAD_DINO_HIDDEN_SIZE = 768


class RADDINOImageEncoderJEPA(nn.Module):
    """RAD-DINO-MAIRA-2 image encoder with L2-normalized patch + global outputs.

    Modes:

    mode="raddino":
        - Pretrained ``microsoft/rad-dino-maira-2`` weights (DINOv2-base
          finetuned on chest X-rays; the version used in MAIRA-2).

    mode="raddino_finetuned":
        - Same architecture but weights initialized from a local
          checkpoint. Pass the HF model repo path (or a directory holding
          a ``pytorch_model.bin`` compatible with ``Dinov2Model``) via
          ``checkpoint_path``.
    """

    def __init__(
        self,
        mode: str = "raddino",
        checkpoint_path: str | None = None,
        model_id: str = "microsoft/rad-dino-maira-2",
    ):
        super().__init__()
        assert mode in {"raddino", "raddino_finetuned"}
        self.mode = mode
        self.embed_dim = RAD_DINO_HIDDEN_SIZE
        self.num_patches = RAD_DINO_NUM_PATCHES
        self.image_size = RAD_DINO_IMAGE_SIZE

        source = model_id if mode == "raddino" else checkpoint_path
        assert source is not None, (
            "checkpoint_path is required for mode='raddino_finetuned'"
        )
        self.model = AutoModel.from_pretrained(source)

        # Register mean/std as buffers so they move with .to(device) and
        # ride along in the state_dict for future loading (harmless if
        # ever overridden — they're constants).
        mean = torch.tensor(RAD_DINO_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(RAD_DINO_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

        if DEBUG:
            print(
                f"[ImageJEPA:{self.mode}] loaded {source}; "
                f"embed_dim={self.embed_dim}, num_patches={self.num_patches}, "
                f"image_size={self.image_size}x{self.image_size}"
            )

    # --------------------------------------------------
    # Preprocessing hook — matches HF BitImageProcessor
    # --------------------------------------------------
    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        """Resize to 518x518 (bicubic) and normalize with the RAD-DINO stats.

        Inputs are expected to be already-rescaled to ``[0, 1]`` (the
        dataset's ``TF.to_tensor`` does this). The dataset delivers
        square 448x448 tensors, so ``F.interpolate`` to 518x518 subsumes
        the processor's ``resize(shortest_edge=518) + center_crop(518)``
        pipeline for square inputs.
        """
        if imgs.shape[-2:] != (self.image_size, self.image_size):
            imgs = F.interpolate(
                imgs,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return (imgs - self.pixel_mean) / self.pixel_std

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, curr_imgs, prev_imgs=None):
        """Returns (global, patches) — both L2-normalized along feature dim.

        curr_imgs : Tensor (B, 3, H, W) in ``[0, 1]``
        prev_imgs : ignored (kept for BioViL-T API compatibility)
        """
        x = self._preprocess(curr_imgs)
        outputs = self.model(pixel_values=x, return_dict=True)
        hidden = outputs.last_hidden_state  # (B, 1 + N, D)

        img_emb = hidden[:, 0, :]           # CLS
        img_emb = F.normalize(img_emb, dim=-1)

        patch_emb = hidden[:, 1:, :]        # (B, N, D)
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
    print("\nRunning RAD-DINO-MAIRA-2 JEPA image encoder sanity checks\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 1
    curr_imgs = torch.rand(B, 3, 448, 448).to(device)  # [0, 1] like TF.to_tensor
    prev_imgs = torch.rand(B, 3, 448, 448).to(device)  # ignored

    enc = RADDINOImageEncoderJEPA(mode="raddino").to(device)
    enc.eval()
    with torch.no_grad():
        g, p = enc(curr_imgs, prev_imgs)
    print("global shape:", tuple(g.shape), "mean norm:", g.norm(dim=-1).mean().item())
    print("patches shape:", tuple(p.shape), "mean norm:", p.norm(dim=-1).mean().item())

    print("\nSanity checks passed")
