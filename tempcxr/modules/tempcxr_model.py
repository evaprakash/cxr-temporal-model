import os
import sys
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Make project root visible so ../../losses.py is importable
# ------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Import encoders (single source of truth)
# ---------------------------------------------------------
from .image_encoder import BioViLTImageEncoder
from .text_encoder import BioViLTTextEncoder

# ---------------------------------------------------------
# Import losses from ../../losses.py
# ---------------------------------------------------------
from losses import (
    global_contrastive_loss,
    local_contrastive_loss,
    mlm_loss,
)


# =========================================================
# TEMPCXR MODEL
# =========================================================
class TempCXR(nn.Module):
    """
    BioViL-T forward orchestration module.

    Forward pass returns representations only.
    Losses are computed externally (clean separation).
    """

    def __init__(self, mode='biovil'):
        super().__init__()

        self.image_encoder = BioViLTImageEncoder(mode=mode)
        self.text_encoder = BioViLTTextEncoder(mode=mode)

    # --------------------------------------------------
    # FULL FORWARD (NO LOSSES)
    # --------------------------------------------------
    def forward(self, curr_imgs, prev_imgs, texts):
        """
        curr_imgs : (B, 3, 448, 448)
        prev_imgs : (B, 3, 448, 448) or None
        texts     : list[str]

        Returns dict with everything needed for loss computation.
        """

        # -------------------------------
        # Image encoding
        # -------------------------------
        img_global, img_patches = self.image_encoder(
            curr_imgs,
            prev_imgs,
        )

        # -------------------------------
        # Text encoding (contrastive)
        # -------------------------------
        txt_global, txt_local, token_mask = (
            self.text_encoder.forward_contrastive(texts)
        )

        # -------------------------------
        # Text encoding (MLM)
        # -------------------------------
        mlm_logits, mlm_labels = (
            self.text_encoder.forward_mlm(texts, img_patches)
        )

        return {
            "img_global": img_global,
            "img_patches": img_patches,
            "txt_global": txt_global,
            "txt_local": txt_local,
            "token_mask": token_mask,
            "mlm_logits": mlm_logits,
            "mlm_labels": mlm_labels,
        }


# =========================================================
# SELF-TEST
# =========================================================
if __name__ == "__main__":
    print("\n🔍 Running TempCXR forward + external loss test\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TempCXR().to(device)
    model.eval()

    # --------------------------------------------------
    # Dummy inputs
    # --------------------------------------------------
    B = 2
    curr_imgs = torch.randn(B, 3, 448, 448).to(device)
    prev_imgs = None#torch.randn(B, 3, 448, 448).to(device)

    texts = [
        "Increased right pleural effusion.",
        "Left lower lobe pneumonia is improving.",
    ]

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    with torch.no_grad():
        out = model(curr_imgs, prev_imgs, texts)

    # --------------------------------------------------
    # Losses (from ../../losses.py)
    # --------------------------------------------------
    g_loss = global_contrastive_loss(
        out["img_global"],
        out["txt_global"],
    )

    l_loss = local_contrastive_loss(
        out["img_patches"],
        out["txt_local"],
        out["token_mask"],
    )

    m_loss = mlm_loss(
        out["mlm_logits"],
        out["mlm_labels"],
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------
    print("Image global :", out["img_global"].shape)
    print("Image patches:", out["img_patches"].shape)
    print("Text global  :", out["txt_global"].shape)
    print("Text local   :", out["txt_local"].shape)
    print("MLM logits   :", out["mlm_logits"].shape)

    print("\n📉 Losses")
    print("Global contrastive:", g_loss.item())
    print("Local contrastive :", l_loss.item())
    print("MLM               :", m_loss.item())

    print("\n✅ TempCXR + external losses wired correctly")

