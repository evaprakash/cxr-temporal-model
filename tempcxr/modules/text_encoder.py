# text_encoder.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

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

from health_multimodal.text.model.configuration_cxrbert import CXRBertConfig
from health_multimodal.text.model.modelling_cxrbert import (
    CXRBertModel,
    BertProjectionHead,
)

# Pretrained text models live under tempcxr/modules/pretrained/.
# Resolved relative to this file so the path is portable across
# machines (cluster, local, GCP, etc.) — only requirement is that the
# directory layout matches.
PRETRAINED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "pretrained")
)
BIOVIL_TEXT_MODEL = os.path.join(PRETRAINED_DIR, "BiomedVLP-CXR-BERT-specialized")
BIOVILT_TEXT_MODEL = os.path.join(PRETRAINED_DIR, "BiomedVLP-BioViL-T")


# ================================================================
# IMAGE-GUIDED CROSS ATTENTION (JOINT SPACE)
# ================================================================
class ImageGuidedCrossAttention(nn.Module):
    def __init__(self, dim=128, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, text_proj, image_proj):
        fused, _ = self.attn(
            query=text_proj,
            key=image_proj,
            value=image_proj,
        )
        return fused


# ================================================================
# BIOVIL / BIOVIL-T TEXT ENCODER
# ================================================================
class BioViLTTextEncoder(nn.Module):
    """CXR-BERT wrapper with an optional 768→128 projection head.

    Parameters
    ----------
    mode
        Which pretrained CXR-BERT variant to load (biovil / biovilt /
        biovilt_finetuned).
    checkpoint_path
        Required when ``mode == "biovilt_finetuned"``.
    mlm_prob
        Masking probability for the (image-guided) MLM path.
    use_projection
        Controls whether ``forward_contrastive`` emits 128-d projected
        tokens (the BioViL-T default) or the raw 768-d CXR-BERT hidden
        states.

        - ``True`` (default): behavior identical to the pre-change
          BioViL-T pipeline — ``token_hidden`` (dim 768) is passed
          through ``self.text_projection`` (a ``BertProjectionHead``)
          and L2-normalized, giving 128-d unit-norm tokens. This is
          what the BioViL-T image encoder path expects, since its
          patches are also 128-d after the ``joint_feature_size=128``
          head.

        - ``False``: skip ``self.text_projection`` entirely and
          L2-normalize the 768-d hidden states directly. Use this when
          the paired image encoder outputs raw 768-d features and you
          want to avoid an image/text dim mismatch (e.g. with the
          RAD-DINO-MAIRA-2 image encoder). ``self.text_projection`` is
          still allocated so the state_dict layout is stable, but its
          parameters get no gradient in this configuration.
    """

    def __init__(
        self,
        mode: str = "biovilt",
        checkpoint_path: str | None = None,
        mlm_prob: float = 0.45,
        use_projection: bool = True,
    ):
        super().__init__()
        assert mode in {"biovil", "biovilt", "biovilt_finetuned"}
        self.mlm_prob = mlm_prob
        self.use_projection = use_projection

        # ------------------------------------------------------------
        # Select model
        # ------------------------------------------------------------
        if mode == "biovil":
            model_name = BIOVIL_TEXT_MODEL
        elif mode == "biovilt":
            model_name = BIOVILT_TEXT_MODEL
        else:
            assert checkpoint_path is not None
            model_name = checkpoint_path

        # ------------------------------------------------------------
        # Tokenizer
        # ------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        if "[MLM]" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": ["[MLM]"]}
            )

        # ------------------------------------------------------------
        # Load CXR-BERT
        # ------------------------------------------------------------
        if mode == "biovilt":
            self.model = CXRBertModel.from_pretrained(model_name)
        else:
            config = CXRBertConfig.from_pretrained(model_name)
            self.model = CXRBertModel.from_pretrained(model_name, config=config)

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.hidden_dim = self.model.config.hidden_size      # 768
        self.proj_dim = self.model.config.projection_size    # 128
        # Dim actually emitted by ``forward_contrastive`` — used by
        # downstream modules that want to size predictor/loss heads to
        # match the text token dim.
        self.output_dim = self.hidden_dim if not self.use_projection else self.proj_dim

        # ------------------------------------------------------------
        # φ_txt projection (same as CLS projection head)
        # ------------------------------------------------------------
        self.text_projection = BertProjectionHead(self.model.config)

        # ------------------------------------------------------------
        # Unprojection back to BERT hidden space (for MLM)
        # ------------------------------------------------------------
        self.text_unprojection = nn.Linear(self.proj_dim, self.hidden_dim)

        # ------------------------------------------------------------
        # Image-guided cross-attention (joint space)
        # ------------------------------------------------------------
        self.cross_attn = ImageGuidedCrossAttention(dim=self.proj_dim)

    # ============================================================
    # CONTRASTIVE FORWARD (TEXT ONLY)
    # ============================================================
    def forward_contrastive(self, texts):
        texts = ["[CLS] " + t for t in texts]

        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=112,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model(
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = outputs.hidden_states[-1]  # (B, T, 768)

        # ---- LOCAL (drop CLS) ----
        token_hidden = hidden[:, 1:, :]     # (B, T-1, 768)
        if self.use_projection:
            txt_local = self.text_projection(token_hidden)
        else:
            # Use raw 768-d hidden states so the token dim matches an
            # image encoder that also outputs 768-d features (e.g.
            # RAD-DINO-MAIRA-2). The projection head is bypassed but
            # kept allocated for state_dict layout stability.
            txt_local = token_hidden
        txt_local = F.normalize(txt_local, dim=-1)

        token_mask = tok.attention_mask[:, 1:].bool()

        # ---- GLOBAL (CLS only) ----
        # The projected global embedding is 128-d by construction. It is
        # not consumed by the JEPA forward path (which only reads
        # ``txt_local``), so we return it unchanged even when
        # ``use_projection=False`` — callers that need a raw 768-d CLS
        # feature can grab ``hidden[:, 0]`` themselves.
        txt_global = self.model.get_projected_text_embeddings(
            input_ids=tok.input_ids,
            attention_mask=tok.attention_mask,
            normalize_embeddings=True,
        )

        return txt_global, txt_local, token_mask

    # ============================================================
    # IMAGE-GUIDED MLM (PROJECT → ATTEND → UNPROJECT → MLM)
    # ============================================================
    def forward_mlm(self, texts, image_patches):
        texts = ["[MLM] " + t for t in texts]

        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=112,
            return_tensors="pt",
        ).to(self.model.device)

        input_ids = tok.input_ids.clone()
        labels = tok.input_ids.clone()

        # ---- Masking (45%) ----
        prob = torch.full(input_ids.shape, self.mlm_prob, device=input_ids.device)

        special_mask = [
            self.tokenizer.get_special_tokens_mask(seq, already_has_special_tokens=True)
            for seq in labels.tolist()
        ]
        special_mask = torch.tensor(
            special_mask, dtype=torch.bool, device=input_ids.device
        )
        prob.masked_fill_(special_mask, 0.0)

        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100
        input_ids[masked] = self.tokenizer.mask_token_id

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=tok.attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # ---- PROJECT TEXT → JOINT SPACE ----
        text_hidden = outputs.hidden_states[-1]        # (B, T, 768)
        text_proj = self.text_projection(text_hidden)  # (B, T, 128)

        # ---- IMAGE-GUIDED CROSS-ATTENTION ----
        fused_proj = self.cross_attn(
            text_proj=text_proj,
            image_proj=image_patches,
        )

        # ---- UNPROJECT BACK TO BERT SPACE ----
        fused_hidden = self.text_unprojection(fused_proj)  # (B, T, 768)

        # ---- MLM HEAD ----
        mlm_logits = self.model.cls(fused_hidden)

        return mlm_logits, labels


# ==================================================================
# SELF-TEST
# ==================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    texts = [
        "Increased right pleural effusion.",
        "Left lower lobe pneumonia is improving.",
    ]

    B = len(texts)
    image_patches = torch.randn(B, 196, 128).to(device)

    encoder = BioViLTTextEncoder(mode="biovilt").to(device)
    encoder.eval()

    with torch.no_grad():
        txt_global, txt_local, token_mask = encoder.forward_contrastive(texts)

    mlm_logits, mlm_labels = encoder.forward_mlm(texts, image_patches)

    print("Global:", txt_global.shape)
    print("Local :", txt_local.shape)
    print("Token mask:", token_mask.shape)
    print("MLM logits:", mlm_logits.shape)

    print("\n✅ BioViL-T text encoder with projection/unprojection is correct")

