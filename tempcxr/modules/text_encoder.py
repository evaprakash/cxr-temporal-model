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
    def __init__(
        self,
        mode: str = "biovilt",
        checkpoint_path: str | None = None,
        mlm_prob: float = 0.45,
    ):
        super().__init__()
        assert mode in {
            "biovil",
            "biovilt",
            "biovilt_finetuned",
            "biovilt_no_pretrained",
        }
        self.mlm_prob = mlm_prob

        # ------------------------------------------------------------
        # Select model
        # ------------------------------------------------------------
        # The text encoder always loads from the local ``pretrained/``
        # directory (no network), so ``biovilt_no_pretrained`` behaves
        # exactly like ``biovilt`` here — it's the image encoder half
        # that actually skips a fetch under this mode. We still accept
        # the mode string so ``TempCXRJEPA(mode=...)`` can pass a single
        # value through to both encoders without special-casing.
        if mode == "biovil":
            model_name = BIOVIL_TEXT_MODEL
        elif mode in ("biovilt", "biovilt_no_pretrained"):
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
        if mode in ("biovilt", "biovilt_no_pretrained"):
            self.model = CXRBertModel.from_pretrained(model_name)
        else:
            config = CXRBertConfig.from_pretrained(model_name)
            self.model = CXRBertModel.from_pretrained(model_name, config=config)

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.hidden_dim = self.model.config.hidden_size      # 768
        self.proj_dim = self.model.config.projection_size    # 128

        # ------------------------------------------------------------
        # Local 768→128 map. BioViL-T only ships ``cls_projection_head``
        # (used for the CLS global). There is no separate local head, so
        # we instantiate the same module class and copy those official
        # weights. GLoRIA + the JEPA predictor consume this, not a
        # random Linear.
        # ------------------------------------------------------------
        self.text_projection = BertProjectionHead(self.model.config)
        self._init_local_proj_from_official_cls(model_name)

        # ------------------------------------------------------------
        # Unprojection back to BERT hidden space (for MLM)
        # ------------------------------------------------------------
        self.text_unprojection = nn.Linear(self.proj_dim, self.hidden_dim)

        # ------------------------------------------------------------
        # Image-guided cross-attention (joint space)
        # ------------------------------------------------------------
        self.cross_attn = ImageGuidedCrossAttention(dim=self.proj_dim)

    def _load_pretrained_cls_proj_tensors(self, model_name: str) -> dict:
        """``cls_projection_head.*`` tensors from the on-disk BioViL-T ckpt."""
        candidates = (
            os.path.join(model_name, "model.safetensors"),
            os.path.join(model_name, "pytorch_model.bin"),
        )
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            raise RuntimeError(
                f"no BioViL-T weights at {model_name} "
                f"(looked for model.safetensors / pytorch_model.bin)"
            )
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            blob = load_file(path)
        else:
            try:
                try:
                blob = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                blob = torch.load(path, map_location="cpu")
            except TypeError:
                blob = torch.load(path, map_location="cpu")
        prefix = "cls_projection_head."
        tensors = {
            k[len(prefix):]: v for k, v in blob.items() if k.startswith(prefix)
        }
        if not tensors:
            raise RuntimeError(
                f"{path} has no '{prefix}*' keys; cannot init local "
                f"text_projection from official BioViL-T. "
                f"keys sample: {list(blob)[:8]}"
            )
        return tensors

    def _init_local_proj_from_official_cls(self, model_name: str) -> None:
        """Copy official ``cls_projection_head`` into ``text_projection``.

        Verifies the live CXR-BERT head matches the file on disk, then
        copies it. Raises if either step would leave a random proj.
        """
        src = getattr(self.model, "cls_projection_head", None)
        if src is None:
            raise RuntimeError(
                "CXRBertModel has no cls_projection_head after "
                "from_pretrained; refusing to leave text_projection random"
            )
        file_tensors = self._load_pretrained_cls_proj_tensors(model_name)
        live = {k: v.detach().cpu() for k, v in src.state_dict().items()}
        if set(live) != set(file_tensors):
            raise RuntimeError(
                f"cls_projection_head keys != checkpoint: "
                f"live={sorted(live)} file={sorted(file_tensors)}"
            )
        for k in live:
            if not torch.allclose(live[k].float(), file_tensors[k].float()):
                raise RuntimeError(
                    f"live cls_projection_head.{k} does not match "
                    f"{model_name} (from_pretrained did not load official proj)"
                )
        incompatible = self.text_projection.load_state_dict(src.state_dict(), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"text_projection load_state_dict mismatch: {incompatible}"
            )
        self.assert_local_proj_matches_official_cls()

    def assert_local_proj_matches_official_cls(self) -> None:
        """Fail if local 768→128 weights != official CLS projector."""
        src = self.model.cls_projection_head
        dst = self.text_projection
        src_sd = src.state_dict()
        dst_sd = dst.state_dict()
        if set(src_sd) != set(dst_sd):
            raise RuntimeError(
                f"proj key mismatch: cls={sorted(src_sd)} "
                f"local={sorted(dst_sd)}"
            )
        for k in src_sd:
            if not torch.equal(src_sd[k], dst_sd[k]):
                raise RuntimeError(
                    f"text_projection.{k} != cls_projection_head.{k} "
                    f"(local proj is not the official BioViL-T head)"
                )

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
        txt_local = self.text_projection(token_hidden)
        txt_local = F.normalize(txt_local, dim=-1)

        token_mask = tok.attention_mask[:, 1:].bool()

        # ---- GLOBAL (CLS only) ----
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

