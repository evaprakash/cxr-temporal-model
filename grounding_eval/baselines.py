"""Zero-shot baselines for the text grounding maps: TILA, BioViL, BiomedCLIP, ALTA.

Each model gives the current film's patch embeddings in its own image-text space
and a sentence encoder for that space, so the text, text_contrast, text_presence
and text_orth maps are built exactly as for BioViL-T: the cosine of each L2-normalized
patch embedding with the text direction, min-max scaled per film.

  tila        BioViL-T architecture fed (current, prior); projected patch embeddings
              (14 x 14 on the 448 crop) against TILA's text encoder.
  biovil      single-image ResNet-50 on the current film (15 x 15 on a 480 crop)
              against CXR-BERT-specialized, BioViL's own phrase grounding.
  biomedclip  ViT-B/16 on the current film (14 x 14 on a 224 crop): patch tokens
              through the image projection that maps the CLS token into the joint
              space, against the PubMedBERT text tower.
  alta        ViT-B/16 on the (current, prior) slots: current-slot patch tokens
              through ALTA's local projection head, the head trained to align patches
              with words, against the mean of the sentence's projected word
              embeddings from the matching text local head.

Every map is resampled from the model's own crop into the BioViL-T 448 frame
(Resize 512, CenterCrop 448), so all models are scored on the same pixels. The
class scores behind the pred column are cosines of a pooled image embedding with
the five class sentences; they are only used with --target pred.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

BASELINES = ["tila", "biovil", "biomedclip", "alta"]


def crop_geometry(w: int, h: int, resize_short: int, crop: int):
    """Scale and crop offset of torchvision Resize(resize_short) + CenterCrop(crop)."""
    if w <= h:
        nw, nh = resize_short, int(resize_short * h / w)
    else:
        nw, nh = int(resize_short * w / h), resize_short
    return nw / w, nh / h, int(round((nw - crop) / 2.0)), int(round((nh - crop) / 2.0))


def crop_to_frame(cam, orig_wh, resize_short: int, crop: int) -> np.ndarray:
    """Grid map over a model's Resize/CenterCrop crop -> [448, 448] on the BioViL-T frame."""
    import torch
    import torch.nn.functional as F

    from grounding_eval.eval_grounding import FRAME, RESIZE_SHORT

    w, h = orig_wh
    sx, sy, left, top = crop_geometry(w, h, RESIZE_SHORT, FRAME)
    msx, msy, mleft, mtop = crop_geometry(w, h, resize_short, crop)
    u = (np.arange(FRAME) + 0.5 + left) / sx
    v = (np.arange(FRAME) + 0.5 + top) / sy
    gx = (u * msx - mleft) / crop * 2 - 1
    gy = (v * msy - mtop) / crop * 2 - 1
    grid = torch.from_numpy(np.stack(np.meshgrid(gx, gy), -1)).float()[None].to(cam.device)
    out = F.grid_sample(cam[None, None].float(), grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out[0, 0].cpu().numpy()


class TextEncoder:
    """``forward_contrastive(texts)`` -> (embeddings, None, None), as sentence_bank expects."""

    def __init__(self, fn):
        self.fn = fn

    def forward_contrastive(self, texts):
        return self.fn(list(texts)), None, None


class BaselineFilm:
    def __init__(self, path: str, tensor):
        from PIL import Image

        from grounding_eval.eval_grounding import FIT

        with Image.open(path) as img:
            self.orig = img.size
            disp = img.copy()
        disp.thumbnail((FIT, FIT))
        self.disp = disp.size
        self.path = path
        self.tensor = tensor


class BaselineGrounder:
    """``encode(prior, current)`` -> (global [D], patches [H, W, D]), both unit-norm."""

    resize_short = 512
    crop = 448

    def __init__(self, device):
        self.device = device
        self.text_encoder = None
        self.phrases = None
        self.token_map = None

    def image(self, path: str):
        raise NotImplementedError

    def encode(self, prior: BaselineFilm, current: BaselineFilm):
        raise NotImplementedError

    def load(self, path: str) -> BaselineFilm:
        return BaselineFilm(path, self.image(path))

    def text(self, prior, current, finding: str, target: Optional[int], method: str):
        import torch

        from grounding_eval.eval_grounding import N_CLS, TEMPERATURE, minmax, phrase_vector

        with torch.no_grad():
            g, p = self.encode(prior, current)
            t_all = self.phrases(finding)
            logits = (t_all[:N_CLS] @ g) / TEMPERATURE
            t = int(logits.argmax()) if target is None else target
            v = phrase_vector(t_all, t, method)
            sims = p @ v
        return None, minmax(sims), logits.cpu().numpy(), t

    def to_frame(self, cam, film: BaselineFilm):
        return None if cam is None else crop_to_frame(cam, film.orig, self.resize_short, self.crop)


def _unit(x):
    import torch.nn.functional as F

    return F.normalize(x.float(), dim=-1)


class TILAGrounder(BaselineGrounder):
    def __init__(self, device, ckpt: Optional[str] = None):
        import torch

        from infer_tila import TILA_REPO, load_model

        super().__init__(device)
        self.m = load_model(device, torch.float32, checkpoint=ckpt or TILA_REPO)
        self.text_encoder = TextEncoder(lambda texts: _unit(self.m.encode_text(texts)))

    def image(self, path):
        import torch

        from infer_tila import _load_image

        return _load_image(path, self.device, torch.float32)

    def encode(self, prior, current):
        out = self.m.image_encoder(current.tensor[None], prior.tensor[None])
        p = _unit(out.projected_patch_embeddings[0].permute(1, 2, 0))
        g = getattr(out, "projected_global_embedding", None)
        g = p.mean((0, 1)) if g is None else g[0]
        return _unit(g), p


class BioViLGrounder(BaselineGrounder):
    crop = 480

    def __init__(self, device, ckpt: Optional[str] = None):
        from infer_biovil import DEFAULT_BIOVIL_IMAGE_CKPT, DEFAULT_CXR_BERT_PATH, load_model

        super().__init__(device)
        self.m, tok, txt = load_model(device, image_ckpt=ckpt or DEFAULT_BIOVIL_IMAGE_CKPT,
                                      cxr_bert_path=DEFAULT_CXR_BERT_PATH)

        def encode_text(texts):
            enc = tok(texts, return_tensors="pt", padding="longest", truncation=True,
                      max_length=128).to(device)
            return _unit(txt.get_projected_text_embeddings(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                normalize_embeddings=True))

        self.text_encoder = TextEncoder(encode_text)

    def image(self, path):
        from infer_biovil import _load_image

        return _load_image(path, self.device)

    def encode(self, prior, current):
        out = self.m(current.tensor[None])
        p = _unit(out.projected_patch_embeddings[0].permute(1, 2, 0))
        return _unit(out.projected_global_embedding[0]), p


class BiomedCLIPGrounder(BaselineGrounder):
    resize_short = 256
    crop = 224

    def __init__(self, device, ckpt: Optional[str] = None):
        from infer_biomedclip import DEFAULT_MODEL_ID, load_model

        super().__init__(device)
        self.m, tok = load_model(device, model_id=ckpt or DEFAULT_MODEL_ID)

        def encode_text(texts):
            tokens = tok(texts)
            if isinstance(tokens, dict):
                feats = self.m.encode_text(**{k: v.to(device) for k, v in tokens.items()})
            else:
                feats = self.m.encode_text(tokens.to(device))
            return _unit(feats)

        self.text_encoder = TextEncoder(encode_text)

    def image(self, path):
        from infer_biomedclip import _load_image

        return _load_image(path, self.device)

    def encode(self, prior, current):
        vis = self.m.visual
        trunk = vis.trunk
        tokens = trunk.forward_features(current.tensor[None])
        tokens = trunk.fc_norm(tokens) if hasattr(trunk, "fc_norm") else tokens
        k = getattr(trunk, "num_prefix_tokens", 1)
        side = int(round((tokens.shape[1] - k) ** 0.5))
        p = _unit(vis.head(tokens[0, k:])).view(side, side, -1)
        return _unit(vis.head(tokens[0, 0])), p


class ALTAGrounder(BaselineGrounder):
    resize_short = 256
    crop = 224

    def __init__(self, device, ckpt: Optional[str] = None):
        import torch

        from infer_alta import DEFAULT_CHECKPOINT_PATH, _tokenize, load_model

        super().__init__(device)
        path = ckpt or DEFAULT_CHECKPOINT_PATH
        if not os.path.isfile(path):
            raise SystemExit(f"[grounding] ALTA checkpoint not found at {path}; infer_alta would "
                             "fall back to random weights. Pass --ckpt.")
        self.m = load_model(device, checkpoint_path=path)

        def encode_text(texts):
            inp = _tokenize(texts, device)
            words = self.m.bert_encoder(None, inp["input_ids"], inp["attention_mask"],
                                        output_cls_projected_embedding=True
                                        ).local_proected_embedding
            n = inp["attention_mask"].sum(1) - 1
            keep = torch.arange(words.shape[1], device=device)[None] < n[:, None]
            words = _unit(words) * keep[..., None]
            return _unit(words.sum(1) / n[:, None])

        self.text_encoder = TextEncoder(encode_text)

    def image(self, path):
        from infer_alta import _load_image

        return _load_image(path, self.device)

    def encode(self, prior, current):
        """infer_alta._encode_pair_global up to the final norm; slot 0 is the current film."""
        import torch
        from einops import rearrange

        m = self.m
        curr, prev = current.tensor, prior.tensor
        zero = torch.zeros_like(curr)
        x = torch.cat([im[None, :, None] for im in (curr, zero, prev, zero)], dim=2)
        missing = (x.sum(dim=[1, 3, 4]) == 0).reshape(-1)
        x = m.patch_embed(rearrange(x, "b c t h w -> (b t) c h w"))
        x[missing] = m.missing_token.to(x.dtype).repeat(int(missing.sum()), 1, 1)
        x = x + m.pos_embed[:, 1:, :]
        cls = (m.cls_token + m.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        n = x.shape[1]
        x = rearrange(x, "(b t) n d -> (b n) t d", t=4)
        x = (x + m.view_embedding.repeat(1, 2, 1)
             + torch.cat([m.temporal_embedding[:, 0:1, :].repeat(1, 2, 1),
                          m.temporal_embedding[:, 1:, :].repeat(1, 2, 1)], dim=1))
        x = rearrange(x, "(b n) t d -> (b t) n d", n=n)
        for blk in m.blocks:
            x = blk(x)
        x = m.norm(x)
        side = int(round((n - 1) ** 0.5))
        p = _unit(m.local_projection_head(x[0, 1:])).view(side, side, -1)
        return _unit(p.mean((0, 1))), p


def load_baseline(name: str, device, ckpt: Optional[str] = None) -> BaselineGrounder:
    from baseline_setmatch import DEFAULT_BASELINES_ROOT, add_baselines_root

    add_baselines_root(os.environ.get("BASELINES_ROOT", DEFAULT_BASELINES_ROOT))
    cls = {"tila": TILAGrounder, "biovil": BioViLGrounder, "biomedclip": BiomedCLIPGrounder,
           "alta": ALTAGrounder}[name]
    model = cls(device, ckpt)
    for prm in model.m.parameters():
        prm.requires_grad = False
    return model
