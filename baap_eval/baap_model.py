"""BAAP v1 inference in plain PyTorch.

Rebuilds the parts of ``AnatomyTemporalFineTuner`` (github.com/chenyuming052/BAAP,
``baap/experiments/code/anatomy_temporal_finetuner.py``) that the released
``baap-v1-paperbest.ckpt`` uses at test time: the MedST ViT-B/16 frontal path,
masked-average ROI pooling over patch tokens, the ROI projection, the anatomy
embedding and the 3-way MLP classifier. Parameter names match the checkpoint,
so its weights load without PyTorch Lightning, timm or the DeiT download that
BAAP's own ``ImageEncoder`` does at construction.

Preprocessing matches BAAP's evaluator: grayscale, aspect-preserving resize to
256 with zero padding, CenterCrop(224), Normalize(0.5, 0.5). Boxes are given in
that 224 frame.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

LABELS = ["improving", "stable", "worsening"]
IMG_SIZE = 224
RESIZE = 256
PATCH = 16
GRID = IMG_SIZE // PATCH

# Chest ImaGenome anatomy order used by the checkpoint's anatomy embedding.
ANATOMY_LIST = [
    "right lung", "left lung", "left lower lung zone", "cardiac silhouette",
    "right hilar structures", "left hilar structures", "right lower lung zone",
    "right costophrenic angle", "left costophrenic angle", "mediastinum",
    "right mid lung zone", "left mid lung zone", "upper mediastinum",
    "right upper lung zone", "right apical zone", "left upper lung zone",
    "aortic arch", "left apical zone", "right chest wall", "left hemidiaphragm",
    "right hemidiaphragm", "left chest wall", "spine", "abdomen", "neck",
    "trachea", "right clavicle", "left clavicle", "right shoulder",
    "left shoulder", "right arm", "svc", "left arm", "right atrium",
    "carina", "right breast", "left breast", "cavoatrial junction",
]
ANATOMY_TO_IDX = {name: i for i, name in enumerate(ANATOMY_LIST)}
ANATOMY_UNK_IDX = len(ANATOMY_LIST)


# ----------------------------------------------------------------------
# Image and box preprocessing
# ----------------------------------------------------------------------
def _to_uint8_gray(img: Image.Image) -> np.ndarray:
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(img, dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
        return a.astype(np.uint8)
    return np.asarray(img.convert("L"), dtype=np.uint8)


def resized_shape(h: int, w: int, scale: int = RESIZE) -> Tuple[int, int, int, int]:
    """(new_h, new_w, top, left) of BAAP's ``resize_img`` before CenterCrop."""
    if h >= w:
        nh, nw = scale, int(w * (scale / float(h)))
        return nh, nw, 0, (scale - nw) // 2
    nh, nw = int(h * (scale / float(w))), scale
    return nh, nw, (scale - nh) // 2, 0


def load_image(path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """[3, 224, 224] tensor in BAAP's frame and the original (h, w)."""
    try:
        import cv2

        gray = cv2.imread(str(path), 0)
        if gray is None:
            raise FileNotFoundError(path)
        resize = lambda a, nh, nw: cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    except ImportError:
        gray = _to_uint8_gray(Image.open(path))
        resize = lambda a, nh, nw: np.asarray(
            Image.fromarray(a).resize((nw, nh), Image.BOX), dtype=np.uint8)
    h, w = gray.shape[:2]
    nh, nw, top, left = resized_shape(h, w)
    canvas = np.zeros((RESIZE, RESIZE), dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resize(gray, nh, nw)
    off = (RESIZE - IMG_SIZE) // 2
    crop = canvas[off:off + IMG_SIZE, off:off + IMG_SIZE].astype(np.float32) / 255.0
    x = torch.from_numpy(crop).unsqueeze(0).repeat(3, 1, 1)
    return (x - 0.5) / 0.5, (h, w)


def frac_box_to_crop(box: Sequence[float], hw: Tuple[int, int]) -> List[float]:
    """Box as fractions of the original image -> BAAP 224 crop coordinates."""
    nh, nw, top, left = resized_shape(*hw)
    off = (RESIZE - IMG_SIZE) / 2
    x1, y1, x2, y2 = box
    out = [x1 * nw + left - off, y1 * nh + top - off, x2 * nw + left - off, y2 * nh + top - off]
    return [min(float(IMG_SIZE), max(0.0, v)) for v in out]


def imagenome224_to_crop(box: Sequence[float]) -> List[float]:
    """Chest ImaGenome 224 (resize + pad) coordinates -> BAAP 224 crop coordinates."""
    s, off = RESIZE / IMG_SIZE, (RESIZE - IMG_SIZE) / 2
    return [min(float(IMG_SIZE), max(0.0, v * s - off)) for v in box]


def box_patch_mask(boxes: torch.Tensor) -> torch.Tensor:
    """[N, 4] boxes in the 224 frame -> [N, 196] patch masks (BAAP ``bbox_to_patch_mask``)."""
    def cell(v):
        return torch.clamp(torch.div(v, PATCH, rounding_mode="floor"), 0, GRID - 1)

    c0, r0 = cell(boxes[:, 0].long()), cell(boxes[:, 1].long())
    c1, r1 = cell(boxes[:, 2].long() - 1), cell(boxes[:, 3].long() - 1)
    c1, r1 = torch.max(c0, c1), torch.max(r0, r1)
    idx = torch.arange(GRID, device=boxes.device)
    rows = (idx[None] >= r0[:, None]) & (idx[None] <= r1[:, None])
    cols = (idx[None] >= c0[:, None]) & (idx[None] <= c1[:, None])
    return (rows.unsqueeze(2) & cols.unsqueeze(1)).reshape(-1, GRID * GRID)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=PATCH, stride=PATCH)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """MedST ViT-B/16, frontal path only (the lateral MLPs are never used here)."""

    def __init__(self, dim: int = 768, depth: int = 12, heads: int = 12):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, GRID * GRID + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:]


class BAAP(nn.Module):
    def __init__(self, emb_dim: int = 128, hidden_dim: int = 256, n_classes: int = 3):
        super().__init__()
        self.vit = ViT()
        self.roi_projection = nn.Module()
        self.roi_projection.head = nn.Sequential(
            nn.Linear(768, emb_dim * 4), nn.BatchNorm1d(emb_dim * 4), nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim), nn.BatchNorm1d(emb_dim, affine=False),
        )
        self.anatomy_embedding = nn.Embedding(len(ANATOMY_LIST) + 1, emb_dim)
        self.classifier = nn.Module()
        self.classifier.block = nn.Sequential(
            nn.Dropout(), nn.Linear(emb_dim * 4, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(), nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2), nn.GELU(), nn.Dropout(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    @torch.no_grad()
    def patches(self, img: torch.Tensor) -> torch.Tensor:
        """[B, 3, 224, 224] -> [B, 196, 768] patch tokens."""
        return self.vit(img)

    @torch.no_grad()
    def logits(self, prior_patches, current_patches, prior_boxes, current_boxes, anatomy_idx):
        """One pair: patches [196, 768], boxes [N, 4] in the 224 frame -> [N, 3] logits."""
        return self.roi_logits(prior_patches, current_patches, prior_boxes, current_boxes, anatomy_idx)

    def roi_logits(self, prior_patches, current_patches, prior_boxes, current_boxes, anatomy_idx):
        def pool(p, boxes):
            m = box_patch_mask(boxes).float()
            return (m @ p) / m.sum(1, keepdim=True).clamp(min=1.0)

        proj = self.roi_projection.head
        prior = F.normalize(proj(pool(prior_patches, prior_boxes)), dim=-1)
        current = F.normalize(proj(pool(current_patches, current_boxes)), dim=-1)
        fused = torch.cat([prior, current, current - prior, self.anatomy_embedding(anatomy_idx)], dim=-1)
        return self.classifier.block(fused)


def load_baap(ckpt_path: str, device) -> Tuple[BAAP, Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = dict(ckpt.get("hyper_parameters", {}))
    for key, want in (("img_encoder", "vit_base"), ("fusion_type", "concat_diff"),
                      ("use_anatomy_emb", True)):
        if hp.get(key, want) != want:
            raise SystemExit(f"[baap] {ckpt_path}: {key}={hp.get(key)!r}, this loader needs {want!r}")
    for key in ("use_crda", "use_attn_roi"):
        if hp.get(key):
            raise SystemExit(f"[baap] {ckpt_path}: {key}=True is not supported by this loader")
    if hp.get("roi_mode", "roi") != "roi":
        raise SystemExit(f"[baap] {ckpt_path}: roi_mode={hp.get('roi_mode')!r}, need 'roi'")

    model = BAAP(emb_dim=int(hp.get("emb_dim", 128)), hidden_dim=int(hp.get("hidden_dim", 256)),
                 n_classes=int(hp.get("num_classes", 3)))
    sd = ckpt["state_dict"]
    mine = {}
    for k, v in sd.items():
        if k.startswith("backbone.model."):
            rest = k[len("backbone.model."):]
            if ".mlp_l." in rest or ".norm2_l." in rest:
                continue
            mine["vit." + rest] = v
        elif k.split(".")[0] in ("roi_projection", "anatomy_embedding", "classifier"):
            mine[k] = v
    missing, unexpected = model.load_state_dict(mine, strict=False)
    if missing or unexpected:
        raise SystemExit(f"[baap] weight mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    meta = dict(ckpt.get("baap_metadata", {}))
    return model.to(device).eval(), {"hyper_parameters": hp, "metadata": meta,
                                     "epoch": ckpt.get("epoch")}


def aggregate(logits: torch.Tensor, how: str = "softmax") -> np.ndarray:
    """[N, 3] per-anatomy logits -> 3 log-probabilities for the pair."""
    if how == "softmax":
        p = logits.softmax(-1).mean(0)
    elif how == "mean_logits":
        p = logits.mean(0).softmax(-1)
    else:
        raise ValueError(how)
    return torch.log(p.clamp(min=1e-12)).double().cpu().numpy()


__all__ = [
    "ANATOMY_TO_IDX", "ANATOMY_UNK_IDX", "BAAP", "IMG_SIZE", "LABELS", "aggregate",
    "frac_box_to_crop", "imagenome224_to_crop", "load_baap", "load_image",
]
