"""Load CheXTemporal ``filtered_masks`` and map them onto JEPA patch weights.

Mask files live under::

    CheXTemporal/filtered_masks/
        chexpert/train/{patient}/{study}/{finding}__{view}.json
        mimic/{pXX}/{subject}/{study}/{finding}__{dicom_id}.json

Each JSON is a tiny COCO-style document with RLE ``segmentation`` masks
in **original image** resolution. Training images go through
``Resize(512) → CenterCrop(448) → synced affine aug``; masks follow the
same *geometric* transforms (nearest-neighbor; no brightness/contrast)
and are then average-pooled onto the BioViL-T 14×14 patch grid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


PATCH_GRID = 14
MODEL_INPUT_SIZE = 448
RESIZE_SHORT = 512
N_PATCHES = PATCH_GRID * PATCH_GRID


def finding_to_mask_key(finding: str) -> str:
    """``lung opacity`` → ``lung_opacity`` (filename convention)."""
    return str(finding).strip().lower().replace(" ", "_")


def normalize_parent_image_rel(dataset: str, parent_image: str) -> Optional[str]:
    """Return a ``{dataset}/.../{stem}.ext`` relative path for mask join."""
    if dataset not in ("chexpert", "mimic"):
        return None

    rel = str(parent_image).strip().lstrip("/")
    for prefix in ("mimic/", "chexpert/", "rexgradient/"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break

    if dataset == "chexpert":
        if not rel.startswith("train/"):
            rel = f"train/{rel}"
        return f"chexpert/{rel}"

    return f"mimic/{rel}"


def resolve_mask_json_path(
    masks_root: Union[str, Path],
    dataset: str,
    parent_image_curr: str,
    finding: str,
) -> Optional[Path]:
    """Locate ``{finding}__{image_stem}.json`` under ``filtered_masks``."""
    if not finding or not parent_image_curr:
        return None

    rel = normalize_parent_image_rel(dataset, parent_image_curr)
    if rel is None:
        return None

    rel_path = Path(rel)
    candidate = (
        Path(masks_root)
        / rel_path.parent
        / f"{finding_to_mask_key(finding)}__{rel_path.stem}.json"
    )
    if candidate.is_file():
        return candidate
    return None


def _decode_rle_to_mask(segmentation: Dict[str, Any]) -> np.ndarray:
    """Decode one COCO RLE dict to a boolean ``(H, W)`` mask."""
    size = segmentation["size"]
    h, w = int(size[0]), int(size[1])
    counts = segmentation["counts"]

    try:
        from pycocotools import mask as mask_utils

        rle = {"size": [h, w], "counts": counts}
        if isinstance(counts, str):
            rle["counts"] = counts.encode("utf-8")
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        return decoded.astype(bool)
    except Exception:
        pass

    if isinstance(counts, (list, tuple)):
        flat = np.zeros(h * w, dtype=np.uint8)
        idx = 0
        val = 0
        for run in counts:
            run = int(run)
            if run > 0:
                flat[idx:idx + run] = val
            idx += run
            val = 1 - val
        return flat.reshape((h, w), order="F").astype(bool)

    raise RuntimeError(
        "Compressed COCO RLE requires pycocotools, or pass uncompressed "
        "counts as a list. Install with: pip install pycocotools"
    )


def load_union_mask_hw(mask_json_path: Union[str, Path]) -> np.ndarray:
    """Load a filtered_masks JSON and union all annotation RLEs → ``(H, W)``."""
    with open(mask_json_path, "r") as f:
        doc = json.load(f)

    anns = doc.get("annotations") or []
    if not anns:
        raise ValueError(f"No annotations in {mask_json_path}")

    union = None
    for ann in anns:
        seg = ann.get("segmentation")
        if seg is None or isinstance(seg, list):
            continue
        m = _decode_rle_to_mask(seg)
        union = m if union is None else np.logical_or(union, m)

    if union is None:
        raise ValueError(f"No RLE segmentations in {mask_json_path}")
    return union


def mask_hw_to_patch_weights(
    mask_hw: np.ndarray,
    aug_params: Optional[Dict[str, float]] = None,
    patch_grid: int = PATCH_GRID,
    model_size: int = MODEL_INPUT_SIZE,
    resize_short: int = RESIZE_SHORT,
) -> torch.Tensor:
    """Warp an original-resolution mask into ``(patch_grid**2,)`` weights."""
    if mask_hw.dtype != np.uint8:
        mask_u8 = (mask_hw.astype(bool).astype(np.uint8)) * 255
    else:
        mask_u8 = mask_hw

    m = Image.fromarray(mask_u8, mode="L")
    m = TF.resize(
        m,
        resize_short,
        interpolation=TF.InterpolationMode.NEAREST,
    )
    m = TF.center_crop(m, model_size)

    if aug_params is not None:
        m = TF.affine(
            m,
            angle=aug_params["angle"],
            translate=(0, 0),
            scale=1.0,
            shear=[aug_params["shear"], 0.0],
            interpolation=TF.InterpolationMode.NEAREST,
            fill=0,
        )

    t = TF.to_tensor(m)  # (1, 448, 448) in [0, 1]
    kernel = model_size // patch_grid
    if kernel * patch_grid != model_size:
        raise ValueError(
            f"model_size={model_size} not divisible by patch_grid={patch_grid}"
        )

    pooled = F.avg_pool2d(t.unsqueeze(0), kernel_size=kernel, stride=kernel)
    return pooled.view(-1).float()


def zero_patch_weights(
    n_patches: int = N_PATCHES,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    return torch.zeros(n_patches, dtype=torch.float32, device=device)


def load_prog_patch_weights(
    masks_root: Union[str, Path],
    dataset: str,
    parent_image_curr: str,
    finding: str,
    aug_params: Optional[Dict[str, float]] = None,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, bool]:
    """Return ``(weights, used_mask)`` for one finding on the current image."""
    path = resolve_mask_json_path(
        masks_root, dataset, parent_image_curr, finding
    )
    if path is None:
        return zero_patch_weights(), False

    try:
        mask_hw = load_union_mask_hw(path)
        weights = mask_hw_to_patch_weights(mask_hw, aug_params=aug_params)
    except Exception:
        return zero_patch_weights(), False

    if float(weights.sum()) <= min_weight_sum:
        return zero_patch_weights(), False

    return weights, True


def load_union_findings_patch_weights(
    masks_root: Union[str, Path],
    dataset: str,
    parent_image_curr: str,
    findings: Sequence[str],
    aug_params: Optional[Dict[str, float]] = None,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, bool]:
    """Soft 14×14 weights from the **union** of all findings' masks.

    Decodes each available ``filtered_masks`` JSON for ``findings`` on
    ``parent_image_curr``, OR-merges pixel masks when shapes match (else
    takes element-wise max of independently warped patch weights), then
    average-pools to float coverage in ``[0, 1]`` per patch.
    """
    if not findings:
        return zero_patch_weights(), False

    union_hw = None
    warped_fallback = []

    for finding in findings:
        if not finding:
            continue
        path = resolve_mask_json_path(
            masks_root, dataset, parent_image_curr, finding
        )
        if path is None:
            continue
        try:
            mask_hw = load_union_mask_hw(path)
        except Exception:
            continue

        if union_hw is None:
            union_hw = mask_hw.astype(bool)
        elif mask_hw.shape == union_hw.shape:
            union_hw = np.logical_or(union_hw, mask_hw)
        else:
            # Rare: inconsistent native sizes — warp each and max later.
            try:
                warped_fallback.append(
                    mask_hw_to_patch_weights(mask_hw, aug_params=aug_params)
                )
            except Exception:
                continue

    if union_hw is not None:
        try:
            weights = mask_hw_to_patch_weights(union_hw, aug_params=aug_params)
        except Exception:
            weights = zero_patch_weights()
        for w in warped_fallback:
            weights = torch.maximum(weights, w)
    elif warped_fallback:
        weights = warped_fallback[0]
        for w in warped_fallback[1:]:
            weights = torch.maximum(weights, w)
    else:
        return zero_patch_weights(), False

    if float(weights.sum()) <= min_weight_sum:
        return zero_patch_weights(), False

    return weights, True


def default_masks_root(chextemporal_dir: Optional[str] = None) -> str:
    """``$CHEXTEMPORAL_DIR/filtered_masks`` (or local CheXTemporal default)."""
    root = chextemporal_dir or os.environ.get(
        "CHEXTEMPORAL_DIR",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "CheXTemporal",
        ),
    )
    return os.path.join(root, "filtered_masks")
