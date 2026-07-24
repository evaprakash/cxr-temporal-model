"""Load CheXTemporal soft masks and map them onto JEPA patch weights.

Two layouts are supported:

1. Finding-filtered masks (``filtered_masks/``)::

       CheXTemporal/filtered_masks/
           chexpert/train/{patient}/{study}/{finding}__{view}.json
           mimic/{pXX}/{subject}/{study}/{finding}__{dicom_id}.json

2. Fixed-anatomy masks (``filtered_masks_anatomy/``)::

       CheXTemporal/filtered_masks_anatomy/
           chexpert/train/{patient}/{study}/{view}.json
           mimic/{pXX}/{subject}/{study}/{dicom_id}.json

   Each retained JSON has the same 22 CXAS anatomy categories.

Each JSON is a tiny COCO-style document with RLE ``segmentation`` masks
in **original image** resolution. Training images go through
``Resize(512) → CenterCrop(448) → synced affine aug``; masks follow the
same *geometric* transforms (nearest-neighbor; no brightness/contrast)
and are then average-pooled onto the BioViL-T 14×14 patch grid as
**float** coverage weights in ``[0, 1]``.

Anatomy dual-mask JEPA pools predicted ``ẑ`` with the **prior** anatomy
mask and ``z_cur`` with the **current** anatomy mask, per category.
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
    parent_image: str,
    finding: str,
) -> Optional[Path]:
    """Locate ``{finding}__{image_stem}.json`` under ``filtered_masks``."""
    if not finding or not parent_image:
        return None

    rel = normalize_parent_image_rel(dataset, parent_image)
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


def category_names_in_mask_json(mask_json_path: Union[str, Path]) -> frozenset:
    """Return the set of ``categories[].name`` present in a filtered mask JSON.

    Only names that have at least one RLE annotation are counted, so empty
    category stubs do not inflate the set.
    """
    with open(mask_json_path, "r") as f:
        doc = json.load(f)
    id_to_name = {
        int(c["id"]): str(c["name"])
        for c in (doc.get("categories") or [])
        if "id" in c and "name" in c
    }
    present = set()
    for ann in doc.get("annotations") or []:
        seg = ann.get("segmentation")
        if seg is None or isinstance(seg, list):
            continue
        cid = ann.get("category_id")
        if cid is None:
            continue
        name = id_to_name.get(int(cid))
        if name is not None:
            present.add(name)
    return frozenset(present)


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
    parent_image: str,
    finding: str,
    aug_params: Optional[Dict[str, float]] = None,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, bool]:
    """Return ``(weights, used_mask)`` for one finding on one image."""
    path = resolve_mask_json_path(
        masks_root, dataset, parent_image, finding
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


def _union_paths_to_patch_weights(
    paths: Sequence[Path],
    aug_params: Optional[Dict[str, float]] = None,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, bool]:
    """OR-merge RLE masks from ``paths`` and warp to 14×14 float weights."""
    if not paths:
        return zero_patch_weights(), False

    union_hw = None
    warped_fallback = []

    for path in paths:
        try:
            mask_hw = load_union_mask_hw(path)
        except Exception:
            continue

        if union_hw is None:
            union_hw = mask_hw.astype(bool)
        elif mask_hw.shape == union_hw.shape:
            union_hw = np.logical_or(union_hw, mask_hw)
        else:
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


def load_union_findings_patch_weights(
    masks_root: Union[str, Path],
    dataset: str,
    parent_image: str,
    findings: Sequence[str],
    aug_params: Optional[Dict[str, float]] = None,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, bool]:
    """Soft 14×14 weights from the **union** of all findings' masks on one image."""
    if not findings:
        return zero_patch_weights(), False

    paths = []
    for finding in findings:
        if not finding:
            continue
        path = resolve_mask_json_path(masks_root, dataset, parent_image, finding)
        if path is not None:
            paths.append(path)
    return _union_paths_to_patch_weights(
        paths, aug_params=aug_params, min_weight_sum=min_weight_sum
    )


def load_dual_findings_patch_weights(
    masks_root: Union[str, Path],
    dataset: str,
    parent_image_prev: str,
    parent_image_curr: str,
    findings: Sequence[str],
    aug_params: Optional[Dict[str, float]] = None,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Prior + current soft weights for dual-mask pooled JEPA.

    Returns ``(prior_weights, current_weights, active)``.

    ``active`` is True only when:
      1. The set of findings with a readable mask JSON is identical (and
         non-empty) for prior and current, and
      2. The union of annotated anatomy category names across those JSONs
         is identical for prior and current, and
      3. Both warped weight tensors have non-trivial mass.

    Prior weights pool ẑ (prior-grid residual prediction); current weights
    pool z_cur (EMA current target).
    """
    z = zero_patch_weights()
    if not findings:
        return z, z, False

    prior_by_finding: Dict[str, Path] = {}
    curr_by_finding: Dict[str, Path] = {}
    prior_cats: set = set()
    curr_cats: set = set()

    for finding in findings:
        if not finding:
            continue
        key = finding_to_mask_key(finding)
        p_path = resolve_mask_json_path(
            masks_root, dataset, parent_image_prev, finding
        )
        c_path = resolve_mask_json_path(
            masks_root, dataset, parent_image_curr, finding
        )
        if p_path is not None:
            try:
                cats = category_names_in_mask_json(p_path)
            except Exception:
                cats = frozenset()
            if cats:
                prior_by_finding[key] = p_path
                prior_cats.update(cats)
        if c_path is not None:
            try:
                cats = category_names_in_mask_json(c_path)
            except Exception:
                cats = frozenset()
            if cats:
                curr_by_finding[key] = c_path
                curr_cats.update(cats)

    prior_keys = frozenset(prior_by_finding)
    curr_keys = frozenset(curr_by_finding)
    if not prior_keys or prior_keys != curr_keys:
        return z, z, False
    if not prior_cats or frozenset(prior_cats) != frozenset(curr_cats):
        return z, z, False

    # Same finding keys on both sides — keep a stable order for union.
    ordered = sorted(prior_keys)
    prior_paths = [prior_by_finding[k] for k in ordered]
    curr_paths = [curr_by_finding[k] for k in ordered]

    prior_w, prior_ok = _union_paths_to_patch_weights(
        prior_paths, aug_params=aug_params, min_weight_sum=min_weight_sum
    )
    curr_w, curr_ok = _union_paths_to_patch_weights(
        curr_paths, aug_params=aug_params, min_weight_sum=min_weight_sum
    )
    if not (prior_ok and curr_ok):
        return z, z, False

    return prior_w, curr_w, True


# Fixed 22 CXAS categories retained by ``filter_masks_by_anatomy_map.py``.
# Order is alphabetical (matches ``allowed_categories.json``).
REQUIRED_CXAS_ANATOMIES: tuple[str, ...] = (
    "aortic arch",
    "cardiomediastinum",
    "clavicle left",
    "clavicle right",
    "heart",
    "heart atrium right",
    "left apical zone lung",
    "left hemidiaphragm",
    "left lung",
    "left lung base",
    "left mid zone lung",
    "left upper zone lung",
    "right apical zone lung",
    "right hemidiaphragm",
    "right lung",
    "right lung base",
    "right mid zone lung",
    "right upper zone lung",
    "spine",
    "trachea",
    "tracheal bifurcation",
    "upper mediastinum",
)
N_ANATOMY_MASKS = len(REQUIRED_CXAS_ANATOMIES)


def _chextemporal_root(chextemporal_dir: Optional[str] = None) -> str:
    return chextemporal_dir or os.environ.get(
        "CHEXTEMPORAL_DIR",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "CheXTemporal",
        ),
    )


def default_masks_root(chextemporal_dir: Optional[str] = None) -> str:
    """``$CHEXTEMPORAL_DIR/filtered_masks`` (finding-filtered)."""
    return os.path.join(_chextemporal_root(chextemporal_dir), "filtered_masks")


def default_anatomy_masks_root(chextemporal_dir: Optional[str] = None) -> str:
    """``$CHEXTEMPORAL_DIR/filtered_masks_anatomy`` (fixed 22 CXAS)."""
    return os.path.join(
        _chextemporal_root(chextemporal_dir), "filtered_masks_anatomy"
    )


def resolve_anatomy_mask_json_path(
    masks_root: Union[str, Path],
    dataset: str,
    parent_image: str,
) -> Optional[Path]:
    """Locate ``{image_stem}.json`` under ``filtered_masks_anatomy``."""
    if not parent_image:
        return None

    rel = normalize_parent_image_rel(dataset, parent_image)
    if rel is None:
        return None

    rel_path = Path(rel)
    candidate = Path(masks_root) / rel_path.parent / f"{rel_path.stem}.json"
    if candidate.is_file():
        return candidate
    return None


def load_per_category_masks_hw(
    mask_json_path: Union[str, Path],
    categories: Sequence[str] = REQUIRED_CXAS_ANATOMIES,
) -> Optional[Dict[str, np.ndarray]]:
    """Decode one RLE mask per required category; None if any are missing."""
    with open(mask_json_path, "r") as f:
        doc = json.load(f)

    id_to_name = {
        int(c["id"]): str(c["name"])
        for c in (doc.get("categories") or [])
        if "id" in c and "name" in c
    }
    want = set(categories)
    by_name: Dict[str, np.ndarray] = {}

    for ann in doc.get("annotations") or []:
        seg = ann.get("segmentation")
        if seg is None or isinstance(seg, list):
            continue
        cid = ann.get("category_id")
        if cid is None:
            continue
        name = id_to_name.get(int(cid))
        if name is None or name not in want:
            continue
        try:
            m = _decode_rle_to_mask(seg)
        except Exception:
            continue
        if name in by_name:
            by_name[name] = np.logical_or(by_name[name], m)
        else:
            by_name[name] = m

    if any(c not in by_name for c in categories):
        return None
    return by_name


def anatomy_categories_present(
    mask_json_path: Union[str, Path],
    categories: Sequence[str] = REQUIRED_CXAS_ANATOMIES,
) -> bool:
    """True iff every required category has at least one RLE annotation.

    Does **not** decode RLEs — cheap enough to filter the silver corpus.
    """
    try:
        with open(mask_json_path, "r") as f:
            doc = json.load(f)
    except Exception:
        return False

    id_to_name = {
        int(c["id"]): str(c["name"])
        for c in (doc.get("categories") or [])
        if "id" in c and "name" in c
    }
    present = set()
    for ann in doc.get("annotations") or []:
        seg = ann.get("segmentation")
        if seg is None or isinstance(seg, list):
            continue
        cid = ann.get("category_id")
        if cid is None:
            continue
        name = id_to_name.get(int(cid))
        if name is not None:
            present.add(name)
    return all(c in present for c in categories)


# Cache: (masks_root, dataset, parent_image) -> has full 22-category inventory.
_ANATOMY_INVENTORY_CACHE: Dict[tuple, bool] = {}


def image_has_full_anatomy_inventory(
    masks_root: Union[str, Path],
    dataset: str,
    parent_image: str,
    categories: Sequence[str] = REQUIRED_CXAS_ANATOMIES,
) -> bool:
    """Whether one image's anatomy JSON covers all required CXAS classes."""
    key = (str(masks_root), dataset, str(parent_image))
    cached = _ANATOMY_INVENTORY_CACHE.get(key)
    if cached is not None:
        return cached

    path = resolve_anatomy_mask_json_path(masks_root, dataset, parent_image)
    ok = path is not None and anatomy_categories_present(path, categories)
    _ANATOMY_INVENTORY_CACHE[key] = ok
    return ok


def pair_has_full_anatomy_inventory(
    masks_root: Union[str, Path],
    dataset: str,
    parent_image_prev: str,
    parent_image_curr: str,
    categories: Sequence[str] = REQUIRED_CXAS_ANATOMIES,
) -> bool:
    """True iff **both** prior and current have the full anatomy inventory."""
    return image_has_full_anatomy_inventory(
        masks_root, dataset, parent_image_prev, categories
    ) and image_has_full_anatomy_inventory(
        masks_root, dataset, parent_image_curr, categories
    )


def load_dual_anatomy_patch_weights(
    masks_root: Union[str, Path],
    dataset: str,
    parent_image_prev: str,
    parent_image_curr: str,
    categories: Sequence[str] = REQUIRED_CXAS_ANATOMIES,
    aug_params: Optional[Dict[str, float]] = None,
    min_weight_sum: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Per-anatomy soft weights for dual-mask pooled JEPA.

    Returns ``(prior_weights, current_weights, active)`` with shapes
    ``(A, N)``, ``(A, N)``, bool where ``A = len(categories)`` (22).

    Row order is exactly ``categories`` (default:
    ``REQUIRED_CXAS_ANATOMIES``) so every sample shares one anatomy axis.

    ``active`` is True only when:
      1. Both prior and current have a readable anatomy JSON, and
      2. Both JSONs contain all ``categories`` with RLE annotations, and
      3. Every warped weight tensor (prior + current, all A) has mass.

    Prior weights pool ``ẑ`` (prior-grid residual); current weights pool
    ``z_cur`` (EMA current target).
    """
    # Freeze to the canonical tuple so callers cannot silently reorder.
    categories = tuple(categories)
    if categories != REQUIRED_CXAS_ANATOMIES:
        raise ValueError(
            "anatomy JEPA requires FIXED order REQUIRED_CXAS_ANATOMIES; "
            f"got {categories!r}"
        )

    a = len(categories)
    z = torch.zeros(a, N_PATCHES, dtype=torch.float32)

    p_path = resolve_anatomy_mask_json_path(
        masks_root, dataset, parent_image_prev
    )
    c_path = resolve_anatomy_mask_json_path(
        masks_root, dataset, parent_image_curr
    )
    if p_path is None or c_path is None:
        return z, z, False

    try:
        prior_hw = load_per_category_masks_hw(p_path, categories)
        curr_hw = load_per_category_masks_hw(c_path, categories)
    except Exception:
        return z, z, False
    if prior_hw is None or curr_hw is None:
        return z, z, False

    prior_rows = []
    curr_rows = []
    for cat in categories:  # fixed order → axis A is aligned across batch
        try:
            pw = mask_hw_to_patch_weights(prior_hw[cat], aug_params=aug_params)
            cw = mask_hw_to_patch_weights(curr_hw[cat], aug_params=aug_params)
        except Exception:
            return z, z, False
        # Tiny structures can vanish after affine/crop. Keep a negligible
        # mass so the pair stays usable instead of dropping all 22.
        if float(pw.sum()) <= min_weight_sum:
            pw = pw.clone()
            pw[pw.numel() // 2] = float(min_weight_sum)
        if float(cw.sum()) <= min_weight_sum:
            cw = cw.clone()
            cw[cw.numel() // 2] = float(min_weight_sum)
        prior_rows.append(pw)
        curr_rows.append(cw)

    return torch.stack(prior_rows, dim=0), torch.stack(curr_rows, dim=0), True
