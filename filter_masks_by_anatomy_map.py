#!/usr/bin/env python3
"""Filter CXAS mask JSONs to a fixed medgemma anatomy inventory.

Unlike filter_masks_by_finding.py, this does NOT use per-finding files.
For each mask JSON under cxas_masks/<ds>/...:
    1. Apply the same quality checks (--min-key-hits, optional --strict).
    2. Map REQUIRED_MEDGEMMA_ANATOMIES → CXAS categories via ANATOMY_MAP
       (some phrases expand to a combo of CXAS classes, e.g. costophrenic
       angle → lung base + hemidiaphragm).
    3. Keep only those CXAS annotations. Skip the image unless EVERY
       required medgemma phrase is fully covered (all of its mapped CXAS
       classes present), so retained images share one fixed anatomy set.
    4. Write the filtered JSON mirroring the mask's relative path (original
       filename, no finding__ prefix).

Outputs:
    <out-root>/allowed_categories.json -- required medgemma phrases + CXAS cats
    <out-root>/stats.json              -- per-dataset bookkeeping
    <out-root>/<ds>/.../<image_stem>.json
    <qa-root>/<ds>/<NN>_<image_stem>.jpg
        -- side-by-side QA overlays (key classes | kept anatomy), like mask_qa/
    <jpgs-root>/<ds>/.../<image_stem>/{id:03d}_{name}.jpg
        -- per-category binary mask JPGs for the same examples, like mask_jpgs/

Usage:
    python filter_masks_by_anatomy_map.py \
        --masks-root cxas_masks \
        --out-root filtered_masks_anatomy \
        --qa-root mask_qa_anatomy \
        --jpgs-root mask_jpgs_anatomy \
        --qa-per-dataset 50 \
        --seed 0
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as cocomask
from scipy import ndimage as ndi
from tqdm import tqdm

DATASETS = ("chexpert", "mimic", "rexgradient")

# Major anatomy classes that a normal frontal chest X-ray should always have.
# Used as a study-level quality filter: studies whose CXAS mask has fewer than
# `--min-key-hits` of these present are considered low-quality and skipped.
KEY_CLASSES_FOR_QC: tuple[str, ...] = (
    "left lung",
    "right lung",
    "heart",
    "spine",
    "clavicle left",
    "clavicle right",
    "trachea",
    "left hemidiaphragm",
    "right hemidiaphragm",
    "aortic arch",
)

# ---------------------------------------------------------------------------
# Fixed medgemma anatomy inventory that every retained image must cover.
# Phrases map to one or more CXAS category names via ANATOMY_MAP (combos OK).
# ---------------------------------------------------------------------------
REQUIRED_MEDGEMMA_ANATOMIES: tuple[str, ...] = (
    "left lung",
    "right lung",
    "cardiac silhouette",
    "mediastinum",
    "left lower lung",
    "right lower lung",
    "right hilar structures",
    "left hilar structures",
    "upper mediastinum",
    "left costophrenic angle",
    "right costophrenic angle",
    "left mid lung",
    "right mid lung",
    "aortic arch",
    "right upper lung",
    "left upper lung",
    "right hemidiaphragm",
    "right clavicle",
    "left clavicle",
    "left hemidiaphragm",
    "right apex",
    "trachea",
    "left apex",
    "carina",
    "svc",
    "right atrium",
    "cavoatrial junction",
    "abdomen",
    "spine",
)

# medgemma phrase (lowercased) -> CXAS category name(s). Combos mean the
# medgemma phrase is only "present" when all listed CXAS classes exist.
ANATOMY_MAP: dict[str, list[str]] = {
    # whole lungs
    "left lung":           ["left lung"],
    "right lung":          ["right lung"],
    "lungs":               ["left lung", "right lung"],
    "bilateral lungs":     ["left lung", "right lung"],
    "lung":                ["lung"],

    # lung zones (medgemma "upper/mid/lower lung" -> CXAS zone classes)
    "left upper lung":     ["left upper zone lung"],
    "right upper lung":    ["right upper zone lung"],
    "left mid lung":       ["left mid zone lung"],
    "right mid lung":      ["right mid zone lung"],
    "left lower lung":     ["left lung base"],
    "right lower lung":    ["right lung base"],

    # apex
    "left apex":           ["left apical zone lung"],
    "right apex":          ["right apical zone lung"],
    "apex":                ["apical zone lung"],

    # costophrenic angle = lower-outer lung meeting diaphragm
    "left costophrenic angle":  ["left lung base", "left hemidiaphragm"],
    "right costophrenic angle": ["right lung base", "right hemidiaphragm"],

    # heart / mediastinum
    "cardiac silhouette":  ["heart"],
    "heart":               ["heart"],
    "mediastinum":         ["cardiomediastinum"],
    "cardiomediastinum":   ["cardiomediastinum"],
    "upper mediastinum":   ["upper mediastinum"],
    "lower mediastinum":   ["lower mediastinum"],
    "right atrium":        ["heart atrium right"],

    # hilar structures (no exact CXAS class - approximate with whole lung)
    "left hilar structures":  ["left lung"],
    "right hilar structures": ["right lung"],
    "hilar structures":       ["left lung", "right lung"],
    "left hilum":          ["left lung"],
    "right hilum":         ["right lung"],

    # diaphragm
    "left hemidiaphragm":  ["left hemidiaphragm"],
    "right hemidiaphragm": ["right hemidiaphragm"],
    "diaphragm":           ["diaphragm"],
    "left diaphragm":      ["left hemidiaphragm"],
    "right diaphragm":     ["right hemidiaphragm"],

    # trachea / carina
    "trachea":             ["trachea"],
    "tracheal bifurcation": ["tracheal bifurcation"],
    "carina":              ["tracheal bifurcation"],

    # clavicles
    "left clavicle":       ["clavicle left"],
    "right clavicle":      ["clavicle right"],

    # spine
    "spine":               ["spine"],
    "thoracic spine":      ["thoracic spine"],
    "cervical spine":      ["cervical spine"],
    "lumbar spine":        ["lumbar spine"],

    # large vessels
    "aorta":               ["aorta"],
    "aortic arch":         ["aortic arch"],
    "ascending aorta":     ["ascending aorta"],
    "descending aorta":    ["descending aorta"],
    "pulmonary artery":    ["pulmonary artery"],
}


def required_medgemma_coverage() -> tuple[dict[str, list[str]], list[str], set[str]]:
    """Return (phrase→CXAS, unmapped_phrases, required_cxas_union).

    Only phrases in REQUIRED_MEDGEMMA_ANATOMIES that appear in ANATOMY_MAP
    are enforceable. Unmapped phrases (no CXAS equivalent) are reported
    but cannot gate retention.
    """
    phrase_to_cxas: dict[str, list[str]] = {}
    unmapped: list[str] = []
    cxas: set[str] = set()
    for phrase in REQUIRED_MEDGEMMA_ANATOMIES:
        key = phrase.strip().lower()
        cats = ANATOMY_MAP.get(key)
        if not cats:
            unmapped.append(key)
            continue
        phrase_to_cxas[key] = list(cats)
        cxas.update(cats)
    return phrase_to_cxas, unmapped, cxas


def allowed_cxas_categories() -> set[str]:
    """CXAS categories required to cover the mapped medgemma inventory."""
    _, _, cxas = required_medgemma_coverage()
    return cxas


def uncovered_medgemma_phrases(
    present_cxas: set[str], phrase_to_cxas: dict[str, list[str]],
) -> list[str]:
    """Medgemma phrases whose mapped CXAS class(es) are not all present."""
    missing = []
    for phrase, cats in phrase_to_cxas.items():
        if any(c not in present_cxas for c in cats):
            missing.append(phrase)
    return missing


# ---------------------------------------------------------------------------
# JSON filtering / QC (same logic as filter_masks_by_finding.py)
# ---------------------------------------------------------------------------

def compute_key_hits(mask_data: dict) -> int:
    """How many KEY_CLASSES_FOR_QC have at least one annotation in this mask?"""
    cat_id_to_name = {c["id"]: c["name"] for c in mask_data.get("categories", [])}
    present = {
        cat_id_to_name[a["category_id"]]
        for a in mask_data.get("annotations", [])
        if a["category_id"] in cat_id_to_name
    }
    return sum(1 for k in KEY_CLASSES_FOR_QC if k in present)


STRICT_CORE_CLASSES: tuple[str, ...] = (
    "left lung",
    "right lung",
    "heart",
    "spine",
    "left hemidiaphragm",
    "right hemidiaphragm",
)

STRICT_DOMINANT_CC_FRAC = 0.95

STRICT_MIN_CLASS_AREA_FRAC = {
    "left lung":          0.04,
    "right lung":         0.04,
    "heart":              0.01,
    "spine":              0.005,
    "left hemidiaphragm": 0.005,
    "right hemidiaphragm": 0.005,
}

STRICT_EXPECTED_CX = {
    "heart":               (0.30, 0.70),
    "right lung":          (0.00, 0.55),
    "left lung":           (0.45, 1.00),
    "spine":               (0.35, 0.65),
    "right hemidiaphragm": (0.00, 0.60),
    "left hemidiaphragm":  (0.40, 1.00),
}

STRICT_MIN_TOTAL_ANNS = 50


def _decode_seg(seg) -> np.ndarray:
    if isinstance(seg, dict):
        rle = seg
        if isinstance(rle["counts"], str):
            rle = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
        m = cocomask.decode(rle)
    elif isinstance(seg, list):
        rles = []
        for s in seg:
            if isinstance(s, dict):
                if isinstance(s["counts"], str):
                    s = {"size": s["size"], "counts": s["counts"].encode("utf-8")}
                rles.append(s)
        m = cocomask.decode(cocomask.merge(rles))
    else:
        return np.zeros((1, 1), dtype=np.uint8)
    if m.ndim == 3:
        m = m.any(axis=-1).astype(np.uint8)
    return m


def passes_strict_filter(
    mask_data: dict,
    min_total_anns: int = STRICT_MIN_TOTAL_ANNS,
) -> tuple[bool, str]:
    """Apply absolute-strictest quality checks to a CXAS mask.

    Requirements (all must hold):
      1. >= `min_total_anns` annotations total.
      2. All STRICT_CORE_CLASSES are present.
      3. Each core class occupies at least the minimum fraction of image area.
      4. Each core class is a single coherent connected component (>= 95% of
         its area in the largest CC).
      5. Each core class has an anatomically-expected horizontal centroid.

    Returns (passes, reason). `reason` is a short tag for stats; "ok" if passed.
    """
    anns = mask_data.get("annotations", [])
    cats = mask_data.get("categories", [])
    if not anns:
        return False, "no_annotations"

    if len(anns) < min_total_anns:
        return False, f"few_anns({len(anns)})"

    name_to_id = {c["name"]: c["id"] for c in cats}
    cat_id_to_name = {c["id"]: c["name"] for c in cats}

    present = {cat_id_to_name[a["category_id"]] for a in anns
               if a["category_id"] in cat_id_to_name}
    missing_core = [k for k in STRICT_CORE_CLASSES if k not in present]
    if missing_core:
        return False, f"missing_core:{missing_core[0]}"

    H, W = anns[0]["segmentation"]["size"]
    img_area = H * W

    cls_mask: dict[str, np.ndarray] = {}
    for cname in STRICT_CORE_CLASSES:
        cid = name_to_id[cname]
        mu = np.zeros((H, W), dtype=bool)
        for a in anns:
            if a["category_id"] == cid:
                mu |= _decode_seg(a["segmentation"]).astype(bool)
        if not mu.any():
            return False, f"empty_mask:{cname}"
        cls_mask[cname] = mu

    for cname, frac in STRICT_MIN_CLASS_AREA_FRAC.items():
        if cls_mask[cname].sum() / img_area < frac:
            return False, f"small:{cname}"

    for cname, mu in cls_mask.items():
        labeled, n = ndi.label(mu)
        if n == 0:
            return False, f"empty_mask:{cname}"
        if n > 1:
            sizes = ndi.sum(mu, labeled, index=range(1, n + 1))
            largest = float(sizes.max())
            total = float(sizes.sum())
            if largest / total < STRICT_DOMINANT_CC_FRAC:
                return False, f"fragmented:{cname}"

    for cname, (x_lo, x_hi) in STRICT_EXPECTED_CX.items():
        ys, xs = np.nonzero(cls_mask[cname])
        cx = xs.mean() / W
        if not (x_lo <= cx <= x_hi):
            return False, f"misplaced:{cname}"

    return True, "ok"


def filter_mask_json(
    mask_data: dict, allowed_cats: set[str],
) -> tuple[dict, set[str], set[str]]:
    """Keep annotations whose category name is in `allowed_cats`.

    Returns (filtered_data, present_cats, missing_cats). Callers that want
    a fixed anatomy inventory should skip when ``missing_cats`` is non-empty.
    """
    name_to_id = {c["name"]: c["id"] for c in mask_data.get("categories", [])}
    wanted_ids = {name_to_id[n] for n in allowed_cats if n in name_to_id}

    kept_anns = [a for a in mask_data.get("annotations", []) if a["category_id"] in wanted_ids]
    present_ids = {a["category_id"] for a in kept_anns}
    id_to_name = {c["id"]: c["name"] for c in mask_data.get("categories", [])}
    present_cats = {id_to_name[i] for i in present_ids}
    missing_cats = allowed_cats - present_cats

    kept_cats = [c for c in mask_data.get("categories", []) if c["id"] in wanted_ids]
    out = {
        "info": mask_data.get("info", {}),
        "licenses": mask_data.get("licenses", []),
        "images": mask_data.get("images", []),
        "categories": kept_cats,
        "annotations": kept_anns,
    }
    return out, present_cats, missing_cats


# ---------------------------------------------------------------------------
# QA / per-category JPG rendering (mirrors mask_qa/ and mask_jpgs/)
# ---------------------------------------------------------------------------

def decode_segmentation(seg) -> np.ndarray:
    if isinstance(seg, dict):
        rle = seg
        if isinstance(rle["counts"], str):
            rle = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
        m = cocomask.decode(rle)
    elif isinstance(seg, list):
        rles = []
        for s in seg:
            if isinstance(s, dict):
                if isinstance(s["counts"], str):
                    s = {"size": s["size"], "counts": s["counts"].encode("utf-8")}
                rles.append(s)
        m = cocomask.decode(cocomask.merge(rles))
    else:
        raise ValueError(f"unsupported segmentation type: {type(seg)!r}")
    if m.ndim == 3:
        m = m.any(axis=-1).astype(np.uint8)
    return m


def _font(size: int) -> ImageFont.ImageFont:
    for c in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _per_category_masks(mask_data: dict) -> tuple[int, int, dict[int, np.ndarray], dict[int, str]]:
    """Decode union mask per category_id. Returns (H, W, masks, id_to_name)."""
    anns = mask_data.get("annotations", [])
    if not anns:
        return 0, 0, {}, {}
    H, W = anns[0]["segmentation"]["size"]
    id_to_name = {c["id"]: c["name"] for c in mask_data.get("categories", [])}
    per_cat: dict[int, np.ndarray] = {}
    for a in anns:
        cid = a["category_id"]
        m = decode_segmentation(a["segmentation"]).astype(bool)
        if cid not in per_cat:
            per_cat[cid] = np.zeros((H, W), dtype=bool)
        per_cat[cid] |= m
    return H, W, per_cat, id_to_name


def _colorize_categories(
    H: int,
    W: int,
    per_cat: dict[int, np.ndarray],
    id_to_name: dict[int, str],
    label_names: set[str] | None = None,
) -> Image.Image:
    """Paint category masks with distinct hues; optionally label centroids."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    cat_ids = sorted(per_cat)
    palette: dict[int, tuple[int, int, int]] = {}
    for i, cid in enumerate(cat_ids):
        hue = (i / max(1, len(cat_ids))) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        palette[cid] = (int(r * 255), int(g * 255), int(b * 255))
    for cid, mu in per_cat.items():
        canvas[mu] = palette[cid]

    img = Image.fromarray(canvas)
    if not label_names:
        return img

    draw = ImageDraw.Draw(img)
    font = _font(max(12, H // 35))
    for cid, mu in per_cat.items():
        name = id_to_name.get(cid, f"cat{cid}")
        if name not in label_names:
            continue
        ys, xs = np.nonzero(mu)
        if xs.size == 0:
            continue
        x, y = int(xs.mean()), int(ys.mean())
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((x + dx, y + dy), name, fill=(0, 0, 0), font=font, anchor="mm")
        draw.text((x, y), name, fill=(255, 255, 255), font=font, anchor="mm")
    return img


def render_qa(
    original: dict,
    filtered: dict,
    out_path: Path,
    rel_path: str,
    dataset: str,
    key_hits: int,
) -> None:
    """Side-by-side QA like mask_qa/: key classes (labeled) | kept anatomy."""
    H, W, orig_masks, orig_names = _per_category_masks(original)
    if H == 0:
        return
    _, _, kept_masks, kept_names = _per_category_masks(filtered)

    key_name_set = set(KEY_CLASSES_FOR_QC)
    key_masks = {
        cid: mu for cid, mu in orig_masks.items()
        if orig_names.get(cid) in key_name_set
    }
    left = _colorize_categories(H, W, key_masks, orig_names, label_names=key_name_set)
    right = _colorize_categories(H, W, kept_masks, kept_names, label_names=None)

    gap = 8
    header_h = max(28, H // 22)
    panel_w = W * 2 + gap
    header = Image.new("RGB", (panel_w, header_h), (20, 20, 20))
    hfont = _font(max(12, header_h - 10))
    hd = ImageDraw.Draw(header)
    n_anns = len(original.get("annotations", []))
    title = (
        f"{dataset}: {rel_path} "
        f"(key hits: {key_hits}/{len(KEY_CLASSES_FOR_QC)}, anns: {n_anns}, "
        f"kept cats: {len(kept_masks)})"
    )
    hd.text((6, header_h // 2), title[:220], fill=(230, 230, 230), font=hfont, anchor="lm")

    body = Image.new("RGB", (panel_w, H), (0, 0, 0))
    body.paste(left, (0, 0))
    body.paste(right, (W + gap, 0))
    final = Image.new("RGB", (panel_w, header_h + H), (0, 0, 0))
    final.paste(header, (0, 0))
    final.paste(body, (0, header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path, quality=92)


def save_category_jpgs(filtered: dict, jpg_dir: Path) -> int:
    """Write per-category binary JPGs like mask_jpgs/<...>/<stem>/{id}_{name}.jpg.

    Returns number of JPGs written.
    """
    H, W, per_cat, id_to_name = _per_category_masks(filtered)
    if H == 0:
        return 0
    jpg_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cid in sorted(per_cat):
        name = id_to_name.get(cid, f"cat{cid}").replace(" ", "_")
        mu = per_cat[cid]
        arr = (mu.astype(np.uint8) * 255)
        Image.fromarray(arr, mode="L").save(jpg_dir / f"{cid:03d}_{name}.jpg", quality=95)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def iter_mask_jsons(ds_root: Path) -> Iterable[Path]:
    """Yield all .json mask files under a dataset root."""
    for p in sorted(ds_root.rglob("*.json")):
        if p.is_file():
            yield p


def relative_out_path(
    out_root: Path, dataset: str, mask_path: Path, masks_root: Path,
) -> Path:
    """Mirror the mask path under <out-root>/<ds>/ (keep original filename)."""
    rel = mask_path.relative_to(masks_root / dataset)
    return out_root / dataset / rel


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--masks-root", default="cxas_masks")
    ap.add_argument("--out-root", default="filtered_masks_anatomy")
    ap.add_argument(
        "--qa-root", default="mask_qa_anatomy",
        help="Folder for side-by-side QA overlays (like mask_qa/).",
    )
    ap.add_argument(
        "--jpgs-root", default="mask_jpgs_anatomy",
        help="Folder for per-category binary mask JPGs (like mask_jpgs/).",
    )
    ap.add_argument(
        "--qa-per-dataset", type=int, default=50,
        help="Number of example QA overlays + JPG bundles per dataset. Default: 50.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--datasets", nargs="*", choices=DATASETS, default=list(DATASETS),
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N mask JSON files per dataset (smoke test).",
    )
    ap.add_argument(
        "--min-key-hits", type=int, default=8,
        help=(
            "Skip masks with fewer than this many of the 10 major "
            f"anatomy classes ({', '.join(KEY_CLASSES_FOR_QC)}). Use 0 to "
            "disable this quality filter. Default: 8."
        ),
    )
    ap.add_argument(
        "--strict", action="store_true",
        help=(
            "Apply the 'absolute strictest' quality filter on top of "
            f"--min-key-hits. Requires all CORE classes ({', '.join(STRICT_CORE_CLASSES)}) "
            f"to be: present, >= {STRICT_MIN_TOTAL_ANNS} total annotations in "
            "the mask, each is a single coherent connected component "
            "(>= 95%% in largest CC), each meets minimum-area thresholds, and "
            "each centroid sits in its anatomically expected horizontal range."
        ),
    )
    args = ap.parse_args()

    masks_root = Path(args.masks_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    qa_root = Path(args.qa_root).expanduser().resolve()
    jpgs_root = Path(args.jpgs_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    phrase_to_cxas, unmapped_required, allowed = required_medgemma_coverage()
    stats: dict[str, dict] = {}
    print(
        f"[filter] requiring coverage of {len(phrase_to_cxas)} medgemma "
        f"anatomies → {len(allowed)} CXAS classes on every kept mask"
    )
    if unmapped_required:
        print(
            f"[filter] WARNING: {len(unmapped_required)} required medgemma "
            f"phrases have no CXAS mapping and cannot be enforced: "
            f"{unmapped_required}",
            file=sys.stderr,
        )

    with (out_root / "allowed_categories.json").open("w") as f:
        json.dump(
            {
                "required_medgemma_anatomies": list(REQUIRED_MEDGEMMA_ANATOMIES),
                "mapped_medgemma_to_cxas": {
                    k: v for k, v in sorted(phrase_to_cxas.items())
                },
                "unmapped_required_medgemma": unmapped_required,
                "n_required_medgemma_mapped": len(phrase_to_cxas),
                "n_required_cxas_categories": len(allowed),
                "required_cxas_categories": sorted(allowed),
                "require_all_mapped_medgemma": True,
            },
            f,
            indent=2,
        )

    for dataset in args.datasets:
        ds_root = masks_root / dataset
        if not ds_root.is_dir():
            print(f"[{dataset}] missing folder: {ds_root}", file=sys.stderr)
            continue

        ds_stats = {
            "n_mask_files": 0,
            "n_kept": 0,
            "n_skipped_unreadable": 0,
            "n_skipped_low_key_hits": 0,
            "n_skipped_strict_qc": 0,
            "strict_skip_reasons": Counter(),
            "n_skipped_no_allowed_anatomy": 0,
            "n_skipped_incomplete_anatomy": 0,
            "n_required_medgemma_mapped": len(phrase_to_cxas),
            "n_required_cxas_categories": len(allowed),
            "unmapped_required_medgemma": list(unmapped_required),
            "n_total_filtered_jsons_written": 0,
            "n_qa_written": 0,
            "n_jpg_bundles_written": 0,
            "n_category_jpgs_written": 0,
            "min_key_hits": args.min_key_hits,
            "strict": args.strict,
            "category_presence_counts": Counter(),
        }

        # Paths of kept filtered JSONs + metadata for later QA/JPG sampling.
        # (out_path, mask_path, key_hits)
        success_records: list[tuple[Path, Path, int]] = []

        mask_paths = list(iter_mask_jsons(ds_root))
        if args.limit is not None:
            mask_paths = mask_paths[: args.limit]
        ds_stats["n_mask_files"] = len(mask_paths)

        pbar = tqdm(
            mask_paths,
            desc=f"[{dataset}] filter",
            unit="mask",
            ncols=120,
            leave=True,
        )
        for mp in pbar:
            try:
                with mp.open() as f:
                    mask_data = json.load(f)
            except Exception:
                ds_stats["n_skipped_unreadable"] += 1
                continue

            key_hits = compute_key_hits(mask_data)
            if key_hits < args.min_key_hits:
                ds_stats["n_skipped_low_key_hits"] += 1
                pbar.set_postfix(kept=ds_stats["n_kept"], refresh=False)
                continue

            if args.strict:
                passes, reason = passes_strict_filter(mask_data)
                if not passes:
                    ds_stats["n_skipped_strict_qc"] += 1
                    bucket = re.split(r"[:(]", reason, 1)[0]
                    ds_stats["strict_skip_reasons"][bucket] += 1
                    pbar.set_postfix(kept=ds_stats["n_kept"], refresh=False)
                    continue

            filtered, present, missing_cxas = filter_mask_json(mask_data, allowed)
            if not present:
                ds_stats["n_skipped_no_allowed_anatomy"] += 1
                pbar.set_postfix(kept=ds_stats["n_kept"], refresh=False)
                continue
            uncovered = uncovered_medgemma_phrases(present, phrase_to_cxas)
            if uncovered or missing_cxas:
                # Every mapped medgemma phrase must be fully covered
                # (including combo CXAS classes like costophrenic angle).
                ds_stats["n_skipped_incomplete_anatomy"] += 1
                pbar.set_postfix(kept=ds_stats["n_kept"], refresh=False)
                continue

            for cname in present:
                ds_stats["category_presence_counts"][cname] += 1

            out_path = relative_out_path(out_root, dataset, mp, masks_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                json.dump(filtered, f)
            ds_stats["n_total_filtered_jsons_written"] += 1
            ds_stats["n_kept"] += 1
            success_records.append((out_path, mp, key_hits))
            pbar.set_postfix(kept=ds_stats["n_kept"], refresh=False)

        if success_records and args.qa_per_dataset > 0:
            picks = (success_records if len(success_records) <= args.qa_per_dataset
                     else rng.sample(success_records, args.qa_per_dataset))
            qa_dir = qa_root / dataset
            qa_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[{dataset}] writing {len(picks)} QA overlays -> {qa_dir}\n"
                f"[{dataset}] writing {len(picks)} JPG bundles -> {jpgs_root / dataset}"
            )
            for k, (out_path, mp, key_hits) in enumerate(
                tqdm(picks, desc=f"[{dataset}] qa+jpgs", unit="ex", ncols=120),
                start=1,
            ):
                try:
                    with out_path.open() as f:
                        filtered = json.load(f)
                    with mp.open() as f:
                        original = json.load(f)
                except Exception as exc:  # noqa: BLE001
                    print(f"  reload failed for {out_path}: {exc}", file=sys.stderr)
                    continue

                rel = mp.relative_to(masks_root / dataset)
                qa_jpg = qa_dir / f"{k:02d}_{out_path.stem}.jpg"
                try:
                    render_qa(
                        original, filtered, qa_jpg,
                        rel_path=str(rel),
                        dataset=dataset,
                        key_hits=key_hits,
                    )
                    ds_stats["n_qa_written"] += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  qa render failed for {out_path}: {exc}", file=sys.stderr)

                # mask_jpgs layout: <ds>/<rel parent>/<stem>/{id}_{name}.jpg
                jpg_dir = jpgs_root / dataset / rel.parent / rel.stem
                try:
                    n_jpgs = save_category_jpgs(filtered, jpg_dir)
                    if n_jpgs:
                        ds_stats["n_jpg_bundles_written"] += 1
                        ds_stats["n_category_jpgs_written"] += n_jpgs
                except Exception as exc:  # noqa: BLE001
                    print(f"  jpg write failed for {out_path}: {exc}", file=sys.stderr)

        stats[dataset] = ds_stats
        print(
            f"[{dataset}] done: kept={ds_stats['n_kept']:,} "
            f"low_qc={ds_stats['n_skipped_low_key_hits']:,} "
            f"strict={ds_stats['n_skipped_strict_qc']:,} "
            f"empty={ds_stats['n_skipped_no_allowed_anatomy']:,} "
            f"incomplete={ds_stats['n_skipped_incomplete_anatomy']:,} "
            f"qa={ds_stats['n_qa_written']} "
            f"jpg_bundles={ds_stats['n_jpg_bundles_written']}"
        )

    stats_serializable = {}
    for ds_name, ds in stats.items():
        ds = dict(ds)
        if isinstance(ds.get("strict_skip_reasons"), Counter):
            ds["strict_skip_reasons"] = dict(ds["strict_skip_reasons"].most_common())
        if isinstance(ds.get("category_presence_counts"), Counter):
            ds["category_presence_counts"] = dict(
                ds["category_presence_counts"].most_common()
            )
        stats_serializable[ds_name] = ds
    with (out_root / "stats.json").open("w") as f:
        json.dump(stats_serializable, f, indent=2)

    print("\nWrote:")
    print(
        f"  {out_root / 'allowed_categories.json'}  "
        f"({len(phrase_to_cxas)} medgemma → {len(allowed)} CXAS)"
    )
    print(f"  {out_root / 'stats.json'}")
    print(f"  {qa_root}/<ds>/  (QA overlays)")
    print(f"  {jpgs_root}/<ds>/  (per-category binary JPGs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
