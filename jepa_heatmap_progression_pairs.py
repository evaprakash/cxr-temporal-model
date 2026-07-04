"""Dynamic phrase grounding heatmaps for the JEPA temporal CXR model.

Direct JEPA analog of ``biovilt_heatmap_progression_pairs.py``: same
gold bboxes CSV, same 448x448 model space, same CNR / Pointing Game
formulas, same rendering. The only differences are how the patch
tokens and text query are computed, because the JEPA training pipeline
uses a shared unit-sphere geometry and does not use BioViL-T's
paired-image cross-attention at feature-extraction time.

What changed vs the BioViL-T script
-----------------------------------
1. **Model.** Loads a JEPA checkpoint (produced by
   ``resume_train_jepa.py``) via ``infer_jepa.load_jepa_model``.
   Pulls image patches from ``model.image_encoder`` and text
   embeddings from ``model.text_encoder`` — both already emit
   L2-normalized 128-D vectors, so ``patch @ text`` is a cosine.

2. **Single-image encoder calls.** ``BioViLTImageEncoderJEPA`` wraps
   ``MultiImageModel`` but the JEPA training loop always calls it as
   ``image_encoder(img)`` (no partner), so the paired-image type
   embeddings that BioViL-T's script had to reason about are never
   exercised. Prev-side patches come from
   ``image_encoder(prev)`` and curr-side from
   ``image_encoder(curr)`` — no argument swap, no
   PROGRESSION_INVERSION.

3. **Same prompt on both sides.** Because there's no directional
   encoding to invert, both prev- and curr-side heatmaps use the raw
   ``"<disease> is <progression>"`` prompt. The prev-side heatmap is
   still meaningful: the L2 local-contrastive loss trains the same
   encoder weights to align patch tokens with report text, and that
   alignment applies to whichever image you feed in.

Why this is the fair BioViL-T comparison
----------------------------------------
The JEPA model retains a cross-modal patch->text alignment through
its L2 (prior_patches ↔ prior_report) and L3 (pred_current_patches ↔
current_report) local-contrastive losses. Using ``patch @ text_emb``
on the encoder's output is the direct probe of that alignment,
identical in spirit to what BioViL-T's script measures — the only
architectural difference being how the patches got there. Reporting
CNR / PG side-by-side against the BioViL-T baseline isolates the
effect of the JEPA training objective on text-grounding.

The rest of the pipeline is unchanged: bilinearly upsample the 14x14
similarity grid to the 448x448 input, overlay on the 1024-fit
display image, draw the single-disease bboxes, and compute BioViL
CNR + Pointing Game in model space.

Inputs
------
* ``--gold-parquet`` (default: ``CheXTemporal/gold_bboxes.parquet``).
  Read with pandas. Expected columns: ``dataset``, ``patient_id``,
  ``study_id_prev``, ``study_id_curr``, ``disease_name``,
  ``comparison``, ``img_path_prev``, ``img_path_curr``,
  ``prior_bboxes``, ``current_bboxes``. Bbox columns may be
  JSON-encoded strings or native list-of-list objects; both work.
* Image roots for ``final_gold_<dataset>_images/`` auto-detect from
  the parquet's directory + its parent, mirroring
  ``eval_progression_jepa.py``'s ``discover_gold_image_roots``.
  ``--image-root DATASET=PATH`` overrides per-dataset.

Notes
-----
* The 14x14 patch grid + 448x448 input assumption matches every
  branch on the BioViL-T backbone (``main``,
  ``prog-loss-all-findings``, ``baseline-jepa-*``, etc.). If you run
  this on the ``raddino-image-encoder`` branch, bump ``INPUT_SIZE``
  to 518 and ``RESIZE_SHORT`` to the RAD-DINO short side — the
  ``patches_to_heatmap`` code already handles the resulting 37x37
  grid via its ``side = round(sqrt(L))`` reshape.
* Writes PNGs + ``cnr.csv`` under ``--out-dir``. Set ``--no-render``
  to compute metrics without rendering the ~2xN PNGs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

from dataset_combined_jepa import DEFAULT_DATASET_DIR
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from progression_classify import (
    DATASETS,
    _resolve_with_fallbacks,
    discover_gold_image_roots,
)
from tempcxr.modules.jepa import TempCXRJEPA

# ===============================================================
# CONFIG
# ===============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Gold bboxes and gold images live next to the CheXTemporal parquets by
# default (same layout as ``eval_progression_jepa.py``): the parquet
# ``CheXTemporal/gold_bboxes.parquet``, the images
# ``final_gold_<dataset>_images/`` as siblings of ``CheXTemporal/`` (or
# inside it — both layouts are handled by ``discover_gold_image_roots``).
DEFAULT_GOLD_PARQUET = os.path.join(DEFAULT_DATASET_DIR, "gold_bboxes.parquet")
DEFAULT_CKPT = os.environ.get(
    "JEPA_CKPT", "checkpoints_jepa_dynamic/best.pt"
)

PROMPT_TEMPLATE = "{} is {}"
CLS_ORDER = ["improving", "stable", "worsening", "new", "resolved"]

INPUT_SIZE = 448           # T.CenterCrop size — matches BASE_TRANSFORM in dataset_combined.py
RESIZE_SHORT = 512         # T.Resize value — matches BASE_TRANSFORM
BBOX_FIT_SIZE = 1024       # bbox coords were captured after thumbnail((1024,1024))

# High-contrast palette chosen so the boxes stand out on top of a jet
# heatmap (red regions = high similarity, blue = low). Same colors as
# the BioViL-T script so cross-model figures render consistently.
BOX_COLORS = {
    "Box1": (0, 255, 255),     # cyan
    "Box2": (255, 255, 255),   # white
    "Box3": (255, 0, 255),     # magenta
    "Box4": (255, 255, 0),     # yellow
    "Box5": (0, 255, 0),       # green
}
DEFAULT_COLOR = (0, 255, 255)

FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def get_font(size: int):
    for fp in FONT_PATHS:
        if os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


# ===============================================================
# IMAGE TRANSFORM (must match training exactly)
# ===============================================================
# Mirrors dataset_combined.BASE_TRANSFORM + apply_augmentation(train=False):
# Resize(512) -> CenterCrop(448) -> ToTensor -> (repeat channel if needed).
transform = T.Compose([
    T.Resize(RESIZE_SHORT),
    T.CenterCrop(INPUT_SIZE),
    T.ToTensor(),
])


def load_image(path) -> tuple[torch.Tensor, Image.Image, tuple[int, int]]:
    """Returns (model_tensor on DEVICE, display_pil, original (W, H)).

    ``path`` may be ``str`` or ``pathlib.Path`` (as returned by
    ``_resolve_with_fallbacks``).

    ``model_tensor`` is what the model sees: the original PIL image after
    ``Resize(512)`` + ``CenterCrop(448)`` + ``ToTensor``.

    ``display_pil`` is what we render to disk: the original PIL image
    after ``Image.thumbnail((1024, 1024))`` — exactly the same convention
    ``draw_bboxes.py`` (and the BioViL-T heatmap script) use, so bbox
    coords from the annotation tool can be drawn with their *raw* values,
    no transform.
    """
    img = Image.open(str(path)).convert("RGB")
    orig_size = img.size  # (W, H)

    img_t = transform(img)
    if img_t.shape[0] == 1:
        img_t = img_t.repeat(3, 1, 1)

    display_pil = img.copy()
    display_pil.thumbnail((BBOX_FIT_SIZE, BBOX_FIT_SIZE))

    return img_t.to(DEVICE), display_pil, orig_size


# ===============================================================
# MODEL LOAD
# ===============================================================
def load_model(ckpt_path: str) -> TempCXRJEPA:
    print(f"🔧 Loading TempCXRJEPA checkpoint: {ckpt_path}")
    device = torch.device(DEVICE)
    model = load_jepa_model(ckpt_path, device)
    model.eval()
    print("✅ Model loaded.")
    return model


# ===============================================================
# CORE: per-image patch embeddings + per-prompt heatmaps
# ===============================================================
@torch.no_grad()
def get_patches(model: TempCXRJEPA, img_t: torch.Tensor) -> torch.Tensor:
    """Single-image call into ``model.image_encoder`` — matches JEPA
    training-time usage. Returns L2-normalized patch embeddings of shape
    ``(L, 128)`` for the BioViL-T backbone (14x14 = 196 patches).

    The image encoder already applies ``F.normalize`` internally, so no
    extra normalization is needed at the call site.
    """
    _, patches = model.image_encoder(img_t.unsqueeze(0))
    return patches.squeeze(0)  # (L, 128)


@torch.no_grad()
def encode_progression_prompt(model: TempCXRJEPA, disease: str,
                              progression: str) -> torch.Tensor:
    """Encode a single ground-truth prompt and return a (128,) embedding.

    ``forward_contrastive`` already returns unit-norm global embeddings
    (see ``BioViLTTextEncoder.forward_contrastive`` in text_encoder.py),
    so this reduces to a shape-fix squeeze.
    """
    prompt = PROMPT_TEMPLATE.format(disease, progression)
    txt_global, _, _ = model.text_encoder.forward_contrastive([prompt])
    txt_global = F.normalize(txt_global, dim=-1)  # defensive; already unit-norm
    return txt_global.squeeze(0)  # (128,)


def patches_to_heatmap(patch_emb: torch.Tensor, text_emb: torch.Tensor,
                       out_size: int) -> np.ndarray:
    """patch_emb: (L, D), text_emb: (D,). Returns (out_size, out_size)
    heatmap upsampled bilinearly from the (H', W') similarity grid.

    Both inputs are unit-norm, so ``patch_emb @ text_emb`` is a per-patch
    cosine similarity in [-1, 1].
    """
    sim = patch_emb @ text_emb  # (L,)
    L = sim.shape[0]
    side = int(round(L ** 0.5))
    assert side * side == L, f"Patch count {L} is not a perfect square"
    grid = sim.view(1, 1, side, side)
    upsampled = F.interpolate(grid, size=(out_size, out_size),
                              mode="bilinear", align_corners=False)
    return upsampled.squeeze().cpu().float().numpy()


# ===============================================================
# MODEL-CROP REGION INSIDE THE 1024-FIT DISPLAY IMAGE
# ===============================================================
def model_crop_in_display(orig_size: tuple[int, int],
                          display_size: tuple[int, int]
                          ) -> tuple[int, int, int, int]:
    """Where in the 1024-fit display image does the 448x448 region the
    model actually sees live? Returns (x0, y0, x1, y1) in display coords.

    The model's pipeline is ``Resize(short=512) + CenterCrop(448)``. In
    original-image space this is a *centered square* of side
    ``448 / s_resize`` (since Resize scales every axis by the same factor).
    Mapping that square through the same scale factor used to produce the
    1024-fit display image gives us the rectangle below.
    """
    W, H = orig_size
    disp_W, disp_H = display_size

    s_thumb = min(BBOX_FIT_SIZE / W, BBOX_FIT_SIZE / H)   # original -> display
    s_resize = RESIZE_SHORT / min(W, H)                   # original -> resized

    crop_side_disp = INPUT_SIZE * s_thumb / s_resize
    x0 = (disp_W - crop_side_disp) / 2
    y0 = (disp_H - crop_side_disp) / 2
    return int(round(x0)), int(round(y0)), \
        int(round(x0 + crop_side_disp)), int(round(y0 + crop_side_disp))


# ===============================================================
# CNR (contrast-to-noise ratio)
# ===============================================================
def boxes_mask_in_model_space(boxes: list,
                              orig_size: tuple[int, int],
                              display_size: tuple[int, int],
                              model_size: int = INPUT_SIZE) -> np.ndarray:
    """Build a boolean mask over the model's 448x448 grid that marks every
    pixel inside the union of the (1024-fit display-space) bboxes.

    Bbox coords live in display (1024-fit) space. The model's receptive
    field is the centered crop returned by ``model_crop_in_display``;
    we map display coords -> model coords with the same affine that was
    used to overlay the heatmap, then clip to [0, model_size].
    """
    H = W = model_size
    mask = np.zeros((H, W), dtype=bool)
    if not boxes:
        return mask

    cx0, cy0, cx1, cy1 = model_crop_in_display(orig_size, display_size)
    crop_w = max(1, cx1 - cx0)
    crop_h = max(1, cy1 - cy0)
    sx = W / crop_w   # display pixels -> model pixels
    sy = H / crop_h

    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        mx1 = (x1 - cx0) * sx
        my1 = (y1 - cy0) * sy
        mx2 = (x2 - cx0) * sx
        my2 = (y2 - cy0) * sy
        ix1 = max(0, min(W, int(round(mx1))))
        iy1 = max(0, min(H, int(round(my1))))
        ix2 = max(0, min(W, int(round(mx2))))
        iy2 = max(0, min(H, int(round(my2))))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        mask[iy1:iy2, ix1:ix2] = True
    return mask


def compute_cnr(heatmap: np.ndarray, mask: np.ndarray) -> float | None:
    """BioViL CNR = |mu_A - mu_A'| / sqrt(var_A + var_A').

    Returns None if either region is empty or the denominator is zero.
    """
    interior = heatmap[mask]
    exterior = heatmap[~mask]
    if interior.size == 0 or exterior.size == 0:
        return None
    mu_a = float(interior.mean())
    mu_b = float(exterior.mean())
    var_a = float(interior.var())
    var_b = float(exterior.var())
    denom = (var_a + var_b) ** 0.5
    if denom < 1e-12:
        return None
    return abs(mu_a - mu_b) / denom


def compute_pointing_game(heatmap: np.ndarray,
                          mask: np.ndarray) -> int | None:
    """Pointing Game: 1 if the heatmap's argmax pixel lies inside the union
    of ground-truth boxes, 0 otherwise. Returns None if the mask is empty
    (e.g. no boxes, or all boxes fell outside the model's receptive field),
    so the caller can skip it exactly like CNR.

    With multiple boxes this is still a single 0/1 per (example, side): we
    compare the max pixel against the union mask, so it counts as a hit if
    it falls in ANY of the ground-truth boxes.
    """
    if not mask.any():
        return None
    flat_idx = int(np.argmax(heatmap))
    return int(bool(mask.flat[flat_idx]))


# ===============================================================
# RENDERING
# ===============================================================
def render_heatmap_png(display_pil: Image.Image, heatmap: np.ndarray,
                       boxes: list, orig_size: tuple[int, int],
                       *, vmin: float, vmax: float, alpha: float,
                       out_path: Path) -> int:
    """Save a single PNG to ``out_path``: 1024-fit display image + heatmap
    overlay (only over the model's centered crop region) + bbox overlay
    (drawn in the SAME 1024-fit coordinate space the bboxes were captured
    in, exactly like ``draw_bboxes.py``).

    Returns the number of boxes drawn.
    """
    base = display_pil.convert("RGBA")
    disp_W, disp_H = base.size

    # Color the 448x448 model heatmap with jet, set a constant alpha.
    cmap = plt.get_cmap("jet")
    if vmax - vmin < 1e-12:
        norm = np.zeros_like(heatmap)
    else:
        norm = np.clip((heatmap - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = (cmap(norm) * 255).astype(np.uint8)  # (448, 448, 4)
    rgba[..., 3] = int(alpha * 255)
    hm_pil_448 = Image.fromarray(rgba, mode="RGBA")

    # Resize the heatmap to the display-space size of the model crop and
    # paste it over the corresponding region of the display image. Pixels
    # outside the crop region get no overlay (since the model never saw
    # them) — they stay as the raw image, which is fine.
    cx0, cy0, cx1, cy1 = model_crop_in_display(orig_size, (disp_W, disp_H))
    crop_w = max(1, cx1 - cx0)
    crop_h = max(1, cy1 - cy0)
    hm_pil = hm_pil_448.resize((crop_w, crop_h), Image.BILINEAR)

    overlay = Image.new("RGBA", (disp_W, disp_H), (0, 0, 0, 0))
    overlay.paste(hm_pil, (cx0, cy0), hm_pil)
    blended = Image.alpha_composite(base, overlay)
    draw = ImageDraw.Draw(blended)

    # Draw bboxes with their RAW coords (1024-fit space = display space),
    # exactly matching draw_bboxes.py. No transform, no clipping.
    line_w = max(3, max(disp_W, disp_H) // 250)
    label_font = get_font(max(14, max(disp_W, disp_H) // 50))
    drawn = 0
    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        label = box[4] if len(box) > 4 else ""
        color = BOX_COLORS.get(label, DEFAULT_COLOR)
        draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1],
                       outline=(0, 0, 0, 255), width=line_w + 2)
        draw.rectangle([x1, y1, x2, y2],
                       outline=color + (255,), width=line_w)
        if label:
            bbox_text = draw.textbbox((0, 0), str(label), font=label_font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]
            pad = max(3, line_w)
            tx0 = x1
            ty0 = max(0, y1 - th - 2 * pad)
            draw.rectangle([tx0, ty0, tx0 + tw + 2 * pad, ty0 + th + 2 * pad],
                           fill=color + (255,))
            draw.text((tx0 + pad, ty0 + pad - 1), str(label),
                      fill=(0, 0, 0, 255), font=label_font)
        drawn += 1

    blended.convert("RGB").save(out_path, "PNG")
    return drawn


# ===============================================================
# BBOX PARSING (parquet-friendly)
# ===============================================================
def _parse_bboxes(raw) -> list:
    """Coerce a ``prior_bboxes`` / ``current_bboxes`` cell into a list.

    The BioViL-T script assumed CSV storage (JSON-encoded strings), but
    ``gold_bboxes.parquet`` can store the same field as a native
    list-of-lists (or as JSON strings depending on how it was written).
    Handle both, plus the various empty representations pandas can
    produce.
    """
    if raw is None:
        return []
    if isinstance(raw, float) and np.isnan(raw):
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return list(parsed) if parsed else []
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)):
        return [list(b) for b in raw]
    return []


# ===============================================================
# EXAMPLE SELECTION
# ===============================================================
def pick_examples(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Pick n examples covering as many progression classes as possible."""
    rng = random.Random(seed)
    by_cls: dict[str, list[int]] = {}
    for idx, row in df.iterrows():
        by_cls.setdefault(str(row["comparison"]).lower(), []).append(idx)
    for ids in by_cls.values():
        rng.shuffle(ids)

    ordered_classes = [c for c in CLS_ORDER if c in by_cls] + \
        [c for c in by_cls if c not in CLS_ORDER]

    chosen: list[int] = []
    while len(chosen) < n and any(by_cls[c] for c in ordered_classes):
        for c in ordered_classes:
            if not by_cls[c]:
                continue
            chosen.append(by_cls[c].pop())
            if len(chosen) >= n:
                break

    return df.loc[chosen].reset_index(drop=True)


# ===============================================================
# MAIN
# ===============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default=DEFAULT_CKPT,
                        help="Path to a JEPA checkpoint (default: "
                             f"{DEFAULT_CKPT}). Override via JEPA_CKPT env var.")
    parser.add_argument("--gold-parquet", default=DEFAULT_GOLD_PARQUET,
                        help=f"Path to gold_bboxes.parquet "
                             f"(default: {DEFAULT_GOLD_PARQUET}). "
                             f"The parquet's directory is used as the search "
                             f"root for auto-detecting "
                             f"final_gold_<dataset>_images/ folders.")
    parser.add_argument("--out-dir", default="heatmaps_progression_pairs_jepa",
                        help="Folder to save the rendered PNGs in")
    parser.add_argument("--image-root", action="append", default=[],
                        metavar="DATASET=PATH",
                        help="Override an image root for one dataset. Can "
                             "repeat. Example: --image-root "
                             "mimic=/data/final_gold_mimic_images. Defaults: "
                             "final_gold_<dataset>_images/ next to the gold "
                             "parquet if present, else IMAGE_ROOTS from "
                             "infer_jepa.py.")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Optional dataset filter, e.g. mimic chexpert")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="If set, only render N examples (one per "
                             "progression class when possible). Default: "
                             "run on every row in the parquet.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="Heatmap opacity (0..1)")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip writing PNGs; only compute metrics.")
    parser.add_argument("--cnr-csv", default=None,
                        help="Path to write per-row CNR CSV "
                             "(default: <out-dir>/cnr.csv)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Image roots: same resolution rule as eval_progression_jepa.py ---
    # Start from the silver IMAGE_ROOTS (so paths like "mimic/p11/..." also
    # resolve under all_data/), prefer final_gold_<dataset>_images/ near the
    # parquet if present, then apply --image-root overrides last.
    parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    auto_gold_roots = discover_gold_image_roots(parquet_dir)
    image_roots: dict[str, str] = {**IMAGE_ROOTS, **auto_gold_roots}
    if auto_gold_roots:
        print("[gold] auto-detected gold image roots:")
        for d, p in auto_gold_roots.items():
            print(f"  {d}: {p}")
    for spec in args.image_root:
        if "=" not in spec:
            raise ValueError(
                f"--image-root expects DATASET=PATH, got: {spec!r}"
            )
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(
                f"--image-root dataset must be one of {DATASETS}, got {d!r}"
            )
        image_roots[d] = p
        print(f"[gold] override: {d} -> {p}")

    # --- Load gold_bboxes parquet ---
    df = pd.read_parquet(args.gold_parquet)
    if args.datasets:
        df = df[df["dataset"].isin(args.datasets)].reset_index(drop=True)
    print(f"📄 Loaded {len(df)} rows from {args.gold_parquet}")

    def _resolve(row, col):
        try:
            return _resolve_with_fallbacks(row["dataset"], row[col], image_roots)
        except FileNotFoundError:
            return None

    df["_prev_resolved"] = df.apply(lambda r: _resolve(r, "img_path_prev"), axis=1)
    df["_curr_resolved"] = df.apply(lambda r: _resolve(r, "img_path_curr"), axis=1)
    df = df[df["_prev_resolved"].notna() & df["_curr_resolved"].notna()]
    df = df.reset_index(drop=True)
    print(f"📄 {len(df)} rows have both images present locally")
    if len(df) == 0:
        raise SystemExit(
            "No usable examples — check --gold-parquet, --image-root, and "
            "the presence of final_gold_<dataset>_images/ next to the parquet."
        )

    if args.num_examples is not None:
        examples = pick_examples(df, n=args.num_examples, seed=args.seed)
    else:
        examples = df.reset_index(drop=True)
    print(f"🎯 Processing {len(examples)} examples")

    model = load_model(args.ckpt)

    cnr_records: list[dict] = []  # one row per (example, side)

    for i, row in examples.iterrows():
        prev_path = row["_prev_resolved"]
        curr_path = row["_curr_resolved"]

        prev_t, prev_disp, prev_orig = load_image(prev_path)
        curr_t, curr_disp, curr_orig = load_image(curr_path)

        # Single-image encoder calls — matches JEPA training-time usage
        # (the encoder's paired-image type embeddings are never activated
        # at training time, so we don't touch them here either).
        prev_patches = get_patches(model, prev_t)
        curr_patches = get_patches(model, curr_t)

        true_label = str(row["comparison"]).lower()

        # Same prompt on both sides — the JEPA encoder does not encode a
        # temporal direction into its patch tokens, so there's no need to
        # invert the progression on the prev-side pass. Reported side
        # asymmetry (if any) reflects real differences in how the encoder
        # localizes the finding on the two visits, not a text/encoding
        # mismatch.
        text_emb = encode_progression_prompt(
            model, disease=row["disease_name"], progression=true_label,
        )

        prev_hm = patches_to_heatmap(prev_patches, text_emb, INPUT_SIZE)
        curr_hm = patches_to_heatmap(curr_patches, text_emb, INPUT_SIZE)

        # Shared color range so prev and curr heatmaps are directly comparable.
        vmin = float(min(prev_hm.min(), curr_hm.min()))
        vmax = float(max(prev_hm.max(), curr_hm.max()))

        prev_boxes = _parse_bboxes(row.get("prior_bboxes"))
        curr_boxes = _parse_bboxes(row.get("current_bboxes"))

        # ----- CNR / PG -----
        prev_mask = boxes_mask_in_model_space(
            prev_boxes, prev_orig, prev_disp.size)
        curr_mask = boxes_mask_in_model_space(
            curr_boxes, curr_orig, curr_disp.size)
        prev_cnr = compute_cnr(prev_hm, prev_mask) if prev_boxes else None
        curr_cnr = compute_cnr(curr_hm, curr_mask) if curr_boxes else None
        prev_pg = compute_pointing_game(prev_hm, prev_mask) if prev_boxes else None
        curr_pg = compute_pointing_game(curr_hm, curr_mask) if curr_boxes else None

        meta_common = {
            "dataset": row["dataset"],
            "patient_id": row["patient_id"],
            "study_id_prev": row["study_id_prev"],
            "study_id_curr": row["study_id_curr"],
            "disease_name": row["disease_name"],
            "comparison": true_label,
        }
        cnr_records.append({**meta_common, "side": "prev",
                            "text_progression": true_label,
                            "n_boxes": len(prev_boxes),
                            "cnr": prev_cnr,
                            "pointing_game": prev_pg})
        cnr_records.append({**meta_common, "side": "curr",
                            "text_progression": true_label,
                            "n_boxes": len(curr_boxes),
                            "cnr": curr_cnr,
                            "pointing_game": curr_pg})

        # ----- render -----
        p_drawn = c_drawn = 0
        if not args.no_render:
            disease_safe = (str(row["disease_name"])
                            .replace(" ", "-").replace("/", "-"))
            base = (
                f"{i:04d}_{row['dataset']}_{row['patient_id']}_"
                f"{row['study_id_prev']}_{row['study_id_curr']}_"
                f"{disease_safe}_{true_label}"
            )
            prev_out = out_dir / f"{base}__prev.png"
            curr_out = out_dir / f"{base}__curr.png"
            p_drawn = render_heatmap_png(
                prev_disp, prev_hm, prev_boxes, prev_orig,
                vmin=vmin, vmax=vmax, alpha=args.alpha, out_path=prev_out,
            )
            c_drawn = render_heatmap_png(
                curr_disp, curr_hm, curr_boxes, curr_orig,
                vmin=vmin, vmax=vmax, alpha=args.alpha, out_path=curr_out,
            )

        if (i + 1) % 25 == 0 or (i + 1) == len(examples) or args.num_examples:
            def _fmt(v):
                return f"{v:.3f}" if v is not None else "  - "
            def _fmt_pg(v):
                return str(v) if v is not None else "-"
            print(
                f"  [{i+1}/{len(examples)}] ds={row['dataset']:11s} "
                f"{str(row['disease_name'])[:18]:18s} | "
                f"prog={true_label:10s} "
                f"prev_box={len(prev_boxes)} curr_box={len(curr_boxes)} "
                f"CNR(prev)={_fmt(prev_cnr)} CNR(curr)={_fmt(curr_cnr)} "
                f"PG(prev)={_fmt_pg(prev_pg)} PG(curr)={_fmt_pg(curr_pg)}"
                + (f"  drew prev={p_drawn} curr={c_drawn}"
                   if not args.no_render else "")
            )

    # ----- aggregate -----
    cnr_df = pd.DataFrame(cnr_records)
    cnr_df["ckpt"] = args.ckpt  # track which checkpoint produced these numbers
    cnr_csv = Path(args.cnr_csv) if args.cnr_csv else (out_dir / "cnr.csv")
    cnr_csv.parent.mkdir(parents=True, exist_ok=True)
    cnr_df.to_csv(cnr_csv, index=False)
    print(f"\n📊 Wrote per-(example,side) CNR table to {cnr_csv}")

    valid = cnr_df.dropna(subset=["cnr"])
    print("\n" + "=" * 70)
    print("CNR SUMMARY (BioViL formula, computed in model 448x448 space)")
    print("=" * 70)
    print(f"  ckpt:           {args.ckpt}")
    print(f"  scored sides:   {len(valid)} / {len(cnr_df)} "
          f"(skipped: no bboxes or empty interior/exterior)")
    if len(valid):
        print(f"  overall CNR:    mean={valid['cnr'].mean():.4f}  "
              f"median={valid['cnr'].median():.4f}  "
              f"std={valid['cnr'].std():.4f}")
        print("\n  per dataset:")
        for ds, sub in valid.groupby("dataset"):
            print(f"    {ds:12s}  n={len(sub):4d}  "
                  f"mean={sub['cnr'].mean():.4f}  "
                  f"median={sub['cnr'].median():.4f}  "
                  f"std={sub['cnr'].std():.4f}")
        print("\n  per side (prev vs curr):")
        for side, sub in valid.groupby("side"):
            print(f"    {side:4s}  n={len(sub):4d}  "
                  f"mean={sub['cnr'].mean():.4f}  "
                  f"median={sub['cnr'].median():.4f}")
        print("\n  per progression label:")
        for lbl, sub in valid.groupby("comparison"):
            print(f"    {lbl:10s}  n={len(sub):4d}  "
                  f"mean={sub['cnr'].mean():.4f}  "
                  f"median={sub['cnr'].median():.4f}")

    valid_pg = cnr_df.dropna(subset=["pointing_game"])
    print("\n" + "=" * 70)
    print("POINTING GAME SUMMARY (argmax in union of boxes, "
          "computed in model 448x448 space)")
    print("=" * 70)
    print(f"  scored sides:   {len(valid_pg)} / {len(cnr_df)} "
          f"(skipped: no bboxes or empty mask in model space)")
    if len(valid_pg):
        print(f"  overall PG:     acc={valid_pg['pointing_game'].mean():.4f}  "
              f"hits={int(valid_pg['pointing_game'].sum())}/"
              f"{len(valid_pg)}")
        print("\n  per dataset:")
        for ds, sub in valid_pg.groupby("dataset"):
            print(f"    {ds:12s}  n={len(sub):4d}  "
                  f"acc={sub['pointing_game'].mean():.4f}  "
                  f"hits={int(sub['pointing_game'].sum())}/{len(sub)}")
        print("\n  per side (prev vs curr):")
        for side, sub in valid_pg.groupby("side"):
            print(f"    {side:4s}  n={len(sub):4d}  "
                  f"acc={sub['pointing_game'].mean():.4f}  "
                  f"hits={int(sub['pointing_game'].sum())}/{len(sub)}")
        print("\n  per progression label:")
        for lbl, sub in valid_pg.groupby("comparison"):
            print(f"    {lbl:10s}  n={len(sub):4d}  "
                  f"acc={sub['pointing_game'].mean():.4f}  "
                  f"hits={int(sub['pointing_game'].sum())}/{len(sub)}")

    if not args.no_render:
        print(f"\n✅ DONE — heatmaps written to {out_dir.resolve()}")
    else:
        print("\n✅ DONE — metrics only (no PNGs written, --no-render set)")


if __name__ == "__main__":
    main()
