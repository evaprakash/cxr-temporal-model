#!/usr/bin/env python3
"""Grounding on the gold box set: does the evidence behind each model's call
land on the radiologist's boxes?

Each row of gold_bboxes.parquet is one pair, one finding and one progression
label, with that finding's boxes on the prior and on the current film. For
every row we build one heatmap per film and score it against that film's boxes
with the change-map metrics of jepa_heatmap_progression_pairs (CNR, pointing
game, mIoU of the hottest box-sized area, pixel AUROC, share of map energy in
the boxes).

Two kinds of map (--method):

  gradcam  gradient x activation of the target class's log-probability, summed
           over channels and ReLU'd, on each film's last spatial feature map
           before the two films are mixed (14 x 14 cells).
           BioViL-T, supervised, VL-JEPA: the backbone_to_vit output of the
           BioViL-T image encoder (current half and prior half).
           BAAP: its ViT patch tokens. BAAP pools patches inside anatomy boxes,
           so its map is zero outside them; that is how BAAP decides.
           Target class: the gold label (--target gt, default) or the model's
           own call (--target pred). BAAP has no new or resolved, so on those
           rows it explains worsening or improving instead.
  swap     1 - cos between prior and current patch features (the role-swap map
           of biovilt_change_map_pairs.py). Tests the image encoder, not the
           decision. The frozen VL-JEPA encoder is BioViL-T's, so its swap map
           is BioViL-T's.

--model anatomy is a control with no model: a flat map over the patch cells BAAP
pools for the finding. If BAAP scores no better, its lead comes from being
restricted to those regions, not from where inside them it looks.

All maps are scored in the BioViL-T 448 crop (Resize 512, CenterCrop 448), so
every model is judged on the same pixels; BAAP's 224 map is resampled into it.

  maps     one model -> grounding_eval/out/<name>_<method>.csv, plus a summary
  compare  several maps CSVs, paired bootstrap over pairs against a reference

    bash grounding_eval/run_ccw.sh maps --model biovilt --name biovilt
    bash grounding_eval/run_ccw.sh maps --model supervised --name supervised \\
        --ckpt checkpoints_supervised_progression_unfrozen/epoch_5.pt
    bash grounding_eval/run_ccw.sh maps --model vljepa --name vljepa \\
        --ckpt checkpoints_vljepa_lora_5way/epoch_5_shifted.pt
    bash grounding_eval/run_ccw.sh maps --model baap --name baap
    bash grounding_eval/run_ccw.sh compare --names biovilt supervised vljepa baap

The CSVs and PNGs hold per-film results on PhysioNet data; grounding_eval/out/
is git-ignored. Only aggregate numbers (--json) belong in the public repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from progression_phrases import CLS_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")
FRAME = 448
RESIZE_SHORT = 512
FIT = 1024
GRID = 14
TEMPERATURE = 0.07
BAAP_STANDIN = {"new": "worsening", "resolved": "improving"}
METRICS = ["cnr", "pointing_game", "iou_eqarea", "pixel_auroc", "energy_in_box"]
METRIC_NAMES = {"cnr": "CNR", "pointing_game": "pointing game", "iou_eqarea": "mIoU",
                "pixel_auroc": "pixel AUROC", "energy_in_box": "energy in box"}
EMPTY_MAP = {"cnr": 0.0, "pointing_game": 0, "iou_eqarea": 0.0, "pixel_auroc": 0.5,
             "energy_in_box": 0.0}


def _default_parquet() -> str:
    from vljepa.cluster_paths import chextemporal_dir

    return os.path.join(chextemporal_dir(), "gold_bboxes.parquet")


# ----------------------------------------------------------------------
# Rows
# ----------------------------------------------------------------------
def grounding_rows(parquet: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    """One dict per gold box row with both images on disk."""
    import pandas as pd

    from baap_eval.eval_baap import gold_key
    from jepa_heatmap_progression_pairs import _parse_bboxes, normalize_gold_bboxes_schema
    from progression_classify import _resolve_with_fallbacks, discover_gold_image_roots
    from vljepa.cluster_paths import gold_image_dirs, image_roots

    parquet = parquet or _default_parquet()
    if not os.path.isfile(parquet):
        raise SystemExit(f"[grounding] no box parquet at {parquet} (pass --parquet)")
    df = normalize_gold_bboxes_schema(pd.read_parquet(parquet))
    roots = {**image_roots(), **discover_gold_image_roots(os.path.dirname(os.path.abspath(parquet))),
             **gold_image_dirs()}
    rows, skipped = [], 0
    for i, r in df.iterrows():
        try:
            prev = str(_resolve_with_fallbacks(r["dataset"], r["parent_image_prev"], roots))
            curr = str(_resolve_with_fallbacks(r["dataset"], r["parent_image_curr"], roots))
        except (FileNotFoundError, OSError):
            skipped += 1
            continue
        rows.append({
            "row_id": int(i),
            "pair_id": f"{r['dataset']}|{r['patient_id']}|{r['study_id_prev']}|{r['study_id_curr']}",
            "dataset": str(r["dataset"]),
            "finding": str(r["finding"]).strip().lower(),
            "label": str(r["progression"]),
            "prev_path": prev,
            "curr_path": curr,
            "prev_key": gold_key(r["dataset"], r["parent_image_prev"]),
            "curr_key": gold_key(r["dataset"], r["parent_image_curr"]),
            "prev_boxes": _parse_bboxes(r["prior_bboxes"]),
            "curr_boxes": _parse_bboxes(r["current_bboxes"]),
        })
        if limit and len(rows) >= limit:
            break
    print(f"[grounding] {parquet}: {len(rows)} rows with both images "
          f"({skipped} skipped for a missing image), {len({r['pair_id'] for r in rows})} pairs")
    return rows


# ----------------------------------------------------------------------
# Geometry: original image <-> 448 model frame <-> 1024-fit box coordinates
# ----------------------------------------------------------------------
def frame_geometry(w: int, h: int):
    """Scale and crop offset of torchvision Resize(512) + CenterCrop(448)."""
    if w <= h:
        nw, nh = RESIZE_SHORT, int(RESIZE_SHORT * h / w)
    else:
        nw, nh = int(RESIZE_SHORT * w / h), RESIZE_SHORT
    return nw / w, nh / h, int(round((nw - FRAME) / 2.0)), int(round((nh - FRAME) / 2.0))


def box_mask(boxes, orig_wh, disp_wh) -> np.ndarray:
    """Union of boxes (1024-fit display coordinates) on the 448 frame."""
    w, h = orig_wh
    sx, sy, left, top = frame_geometry(w, h)
    fx, fy = sx * w / disp_wh[0], sy * h / disp_wh[1]
    mask = np.zeros((FRAME, FRAME), dtype=bool)
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        c0, c1 = (max(0, min(FRAME, int(round(v * fx - left)))) for v in (x1, x2))
        r0, r1 = (max(0, min(FRAME, int(round(v * fy - top)))) for v in (y1, y2))
        if c1 > c0 and r1 > r0:
            mask[r0:r1, c0:c1] = True
    return mask


def grid_to_frame(cam):
    """[14, 14] map in the 448 frame's patch grid -> [448, 448]."""
    import torch.nn.functional as F

    return F.interpolate(cam[None, None].float(), size=(FRAME, FRAME), mode="bilinear",
                         align_corners=False)[0, 0].cpu().numpy()


def baap_to_frame(cam, orig_wh):
    """[14, 14] map on BAAP's 224 crop -> [448, 448] on the BioViL-T frame of the same image."""
    import torch
    import torch.nn.functional as F

    from baap_eval.baap_model import IMG_SIZE, RESIZE, resized_shape

    w, h = orig_wh
    sx, sy, left, top = frame_geometry(w, h)
    nh, nw, btop, bleft = resized_shape(h, w)
    off = (RESIZE - IMG_SIZE) / 2
    u = (np.arange(FRAME) + 0.5 + left) / sx / w
    v = (np.arange(FRAME) + 0.5 + top) / sy / h
    gx = (u * nw + bleft - off) / IMG_SIZE * 2 - 1
    gy = (v * nh + btop - off) / IMG_SIZE * 2 - 1
    grid = torch.from_numpy(np.stack(np.meshgrid(gx, gy), -1)).float()[None].to(cam.device)
    out = F.grid_sample(cam[None, None].float(), grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)
    return out[0, 0].cpu().numpy()


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class Film:
    """One image as the BioViL-T family sees it, plus the sizes the boxes need."""

    def __init__(self, path: str, device):
        from PIL import Image

        from dataset_combined import BASE_TRANSFORM, apply_augmentation

        img = Image.open(path).convert("RGB")
        self.orig = img.size
        disp = img.copy()
        disp.thumbnail((FIT, FIT))
        self.disp = disp.size
        self.tensor = apply_augmentation(BASE_TRANSFORM(img), None).unsqueeze(0).to(device)


class BioViLFamily:
    """BioViL-T image encoder + a class scorer. ``score(global, patches, finding)`` -> 5 logits."""

    frame = "biovil"

    def __init__(self, image_encoder, score, device):
        self.enc = image_encoder
        self.score = score
        self.device = device

    def load(self, path):
        return Film(path, self.device)

    def gradcam(self, prior: Film, current: Film, finding: str, target: Optional[int]):
        import torch
        import torch.nn.functional as F

        store = {}
        hook = self.enc.model.encoder.backbone_to_vit.register_forward_hook(
            lambda m, i, o: store.__setitem__("x", o))
        try:
            with torch.enable_grad():
                cur = current.tensor.clone().requires_grad_(True)
                pri = prior.tensor.clone().requires_grad_(True)
                g, p = self.enc(cur, pri)
                logits = self.score(g, p, finding).float()
                t = int(logits.argmax()) if target is None else target
                (grad,) = torch.autograd.grad(F.log_softmax(logits, -1)[t], store["x"])
        finally:
            hook.remove()
        cam = F.relu((grad * store["x"]).sum(1)).detach()
        return cam[1], cam[0], logits.detach().cpu().numpy(), t

    def swap(self, prior: Film, current: Film, finding: str):
        import torch

        with torch.no_grad():
            g, pc = self.enc(current.tensor, prior.tensor)
            _, pp = self.enc(prior.tensor, current.tensor)
            logits = self.score(g, pc, finding).float().cpu().numpy()
        change = (1 - (pc[0] * pp[0]).sum(-1)).view(GRID, GRID)
        return change, change, logits

    def to_frame(self, cam, film: Film):
        return grid_to_frame(cam)


def load_biovilt(device):
    import torch
    import torch.nn.functional as F

    from eval_progression_biovilt import PROMPT_TEMPLATE, BioViLTPairModel, build_phrase_bank

    m = BioViLTPairModel(device)
    for p in list(m.image_encoder.parameters()) + list(m.text_encoder.parameters()):
        p.requires_grad = False
    bank = {}

    def score(g, p, finding):
        if finding not in bank:
            prompts, owner = build_phrase_bank(finding, PROMPT_TEMPLATE)
            with torch.no_grad():
                t, _, _ = m.text_encoder.forward_contrastive(prompts)
            idx = [[i for i, c in enumerate(owner) if c == cls] for cls in CLS_ORDER]
            bank[finding] = (F.normalize(t.float(), dim=-1), idx)
        txt, idx = bank[finding]
        sims = (F.normalize(g.float(), dim=-1) @ txt.T)[0]
        return torch.stack([sims[i].max() for i in idx]) / TEMPERATURE

    return BioViLFamily(m.image_encoder, score, device)


def load_supervised(ckpt, device):
    import torch
    import torch.nn.functional as F

    from vljepa.check_class_signal import SupervisedScorer

    s = SupervisedScorer(ckpt, device)

    def score(g, p, finding):
        with torch.no_grad():
            txt = s._text(finding)
        return s.head(torch.cat([F.normalize(g.float(), dim=-1), txt], dim=-1))[0]

    return BioViLFamily(s.image_encoder, score, device)


def load_vljepa(ckpt, shift_json, device):
    import torch
    import torch.nn.functional as F

    from vljepa.check_class_signal import VLJEPAScorer
    from vljepa.eval_gold import load_vljepa_from_ckpt
    from vljepa.prompts import query_text

    model = load_vljepa_from_ckpt(ckpt, device, shift_json=shift_json)
    for p in model.parameters():
        p.requires_grad = False
    targets = VLJEPAScorer(model)
    shift = model.class_shift.to(device) if model.class_shift is not None else None

    def score(g, p, finding):
        with torch.no_grad():
            tgt = targets._targets(finding)
        pred = F.normalize(model.predict(g, p, [query_text(finding)]).float(), dim=-1)
        cos = (pred @ tgt.T)[0]
        return (cos + shift if shift is not None else cos) / TEMPERATURE

    return BioViLFamily(model.image_encoder, score, device)


class BAAPModel:
    frame = "baap"

    def __init__(self, ckpt, boxes_json, device):
        from baap_eval.eval_baap import BoxSource, PairScorer

        if not os.path.isfile(boxes_json):
            raise SystemExit(f"[grounding] no BAAP boxes at {boxes_json}. Run: "
                             "bash baap_eval/run_ccw.sh boxes --set gold_bboxes")
        self.ps = PairScorer(ckpt, BoxSource(boxes_json, None), "softmax", device)
        for p in self.ps.model.parameters():
            p.requires_grad = False
        self.device = device
        self.row = None

    def load(self, path):
        return Film(path, "cpu")

    def _inputs(self, prior: Film, current: Film, finding: str):
        import torch

        from baap_eval.baap_model import ANATOMY_TO_IDX, ANATOMY_UNK_IDX

        r = self.row
        names, bp, bc = self.ps.anatomy_boxes(r["prev_path"], r["curr_path"], r["prev_key"],
                                              r["curr_key"], finding)
        if names == ["(whole image)"]:
            self.ps.fallbacks += 1
        dev = self.device
        idx = torch.tensor([ANATOMY_TO_IDX.get(a, ANATOMY_UNK_IDX) for a in names], device=dev)
        pp = self.ps._image(r["prev_path"])[0].detach().clone()
        pc = self.ps._image(r["curr_path"])[0].detach().clone()
        return pp, pc, torch.tensor(bp, device=dev), torch.tensor(bc, device=dev), idx

    @staticmethod
    def _five(logp3):
        from baap_eval.eval_baap import NEVER, THREE_IDX

        out = np.full(len(CLS_ORDER), NEVER)
        out[THREE_IDX] = logp3
        return out

    def gradcam(self, prior: Film, current: Film, finding: str, target: Optional[int]):
        import torch
        import torch.nn.functional as F

        from baap_eval.eval_baap import THREE

        pp, pc, bp, bc, idx = self._inputs(prior, current, finding)
        pp.requires_grad_(True)
        pc.requires_grad_(True)
        with torch.enable_grad():
            logp = torch.log(self.ps.model.roi_logits(pp, pc, bp, bc, idx).softmax(-1).mean(0)
                             .clamp(min=1e-12))
            if target is None:
                t3 = int(logp.argmax())
            else:
                t3 = THREE.index(BAAP_STANDIN.get(CLS_ORDER[target], CLS_ORDER[target]))
            gp, gc = torch.autograd.grad(logp[t3], (pp, pc))
        cam_p = F.relu((gp * pp).sum(-1)).view(GRID, GRID).detach()
        cam_c = F.relu((gc * pc).sum(-1)).view(GRID, GRID).detach()
        return cam_p, cam_c, self._five(logp.detach().cpu().numpy()), CLS_ORDER.index(THREE[t3])

    def swap(self, prior: Film, current: Film, finding: str):
        import torch
        import torch.nn.functional as F

        pp, pc, bp, bc, idx = self._inputs(prior, current, finding)
        with torch.no_grad():
            logp = torch.log(self.ps.model.roi_logits(pp, pc, bp, bc, idx).softmax(-1).mean(0)
                             .clamp(min=1e-12))
        change = (1 - F.cosine_similarity(pp, pc, dim=-1)).view(GRID, GRID)
        return change, change, self._five(logp.cpu().numpy())

    def to_frame(self, cam, film: Film):
        return baap_to_frame(cam, film.orig)


class AnatomyPrior(BAAPModel):
    """Control: a flat map over the patch cells BAAP pools for this finding, no model.
    Its pred column is meaningless."""

    def _flat(self, prior: Film, current: Film, finding: str):
        from baap_eval.baap_model import box_patch_mask

        _, _, bp, bc, _ = self._inputs(prior, current, finding)
        return tuple(box_patch_mask(b).any(0).float().view(GRID, GRID) for b in (bp, bc))

    def gradcam(self, prior: Film, current: Film, finding: str, target: Optional[int]):
        mp, mc = self._flat(prior, current, finding)
        return mp, mc, np.zeros(len(CLS_ORDER)), -1 if target is None else target

    def swap(self, prior: Film, current: Film, finding: str):
        mp, mc = self._flat(prior, current, finding)
        return mp, mc, np.zeros(len(CLS_ORDER))


# ----------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------
def side_metrics(heat: np.ndarray, mask: np.ndarray) -> Dict:
    from jepa_heatmap_progression_pairs import compute_change_map_side_metrics

    if not mask.any():
        return {k: None for k in METRICS} | {"empty_map": None}
    if not np.isfinite(heat).all() or heat.max() <= 0:
        return dict(EMPTY_MAP) | {"empty_map": 1}
    m = compute_change_map_side_metrics(heat, mask)
    return {k: m[k] for k in METRICS} | {"empty_map": 0}


def render(path, prior: Film, current: Film, heats, masks, title):
    from matplotlib import cm
    from PIL import Image, ImageDraw

    from dataset_combined import BASE_TRANSFORM

    panels = []
    for film, heat, mask in zip((prior, current), heats, masks):
        base = np.asarray(BASE_TRANSFORM(Image.open(film.path).convert("L")), dtype=np.float32) / 255
        h = heat / heat.max() if heat.max() > 0 else heat
        rgb = 0.55 * np.repeat(base[..., None], 3, -1) + 0.45 * cm.jet(h)[..., :3]
        im = Image.fromarray((rgb * 255).astype(np.uint8))
        edge = mask ^ np.roll(mask, 1, 0) | mask ^ np.roll(mask, 1, 1)
        arr = np.asarray(im).copy()
        arr[edge] = (255, 255, 255)
        panels.append(Image.fromarray(arr))
    out = Image.new("RGB", (FRAME * 2 + 8, FRAME + 24), "black")
    out.paste(panels[0], (0, 24))
    out.paste(panels[1], (FRAME + 8, 24))
    ImageDraw.Draw(out).text((4, 4), title, fill="white")
    out.save(path)


def maps(args) -> int:
    import pandas as pd
    import torch
    from tqdm import tqdm

    if args.model in ("supervised", "vljepa") and not args.ckpt:
        raise SystemExit(f"[grounding] --model {args.model} needs --ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "biovilt":
        model = load_biovilt(device)
    elif args.model == "supervised":
        model = load_supervised(args.ckpt, device)
    elif args.model == "vljepa":
        model = load_vljepa(args.ckpt, args.shift_json, device)
    else:
        from baap_eval.eval_baap import DEFAULT_CKPT, default_boxes

        cls = AnatomyPrior if args.model == "anatomy" else BAAPModel
        model = cls(args.ckpt or DEFAULT_CKPT, args.boxes or default_boxes("gold"), device)

    rows = grounding_rows(args.parquet, args.limit)
    name = f"{args.name}_{args.method}"
    os.makedirs(OUT_DIR, exist_ok=True)
    png_dir = os.path.join(OUT_DIR, "png", name)
    if args.render:
        os.makedirs(png_dir, exist_ok=True)
    records, small = [], 0
    for n, r in enumerate(tqdm(rows, desc=name, file=sys.stdout)):
        prior, current = model.load(r["prev_path"]), model.load(r["curr_path"])
        small += sum(max(f.orig) <= FIT for f in (prior, current))
        prior.path, current.path = r["prev_path"], r["curr_path"]
        model.row = r
        gt = CLS_ORDER.index(r["label"])
        if args.method == "gradcam":
            cam_p, cam_c, logits, target = model.gradcam(
                prior, current, r["finding"], None if args.target == "pred" else gt)
        else:
            cam_p, cam_c, logits = model.swap(prior, current, r["finding"])
            target = -1
        heats = (model.to_frame(cam_p, prior), model.to_frame(cam_c, current))
        masks = (box_mask(r["prev_boxes"], prior.orig, prior.disp),
                 box_mask(r["curr_boxes"], current.orig, current.disp))
        pred = CLS_ORDER[int(np.argmax(logits))]
        for side, heat, mask, boxes in zip(("prev", "curr"), heats, masks,
                                           (r["prev_boxes"], r["curr_boxes"])):
            records.append({
                "row_id": r["row_id"], "pair_id": r["pair_id"], "dataset": r["dataset"],
                "finding": r["finding"], "label": r["label"], "side": side,
                "target": CLS_ORDER[target] if target >= 0 else "", "pred": pred,
                "n_boxes": len(boxes), "box_area": float(mask.mean()),
                **side_metrics(heat, mask),
            })
        if n < args.render:
            render(os.path.join(png_dir, f"{n:04d}_{r['dataset']}_{r['finding'].replace(' ', '-')}"
                                f"_{r['label']}.png"), prior, current, heats, masks,
                   f"{args.name} {args.method}  {r['finding']}: gold {r['label']}, "
                   f"pred {pred}  (prior | current)")
    df = pd.DataFrame(records)
    out = args.out or os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(out, index=False)
    print(f"[grounding] wrote {out}")
    print(f"[grounding] films no larger than {FIT} px (boxes are in their original pixels): {small}")
    if isinstance(model, BAAPModel):
        print(f"[grounding] BAAP boxes from {dict(model.ps.boxes.used)}; rows scored on the "
              f"whole image (no anatomy box): {model.ps.fallbacks}")
    summarize(df, name)
    return 0


def summarize(df, name: str) -> None:
    v = df.dropna(subset=["cnr"])
    print(f"\n=== {name}: {len(v)} scored films ({len(df) - len(v)} without a box), "
          f"{v['pair_id'].nunique()} pairs, empty maps {int(v['empty_map'].sum())}")
    print("  " + f"{'':16}{'n':>6}" + "".join(f"{METRIC_NAMES[m]:>15}" for m in METRICS))

    def line(label, sub):
        print("  " + f"{label:16}{len(sub):>6}" + "".join(f"{sub[m].mean():>15.4f}" for m in METRICS))

    line("all", v)
    for col in ("dataset", "side", "label"):
        for key, sub in v.groupby(col):
            line(f"{col} {key}"[:16], sub)
    print(f"  mean box area (share of frame) {v['box_area'].mean():.3f}")


# ----------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------
def compare(args) -> int:
    import pandas as pd

    frames = {}
    for n in args.names:
        path = n if n.endswith(".csv") else os.path.join(OUT_DIR, f"{n}_{args.method}.csv")
        name = os.path.splitext(os.path.basename(path))[0] if n.endswith(".csv") else n
        frames[name] = pd.read_csv(path).dropna(subset=["cnr"]).set_index(["row_id", "side"])
    names = list(frames)
    if args.ref not in frames:
        raise SystemExit(f"[compare] --ref {args.ref} is not among {names}")
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    for k, f in frames.items():
        if len(f) != len(common):
            print(f"[compare] {k}: {len(f) - len(common)} films not scored by every model, dropped")
    base = frames[args.ref].loc[common]
    pairs = base["pair_id"].to_numpy()
    uniq, pid = np.unique(pairs, return_inverse=True)
    n_pairs = len(uniq)
    cnt = np.bincount(pid, minlength=n_pairs).astype(float)
    rng = np.random.default_rng(args.seed)
    w = np.stack([np.bincount(rng.integers(0, n_pairs, n_pairs), minlength=n_pairs)
                  for _ in range(args.boot)]).astype(float)

    def sums(k, m):
        return np.bincount(pid, weights=frames[k].loc[common, m].to_numpy(float), minlength=n_pairs)

    print(f"\n=== Grounding ({args.method}) on {len(common)} films from {n_pairs} pairs")
    print("  " + f"{'metric':16}" + "".join(f"{k:>14}" for k in names))
    result = {"method": args.method, "films": len(common), "pairs": int(n_pairs), "means": {},
              "vs_ref": {}}
    for m in METRICS:
        means = {k: float(sums(k, m).sum() / cnt.sum()) for k in names}
        result["means"][m] = means
        print("  " + f"{METRIC_NAMES[m]:16}" + "".join(f"{means[k]:>14.4f}" for k in names))
    print(f"\n=== Paired bootstrap over pairs ({args.boot} resamples), model minus {args.ref}")
    print("  95% range spanning 0 = tie. P = share of resamples where the model is ahead.")
    for k in names:
        if k == args.ref:
            continue
        print(f"\n  {k} minus {args.ref}")
        result["vs_ref"][k] = {}
        for m in METRICS:
            d = sums(k, m) - sums(args.ref, m)
            boot = (w @ d) / (w @ cnt)
            delta = float(d.sum() / cnt.sum())
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p = float((boot > 0).mean())
            result["vs_ref"][k][m] = {"delta": delta, "lo": float(lo), "hi": float(hi), "p": p}
            print(f"  {METRIC_NAMES[m]:16}{delta:+.4f}   [{lo:+.4f}, {hi:+.4f}]   P {p:.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[compare] wrote {args.json}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("maps")
    a.add_argument("--model", required=True,
                   choices=["biovilt", "supervised", "vljepa", "baap", "anatomy"],
                   help="anatomy: control, a flat map over the anatomy cells BAAP pools")
    a.add_argument("--name", required=True)
    a.add_argument("--ckpt")
    a.add_argument("--shift-json", help="VL-JEPA class shift, if the checkpoint has none")
    a.add_argument("--boxes", help="BAAP CXAS boxes (default baap_eval/boxes/cxas_gold.json)")
    a.add_argument("--method", default="gradcam", choices=["gradcam", "swap"])
    a.add_argument("--target", default="gt", choices=["gt", "pred"])
    a.add_argument("--parquet", help="gold box parquet (default CheXTemporal/gold_bboxes.parquet)")
    a.add_argument("--render", type=int, default=0, help="write PNGs for the first N rows")
    a.add_argument("--limit", type=int)
    a.add_argument("--out")

    a = sub.add_parser("compare")
    a.add_argument("--names", nargs="+", required=True,
                   help="model names given to maps (or paths to their CSVs)")
    a.add_argument("--method", default="gradcam", choices=["gradcam", "swap"])
    a.add_argument("--ref", default="supervised")
    a.add_argument("--boot", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--json", help="also write the aggregate numbers here")

    args = p.parse_args()
    return {"maps": maps, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
