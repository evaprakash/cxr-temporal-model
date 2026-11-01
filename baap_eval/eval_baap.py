#!/usr/bin/env python3
"""BAAP (github.com/chenyuming052/BAAP, release baap-v1.0) on our gold set and
on the clean MS-CXR-T pairs.

BAAP is a 3-way (improving / stable / worsening) classifier per anatomy box.
It has no finding input: for each (pair, finding) we feed the anatomy boxes
BAAP's own evaluator uses for that finding, average the per-box softmax, and
take the result as the pair's prediction. It cannot say "new" or "resolved";
those two scores are fixed far below the others, so on gold every new or
resolved film counts as a miss and their F1 is 0.

Anatomy boxes. BAAP was trained and evaluated with Chest ImaGenome scene-graph
boxes, which exist only for MIMIC images. Most of gold is CheXpert and
ReXGradient, so boxes come from CXAS anatomy masks (the same segmenter as our
filtered_masks_anatomy) for every image. ``--scene-graph-dir`` switches MIMIC
images to their scene-graph boxes instead.

  boxes         CXAS boxes for every gold or MS-CXR-T image (GPU; needs cxas)
  gold          score gold, save baap_eval/scores/gold_baap.npz, print metrics
  mscxrt        score mscxrt_clean.csv, save mscxrt_scores/baap.npz so that
                ``python -m vljepa.eval_mscxrt compare`` includes BAAP
  compare-gold  BAAP vs other gold score files, with a paired bootstrap
  qa            draw the boxes BAAP sees on a few pairs, to eyeball alignment

    bash baap_eval/run_ccw.sh boxes --set gold
    bash baap_eval/run_ccw.sh boxes --set mscxrt
    bash baap_eval/run_ccw.sh gold
    bash baap_eval/run_ccw.sh mscxrt
    bash baap_eval/run_ccw.sh compare-gold \\
        --model supervised=checkpoints_supervised_progression_unfrozen/class_signal_epoch_5.npz \\
        --model vljepa=checkpoints_vljepa_lora_5way/class_signal_epoch_5.npz:vljepa/results/checkpoints_vljepa_lora_5way/epoch_5.shift.json

Box files, score files and QA images hold PhysioNet-derived data; they are
git-ignored. Keep them out of the public repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from progression_phrases import CLS_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(HERE, "ckpt", "baap-v1-paperbest.ckpt")
BOX_DIR = os.path.join(HERE, "boxes")
SCORE_DIR = os.path.join(HERE, "scores")
DEFAULT_CLEAN = os.path.join(REPO_ROOT, "mscxrt_clean.csv")
DEFAULT_MSCXRT_SCORES = os.path.join(REPO_ROOT, "mscxrt_scores")
N_CLS = len(CLS_ORDER)
THREE = ["improving", "stable", "worsening"]
THREE_IDX = [CLS_ORDER.index(c) for c in THREE]
NEVER = -1e9

# ----------------------------------------------------------------------
# Finding -> anatomy boxes
# ----------------------------------------------------------------------
LUNG_ZONES = [
    "right lung", "left lung",
    "right lower lung zone", "left lower lung zone",
    "right mid lung zone", "left mid lung zone",
    "right upper lung zone", "left upper lung zone",
]
PLEURAL = [
    "right costophrenic angle", "left costophrenic angle",
    "right lower lung zone", "left lower lung zone",
    "right lung", "left lung",
]
# The first five are BAAP's DISEASE_ANATOMY_MAP, unchanged. The rest are the
# other gold findings, mapped to the nearest of those regions.
FINDING_ANATOMIES: Dict[str, List[str]] = {
    "consolidation": LUNG_ZONES,
    "pneumonia": LUNG_ZONES,
    "edema": [
        "right lung", "left lung",
        "right lower lung zone", "left lower lung zone",
        "right mid lung zone", "left mid lung zone",
        "right costophrenic angle", "left costophrenic angle",
    ],
    "pleural effusion": PLEURAL,
    "pneumothorax": [
        "right apical zone", "left apical zone",
        "right upper lung zone", "left upper lung zone",
        "right lung", "left lung",
    ],
    "lung opacity": LUNG_ZONES,
    "atelectasis": LUNG_ZONES,
    "lung lesion": LUNG_ZONES,
    "pleural other": PLEURAL,
    "cardiomegaly": ["cardiac silhouette"],
    "enlarged cardiomediastinum": ["mediastinum", "upper mediastinum", "cardiac silhouette"],
}

# CXAS class -> Chest ImaGenome anatomy name used by BAAP.
CXAS_TO_BAAP = {
    "right lung": "right lung",
    "left lung": "left lung",
    "right lung base": "right lower lung zone",
    "left lung base": "left lower lung zone",
    "right mid zone lung": "right mid lung zone",
    "left mid zone lung": "left mid lung zone",
    "right upper zone lung": "right upper lung zone",
    "left upper zone lung": "left upper lung zone",
    "right apical zone lung": "right apical zone",
    "left apical zone lung": "left apical zone",
    "heart": "cardiac silhouette",
    "cardiomediastinum": "mediastinum",
    "upper mediastinum": "upper mediastinum",
}
# CXAS has no costophrenic-angle class. Approximated as the lateral 45% and
# lowest 20% of the lung box (patient right = image left on a frontal film).
CP_WIDTH, CP_HEIGHT = 0.45, 0.20


def norm_finding(f: str) -> str:
    return str(f).strip().lower().replace("_", " ")


def mimic_key(path: str) -> str:
    i = str(path).find("files/p")
    if i < 0:
        raise ValueError(f"no files/p in MIMIC path: {path}")
    return "mimic/" + str(path)[i + len("files/"):]


def gold_key(dataset: str, rel: str) -> str:
    return f"gold/{dataset}/{str(rel).strip().lstrip('/')}"


class BoxSource:
    """Anatomy boxes in BAAP's 224 crop frame for one image."""

    def __init__(self, cxas_json: Optional[str], scene_graph_dir: Optional[str]):
        self.cxas = {}
        if cxas_json:
            with open(cxas_json) as f:
                self.cxas = json.load(f)["boxes"]
        self.sg_dir = scene_graph_dir
        self.used = Counter()

    def _scene_graph(self, key: str):
        from baap_eval.baap_model import imagenome224_to_crop

        if not self.sg_dir or not (key.startswith("mimic/") or key.startswith("gold/mimic/")):
            return None
        dicom = os.path.splitext(os.path.basename(key))[0]
        path = os.path.join(self.sg_dir, f"{dicom}_SceneGraph.json")
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            sg = json.load(f)
        out = {}
        for obj in sg.get("objects", []):
            name = str(obj.get("bbox_name", "")).strip().lower()
            if name:
                out[name] = imagenome224_to_crop([obj["x1"], obj["y1"], obj["x2"], obj["y2"]])
        return out

    def _cxas(self, key: str, hw):
        from baap_eval.baap_model import frac_box_to_crop

        raw = self.cxas.get(key)
        if raw is None:
            return None
        frac = {CXAS_TO_BAAP[k]: v for k, v in raw.items() if k in CXAS_TO_BAAP}
        for side in ("right", "left"):
            lung = frac.get(f"{side} lung")
            if lung is None:
                continue
            x1, y1, x2, y2 = lung
            w, h = x2 - x1, y2 - y1
            if side == "right":
                frac[f"{side} costophrenic angle"] = [x1, y2 - CP_HEIGHT * h, x1 + CP_WIDTH * w, y2]
            else:
                frac[f"{side} costophrenic angle"] = [x2 - CP_WIDTH * w, y2 - CP_HEIGHT * h, x2, y2]
        return {k: frac_box_to_crop(v, hw) for k, v in frac.items()}

    def get(self, key: str, hw) -> Dict[str, List[float]]:
        boxes = self._scene_graph(key)
        if boxes is not None:
            self.used["scene graph"] += 1
            return boxes
        boxes = self._cxas(key, hw)
        if boxes is not None:
            self.used["cxas"] += 1
            return boxes
        self.used["none"] += 1
        return {}


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
class PairScorer:
    def __init__(self, ckpt: str, boxes: BoxSource, aggregation: str, device):
        from baap_eval.baap_model import load_baap

        self.model, self.info = load_baap(ckpt, device)
        self.boxes = boxes
        self.aggregation = aggregation
        self.device = device
        self.cache: Dict[str, Tuple] = {}
        self.fallbacks = 0

    def _image(self, path: str):
        from baap_eval.baap_model import load_image

        if path not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            img, hw = load_image(path)
            patches = self.model.patches(img.unsqueeze(0).to(self.device))[0]
            self.cache[path] = (patches, hw)
        return self.cache[path]

    def anatomy_boxes(self, prior_path, current_path, prior_key, current_key, finding):
        """(anatomy names, prior boxes, current boxes) BAAP sees for this pair."""
        from baap_eval.baap_model import IMG_SIZE

        _, hw_p = self._image(prior_path)
        _, hw_c = self._image(current_path)
        bp = self.boxes.get(prior_key, hw_p)
        bc = self.boxes.get(current_key, hw_c)
        want = FINDING_ANATOMIES.get(norm_finding(finding), LUNG_ZONES)
        names = [a for a in want if a in bp and a in bc]
        if not names:
            full = [0.0, 0.0, float(IMG_SIZE), float(IMG_SIZE)]
            return ["(whole image)"], [full], [full]
        return names, [bp[a] for a in names], [bc[a] for a in names]

    def score(self, prior_path, current_path, prior_key, current_key, finding) -> Tuple[np.ndarray, int]:
        """Five scores in CLS_ORDER (log-probabilities; new / resolved = NEVER) and #boxes."""
        import torch

        from baap_eval.baap_model import ANATOMY_TO_IDX, ANATOMY_UNK_IDX, aggregate

        pp, _ = self._image(prior_path)
        pc, _ = self._image(current_path)
        names, bp, bc = self.anatomy_boxes(prior_path, current_path, prior_key, current_key, finding)
        if names == ["(whole image)"]:
            self.fallbacks += 1
        dev = self.device
        idx = torch.tensor([ANATOMY_TO_IDX.get(a, ANATOMY_UNK_IDX) for a in names], device=dev)
        logits = self.model.logits(pp, pc, torch.tensor(bp, device=dev), torch.tensor(bc, device=dev), idx)
        out = np.full(N_CLS, NEVER)
        out[THREE_IDX] = aggregate(logits, self.aggregation)
        return out, len(names)


def default_boxes(which: str) -> str:
    return os.path.join(BOX_DIR, f"cxas_{which}.json")


def gold_items(limit=None):
    """(group row, prior path, current path) for every gold group with both images."""
    from dataset_combined_jepa import DEFAULT_FINDINGS
    from gold_progression_setmatch import group_gold_by_pair_finding
    from progression_classify import (
        DEFAULT_GOLD_PARQUET,
        _resolve_with_fallbacks,
        discover_gold_image_roots,
        load_gold_pairs,
    )
    from vljepa.cluster_paths import gold_image_dirs, image_roots

    groups = group_gold_by_pair_finding(load_gold_pairs(DEFAULT_GOLD_PARQUET, DEFAULT_FINDINGS))
    if limit:
        groups = groups.iloc[:limit].reset_index(drop=True)
    roots = {
        **image_roots(),
        **discover_gold_image_roots(os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))),
        **gold_image_dirs(),
    }
    items, skipped = [], 0
    for i in range(len(groups)):
        row = groups.iloc[i]
        try:
            prior = str(_resolve_with_fallbacks(row["dataset"], row["parent_image_prev"], roots))
            current = str(_resolve_with_fallbacks(row["dataset"], row["parent_image_curr"], roots))
        except (FileNotFoundError, OSError):
            skipped += 1
            continue
        items.append((row, prior, current))
    if skipped:
        print(f"[baap] gold groups skipped for a missing image: {skipped}")
    return items


def mscxrt_items(clean: str, mimic_roots: List[str], limit=None):
    """(row, prior path, current path) for each clean MS-CXR-T pair."""
    import pandas as pd

    df = pd.read_csv(clean, dtype=str)
    if limit:
        df = df.iloc[:limit]

    def find(local, original):
        if isinstance(local, str) and os.path.isfile(local):
            return local
        rel = mimic_key(original)[len("mimic/"):]
        for root in mimic_roots:
            for cand in (os.path.join(root, "files", rel), os.path.join(root, rel)):
                if os.path.isfile(cand):
                    return cand
        raise FileNotFoundError(original)

    return [(row, find(row["prev_path"], row["img_path_prev"]), find(row["curr_path"], row["img_path_curr"]))
            for _, row in df.iterrows()]


# ----------------------------------------------------------------------
# boxes
# ----------------------------------------------------------------------
def place_cxas_weights(weights: str, model_name: str) -> None:
    from pathlib import Path

    root = os.environ.get("CXAS_PATH") or os.environ.get("HOME", ".")
    expected = Path(root) / ".cxas" / "weights" / f"{model_name}.pth"
    if expected.exists():
        return
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.symlink_to(Path(weights).expanduser().resolve())
    print(f"[boxes] linked {expected} -> {weights}")


def boxes(args) -> int:
    import torch
    from tqdm import tqdm

    if args.set == "gold":
        pairs = [((gold_key(r["dataset"], r["parent_image_prev"]), p),
                  (gold_key(r["dataset"], r["parent_image_curr"]), c))
                 for r, p, c in gold_items(args.limit)]
    elif args.set == "gold_bboxes":
        from grounding_eval.eval_grounding import grounding_rows

        pairs = [((r["prev_key"], r["prev_path"]), (r["curr_key"], r["curr_path"]))
                 for r in grounding_rows(None, args.limit)]
    else:
        pairs = [((mimic_key(r["img_path_prev"]), p), (mimic_key(r["img_path_curr"]), c))
                 for r, p, c in mscxrt_items(args.clean, args.mimic_root, args.limit)]
    todo = dict(kv for pair in pairs for kv in pair)
    out_path = args.out or default_boxes("gold" if args.set == "gold_bboxes" else args.set)
    done = {}
    if os.path.isfile(out_path) and not args.overwrite:
        with open(out_path) as f:
            done = json.load(f)["boxes"]
    todo = {k: v for k, v in todo.items() if k not in done}
    print(f"[boxes] {args.set}: {len(todo) + len(done)} images, {len(done)} already in {out_path}")
    if not todo:
        return 0

    try:
        from cxas import CXAS
        from cxas.label_mapper import id2label_dict
    except ImportError:
        raise SystemExit("[boxes] needs the cxas package (pip install cxas, or the "
                         "ChestXRayAnatomySegmentation checkout on PYTHONPATH)")
    if args.cxas_weights:
        place_cxas_weights(args.cxas_weights, args.cxas_model)
    gpus = args.gpus if torch.cuda.is_available() else "cpu"
    model = CXAS(model_name=args.cxas_model, gpus=gpus)
    channel = {name: int(k) for k, name in id2label_dict.items() if name in CXAS_TO_BAAP}
    missing = sorted(set(CXAS_TO_BAAP) - set(channel))
    if missing:
        raise SystemExit(f"[boxes] CXAS labels not found: {missing}")

    def save():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"source": "cxas", "cxas_model": args.cxas_model,
                       "box_format": "x1,y1,x2,y2 as fractions of the image", "boxes": done}, f)

    failed = 0
    with torch.no_grad():
        for n, (key, path) in enumerate(tqdm(sorted(todo.items()), desc="cxas", file=sys.stdout)):
            sample = model.fileloader.load_file(path)
            if sample["file_size"] == -1:
                failed += 1
                continue
            sample["filename"], sample["file_size"] = [sample["filename"]], [sample["file_size"]]
            seg = model.model(sample)["segmentation_preds"][0]
            h, w = seg.shape[-2:]
            found = {}
            for name, ch in channel.items():
                m = seg[ch].bool()
                rows, cols = torch.nonzero(m.any(1)), torch.nonzero(m.any(0))
                if len(rows) == 0:
                    continue
                found[name] = [round(int(cols[0]) / w, 5), round(int(rows[0]) / h, 5),
                               round((int(cols[-1]) + 1) / w, 5), round((int(rows[-1]) + 1) / h, 5)]
            done[key] = found
            if (n + 1) % 200 == 0:
                save()
    save()
    short = Counter(k for k, v in done.items() if len(v) < len(CXAS_TO_BAAP))
    print(f"[boxes] wrote {out_path}: {len(done)} images, {failed} failed to load, "
          f"{len(short)} missing at least one anatomy")
    return 1 if failed else 0


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def group_scores(scores, gts, findings):
    from gold_progression_setmatch import topk_set_match

    return np.array([topk_set_match(scores[i].tolist(), gts[i].split("|"), CLS_ORDER,
                                    finding=findings[i]).group_score for i in range(len(gts))])


def f1_from_counts(tp, n_pred, n_true):
    denom = n_pred + n_true
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)


class GoldTable:
    """Per-group indicators so every metric is a weighted sum (cheap to bootstrap)."""

    def __init__(self, scores, gts, findings):
        self.n = len(gts)
        self.group = group_scores(scores, gts, findings)
        single = np.array(["|" not in g for g in gts])
        label = np.array([CLS_ORDER.index(g) if s else -1 for g, s in zip(gts, single)])
        pred5 = scores.argmax(1)
        pred3 = np.array(THREE_IDX)[scores[:, THREE_IDX].argmax(1)]
        in3 = single & np.isin(label, THREE_IDX)
        self.cols = {}
        for c in range(N_CLS):
            self.cols[("tp5", c)] = single & (pred5 == c) & (label == c)
            self.cols[("pred5", c)] = single & (pred5 == c)
            self.cols[("true5", c)] = single & (label == c)
        for c in THREE_IDX:
            self.cols[("tp3", c)] = in3 & (pred3 == c) & (label == c)
            self.cols[("pred3", c)] = in3 & (pred3 == c)
            self.cols[("true3", c)] = in3 & (label == c)
        self.cols[("single",)] = single
        self.cols[("single_ok",)] = single & (pred5 == label)
        self.cols[("in3",)] = in3
        self.cols[("in3_ok",)] = in3 & (pred3 == label)
        self.mat = np.stack([self.group] + [v.astype(float) for v in self.cols.values()], axis=1)
        self.pos = {k: i + 1 for i, k in enumerate(self.cols)}

    def metrics(self, w=None):
        """Metrics for row weights w (counts per group; None = each group once)."""
        w = np.ones((1, self.n)) if w is None else np.atleast_2d(w)
        sums = w @ self.mat
        n = w.sum(1)
        get = lambda *k: sums[:, self.pos[k]]
        f15 = np.stack([f1_from_counts(get("tp5", c), get("pred5", c), get("true5", c))
                        for c in range(N_CLS)], 1)
        f13 = np.stack([f1_from_counts(get("tp3", c), get("pred3", c), get("true3", c))
                        for c in THREE_IDX], 1)
        out = {
            "combined": sums[:, 0] / n,
            "single_acc": get("single_ok") / np.maximum(get("single"), 1),
            "macro_f1": f15.mean(1),
            "acc_3class": get("in3_ok") / np.maximum(get("in3"), 1),
            "macro_f1_3class": f13.mean(1),
        }
        for c, cls in enumerate(CLS_ORDER):
            out[f"f1_{cls}"] = f15[:, c]
        return out


METRIC_ROWS = ["combined", "single_acc", "macro_f1"] + [f"f1_{c}" for c in CLS_ORDER] + [
    "acc_3class", "macro_f1_3class"]


def print_gold_metrics(name, scores, gts, findings, datasets=None):
    t = GoldTable(scores, gts, findings)
    m = {k: float(v[0]) for k, v in t.metrics().items()}
    n_single = int(t.cols[("single",)].sum())
    print(f"\n=== GOLD: {name} ({t.n} groups, {n_single} single-label, "
          f"{int(t.cols[('in3',)].sum())} single-label improving/stable/worsening)")
    print(f"  combined {m['combined']:.4f}   single acc {m['single_acc']:.4f}   "
          f"macro F1 (5 classes) {m['macro_f1']:.4f}")
    print(f"  {'class':<10} {'F1':>7} {'called':>7} {'true':>6}")
    for c, cls in enumerate(CLS_ORDER):
        print(f"  {cls:<10} {m['f1_' + cls]:>7.3f} {int(t.cols[('pred5', c)].sum()):>7d} "
              f"{int(t.cols[('true5', c)].sum()):>6d}")
    print(f"  improving/stable/worsening films only, argmax over those 3: "
          f"acc {m['acc_3class']:.4f}, macro F1 {m['macro_f1_3class']:.4f}")
    if datasets is not None:
        print(f"  {'dataset':<12} {'groups':>6} {'combined':>9} {'3-class acc':>12}")
        for ds in sorted(set(datasets)):
            keep = datasets == ds
            sub = GoldTable(scores[keep], gts[keep], findings[keep]).metrics()
            print(f"  {ds:<12} {int(keep.sum()):>6d} {float(sub['combined'][0]):>9.4f} "
                  f"{float(sub['acc_3class'][0]):>12.4f}")
    return m


# ----------------------------------------------------------------------
# gold / mscxrt scoring
# ----------------------------------------------------------------------
def make_scorer(args, which):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    box_path = args.boxes or default_boxes(which)
    if not os.path.isfile(box_path) and not args.scene_graph_dir:
        raise SystemExit(f"[baap] no boxes at {box_path}. Run: bash baap_eval/run_ccw.sh boxes --set {which}")
    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"[baap] no checkpoint at {args.ckpt} (see the download step in baap_eval/run_ccw.sh)")
    src = BoxSource(box_path if os.path.isfile(box_path) else None, args.scene_graph_dir)
    scorer = PairScorer(args.ckpt, src, args.aggregation, device)
    hp = scorer.info["hyper_parameters"]
    print(f"[baap] {args.ckpt} (epoch {scorer.info['epoch']}, emb {hp.get('emb_dim')}, "
          f"anatomy emb {hp.get('use_anatomy_emb')}) on {device}; boxes: {box_path}"
          + (f" + scene graphs {args.scene_graph_dir}" if args.scene_graph_dir else "")
          + f"; aggregation: {args.aggregation}")
    return scorer


def gold(args) -> int:
    from tqdm import tqdm

    from gold_progression_setmatch import print_setmatch_report, topk_set_match

    scorer = make_scorer(args, "gold")
    items = gold_items(args.limit)
    scores, gts, findings, keys, datasets, n_boxes = [], [], [], [], [], []
    for row, prior, current in tqdm(items, desc="gold", file=sys.stdout):
        s, k = scorer.score(prior, current, gold_key(row["dataset"], row["parent_image_prev"]),
                            gold_key(row["dataset"], row["parent_image_curr"]), row["finding"])
        scores.append(s)
        n_boxes.append(k)
        gts.append("|".join(sorted(str(x).strip().lower() for x in row["gt_labels"])))
        findings.append(str(row["finding"]).strip().lower())
        datasets.append(str(row["dataset"]))
        keys.append("|".join(str(row[c]) for c in ("dataset", "patient_id", "study_id_prev",
                                                   "study_id_curr", "finding")))
    scores = np.stack(scores)
    gts, findings = np.array(gts, dtype=object), np.array(findings, dtype=object)
    datasets, keys = np.array(datasets, dtype=object), np.array(keys, dtype=object)
    print(f"[baap] boxes from: {dict(scorer.boxes.used)}; pairs with no usable anatomy box "
          f"(scored on the whole image): {scorer.fallbacks}; boxes per pair: median "
          f"{int(np.median(n_boxes))}")

    out = args.out or os.path.join(SCORE_DIR, "gold_baap.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, gold_scores=scores, gold_gts=gts, gold_findings=findings, gold_keys=keys,
             gold_datasets=datasets, n_boxes=np.array(n_boxes), score_kind="logprob",
             model_name="baap", ckpt=args.ckpt, aggregation=args.aggregation)
    print(f"[baap] saved {out}")
    results = [topk_set_match(scores[i].tolist(), gts[i].split("|"), CLS_ORDER, finding=findings[i])
               for i in range(len(gts))]
    print_setmatch_report(results, "baap", f", aggregation={args.aggregation}")
    print_gold_metrics("baap", scores, gts, findings, datasets)
    return 0


def mscxrt(args) -> int:
    from tqdm import tqdm

    from vljepa.eval_mscxrt import print_metrics

    scorer = make_scorer(args, "mscxrt")
    items = mscxrt_items(args.clean, args.mimic_root, args.limit)
    scores, n_boxes = [], []
    for row, prior, current in tqdm(items, desc="mscxrt", file=sys.stdout):
        s, k = scorer.score(prior, current, mimic_key(row["img_path_prev"]),
                            mimic_key(row["img_path_curr"]), row["finding"])
        scores.append(s)
        n_boxes.append(k)
    scores = np.stack(scores)
    labels = np.array([CLS_ORDER.index(r["label"]) for r, _, _ in items], dtype=np.int64)
    findings = np.array([r["finding"] for r, _, _ in items], dtype=object)
    keys = np.array([r["key"] for r, _, _ in items], dtype=object)
    print(f"[baap] boxes from: {dict(scorer.boxes.used)}; pairs scored on the whole image: "
          f"{scorer.fallbacks}")
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.name}.npz")
    np.savez(out, scores=scores, labels=labels, findings=findings, keys=keys, name=args.name,
             model="baap", ckpt=args.ckpt, score_kind="logprob", n_boxes=np.array(n_boxes))
    print(f"[baap] saved {out}")
    print_metrics({args.name: (scores, labels, findings)})
    return 0


# ----------------------------------------------------------------------
# compare-gold
# ----------------------------------------------------------------------
def load_gold_npz(spec: str):
    """NAME=NPZ or NAME=NPZ:SHIFT_JSON -> (name, scores with shift, gts, findings)."""
    name, rest = spec.split("=", 1)
    path, shift_path = (rest.split(":", 1) + [None])[:2]
    d = np.load(path, allow_pickle=True)
    scores = d["gold_scores"].astype(np.float64)
    if shift_path:
        with open(shift_path) as f:
            info = json.load(f)
        if list(info["class_order"]) != list(CLS_ORDER):
            raise SystemExit(f"[compare] class order mismatch in {shift_path}")
        scores = scores + np.array(info["class_shift"], dtype=np.float64)
    return name, scores, d["gold_gts"], d["gold_findings"]


def compare_gold(args) -> int:
    runs = {}
    b = np.load(args.baap, allow_pickle=True)
    runs["baap"] = (b["gold_scores"].astype(np.float64), b["gold_gts"], b["gold_findings"])
    for spec in args.model:
        name, s, g, f = load_gold_npz(spec)
        runs[name] = (s, g, f)
    ref = args.ref
    if ref not in runs:
        raise SystemExit(f"[compare] --ref {ref!r} is not among {sorted(runs)}")
    _, ref_g, ref_f = runs[ref]
    for name, (s, g, f) in runs.items():
        if len(g) != len(ref_g) or not (np.array_equal(g, ref_g) and np.array_equal(f, ref_f)):
            raise SystemExit(
                f"[compare] {name} and {ref} do not hold the same gold groups in the same order "
                f"({len(g)} vs {len(ref_g)}). Both must be scored on the same gold images.")

    tables = {name: GoldTable(s, g, f) for name, (s, g, f) in runs.items()}
    point = {name: {k: float(v[0]) for k, v in t.metrics().items()} for name, t in tables.items()}
    print(f"\n=== GOLD ({len(ref_g)} groups). macro F1 is over the 5 single-label classes; "
          f"the 3-class columns use only improving/stable/worsening films and argmax over those 3")
    print(f"  {'metric':<18}" + "".join(f"{n[:14]:>15}" for n in runs))
    for k in METRIC_ROWS:
        print(f"  {k:<18}" + "".join(f"{point[n][k]:>15.4f}" for n in runs))

    rng = np.random.default_rng(args.seed)
    n = len(ref_g)
    w = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(args.boot)]).astype(float)
    boot = {name: t.metrics(w) for name, t in tables.items()}
    print(f"\n=== Paired bootstrap ({args.boot} resamples of the same gold groups), model minus {ref}")
    print("  95% range = where the true difference most likely lies; if it spans 0, it is a tie.")
    print("  P = share of resamples where the model is ahead.")
    report = {"point": point, "diff_vs_" + ref: {}}
    for name in runs:
        if name == ref:
            continue
        print(f"\n  {name} minus {ref}")
        report["diff_vs_" + ref][name] = {}
        for k in METRIC_ROWS:
            d = boot[name][k] - boot[ref][k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = float((d > 0).mean())
            diff = point[name][k] - point[ref][k]
            print(f"  {k:<18} {diff:>+8.4f}   [{lo:>+7.4f}, {hi:>+7.4f}]   P {p:.2f}")
            report["diff_vs_" + ref][name][k] = {"diff": diff, "lo": float(lo), "hi": float(hi), "p": p}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[compare] wrote {args.json}")
    return 0


# ----------------------------------------------------------------------
# qa
# ----------------------------------------------------------------------
def qa(args) -> int:
    from PIL import Image, ImageDraw

    scorer = make_scorer(args, args.set)
    if args.set == "gold":
        items = [(r, p, c, gold_key(r["dataset"], r["parent_image_prev"]),
                  gold_key(r["dataset"], r["parent_image_curr"])) for r, p, c in gold_items()]
    else:
        items = [(r, p, c, mimic_key(r["img_path_prev"]), mimic_key(r["img_path_curr"]))
                 for r, p, c in mscxrt_items(args.clean, args.mimic_root)]
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(items), size=min(args.n, len(items)), replace=False)
    os.makedirs(args.out_dir, exist_ok=True)
    from baap_eval.baap_model import load_image

    for j in pick:
        row, prior, current, kp, kc = items[j]
        names, bp, bc = scorer.anatomy_boxes(prior, current, kp, kc, row["finding"])
        tiles = []
        for path, boxes in ((prior, bp), (current, bc)):
            x, _ = load_image(path)
            arr = ((x[0].numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
            tile = Image.fromarray(arr).convert("RGB").resize((448, 448))
            draw = ImageDraw.Draw(tile)
            for name, bx in zip(names, boxes):
                draw.rectangle([v * 2 for v in bx], outline=(255, 64, 64), width=2)
                draw.text((bx[0] * 2 + 3, bx[1] * 2 + 2), name, fill=(255, 255, 0))
            tiles.append(tile)
        out = Image.new("RGB", (896, 448))
        out.paste(tiles[0], (0, 0))
        out.paste(tiles[1], (448, 0))
        fn = os.path.join(args.out_dir, f"{args.set}_{j}_{norm_finding(row['finding']).replace(' ', '-')}.png")
        out.save(fn)
        print(f"[qa] {fn}: {row['finding']} -> {', '.join(names)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    mimic_default = [os.environ.get("MIMIC_CXR_JPG", "/nvme/data/cxr/mimic-cxr-jpg/2.0.0")]

    def scoring_args(a):
        a.add_argument("--ckpt", default=DEFAULT_CKPT)
        a.add_argument("--boxes", help="CXAS box file (default baap_eval/boxes/cxas_<set>.json)")
        a.add_argument("--scene-graph-dir",
                       help="Chest ImaGenome silver_dataset/scene_graph; MIMIC images use these boxes")
        a.add_argument("--aggregation", default="softmax", choices=["softmax", "mean_logits"],
                       help="how per-anatomy outputs become one pair prediction (BAAP paper: softmax)")

    a = sub.add_parser("boxes")
    a.add_argument("--set", required=True, choices=["gold", "mscxrt", "gold_bboxes"],
                   help="gold_bboxes: images of the grounding box set, added to cxas_gold.json")
    a.add_argument("--out")
    a.add_argument("--clean", default=DEFAULT_CLEAN)
    a.add_argument("--mimic-root", nargs="*", default=mimic_default)
    a.add_argument("--gpus", default="0")
    a.add_argument("--cxas-model", default="UNet_ResNet50_default")
    a.add_argument("--cxas-weights", help="pre-downloaded CXAS .pth (skips its Google Drive download)")
    a.add_argument("--overwrite", action="store_true")
    a.add_argument("--limit", type=int)

    a = sub.add_parser("gold")
    scoring_args(a)
    a.add_argument("--out")
    a.add_argument("--limit", type=int)

    a = sub.add_parser("mscxrt")
    scoring_args(a)
    a.add_argument("--clean", default=DEFAULT_CLEAN)
    a.add_argument("--mimic-root", nargs="*", default=mimic_default)
    a.add_argument("--out-dir", default=DEFAULT_MSCXRT_SCORES)
    a.add_argument("--name", default="baap")
    a.add_argument("--limit", type=int)

    a = sub.add_parser("compare-gold")
    a.add_argument("--baap", default=os.path.join(SCORE_DIR, "gold_baap.npz"))
    a.add_argument("--model", action="append", default=[], metavar="NAME=NPZ[:SHIFT_JSON]",
                   help="gold score file (class_signal_*.npz); add :<epoch>.shift.json for a VL-JEPA shift")
    a.add_argument("--ref", default="supervised")
    a.add_argument("--boot", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--json", help="also write the numbers here")

    a = sub.add_parser("qa")
    scoring_args(a)
    a.add_argument("--set", required=True, choices=["gold", "mscxrt"])
    a.add_argument("--clean", default=DEFAULT_CLEAN)
    a.add_argument("--mimic-root", nargs="*", default=mimic_default)
    a.add_argument("--n", type=int, default=12)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out-dir", default=os.path.join(HERE, "qa"))

    args = p.parse_args()
    return {"boxes": boxes, "gold": gold, "mscxrt": mscxrt, "compare-gold": compare_gold,
            "qa": qa}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
