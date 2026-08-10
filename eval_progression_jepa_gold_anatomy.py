#!/usr/bin/env python3
"""Gold 5-way progression eval with anatomy dual-mask scoring.

Matches the ``experiment/anatomy-jepa-prog-ce`` train rule:

  * prior CXAS anatomy masks soft-pool ``ẑ^c``
  * current masks soft-pool ``z_cur``
  * logit = mean cosine over the 22 ``REQUIRED_CXAS_ANATOMIES``
  * only pairs where **both** images pass the same keep rules used to
    build silver ``filtered_masks_anatomy``:
      - ``min_key_hits`` major-anatomy QC (default 8/10)
      - full allowed CXAS inventory (22 train anatomies, non-empty mass)
      - optional ``--strict`` geometric QC

Gold CXAS masks (from ``run_cxas_gold.sh`` in mask_silver /
ChestXRayAnatomySegmentation) live under::

    <gold-masks-base>/final_gold_{chexpert,mimic,rexgradient}_masks/

CXAS may write either:

  1. one JSON per image: ``{stem}.json`` (flat), or mirrored rel-path, or
  2. one mega COCO JSON for the whole folder (stock ``cxas_segment -ot json``
     on a directory) named like ``final_gold_chexpert_images.json``.

This script resolves both layouts. RexGradient pairs often lack a full
22-inventory and are skipped — same as the silver anatomy filter.

Usage
-----
    # On the cluster (after gold CXAS finished):
    cd /scratch/m000081-pm06/eprakash/anatomy-cond/cxr-temporal-model
    conda activate roentgen
    export PYTHONPATH=$PWD/tempcxr/modules/hi-ml/hi-ml-multimodal/src:$PYTHONPATH

    python eval_progression_jepa_gold_anatomy.py --eval \\
      --ckpt checkpoints_jepa_dynamic_cbw99999_anatjepaonly100_anatprog/best.pt \\
      --gold-masks-base /scratch/m000081-pm06/eprakash/ChestXRayAnatomySegmentation

    # Local default looks for ../mask_silver/ChestXRayAnatomySegmentation
    python eval_progression_jepa_gold_anatomy.py --demo --idx 0 \\
      --ckpt /path/to/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from dataset_combined_jepa import DEFAULT_FINDINGS
from eval_progression_jepa import (
    PROMPT_TEMPLATE,
    _print_eval_summary,
    score_one_pair,
)
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    _normalize_label,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER
from filter_masks_by_anatomy_map import (
    KEY_CLASSES_FOR_QC,
    allowed_cxas_categories,
    compute_key_hits,
    filter_mask_json,
    passes_strict_filter,
)
from silver_masks import (
    N_PATCHES,
    REQUIRED_CXAS_ANATOMIES,
    load_per_category_masks_hw,
    mask_hw_to_patch_weights,
)

_HERE = Path(__file__).resolve().parent

# Where run_cxas_gold.sh writes masks (cluster) / local mask_silver checkout.
_DEFAULT_GOLD_MASKS_BASE_CANDIDATES = [
    os.environ.get("CXAS_GOLD_MASKS_BASE", ""),
    "/scratch/m000081-pm06/eprakash/ChestXRayAnatomySegmentation",
    str(_HERE.parent / "mask_silver" / "ChestXRayAnatomySegmentation"),
    str(_HERE / ".." / "mask_silver" / "ChestXRayAnatomySegmentation"),
    str(_HERE.parent / "ChestXRayAnatomySegmentation"),
]


def _discover_gold_masks_base() -> str:
    for cand in _DEFAULT_GOLD_MASKS_BASE_CANDIDATES:
        if not cand:
            continue
        p = Path(cand).expanduser().resolve()
        # Accept if any final_gold_*_masks dir exists.
        if any((p / f"final_gold_{ds}_masks").is_dir() for ds in DATASETS):
            return str(p)
    # Fall back to the cluster path even if missing (clearer error later).
    return "/scratch/m000081-pm06/eprakash/ChestXRayAnatomySegmentation"


def dataset_masks_root(gold_masks_base: str, dataset: str) -> Path:
    return Path(gold_masks_base).expanduser().resolve() / f"final_gold_{dataset}_masks"


# ---------------------------------------------------------------------------
# Path resolution for gold CXAS JSONs
# ---------------------------------------------------------------------------
class GoldMaskIndex:
    """Lazy index over one dataset's ``final_gold_{ds}_masks`` tree.

    Important: CheXpert stems are often the non-unique ``view1_frontal``.
    We never resolve by bare stem when that would be ambiguous — only by
    full relative path, unique basename, or mega-COCO ``file_name``.
    """

    def __init__(self, root: Path):
        self.root = root
        self._rel_to_path: Optional[Dict[str, Path]] = None
        self._basename_to_paths: Optional[Dict[str, List[Path]]] = None
        self._mega: Optional[dict] = None
        self._mega_by_key: Optional[Dict[str, int]] = None
        self._tmpdir: Optional[tempfile.TemporaryDirectory] = None
        self._extracted: Dict[int, Path] = {}

    def _ensure_file_index(self) -> None:
        if self._rel_to_path is not None:
            return
        self._rel_to_path = {}
        self._basename_to_paths = defaultdict(list)
        if not self.root.is_dir():
            return
        for p in self.root.rglob("*.json"):
            try:
                rel = str(p.relative_to(self.root).with_suffix("")).replace("\\", "/")
            except ValueError:
                rel = p.stem
            self._rel_to_path[rel] = p
            self._basename_to_paths[p.name].append(p)

    def _ensure_mega(self) -> None:
        if self._mega_by_key is not None:
            return
        self._mega_by_key = {}
        self._mega = None
        if not self.root.is_dir():
            return
        # Stock cxas_segment on a directory writes one JSON named after
        # the input folder, e.g. final_gold_chexpert_images.json.
        for p in sorted(self.root.glob("*.json")):
            try:
                with p.open() as f:
                    doc = json.load(f)
            except Exception:
                continue
            images = doc.get("images") or []
            flat = []
            for im in images:
                if isinstance(im, list):
                    flat.extend(x for x in im if isinstance(x, dict))
                elif isinstance(im, dict):
                    flat.append(im)
            if len(flat) < 2 or not doc.get("annotations"):
                continue
            doc["_flat_images"] = flat
            self._mega = doc
            for im in flat:
                fn = str(im.get("file_name", "")).replace("\\", "/")
                iid = int(im["id"])
                # Index unique keys only — never bare stem (collides).
                self._mega_by_key[fn] = iid
                self._mega_by_key[Path(fn).name] = iid
                stem_path = str(Path(fn).with_suffix("")).replace("\\", "/")
                self._mega_by_key[stem_path] = iid
            break

    def resolve_json_path(self, parent_image: str) -> Optional[Path]:
        """Return a on-disk single-image COCO JSON for ``parent_image``."""
        rel = str(parent_image).strip().lstrip("/").replace("\\", "/")
        for prefix in ("chexpert/train/", "chexpert/", "mimic/", "rexgradient/"):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
                break
        rel_nosuffix = str(Path(rel).with_suffix("")).replace("\\", "/")
        name_json = Path(rel).with_suffix(".json").name

        self._ensure_file_index()
        assert self._rel_to_path is not None and self._basename_to_paths is not None

        # 1) Exact relative path (mirrored tree) — unambiguous.
        for key in (rel_nosuffix, rel):
            hit = self._rel_to_path.get(key)
            if hit is not None and hit.is_file():
                return hit
        direct = self.root / Path(rel).with_suffix(".json")
        if direct.is_file():
            return direct

        # 2) Unique basename only (e.g. unique dicom_id.json). Never use
        #    CheXpert-style colliding stems even if only one file remains
        #    on disk (overwrites would silently reuse the wrong mask).
        _COLLIDING = {"view1_frontal", "view2_frontal", "view1_lateral",
                      "view2_lateral", "view3_frontal"}
        hits = self._basename_to_paths.get(name_json, [])
        if (
            Path(name_json).stem not in _COLLIDING
            and len(hits) == 1
            and hits[0].is_file()
        ):
            return hits[0]

        # 3) Mega COCO: match full relative file_name, not bare stem.
        self._ensure_mega()
        if self._mega is None:
            return None
        iid = None
        for key in (rel, Path(rel).name, rel_nosuffix, name_json):
            if key in self._mega_by_key:
                iid = self._mega_by_key[key]
                break
        if iid is None:
            # Unique endswith on full relative path (not stem alone).
            matches = [
                v for k, v in self._mega_by_key.items()
                if k.endswith("/" + Path(rel).name) or k.endswith(rel)
            ]
            uniq = list(dict.fromkeys(matches))
            if len(uniq) == 1:
                iid = uniq[0]
        if iid is None:
            return None
        return self._extract_mega_image(iid)

    def _extract_mega_image(self, image_id: int) -> Path:
        if image_id in self._extracted:
            return self._extracted[image_id]
        assert self._mega is not None
        if self._tmpdir is None:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="gold_cxas_")
        flat = self._mega["_flat_images"]
        images = [im for im in flat if int(im["id"]) == int(image_id)]
        anns = [
            a for a in (self._mega.get("annotations") or [])
            if int(a.get("image_id", -1)) == int(image_id)
        ]
        doc = {
            "categories": self._mega.get("categories") or [],
            "images": images,
            "annotations": anns,
        }
        out = Path(self._tmpdir.name) / f"image_{image_id}.json"
        with out.open("w") as f:
            json.dump(doc, f)
        self._extracted[image_id] = out
        return out


_INDEX_CACHE: Dict[str, GoldMaskIndex] = {}


def get_index(masks_root: Path) -> GoldMaskIndex:
    key = str(masks_root)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = GoldMaskIndex(masks_root)
    return _INDEX_CACHE[key]


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def passes_silver_anatomy_filter(
    mask_data: dict,
    min_key_hits: int = 8,
    strict: bool = False,
) -> Tuple[bool, str]:
    """Same keep rules as ``filter_masks_by_anatomy_map.py`` (default mode).

    1. ``compute_key_hits`` >= ``min_key_hits`` (default 8 / 10 majors)
    2. Every allowed CXAS class for the fixed medgemma inventory is present
    3. Optional ``--strict`` geometric QC
    """
    if min_key_hits > 0:
        hits = compute_key_hits(mask_data)
        if hits < min_key_hits:
            return False, f"key_hits({hits}<{min_key_hits})"

    if strict:
        ok, reason = passes_strict_filter(mask_data)
        if not ok:
            return False, f"strict:{reason}"

    allowed = allowed_cxas_categories()
    _filtered, _present, missing = filter_mask_json(mask_data, allowed)
    if missing:
        # Also require the exact 22 used at train time.
        return False, f"missing_cxas({len(missing)})"

    # Train-time inventory is REQUIRED_CXAS_ANATOMIES (22). Enforce both
    # the filter-script allowed set and that 22-tuple.
    present_names = {
        c["name"]
        for c in mask_data.get("categories", [])
        if "name" in c
    }
    # Categories listed is not enough — need at least one annotation each.
    id_to_name = {
        int(c["id"]): str(c["name"])
        for c in (mask_data.get("categories") or [])
        if "id" in c and "name" in c
    }
    annotated = set()
    for ann in mask_data.get("annotations") or []:
        seg = ann.get("segmentation")
        if seg is None or isinstance(seg, list):
            continue
        name = id_to_name.get(int(ann.get("category_id", -1)))
        if name is not None:
            annotated.add(name)
    missing_22 = [c for c in REQUIRED_CXAS_ANATOMIES if c not in annotated]
    if missing_22:
        return False, f"missing_22({missing_22[0]})"

    # Non-empty mask mass for each of the 22 (raw CXAS often emits empty RLEs).
    try:
        # Write a tiny temp view isn't needed — decode via helper below in
        # caller. Here just check annotation presence; mass checked after decode.
        _ = present_names
    except Exception:
        pass
    return True, "ok"


def load_gold_dual_anatomy_weights(
    gold_masks_base: str,
    dataset: str,
    parent_image_prev: str,
    parent_image_curr: str,
    min_key_hits: int = 8,
    strict: bool = False,
    min_weight_sum: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, bool, str]:
    """Load dual 22-CXAS patch weights for a gold pair.

    Applies the **same keep rules** as silver ``filter_masks_by_anatomy_map``
    (key-hits QC + full allowed CXAS inventory) on **both** prior and
    current, then requires non-empty patch mass per anatomy after warp.
    """
    a = len(REQUIRED_CXAS_ANATOMIES)
    z = torch.zeros(a, N_PATCHES, dtype=torch.float32)
    root = dataset_masks_root(gold_masks_base, dataset)
    if not root.is_dir():
        return z, z, False, f"missing masks dir {root}"

    idx = get_index(root)
    p_path = idx.resolve_json_path(parent_image_prev)
    c_path = idx.resolve_json_path(parent_image_curr)
    if p_path is None or c_path is None:
        return z, z, False, "json_not_found"

    try:
        p_doc = _load_json(p_path)
        c_doc = _load_json(c_path)
    except Exception as exc:
        return z, z, False, f"json_load_failed:{exc}"

    ok_p, reason_p = passes_silver_anatomy_filter(
        p_doc, min_key_hits=min_key_hits, strict=strict,
    )
    if not ok_p:
        return z, z, False, f"prior:{reason_p}"
    ok_c, reason_c = passes_silver_anatomy_filter(
        c_doc, min_key_hits=min_key_hits, strict=strict,
    )
    if not ok_c:
        return z, z, False, f"curr:{reason_c}"

    try:
        prior_hw = load_per_category_masks_hw(p_path)
        curr_hw = load_per_category_masks_hw(c_path)
    except Exception as exc:
        return z, z, False, f"decode_failed:{exc}"
    if prior_hw is None or curr_hw is None:
        return z, z, False, "missing_category_after_decode"

    prior_rows = []
    curr_rows = []
    for cat in REQUIRED_CXAS_ANATOMIES:
        try:
            pw = mask_hw_to_patch_weights(prior_hw[cat], aug_params=None)
            cw = mask_hw_to_patch_weights(curr_hw[cat], aug_params=None)
        except Exception as exc:
            return z, z, False, f"warp_failed:{cat}:{exc}"
        # Do **not** fake mass for empty organs — that would defeat filtering.
        if float(pw.sum()) <= min_weight_sum:
            return z, z, False, f"prior_empty_mass:{cat}"
        if float(cw.sum()) <= min_weight_sum:
            return z, z, False, f"curr_empty_mass:{cat}"
        prior_rows.append(pw)
        curr_rows.append(cw)

    return (
        torch.stack(prior_rows, dim=0),
        torch.stack(curr_rows, dim=0),
        True,
        "ok",
    )


# ---------------------------------------------------------------------------
# Eval / demo
# ---------------------------------------------------------------------------
def run_demo(args, model, gold_df, device):
    import random

    if args.idx is None:
        rng = random.Random(args.seed)
        args.idx = rng.randrange(len(gold_df))
    row = gold_df.iloc[args.idx]
    finding = str(row["finding"])
    gt_label = _normalize_label(row["progression"])
    print(f"\n=== Gold sample {args.idx} of {len(gold_df)} ===")
    print(f"  dataset:       {row['dataset']}")
    print(f"  finding:       {finding}")
    print(f"  ground-truth:  {gt_label}")
    print(f"  prev:          {row['parent_image_prev']}")
    print(f"  curr:          {row['parent_image_curr']}")

    prior = load_image_tensor(
        row["dataset"], row["parent_image_prev"], args.image_roots,
    )
    current = load_image_tensor(
        row["dataset"], row["parent_image_curr"], args.image_roots,
    )
    w_prior, w_curr, active, reason = load_gold_dual_anatomy_weights(
        args.gold_masks_base,
        str(row["dataset"]),
        str(row["parent_image_prev"]),
        str(row["parent_image_curr"]),
        min_key_hits=args.min_key_hits,
        strict=args.strict,
    )
    print(f"  anatomy:       active={active} ({reason})")
    if not active:
        raise RuntimeError(
            "Demo row failed the silver-style anatomy filter. "
            "Pick another --idx or check --gold-masks-base."
        )

    out = score_one_pair(
        model, prior, current, finding, args.prompt_template, device,
        mask_patch_weights_prior=w_prior,
        mask_patch_weights_curr=w_curr,
    )
    best = out["pred_class"]
    pred_label = CLS_ORDER[best]
    print("\nPer-class anatomy dual-mask cosines:")
    for k, cls in enumerate(CLS_ORDER):
        marker = "  <-- argmax" if k == best else ""
        print(
            f"  {cls:<10}  {out['cos_class_scores'][k]:>+.4f}  "
            f"Δnaive={out['cos_class_scores'][k] - out['cos_naive']:>+.4f}"
            f"{marker}"
        )
    print(
        f"\n  Prediction: {pred_label} vs gt {gt_label} => "
        f"{'CORRECT' if pred_label == gt_label else 'WRONG'}"
    )


def run_eval(args, model, gold_df, device):
    if args.limit is not None:
        gold_df = gold_df.head(args.limit).reset_index(drop=True)

    n_correct = n_seen = n_above_naive = 0
    skipped_img = skipped_mask = 0
    skip_reasons: Counter = Counter()
    skip_by_ds: Counter = Counter()
    kept_by_ds: Counter = Counter()
    confusion: Dict[str, Counter] = defaultdict(Counter)
    per_finding: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    cos_class_sums = [0.0] * len(CLS_ORDER)
    naive_sum = 0.0
    text_cache: Dict[str, Tuple] = {}

    print(
        f"\n[eval] gold anatomy dual-mask scoring on {len(gold_df)} rows"
    )
    print(f"[eval] gold_masks_base={args.gold_masks_base}")
    print(
        f"[eval] filter = silver filtered_masks_anatomy rules "
        f"(min_key_hits={args.min_key_hits}/{len(KEY_CLASSES_FOR_QC)}, "
        f"strict={args.strict}, full 22 CXAS with non-empty mass on BOTH sides)"
    )
    for ds in DATASETS:
        root = dataset_masks_root(args.gold_masks_base, ds)
        print(f"[eval]   {ds}: {root}  exists={root.is_dir()}")

    for i in range(len(gold_df)):
        row = gold_df.iloc[i]
        finding = str(row["finding"])
        gt_label = _normalize_label(row["progression"])
        ds = str(row["dataset"])
        try:
            prior = load_image_tensor(
                ds, row["parent_image_prev"], args.image_roots,
            )
            current = load_image_tensor(
                ds, row["parent_image_curr"], args.image_roots,
            )
        except (FileNotFoundError, OSError) as e:
            skipped_img += 1
            if skipped_img <= 5:
                print(f"[eval] skip row {i} missing image: {e}")
            continue

        w_prior, w_curr, active, reason = load_gold_dual_anatomy_weights(
            args.gold_masks_base,
            ds,
            str(row["parent_image_prev"]),
            str(row["parent_image_curr"]),
            min_key_hits=args.min_key_hits,
            strict=args.strict,
        )
        if not active:
            skipped_mask += 1
            skip_reasons[reason.split(":")[0]] += 1
            skip_by_ds[ds] += 1
            if skipped_mask <= 8:
                print(f"[eval] skip row {i} ({ds}): {reason}")
            continue

        out = score_one_pair(
            model, prior, current, finding, args.prompt_template, device,
            text_cache=text_cache,
            mask_patch_weights_prior=w_prior,
            mask_patch_weights_curr=w_curr,
        )
        pred_idx = out["pred_class"]
        pred_label = CLS_ORDER[pred_idx]

        n_seen += 1
        kept_by_ds[ds] += 1
        n_correct += int(pred_label == gt_label)
        confusion[gt_label][pred_label] += 1
        per_finding[finding.lower()][1] += 1
        per_finding[finding.lower()][0] += int(pred_label == gt_label)
        if out["cos_class_scores"][pred_idx] > out["cos_naive"]:
            n_above_naive += 1
        for k in range(len(CLS_ORDER)):
            cos_class_sums[k] += out["cos_class_scores"][k]
        naive_sum += out["cos_naive"]

        if (i + 1) % max(1, len(gold_df) // 20) == 0:
            acc = n_correct / max(1, n_seen)
            print(
                f"[eval]   {i + 1}/{len(gold_df)}  "
                f"kept={n_seen} acc={acc:.4f}  "
                f"skip_img={skipped_img} skip_mask={skipped_mask}"
            )

    print(
        f"\n[eval] kept {n_seen}/{len(gold_df)} pairs with full dual "
        f"22-CXAS inventory"
    )
    print(f"[eval] skipped missing images: {skipped_img}")
    print(f"[eval] skipped incomplete masks: {skipped_mask}")
    if skip_by_ds:
        print("[eval] mask-skips by dataset:")
        for ds, n in sorted(skip_by_ds.items()):
            print(f"         {ds:<12} {n}")
    if skip_reasons:
        print("[eval] mask-skip reasons:")
        for r, n in skip_reasons.most_common():
            print(f"         {n:>5}  {r}")
    if kept_by_ds:
        print("[eval] kept by dataset:")
        for ds, n in sorted(kept_by_ds.items()):
            print(f"         {ds:<12} {n}")

    if n_seen == 0:
        print("No samples evaluated — check --gold-masks-base and CXAS output.")
        return

    _print_eval_summary(
        n_correct=n_correct,
        n_seen=n_seen,
        confusion=confusion,
        per_finding=per_finding,
        cos_class_sums=cos_class_sums,
        naive_sum=naive_sum,
        n_above_naive=n_above_naive,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        default=os.environ.get(
            "JEPA_CKPT",
            "checkpoints_jepa_dynamic_cbw99999_anatjepaonly100_anatprog/best.pt",
        ),
    )
    parser.add_argument("--gold-parquet", default=DEFAULT_GOLD_PARQUET)
    parser.add_argument("--findings-parquet", default=DEFAULT_FINDINGS)
    parser.add_argument(
        "--gold-masks-base",
        default=_discover_gold_masks_base(),
        help="Directory containing final_gold_{ds}_masks/ "
             "(default: auto-detect cluster or ../mask_silver/...).",
    )
    parser.add_argument(
        "--min-key-hits",
        type=int,
        default=8,
        help="Same as filter_masks_by_anatomy_map.py (default 8). "
             "Set 0 to disable key-class QC.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also apply filter_masks_by_anatomy_map --strict geometric QC.",
    )
    parser.add_argument("--prompt-template", default=PROMPT_TEMPLATE)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--eval", action="store_true")
    parser.add_argument("--idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Image roots: prefer final_gold_*_images next to parquet / cwd / script.
    parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    auto_gold_roots = discover_gold_image_roots(parquet_dir)
    # Also search gold-masks-base parent and cwd (cluster CXAS tree often
    # co-locates final_gold_*_images with final_gold_*_masks).
    for base in (
        args.gold_masks_base,
        str(Path(args.gold_masks_base).resolve().parent),
        str(_HERE),
        os.getcwd(),
    ):
        extra = discover_gold_image_roots(base)
        for d, p in extra.items():
            auto_gold_roots.setdefault(d, p)
    image_roots: Dict[str, str] = {**IMAGE_ROOTS, **auto_gold_roots}
    for spec in args.image_root:
        d, p = spec.split("=", 1)
        image_roots[d] = p
    args.image_roots = image_roots
    print("[gold] image roots:")
    for d in DATASETS:
        print(f"  {d}: {image_roots.get(d, '<missing>')}")

    device = torch.device(args.device)
    model = load_jepa_model(args.ckpt, device)
    gold_df = load_gold_pairs(
        args.gold_parquet,
        args.findings_parquet,
    )
    if len(gold_df) == 0:
        raise RuntimeError("No usable gold rows after filtering.")

    if args.demo:
        run_demo(args, model, gold_df, device)
    else:
        run_eval(args, model, gold_df, device)


if __name__ == "__main__":
    main()
