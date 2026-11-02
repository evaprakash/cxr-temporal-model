#!/usr/bin/env python3
"""Score zero-shot baselines on CheXTemporal gold and save a score file that
``baap_eval/eval_baap.py compare-gold`` reads (single-label, multi-label
set-match, 3-class, per dataset, paired bootstrap).

Every backend uses the 37-phrase bank ``"{finding} is {phrase}"``; a class's
score is its best phrase cosine. Gold groups and their order match the BAAP,
VL-JEPA and supervised score files (groups with a missing image are skipped
the same way).

  biovilt     official BioViL-T, pair-conditioned global embedding
  tila        TILA (lukeingawesome/TILA), pair-conditioned
  alta        ALTA, pair-conditioned (needs ALTA.pth, see --ckpt)
  biovil      BioViL, single image: normalize(current - prior) by default
  biomedclip  BiomedCLIP, single image: normalize(current - prior) by default

The non-BioViL-T backends import ``infer_<backend>.py`` from
``--baselines-root`` (default ``baselines_ext``) through ``baseline_setmatch``.

    python -m vljepa.score_gold_baselines --backend biovilt
    python -m vljepa.score_gold_baselines --backend alta --ckpt /path/to/ALTA.pth

The score files hold per-pair gold labels; keep them out of git.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from progression_phrases import CLS_ORDER

BACKENDS = ["biovilt", "tila", "alta", "biovil", "biomedclip"]
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "baap_eval", "scores")


def gold_roots():
    from progression_classify import DEFAULT_GOLD_PARQUET, discover_gold_image_roots

    from vljepa.cluster_paths import gold_image_dirs, image_roots

    return {
        **image_roots(),
        **discover_gold_image_roots(os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))),
        **gold_image_dirs(),
    }


def make_scorer(args, device):
    """fn(row, prior_path, current_path) -> 5 class scores in CLS_ORDER."""
    if args.backend == "biovilt":
        from eval_progression_biovilt import PROMPT_TEMPLATE, BioViLTPairModel, score_one_pair
        from progression_classify import load_image_tensor

        model = BioViLTPairModel(device)
        cache = {}

        def fn(row, prior, current):
            p = load_image_tensor(row["dataset"], prior, {row["dataset"]: "/"}).to(device)
            c = load_image_tensor(row["dataset"], current, {row["dataset"]: "/"}).to(device)
            return score_one_pair(model, p, c, str(row["finding"]), PROMPT_TEMPLATE,
                                  device, cache)["cos_class_scores"]

        return fn

    from baseline_setmatch import load_path_backend

    score_fn, note = load_path_backend(args.backend, device, args.baselines_root,
                                       ckpt=args.ckpt, image_mode=args.image_mode)
    print(f"[baselines] {args.backend}{note}")
    roots = gold_roots()
    cache = {}
    return lambda row, prior, current: score_fn(row, cache, roots)["cos_class_scores"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--backend", required=True, choices=BACKENDS)
    p.add_argument("--ckpt", help="override the backend's default weights")
    p.add_argument("--image-mode", default="diff", choices=["diff", "curr"],
                   help="single-image backends (biovil, biomedclip): current minus prior, or current only")
    p.add_argument("--baselines-root", default=os.environ.get(
        "BASELINES_ROOT", os.path.join(REPO_ROOT, "baselines_ext")))
    p.add_argument("--name", help="score file name (default: the backend)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    import torch
    from tqdm import tqdm

    from baap_eval.eval_baap import gold_items, print_gold_metrics
    from gold_progression_setmatch import print_setmatch_report, topk_set_match

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.name or args.backend
    fn = make_scorer(args, device)
    items = gold_items(args.limit)

    scores, gts, findings, keys, datasets = [], [], [], [], []
    with torch.no_grad():
        for row, prior, current in tqdm(items, desc=name, file=sys.stdout):
            scores.append(np.asarray(fn(row, prior, current), dtype=np.float64))
            gts.append("|".join(sorted(str(x).strip().lower() for x in row["gt_labels"])))
            findings.append(str(row["finding"]).strip().lower())
            datasets.append(str(row["dataset"]))
            keys.append("|".join(str(row[c]) for c in ("dataset", "patient_id", "study_id_prev",
                                                       "study_id_curr", "finding")))
    scores = np.stack(scores)
    gts, findings = np.array(gts, dtype=object), np.array(findings, dtype=object)
    datasets, keys = np.array(datasets, dtype=object), np.array(keys, dtype=object)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"gold_{name}.npz")
    np.savez(out, gold_scores=scores, gold_gts=gts, gold_findings=findings, gold_keys=keys,
             gold_datasets=datasets, score_kind="cosine", model_name=name,
             backend=args.backend, ckpt=str(args.ckpt), image_mode=args.image_mode)
    print(f"[baselines] saved {out}")
    results = [topk_set_match(scores[i].tolist(), gts[i].split("|"), CLS_ORDER, finding=findings[i])
               for i in range(len(gts))]
    print_setmatch_report(results, name, "")
    print_gold_metrics(name, scores, gts, findings, datasets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
