#!/usr/bin/env python3
"""Choose the resolved-only class shift on silver val and save it with a checkpoint.

The shift is the fixed rule from check_class_signal.py --resolved-only:
add one number to the resolved score before argmax, chosen as the value
that minimizes the training loss (class-weighted CE, T = 0.07) on silver
val. Gold labels are never used to choose it; gold metrics are only
reported.

Outputs:
  * vljepa/results/<ckpt dir>/<epoch>.shift.json   (small; tracked in git)
  * vljepa/results/<ckpt dir>/class_signal_<epoch>.npz (copy of the scores)
  * <ckpt dir>/<epoch>_shifted.pt  model weights + config + shift, no
    optimizer state (skip with --no-slim)

    python -m vljepa.save_class_shift --ckpt checkpoints_vljepa_lora_5way/epoch_5.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from gold_progression_setmatch import summarize_setmatch
from progression_phrases import CLS_ORDER

from vljepa.check_class_signal import (
    N_CLS,
    gold_results,
    gold_single,
    per_class_f1,
    weighted_ce,
)

RULE = ("resolved-only shift minimizing class-weighted training CE "
        "(T = 0.07) on silver val")
TRAIN_CLASS_WEIGHTS = [0.366, 0.140, 0.233, 0.432, 3.829]
TEMPERATURE = 0.07
STEP = 0.0025
GRID = np.round(np.arange(-40 * STEP, 120 * STEP + 1e-9, STEP), 4)
RESOLVED = CLS_ORDER.index("resolved")


def choose_resolved_shift(val_scores, val_labels, class_weights):
    def loss(v):
        s = val_scores.copy()
        s[:, RESOLVED] += v
        return weighted_ce(s, val_labels, TEMPERATURE, class_weights)

    best = min(GRID, key=loss)
    shift = np.zeros(N_CLS)
    shift[RESOLVED] = float(best)
    return shift, loss(0.0), loss(best)


def gold_summary(gold_scores, gold_gts, gold_findings, shift):
    single_scores, single_labels = gold_single(gold_scores, gold_gts)
    f1 = per_class_f1((single_scores + shift).argmax(1), single_labels)
    s = summarize_setmatch(gold_results(gold_scores, gold_gts, gold_findings, shift))
    return {
        "combined": round(float(s["combined_score"]), 4),
        "single_acc": round(float(s["single_acc"]), 4),
        "multi_jaccard": round(float(s["multi_jaccard"]), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "f1": {c: round(float(x), 4) for c, x in zip(CLS_ORDER, f1)},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True, help="VL-JEPA epoch_N.pt")
    p.add_argument("--scores", help="class_signal npz (default: next to the checkpoint)")
    p.add_argument("--class-weights", type=float, nargs=N_CLS, default=TRAIN_CLASS_WEIGHTS)
    p.add_argument("--no-slim", action="store_true", help="do not write <epoch>_shifted.pt")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    ckpt_dir = os.path.dirname(ckpt_path)
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    scores_path = args.scores or os.path.join(ckpt_dir, f"class_signal_{stem}.npz")
    if not os.path.isfile(scores_path):
        raise SystemExit(f"[shift] no scores at {scores_path}. Run: "
                         f"bash vljepa/run_ccw.sh check --ckpt {args.ckpt}")

    data = np.load(scores_path, allow_pickle=True)
    kind = str(data["score_kind"]) if "score_kind" in data else "cosine"
    if kind != "cosine":
        raise SystemExit(f"[shift] {scores_path} holds {kind}s; this rule is for VL-JEPA cosines")
    val_scores, val_labels = data["val_scores"], data["val_labels"]
    gold_scores, gold_gts, gold_findings = (
        data["gold_scores"], data["gold_gts"], data["gold_findings"])

    shift, loss0, loss1 = choose_resolved_shift(val_scores, val_labels, args.class_weights)
    raw = gold_summary(gold_scores, gold_gts, gold_findings, np.zeros(N_CLS))
    shifted = gold_summary(gold_scores, gold_gts, gold_findings, shift)

    print(f"[shift] {stem}: resolved shift = {shift[RESOLVED]:+.4f} "
          f"(silver val loss {loss0:.4f} -> {loss1:.4f}, n={len(val_labels)})")
    print(f"  {'gold':<14} {'raw':>8} {'shifted':>8}")
    for key in ("combined", "macro_f1", "single_acc", "multi_jaccard"):
        print(f"  {key:<14} {raw[key]:>8.4f} {shifted[key]:>8.4f}")
    for c in CLS_ORDER:
        print(f"  {c + ' F1':<14} {raw['f1'][c]:>8.4f} {shifted['f1'][c]:>8.4f}")

    info = {
        "source_checkpoint": ckpt_path,
        "class_order": list(CLS_ORDER),
        "class_shift": [round(float(x), 4) for x in shift],
        "rule": RULE,
        "temperature": TEMPERATURE,
        "class_weights": [float(w) for w in args.class_weights],
        "grid": [float(GRID[0]), float(GRID[-1]), STEP],
        "silver_val_n": int(len(val_labels)),
        "silver_val_loss_raw": round(float(loss0), 6),
        "silver_val_loss_shifted": round(float(loss1), 6),
        "gold_raw": raw,
        "gold_shifted": shifted,
        "note": "shift chosen on silver val only; gold metrics are reported, not used",
    }
    out_dir = os.path.join(REPO_ROOT, "vljepa", "results", os.path.basename(ckpt_dir))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{stem}.shift.json")
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)
    npz_copy = os.path.join(out_dir, os.path.basename(scores_path))
    if os.path.abspath(scores_path) != os.path.abspath(npz_copy):
        shutil.copy2(scores_path, npz_copy)
    print(f"[shift] wrote {json_path}")
    print(f"[shift] copied scores to {npz_copy}")

    if not args.no_slim:
        slim_path = os.path.join(ckpt_dir, f"{stem}_shifted.pt")
        if os.path.exists(slim_path) and not args.overwrite:
            print(f"[shift] {slim_path} exists; pass --overwrite to replace it")
        else:
            import torch

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            torch.save({
                "epoch": ckpt.get("epoch"),
                "model": ckpt["model"],
                "vljepa_cfg": ckpt.get("vljepa_cfg", {}),
                "class_shift": info["class_shift"],
                "class_shift_info": {k: v for k, v in info.items()
                                     if k not in ("gold_raw", "gold_shifted")},
            }, slim_path)
            print(f"[shift] wrote {slim_path} (no optimizer state)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
