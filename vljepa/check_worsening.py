#!/usr/bin/env python3
"""Where does VL-JEPA lose worsening against the supervised model?

Part 1 (saved class_signal npz files only, no model, no GPU): for every
finding on gold single-label films, worsening recall of both models, what
worsening films get called instead, and the worsening-vs-stable AUROC
(ranking quality, independent of any cutoff).

Part 2 (--ckpt, loads the VL-JEPA checkpoint): cosine between the class
sentences "{finding} is {class}." under the checkpoint's text encoder, per
finding. Worsening sitting very close to stable or new means the fixed
sentences leave little room to separate them.

    python -m vljepa.check_worsening \\
        --vljepa checkpoints_vljepa_lora_5way/class_signal_epoch_5.npz \\
        --shift-json vljepa/results/checkpoints_vljepa_lora_5way/epoch_5.shift.json \\
        --supervised checkpoints_supervised_progression_unfrozen/class_signal_epoch_5.npz \\
        [--ckpt checkpoints_vljepa_lora_5way/epoch_5.pt]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from progression_phrases import CLS_ORDER

from vljepa.check_class_signal import N_CLS, STABLE, auroc

WORSENING = CLS_ORDER.index("worsening")


def load_single(path, shift=None):
    data = np.load(path, allow_pickle=True)
    scores, gts, findings = data["gold_scores"], data["gold_gts"], data["gold_findings"]
    keep = np.array(["|" not in g for g in gts])
    scores = scores[keep] + (0 if shift is None else np.asarray(shift)[None, :])
    labels = np.array([CLS_ORDER.index(g) for g in gts[keep]], dtype=np.int64)
    return scores, labels, findings[keep]


def worsening_stats(scores, labels, mask):
    pred = scores.argmax(1)
    w = mask & (labels == WORSENING)
    n = int(w.sum())
    recall = float((pred[w] == WORSENING).mean()) if n else float("nan")
    n_called = int((mask & (pred == WORSENING)).sum())
    precision = float((labels[mask & (pred == WORSENING)] == WORSENING).mean()) if n_called else float("nan")
    ws = mask & ((labels == WORSENING) | (labels == STABLE))
    rank = auroc(scores[ws, WORSENING] - scores[ws, STABLE], labels[ws] == WORSENING)
    called_as = Counter(CLS_ORDER[p] for p in pred[w])
    return n, recall, precision, rank, called_as


def per_finding_table(vj, sup):
    vj_scores, vj_labels, vj_findings = vj
    sup_scores, sup_labels, sup_findings = sup
    if len(vj_labels) != len(sup_labels) or not (vj_findings == sup_findings).all():
        raise SystemExit("[worsening] the two npz files do not hold the same gold films")

    print("\n=== Part 1: worsening on gold single-label films, per finding")
    print("  recall = share of true worsening called worsening; prec = share of worsening calls")
    print("  that are right; AUROC = worsening minus stable score on worsening/stable films")
    print("  (ranking only, no cutoff). 'VJ called' = what VL-JEPA calls true worsening films.")
    head = (f"  {'finding':<27} {'n_w':>4} {'VJ rec':>7} {'SUP rec':>7} {'VJ prec':>7} "
            f"{'SUP prec':>8} {'VJ auc':>7} {'SUP auc':>7}  VJ called (impr/stab/wors/new/res)")
    print(head)
    rows = sorted(set(vj_findings), key=lambda f: -int(((vj_findings == f) & (vj_labels == WORSENING)).sum()))
    for f in rows + ["ALL"]:
        mask = np.ones(len(vj_labels), bool) if f == "ALL" else vj_findings == f
        n, vr, vp, va, called = worsening_stats(vj_scores, vj_labels, mask)
        _, sr, sp, sa, _ = worsening_stats(sup_scores, sup_labels, mask)
        if n == 0:
            continue
        called_s = "/".join(str(called.get(c, 0)) for c in CLS_ORDER)
        if f == "ALL":
            print("  " + "-" * (len(head) - 2))
        print(f"  {f:<27} {n:>4d} {vr:>7.3f} {sr:>7.3f} {vp:>7.3f} {sp:>8.3f} "
              f"{va:>7.3f} {sa:>7.3f}  {called_s}")


def class_rates(scores, labels):
    pred = scores.argmax(1)
    out = []
    for c in range(N_CLS):
        n_pred = int((pred == c).sum())
        tp = int(((pred == c) & (labels == c)).sum())
        n_true = int((labels == c).sum())
        out.append((n_true, n_pred, tp / n_true if n_true else float("nan"),
                    tp / n_pred if n_pred else float("nan")))
    return out


def silver_table(vj_path, sup_path, shift):
    vj, sup = np.load(vj_path, allow_pickle=True), np.load(sup_path, allow_pickle=True)
    vj_scores = vj["val_scores"] + (0 if shift is None else np.asarray(shift)[None, :])
    labels = vj["val_labels"]
    if len(labels) != len(sup["val_labels"]) or not (labels == sup["val_labels"]).all():
        raise SystemExit("[worsening] the two npz files do not hold the same silver val films")
    sup_scores = sup["val_scores"]

    print(f"\n=== Part 1b: silver val (n={len(labels)}), how often each model calls each class")
    print(f"  {'class':<10} {'n_true':>7} {'VJ n_pred':>9} {'SUP n_pred':>10} "
          f"{'VJ rec':>7} {'SUP rec':>7} {'VJ prec':>7} {'SUP prec':>8}")
    for c, (v, s) in enumerate(zip(class_rates(vj_scores, labels), class_rates(sup_scores, labels))):
        print(f"  {CLS_ORDER[c]:<10} {v[0]:>7d} {v[1]:>9d} {s[1]:>10d} "
              f"{v[2]:>7.3f} {s[2]:>7.3f} {v[3]:>7.3f} {s[3]:>8.3f}")
    ws = (labels == WORSENING) | (labels == STABLE)
    rank = [auroc(x[ws, WORSENING] - x[ws, STABLE], labels[ws] == WORSENING)
            for x in (vj_scores, sup_scores)]
    print(f"  worsening vs stable AUROC: VJ {rank[0]:.3f}   SUP {rank[1]:.3f}")


def sentence_table(ckpt_path, findings):
    import torch
    import torch.nn.functional as F

    from .eval_gold import load_vljepa_from_ckpt
    from .prompts import class_target_texts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vljepa_from_ckpt(ckpt_path, device, use_shift=False)
    pairs = [("worsening", "stable"), ("worsening", "new"), ("worsening", "improving"),
             ("improving", "stable"), ("resolved", "stable"), ("new", "stable")]
    print(f"\n=== Part 2: cosine between class sentences ({os.path.basename(ckpt_path)} text encoder)")
    print("  1.0 = identical direction. Higher = harder to tell the two classes apart.")
    print(f"  {'finding':<27} " + " ".join(f"{a[:4] + '-' + b[:4]:>9}" for a, b in pairs))
    totals = np.zeros(len(pairs))
    with torch.no_grad():
        for f in findings:
            g, _, _ = model.encode_targets(class_target_texts(f))
            g = F.normalize(g.float(), dim=-1)
            sim = (g @ g.T).cpu().numpy()
            vals = [sim[CLS_ORDER.index(a), CLS_ORDER.index(b)] for a, b in pairs]
            totals += vals
            print(f"  {f:<27} " + " ".join(f"{v:>9.3f}" for v in vals))
    print(f"  {'MEAN':<27} " + " ".join(f"{v:>9.3f}" for v in totals / len(findings)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--vljepa", required=True, help="VL-JEPA class_signal npz")
    p.add_argument("--supervised", required=True, help="supervised class_signal npz")
    p.add_argument("--shift-json", help="apply this saved VL-JEPA class shift")
    p.add_argument("--ckpt", help="VL-JEPA checkpoint for the sentence cosines (part 2)")
    args = p.parse_args()

    shift = None
    if args.shift_json:
        with open(args.shift_json) as f:
            shift = json.load(f)["class_shift"]
        print(f"[worsening] VL-JEPA class shift {shift}")
    vj = load_single(args.vljepa, shift)
    per_finding_table(vj, load_single(args.supervised))
    silver_table(args.vljepa, args.supervised, shift)
    if args.ckpt:
        sentence_table(args.ckpt, sorted(set(vj[2])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
