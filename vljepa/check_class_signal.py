#!/usr/bin/env python3
"""Does a VL-JEPA checkpoint already see each class, or does argmax hide it?

For one checkpoint, every film gets its five sentence cosines on silver val
and on gold. Then:

  1. Ranking checks, no tuning. AUROC of "how close class c is to winning"
     (one class vs the rest), and of "class c minus stable" on films that
     are class c or stable. 0.5 is a random ranking, 1.0 is perfect.
  2. Per-class bonuses added to the cosines before argmax. Stable stays at
     0. The bonuses maximize macro F1 on SILVER VAL only; gold is then
     scored once with them fixed.

High AUROC with few correct calls means the signal is there and the
decision rule hides it. AUROC near 0.5 means the scores do not separate
that class.

    python -m vljepa.check_class_signal --ckpt checkpoints_vljepa_lora_5way/epoch_9.pt
    python -m vljepa.check_class_signal --reuse <scores.npz>   # rerun the math only
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from gold_progression_setmatch import (
    _compute_balanced_metrics,
    print_setmatch_report,
    summarize_setmatch,
    topk_set_match,
)
from progression_phrases import CLS_ORDER

N_CLS = len(CLS_ORDER)
STABLE = CLS_ORDER.index("stable")
BONUS_GRID = np.round(np.arange(-0.15, 0.3001, 0.005), 4)


def _print(msg: str = "") -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------
# Scoring (needs the model; imported lazily so --reuse runs anywhere)
# ----------------------------------------------------------------------
def _target_bank(model, finding, cache):
    import torch.nn.functional as F

    from .prompts import class_target_texts

    if finding not in cache:
        g, _, _ = model.encode_targets(class_target_texts(finding))
        cache[finding] = F.normalize(g.float(), dim=-1)
    return cache[finding]


def score_silver_val(model, device, batch_size, num_workers, limit):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from .cluster_paths import image_roots
    from .dataset import VLJEPAFindingDataset, vljepa_collate_fn

    ds = VLJEPAFindingDataset(
        image_roots(), split="val", train=False, val_fraction=0.1, split_seed=42,
    )
    if limit:
        ds = Subset(ds, range(min(limit, len(ds))))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=vljepa_collate_fn,
    )
    cache = {}
    scores, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="silver val", file=sys.stdout):
            prior = batch["prior_image"].to(device, non_blocking=True)
            current = batch["current_image"].to(device, non_blocking=True)
            img_g, img_p = model.encode_images(prior, current)
            pred = F.normalize(
                model.predict(img_g, img_p, batch["query_text"]).float(), dim=-1,
            )
            tgt = torch.stack(
                [_target_bank(model, f, cache) for f in batch["finding"]]
            )
            scores.append(torch.einsum("bd,bcd->bc", pred, tgt).cpu().numpy())
            labels.append(batch["cls_idx"].numpy())
    return np.concatenate(scores), np.concatenate(labels)


def score_gold(model, device, limit):
    import torch
    from tqdm import tqdm

    from progression_classify import (
        DEFAULT_GOLD_PARQUET,
        discover_gold_image_roots,
        load_image_tensor,
    )

    from .cluster_paths import gold_image_dirs, image_roots
    from .eval_gold import load_gold_groups

    groups = load_gold_groups()
    if limit:
        groups = groups.iloc[:limit].reset_index(drop=True)
    roots = {
        **image_roots(),
        **discover_gold_image_roots(os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))),
        **gold_image_dirs(),
    }
    text_cache = {}
    scores, gts, findings = [], [], []
    skipped = 0
    with torch.no_grad():
        for i in tqdm(range(len(groups)), desc="gold", file=sys.stdout):
            row = groups.iloc[i]
            try:
                prior = load_image_tensor(row["dataset"], row["parent_image_prev"], roots)
                current = load_image_tensor(row["dataset"], row["parent_image_curr"], roots)
            except (FileNotFoundError, OSError):
                skipped += 1
                continue
            s = model.score_classes(
                prior.unsqueeze(0).to(device),
                current.unsqueeze(0).to(device),
                str(row["finding"]),
                text_cache=text_cache,
            )
            scores.append(s.float().cpu().numpy())
            gts.append("|".join(sorted(str(x).strip().lower() for x in row["gt_labels"])))
            findings.append(str(row["finding"]).strip().lower())
    if skipped:
        _print(f"[check] gold skipped missing images: {skipped}")
    return np.stack(scores), np.array(gts, dtype=object), np.array(findings, dtype=object)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def auroc(scores, is_pos) -> float:
    """Mann-Whitney AUROC with tied scores sharing the average rank."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(is_pos, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def win_margin(scores: np.ndarray) -> np.ndarray:
    """Per film and class: cosine to that class minus the best other class."""
    out = np.empty_like(scores)
    for c in range(N_CLS):
        others = np.delete(scores, c, axis=1)
        out[:, c] = scores[:, c] - others.max(axis=1)
    return out


def macro_f1(pred: np.ndarray, labels: np.ndarray) -> float:
    f1s = []
    for c in range(N_CLS):
        tp = int(((pred == c) & (labels == c)).sum())
        n_pred = int((pred == c).sum())
        n_true = int((labels == c).sum())
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_true if n_true else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def macro_recall(pred: np.ndarray, labels: np.ndarray) -> float:
    """Mean per-class recall. Unlike macro F1, it does not depend on class shares."""
    recalls = []
    for c in range(N_CLS):
        n_true = int((labels == c).sum())
        if n_true:
            recalls.append(int(((pred == c) & (labels == c)).sum()) / n_true)
    return float(np.mean(recalls)) if recalls else 0.0


def tune_bonus(scores: np.ndarray, labels: np.ndarray, objective=macro_f1,
               rounds: int = 3) -> np.ndarray:
    """Coordinate ascent on ``objective``. Stable is held at 0."""
    bonus = np.zeros(N_CLS)
    best = objective((scores + bonus).argmax(1), labels)
    for _ in range(rounds):
        changed = False
        for c in range(N_CLS):
            if c == STABLE:
                continue
            for v in BONUS_GRID:
                trial = bonus.copy()
                trial[c] = v
                f = objective((scores + trial).argmax(1), labels)
                if f > best + 1e-9:
                    best, bonus, changed = f, trial, True
        if not changed:
            break
    return bonus


def print_rank_checks(name: str, scores: np.ndarray, labels: np.ndarray) -> None:
    margin = win_margin(scores)
    counts = Counter(labels.tolist())
    _print(f"\n=== {name}: ranking checks (0.5 = random, 1.0 = perfect)")
    _print(f"  {'class':<10} {'n':>6} {'vs rest':>9} {'vs stable':>10} {'n stable':>9}")
    for c, cls in enumerate(CLS_ORDER):
        vs_rest = auroc(margin[:, c], labels == c)
        if c == STABLE:
            vs_stable = "      -"
            n_st = "        -"
        else:
            keep = (labels == c) | (labels == STABLE)
            vs_stable = f"{auroc(scores[keep, c] - scores[keep, STABLE], labels[keep] == c):>10.4f}"
            n_st = f"{counts.get(STABLE, 0):>9d}"
        _print(f"  {cls:<10} {counts.get(c, 0):>6d} {vs_rest:>9.4f} {vs_stable} {n_st}")


def single_label_table(scores, labels, title):
    pred = scores.argmax(1)
    confusion = defaultdict(Counter)
    for g, p in zip(labels, pred):
        confusion[CLS_ORDER[g]][CLS_ORDER[p]] += 1
    n_correct = int((pred == labels).sum())
    m = _compute_balanced_metrics(confusion, list(CLS_ORDER), n_correct)
    _print(f"\n--- {title}")
    _print(f"  accuracy {n_correct}/{len(labels)} = {n_correct / max(len(labels), 1):.4f}   "
           f"macro F1 {m['macro_f1']:.4f}   kappa {m['cohen_kappa']:.4f}")
    _print(f"  {'class':<10} {'n_true':>7} {'n_pred':>7} {'prec':>7} {'recall':>7} {'F1':>7}")
    for k, cls in enumerate(CLS_ORDER):
        _print(f"  {cls:<10} {m['n_true'][cls]:>7d} {m['n_pred'][cls]:>7d} "
               f"{m['per_class_precision'][k]:>7.4f} {m['per_class_recall'][k]:>7.4f} "
               f"{m['per_class_f1'][k]:>7.4f}")
    _print("  confusion (rows = true, cols = called):")
    _print("  " + " " * 10 + "".join(f"{c[:9]:>10}" for c in CLS_ORDER))
    for gt in CLS_ORDER:
        _print(f"  {gt:<10}" + "".join(f"{confusion[gt].get(p, 0):>10d}" for p in CLS_ORDER))
    return m


def gold_results(scores, gts, findings, bonus):
    adj = scores + bonus[None, :]
    return [
        topk_set_match(adj[i].tolist(), gts[i].split("|"), CLS_ORDER, finding=findings[i])
        for i in range(len(adj))
    ]


def report(val_scores, val_labels, gold_scores, gold_gts, gold_findings, show_full):
    _print(f"\n[check] silver val films: {len(val_labels)}   gold groups: {len(gold_gts)}")
    gold_single = np.array(["|" not in g for g in gold_gts])
    gold_single_labels = np.array(
        [CLS_ORDER.index(g) for g in gold_gts[gold_single]], dtype=np.int64,
    )
    gold_single_scores = gold_scores[gold_single]

    print_rank_checks("SILVER VAL", val_scores, val_labels)
    print_rank_checks("GOLD single-label", gold_single_scores, gold_single_labels)

    # Resolved is ~1% of silver but ~6% of gold. Macro F1 rewards rarely
    # calling a rare class, so it can push resolved down; macro recall
    # ignores class shares.
    bonuses = {
        "F1 bonus": tune_bonus(val_scores, val_labels, macro_f1),
        "recall bonus": tune_bonus(val_scores, val_labels, macro_recall),
    }
    _print("\n=== Per-class bonuses chosen on silver val (stable fixed at 0)")
    _print(f"  {'class':<10}" + "".join(f"{k:>14}" for k in bonuses))
    for c, cls in enumerate(CLS_ORDER):
        _print(f"  {cls:<10}" + "".join(f"{b[c]:>+14.3f}" for b in bonuses.values()))
    base_pred = val_scores.argmax(1)
    _print(f"  silver val macro F1 {macro_f1(base_pred, val_labels):.4f} -> "
           + ", ".join(f"{k} {macro_f1((val_scores + b).argmax(1), val_labels):.4f}"
                       for k, b in bonuses.items()))
    _print(f"  silver val macro recall {macro_recall(base_pred, val_labels):.4f} -> "
           + ", ".join(f"{k} {macro_recall((val_scores + b).argmax(1), val_labels):.4f}"
                       for k, b in bonuses.items()))

    single_label_table(gold_single_scores, gold_single_labels, "GOLD single-label, no bonus")
    for name, b in bonuses.items():
        single_label_table(gold_single_scores + b, gold_single_labels,
                           f"GOLD single-label, silver-val {name}")

    runs = {"no bonus": np.zeros(N_CLS), **bonuses}
    summaries = {
        name: summarize_setmatch(gold_results(gold_scores, gold_gts, gold_findings, b))
        for name, b in runs.items()
    }
    _print("\n=== GOLD set-match")
    _print(f"  {'':<14}" + "".join(f"{k:>14}" for k in runs))
    for key, label in (("combined_score", "combined"), ("single_acc", "single acc"),
                       ("multi_jaccard", "multi Jaccard")):
        _print(f"  {label:<14}" + "".join(f"{summaries[k][key]:>14.4f}" for k in runs))
    if show_full:
        for name, b in bonuses.items():
            print_setmatch_report(
                gold_results(gold_scores, gold_gts, gold_findings, b),
                f"vljepa + silver-val {name}",
                ", bonus=" + ",".join(f"{x:+.3f}" for x in b),
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", help="VL-JEPA checkpoint to score")
    p.add_argument("--reuse", help="scores .npz from an earlier run; skips the model")
    p.add_argument("--out", help="where to save scores (default next to the checkpoint)")
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--gold-limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-full-report", action="store_true")
    args = p.parse_args()

    if args.reuse:
        data = np.load(args.reuse, allow_pickle=True)
        _print(f"[check] reusing scores from {args.reuse}")
        report(data["val_scores"], data["val_labels"], data["gold_scores"],
               data["gold_gts"], data["gold_findings"], not args.no_full_report)
        return 0
    if not args.ckpt:
        p.error("--ckpt or --reuse is required")

    import torch

    from .eval_gold import load_vljepa_from_ckpt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print(f"[check] ckpt   = {args.ckpt}")
    _print(f"[check] device = {device}")
    model = load_vljepa_from_ckpt(args.ckpt, device)
    _print(f"[check] text lora = {getattr(model, 'text_lora_path', None)}")

    val_scores, val_labels = score_silver_val(
        model, device, args.batch_size, args.num_workers, args.val_limit,
    )
    gold_scores, gold_gts, gold_findings = score_gold(model, device, args.gold_limit)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)),
        "class_signal_" + os.path.splitext(os.path.basename(args.ckpt))[0] + ".npz",
    )
    np.savez(out, val_scores=val_scores, val_labels=val_labels,
             gold_scores=gold_scores, gold_gts=gold_gts, gold_findings=gold_findings)
    _print(f"[check] saved scores to {out}")
    report(val_scores, val_labels, gold_scores, gold_gts, gold_findings,
           not args.no_full_report)
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        __package__ = "vljepa"
    raise SystemExit(main())
