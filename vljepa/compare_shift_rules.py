#!/usr/bin/env python3
"""Same cutoff rules for every model on gold, with paired bootstrap CIs.

Works from saved class_signal npz files only (no model, no GPU). Every rule
is applied the same way to every model, in that model's own score units
(cosines with T = 0.07, or logits with T = 1). Rules, from cleanest to least
clean:

  none                  raw argmax
  silver resolved-only  resolved shift minimizing the model's training CE on
                        silver val (VL-JEPA: CBW weights; supervised: balanced)
  silver all-class      per-class bias minimizing class-balanced CE on silver val
  EM prior              silver-calibrated posteriors moved to gold class shares
                        estimated without gold labels (Saerens et al. 2002)
  gold mix prior        same, but with gold's true class shares (uses gold label
                        counts: five numbers, no per-film labels)
  gold cross-fit        per-class bonuses maximizing macro F1 on 4/5 of gold,
                        applied to the held-out 1/5; folds stratified by label
  GOLD oracle           bonuses maximizing macro F1 on all of gold (upper bound;
                        never report it as a result)

The bootstrap resamples gold films (same draw for both models) and reports
model minus supervised for combined, macro F1, and per-class AUROC vs stable.

    python -m vljepa.compare_shift_rules \\
        --vljepa checkpoints_vljepa_lora_5way/class_signal_epoch_5.npz \\
        --supervised checkpoints_supervised_progression_unfrozen/class_signal_epoch_5.npz \\
        --extra imgft_ep4=checkpoints_vljepa_lora_5way_imgft/class_signal_epoch_4.npz
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

from gold_progression_setmatch import topk_set_match
from progression_phrases import CLS_ORDER

from vljepa.check_class_signal import (
    BONUS_GRIDS,
    N_CLS,
    STABLE,
    em_prior_bonus,
    fit_calibration,
    fit_ce_bias,
    macro_f1,
    per_class_f1,
    tune_bonus,
    weighted_ce,
)

VLJEPA_CLASS_WEIGHTS = [0.366, 0.140, 0.233, 0.432, 3.829]
RESOLVED = CLS_ORDER.index("resolved")
RULES = ["none", "silver resolved-only", "silver all-class", "EM prior",
         "gold mix prior", "gold cross-fit", "GOLD oracle"]
USES_GOLD = {"gold mix prior": "gold class counts", "gold cross-fit": "other folds",
             "GOLD oracle": "all of gold"}


def load_scores(path):
    d = np.load(path, allow_pickle=True)
    kind = str(d["score_kind"]) if "score_kind" in d else "cosine"
    return {
        "path": path,
        "kind": kind,
        "temperature": 0.07 if kind == "cosine" else 1.0,
        "val_scores": d["val_scores"].astype(np.float64),
        "val_labels": d["val_labels"].astype(np.int64),
        "gold_scores": d["gold_scores"].astype(np.float64),
        "gold_gts": d["gold_gts"],
        "gold_findings": d["gold_findings"],
    }


def balanced_weights(labels):
    counts = np.bincount(labels, minlength=N_CLS).astype(np.float64)
    return counts.sum() / (N_CLS * np.maximum(counts, 1))


def resolved_only(m, class_weights):
    step = 0.0025 if m["kind"] == "cosine" else 0.025
    grid = np.round(np.arange(-40 * step, 120 * step + 1e-9, step), 4)

    def loss(v):
        s = m["val_scores"].copy()
        s[:, RESOLVED] += v
        return weighted_ce(s, m["val_labels"], m["temperature"], class_weights)

    b = np.zeros(N_CLS)
    b[RESOLVED] = float(min(grid, key=loss))
    return b


def gold_true_prior(gts):
    counts = Counter(c for g in gts for c in g.split("|"))
    p = np.array([counts.get(c, 0) for c in CLS_ORDER], dtype=np.float64)
    return p / p.sum()


def gold_mix_bonus(m):
    t, b = fit_calibration(m["val_scores"], m["val_labels"], m["temperature"])
    prior_src = np.bincount(m["val_labels"], minlength=N_CLS) / len(m["val_labels"])
    prior_gold = np.maximum(gold_true_prior(m["gold_gts"]), 1e-6)
    bonus = t * (b + np.log(prior_gold / prior_src))
    return bonus - bonus[STABLE]


def single_mask(gts):
    return np.array(["|" not in g for g in gts])


def single_labels(gts):
    return np.array([CLS_ORDER.index(g) if "|" not in g else -1 for g in gts], dtype=np.int64)


def make_folds(gts, k, seed):
    rng = np.random.default_rng(seed)
    folds = np.empty(len(gts), dtype=np.int64)
    for g in sorted(set(gts)):
        idx = np.flatnonzero(gts == g)
        rng.shuffle(idx)
        folds[idx] = (np.arange(len(idx)) + rng.integers(k)) % k
    return folds


def cross_fit_bonus(m, folds, k):
    """Per-film bonus rows; each film gets the bonus fit on the other folds."""
    grid = BONUS_GRIDS[m["kind"]]
    labels = single_labels(m["gold_gts"])
    single = labels >= 0
    rows = np.zeros((len(labels), N_CLS))
    per_fold = []
    for f in range(k):
        fit = single & (folds != f)
        b = tune_bonus(m["gold_scores"][fit], labels[fit], macro_f1, grid)
        rows[folds == f] = b
        per_fold.append(b)
    return rows, np.array(per_fold)


def rule_bonuses(m, folds, k, class_weights):
    """Rule name -> (n_gold, 5) bonus rows, plus the summary bonus per rule."""
    n = len(m["gold_gts"])
    labels = single_labels(m["gold_gts"])
    single = labels >= 0
    fixed = {
        "none": np.zeros(N_CLS),
        "silver resolved-only": resolved_only(m, class_weights),
        "silver all-class": fit_ce_bias(m["val_scores"], m["val_labels"], m["temperature"]),
        "EM prior": em_prior_bonus(m["val_scores"], m["val_labels"],
                                   m["gold_scores"], m["temperature"])[0],
        "gold mix prior": gold_mix_bonus(m),
        "GOLD oracle": tune_bonus(m["gold_scores"][single], labels[single], macro_f1,
                                  BONUS_GRIDS[m["kind"]]),
    }
    rows = {name: np.tile(b, (n, 1)) for name, b in fixed.items()}
    rows["gold cross-fit"], per_fold = cross_fit_bonus(m, folds, k)
    fixed["gold cross-fit"] = per_fold.mean(0)
    return rows, fixed, per_fold


def film_outcomes(m, bonus_rows):
    """Per film: set-match group score and argmax class."""
    adj = m["gold_scores"] + bonus_rows
    scores = np.empty(len(adj))
    pred = adj.argmax(1)
    for i in range(len(adj)):
        r = topk_set_match(adj[i].tolist(), m["gold_gts"][i].split("|"), CLS_ORDER,
                           finding=m["gold_findings"][i])
        scores[i] = r.group_score
    return scores, pred


def fast_auroc(s, y):
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    start = np.cumsum(counts) - counts
    ranks = (start + (counts + 1) / 2.0)[inv]
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_vs_stable(scores, labels, idx=None):
    if idx is not None:
        scores, labels = scores[idx], labels[idx]
    out = np.full(N_CLS, np.nan)
    for c in range(N_CLS):
        if c == STABLE:
            continue
        keep = (labels == c) | (labels == STABLE)
        out[c] = fast_auroc(scores[keep, c] - scores[keep, STABLE], labels[keep] == c)
    return out


def ci(x):
    x = x[~np.isnan(x)]
    if not len(x):
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi), float((x > 0).mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--vljepa", required=True, help="VL-JEPA class_signal npz")
    p.add_argument("--supervised", required=True, help="supervised class_signal npz")
    p.add_argument("--extra", action="append", default=[], metavar="NAME=NPZ",
                   help="another model to score the same way (repeatable)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", help="also write all numbers here")
    args = p.parse_args()

    models = {"vljepa": load_scores(args.vljepa), "supervised": load_scores(args.supervised)}
    for item in args.extra:
        name, _, path = item.partition("=")
        if not path:
            p.error(f"--extra expects NAME=NPZ, got {item!r}")
        models[name] = load_scores(path)
    ref = models["supervised"]
    for name, m in models.items():
        if not (np.array_equal(m["gold_gts"], ref["gold_gts"])
                and np.array_equal(m["gold_findings"], ref["gold_findings"])):
            raise SystemExit(f"[rules] {name} and supervised do not hold the same gold films")
        if not np.array_equal(m["val_labels"], ref["val_labels"]):
            raise SystemExit(f"[rules] {name} and supervised do not hold the same silver val films")

    gts = ref["gold_gts"]
    labels = single_labels(gts)
    single = labels >= 0
    true_n = np.bincount(labels[single], minlength=N_CLS)
    folds = make_folds(gts, args.folds, args.seed)
    print(f"[rules] gold films {len(gts)} (single-label {int(single.sum())}), "
          f"silver val {len(ref['val_labels'])}, folds {args.folds}, bootstrap {args.boot}")
    prior_silver = np.bincount(ref["val_labels"], minlength=N_CLS) / len(ref["val_labels"])
    prior_gold = gold_true_prior(gts)
    print("  class shares   " + "".join(f"{c[:9]:>10}" for c in CLS_ORDER))
    print("  silver val     " + "".join(f"{x:>10.3f}" for x in prior_silver))
    print("  gold           " + "".join(f"{x:>10.3f}" for x in prior_gold))

    outcomes, results = {}, {"class_order": list(CLS_ORDER), "models": {}}
    for name, m in models.items():
        weights = (np.array(VLJEPA_CLASS_WEIGHTS) if m["kind"] == "cosine"
                   else balanced_weights(m["val_labels"]))
        rows, fixed, per_fold = rule_bonuses(m, folds, args.folds, weights)
        outcomes[name] = {}
        results["models"][name] = {"npz": m["path"], "kind": m["kind"], "rules": {}}
        print(f"\n=== {name} ({m['kind']}s, T={m['temperature']}, "
              f"training weights {np.round(weights, 3).tolist()})")
        print(f"  {'rule':<21} {'uses gold':<18}"
              + "".join(f"{c[:9]:>10}" for c in CLS_ORDER) + "   <- bonus")
        for rule in RULES:
            print(f"  {rule:<21} {USES_GOLD.get(rule, 'no'):<18}"
                  + "".join(f"{x:>+10.3f}" for x in fixed[rule]))
        print(f"  cross-fit spread across folds (sd): "
              + " ".join(f"{c[:4]} {s:.3f}" for c, s in zip(CLS_ORDER, per_fold.std(0))))
        print(f"\n  {'rule':<21} {'combined':>9} {'macroF1':>8}"
              + "".join(f"{c[:4] + ' F1':>9}" for c in CLS_ORDER)
              + "".join(f"{c[:4] + ' n':>7}" for c in CLS_ORDER))
        for rule in RULES:
            group, pred = film_outcomes(m, rows[rule])
            outcomes[name][rule] = (group, pred)
            f1 = per_class_f1(pred[single], labels[single])
            calls = np.bincount(pred[single], minlength=N_CLS)
            print(f"  {rule:<21} {group.mean():>9.4f} {f1.mean():>8.4f}"
                  + "".join(f"{x:>9.3f}" for x in f1) + "".join(f"{x:>7d}" for x in calls))
            results["models"][name]["rules"][rule] = {
                "bonus": [round(float(x), 4) for x in fixed[rule]],
                "combined": round(float(group.mean()), 4),
                "macro_f1": round(float(f1.mean()), 4),
                "f1": {c: round(float(x), 4) for c, x in zip(CLS_ORDER, f1)},
                "calls": {c: int(x) for c, x in zip(CLS_ORDER, calls)},
            }
        print(f"  {'true (single-label)':<21} {'':>9} {'':>8}" + " " * 9 * N_CLS
              + "".join(f"{x:>7d}" for x in true_n))

    rng = np.random.default_rng(args.seed)
    boots = rng.integers(0, len(gts), size=(args.boot, len(gts)))
    single_idx = np.flatnonzero(single)
    boots_single = rng.integers(0, len(single_idx), size=(args.boot, len(single_idx)))

    def f1_rows(pred_single):
        return np.array([per_class_f1(pred_single[b], labels[single_idx][b])
                         for b in boots_single])

    def macro_f1_rows(pred_single):
        return f1_rows(pred_single).mean(1)

    print(f"\n=== Paired bootstrap on gold, model minus supervised "
          f"(95% CI; P = share of draws where the model is ahead)")
    results["bootstrap"] = {}
    for name in models:
        if name == "supervised":
            continue
        results["bootstrap"][name] = {}
        print(f"\n  {name} vs supervised")
        print(f"  {'rule':<21} {'d combined':>11} {'95% CI':>17} {'P':>5}"
              f" {'d macroF1':>10} {'95% CI':>17} {'P':>5}")
        for rule in RULES:
            g_m, p_m = outcomes[name][rule]
            g_s, p_s = outcomes["supervised"][rule]
            d_comb = (g_m[boots] - g_s[boots]).mean(1)
            d_mf1 = (macro_f1_rows(p_m[single_idx]) - macro_f1_rows(p_s[single_idx]))
            point_c = g_m.mean() - g_s.mean()
            point_f = (per_class_f1(p_m[single], labels[single]).mean()
                       - per_class_f1(p_s[single], labels[single]).mean())
            lo_c, hi_c, pr_c = ci(d_comb)
            lo_f, hi_f, pr_f = ci(d_mf1)
            print(f"  {rule:<21} {point_c:>+11.4f} [{lo_c:>+7.4f},{hi_c:>+7.4f}] {pr_c:>5.2f}"
                  f" {point_f:>+10.4f} [{lo_f:>+7.4f},{hi_f:>+7.4f}] {pr_f:>5.2f}")
            results["bootstrap"][name][rule] = {
                "d_combined": [round(point_c, 4), round(lo_c, 4), round(hi_c, 4), round(pr_c, 3)],
                "d_macro_f1": [round(point_f, 4), round(lo_f, 4), round(hi_f, 4), round(pr_f, 3)],
            }

    print("\n=== Headline: model with its silver resolved-only shift minus supervised with no shift")
    print("  (each model in the form you would report it; 95% CI; P = share of draws ahead)")
    g_s, p_s = outcomes["supervised"]["none"]
    f1_s = f1_rows(p_s[single_idx])
    results["headline"] = {}
    for name in models:
        if name == "supervised":
            continue
        g_m, p_m = outcomes[name]["silver resolved-only"]
        f1_m = f1_rows(p_m[single_idx])
        point_f1 = per_class_f1(p_m[single], labels[single]) - per_class_f1(p_s[single], labels[single])
        rows = [("combined", g_m.mean() - g_s.mean(), (g_m[boots] - g_s[boots]).mean(1)),
                ("macro F1", point_f1.mean(), f1_m.mean(1) - f1_s.mean(1))]
        rows += [(f"{c} F1", point_f1[i], f1_m[:, i] - f1_s[:, i]) for i, c in enumerate(CLS_ORDER)]
        print(f"\n  {name} + silver resolved-only  minus  supervised")
        results["headline"][name] = {}
        for label, point_d, draws in rows:
            lo, hi, pr = ci(draws)
            print(f"  {label:<14} {point_d:>+8.4f}   [{lo:>+7.4f}, {hi:>+7.4f}]   P {pr:.2f}")
            results["headline"][name][label] = [round(float(point_d), 4), round(lo, 4),
                                                round(hi, 4), round(pr, 3)]

    print("\n=== Ranking on gold single-label, AUROC of (class - stable) score; no cutoff involved")
    s_scores = {name: m["gold_scores"][single] for name, m in models.items()}
    s_labels = labels[single]
    point = {name: auroc_vs_stable(s, s_labels) for name, s in s_scores.items()}
    print(f"  {'class':<10}" + "".join(f"{n[:12]:>13}" for n in models))
    for c, cls in enumerate(CLS_ORDER):
        if c == STABLE:
            continue
        print(f"  {cls:<10}" + "".join(f"{point[n][c]:>13.4f}" for n in models))
    results["auroc"] = {n: {c: round(float(point[n][i]), 4) for i, c in enumerate(CLS_ORDER)
                            if i != STABLE} for n in models}
    boot_auc = {name: np.array([auroc_vs_stable(s, s_labels, b) for b in boots_single])
                for name, s in s_scores.items()}
    for name in models:
        if name == "supervised":
            continue
        d = boot_auc[name] - boot_auc["supervised"]
        print(f"\n  {name} minus supervised AUROC (95% CI, P ahead)")
        for c, cls in enumerate(CLS_ORDER):
            if c == STABLE:
                continue
            lo, hi, pr = ci(d[:, c])
            print(f"  {cls:<10} {point[name][c] - point['supervised'][c]:>+8.4f} "
                  f"[{lo:>+7.4f},{hi:>+7.4f}] {pr:>5.2f}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[rules] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
