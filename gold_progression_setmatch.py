"""Top-|GT| set-match metrics for multi-progression CheXTemporal gold groups.

For a ``(pair, finding)`` with ground-truth label set ``GT``:

  1. Rank the 5 class cosine scores
  2. Take top ``k = |GT|`` classes as prediction set ``Ŷ``
  3. Recall = |Ŷ ∩ GT| / |GT|
     Precision = |Ŷ ∩ GT| / k
     Jaccard = |Ŷ ∩ GT| / |Ŷ ∪ GT|

Single-label groups (|GT|=1) keep ordinary argmax accuracy.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

from progression_phrases import CLS_ORDER


def _compute_balanced_metrics(
    confusion: Dict,
    classes: List[str],
    n_correct: int,
) -> Dict:
    """Macro P/R/F1 + Cohen's kappa (same convention as eval_progression_jepa)."""
    n_true = {gt: sum(confusion[gt].values()) for gt in classes}
    n_pred = {
        p: sum(confusion[gt].get(p, 0) for gt in classes) for p in classes
    }
    total = sum(n_true.values())

    per_class_recall: List[float] = []
    per_class_precision: List[float] = []
    per_class_f1: List[float] = []
    for cls in classes:
        tp = confusion[cls].get(cls, 0)
        rec = tp / n_true[cls] if n_true[cls] else 0.0
        prec = tp / n_pred[cls] if n_pred[cls] else 0.0
        f1 = (
            2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        )
        per_class_recall.append(rec)
        per_class_precision.append(prec)
        per_class_f1.append(f1)

    n = max(1, len(classes))
    if total > 0:
        p_o = n_correct / total
        p_e = sum((n_true[c] / total) * (n_pred[c] / total) for c in classes)
        kappa = (
            (p_o - p_e) / (1.0 - p_e) if abs(1.0 - p_e) > 1e-12 else 0.0
        )
        majority_class = max(classes, key=lambda c: n_true[c])
        majority_acc = n_true[majority_class] / total
    else:
        kappa = float("nan")
        majority_class = ""
        majority_acc = float("nan")

    return {
        "n_true": n_true,
        "n_pred": n_pred,
        "total": total,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "macro_recall": sum(per_class_recall) / n,
        "macro_precision": sum(per_class_precision) / n,
        "macro_f1": sum(per_class_f1) / n,
        "cohen_kappa": kappa,
        "majority_class": majority_class,
        "majority_acc": majority_acc,
    }

PAIR_FINDING_KEYS = [
    "dataset",
    "patient_id",
    "study_id_curr",
    "study_id_prev",
    "finding_lc",
]


@dataclass(frozen=True)
class SetMatchResult:
    gt: Tuple[str, ...]
    pred_set: Tuple[str, ...]
    argmax: str
    k: int
    recall: float
    precision: float
    jaccard: float
    # 0/1 exact set match via top-k (recall == 1 and precision == 1)
    exact: float
    # Combined-table score: 0/1 for |GT|=1, Jaccard for |GT|>1
    group_score: float
    is_multi: bool
    finding: str = ""
    cos_class_scores: Tuple[float, ...] = field(default_factory=tuple)


def topk_set_match(
    cos_class_scores: Sequence[float],
    gt_labels: Iterable[str],
    classes: Sequence[str] = CLS_ORDER,
    finding: str = "",
) -> SetMatchResult:
    """Compute top-|GT| set metrics from per-class cosine scores."""
    gt_set: Set[str] = set(str(x).strip().lower() for x in gt_labels)
    if not gt_set:
        raise ValueError("gt_labels must be non-empty")
    unknown = gt_set - set(classes)
    if unknown:
        raise ValueError(f"Unknown progression labels in GT: {sorted(unknown)}")

    k = len(gt_set)
    # Stable sort: higher score first; ties broken by class order index.
    ranked = sorted(
        range(len(classes)),
        key=lambda i: (-float(cos_class_scores[i]), i),
    )
    pred_idxs = ranked[:k]
    pred_set = {classes[i] for i in pred_idxs}
    argmax = classes[ranked[0]]

    inter = pred_set & gt_set
    union = pred_set | gt_set
    recall = len(inter) / k
    precision = len(inter) / k  # |Ŷ| == k by construction
    jaccard = len(inter) / len(union) if union else 0.0
    exact = 1.0 if pred_set == gt_set else 0.0
    is_multi = k > 1
    if is_multi:
        group_score = jaccard
    else:
        group_score = 1.0 if argmax in gt_set else 0.0

    return SetMatchResult(
        gt=tuple(sorted(gt_set)),
        pred_set=tuple(sorted(pred_set)),
        argmax=argmax,
        k=k,
        recall=recall,
        precision=precision,
        jaccard=jaccard,
        exact=exact,
        group_score=group_score,
        is_multi=is_multi,
        finding=str(finding).strip().lower(),
        cos_class_scores=tuple(float(x) for x in cos_class_scores),
    )


def group_gold_by_pair_finding(gold: pd.DataFrame) -> pd.DataFrame:
    """Collapse gold rows to one row per (pair, finding) with GT label set.

    Returns columns: PAIR_FINDING_KEYS + image path cols +
    ``gt_labels`` (tuple), ``n_rows``, ``is_multi``.
    """
    need = [
        "dataset",
        "patient_id",
        "study_id_curr",
        "study_id_prev",
        "finding",
        "progression",
        "parent_image_prev",
        "parent_image_curr",
    ]
    missing = [c for c in need if c not in gold.columns]
    if missing:
        raise ValueError(f"gold df missing {missing}")

    g = gold.copy()
    g["finding_lc"] = g["finding"].astype(str).str.strip().str.lower()
    g["prog_lc"] = g["progression"].astype(str).str.strip().str.lower()

    rows = []
    for key, sub in g.groupby(PAIR_FINDING_KEYS, dropna=False, sort=False):
        key_t = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(PAIR_FINDING_KEYS, key_t))
        gt = tuple(sorted(sub["prog_lc"].unique()))
        rows.append(
            {
                **meta,
                "finding": meta["finding_lc"],  # canonical lower finding
                "parent_image_prev": sub["parent_image_prev"].iloc[0],
                "parent_image_curr": sub["parent_image_curr"].iloc[0],
                "gt_labels": gt,
                "n_rows": len(sub),
                "is_multi": len(gt) > 1,
            }
        )
    out = pd.DataFrame(rows)
    n_multi = int(out["is_multi"].sum())
    print(
        f"[setmatch] grouped {len(gold)} gold rows → {len(out)} "
        f"(pair, finding) groups ({n_multi} multi-label, "
        f"{len(out) - n_multi} single-label)"
    )
    return out


def summarize_setmatch(
    results: List[SetMatchResult],
) -> Dict[str, float]:
    """Aggregate multi-only and combined (single 0/1 + multi Jaccard)."""
    if not results:
        return {
            "n_groups": 0,
            "n_single": 0,
            "n_multi": 0,
            "multi_recall": float("nan"),
            "multi_precision": float("nan"),
            "multi_jaccard": float("nan"),
            "multi_exact": float("nan"),
            "single_acc": float("nan"),
            "combined_score": float("nan"),
            "combined_score_recall": float("nan"),
        }

    multi = [r for r in results if r.is_multi]
    single = [r for r in results if not r.is_multi]

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "n_groups": len(results),
        "n_single": len(single),
        "n_multi": len(multi),
        "multi_recall": _mean([r.recall for r in multi]),
        "multi_precision": _mean([r.precision for r in multi]),
        "multi_jaccard": _mean([r.jaccard for r in multi]),
        "multi_exact": _mean([r.exact for r in multi]),
        "single_acc": _mean([r.group_score for r in single]),
        "combined_score": _mean([r.group_score for r in results]),
        "combined_score_recall": _mean(
            [
                (r.group_score if not r.is_multi else r.recall)
                for r in results
            ]
        ),
    }


def _print_single_label_breakdown(
    results: List[SetMatchResult],
    title: str,
) -> None:
    """Full regular-eval-style tables on single-label groups (argmax)."""
    single = [r for r in results if not r.is_multi]
    print(f"\n{'-' * 60}")
    print(title)
    print(f"{'-' * 60}")
    if not single:
        print("  (no single-label groups)")
        return

    confusion: Dict[str, Counter] = defaultdict(Counter)
    per_finding: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    cos_sums = [0.0] * len(CLS_ORDER)
    n_correct = 0
    for r in single:
        gt = r.gt[0]
        pred = r.argmax
        confusion[gt][pred] += 1
        n_correct += int(pred == gt)
        if r.finding:
            per_finding[r.finding][1] += 1
            per_finding[r.finding][0] += int(pred == gt)
        if r.cos_class_scores:
            for k, s in enumerate(r.cos_class_scores):
                cos_sums[k] += s

    n_seen = len(single)
    acc = n_correct / n_seen
    print(
        f"Overall accuracy: {n_correct}/{n_seen} = {acc:.4f}    "
        f"(chance = {1.0 / len(CLS_ORDER):.3f})"
    )

    print("\nPer-progression accuracy (SINGLE; = per-class recall):")
    print(f"  {'gt class':<10} {'n':>6} {'acc':>8}")
    for cls in CLS_ORDER:
        n = sum(confusion[cls].values())
        c = confusion[cls].get(cls, 0)
        a = c / n if n else float("nan")
        print(f"  {cls:<10} {n:>6} {a:>8.4f}")

    print("\nConfusion matrix (rows=gt, cols=pred):")
    header = " ".join(f"{c[:9]:>9}" for c in CLS_ORDER)
    print(f"  {'':<10} {header}")
    for gt in CLS_ORDER:
        cells = " ".join(f"{confusion[gt].get(p, 0):>9}" for p in CLS_ORDER)
        print(f"  {gt:<10} {cells}")

    print("\nMean class cosine (avg over single-label groups):")
    print(f"  {'class':<10} {'mean_cos':>10}")
    for k, cls in enumerate(CLS_ORDER):
        print(f"  {cls:<10} {cos_sums[k] / n_seen:>10.4f}")

    print("\nPer-disease accuracy (SINGLE):")
    print(f"  {'finding':<26} {'n':>6} {'acc':>8}")
    for finding in sorted(per_finding):
        c, n = per_finding[finding]
        print(f"  {finding:<26} {n:>6} {c / n if n else float('nan'):>8.4f}")

    m = _compute_balanced_metrics(confusion, CLS_ORDER, n_correct)
    print("\nBalanced (imbalance-corrected) metrics:")
    print(f"  macro recall       {m['macro_recall']:>8.4f}")
    print(f"  macro precision    {m['macro_precision']:>8.4f}")
    print(f"  macro F1           {m['macro_f1']:>8.4f}")
    print(f"  Cohen's kappa      {m['cohen_kappa']:>8.4f}")
    print(
        f"  majority baseline  {m['majority_acc']:>8.4f}   "
        f"(always predict {m['majority_class']!r})"
    )

    print("\nPer-class precision / recall / F1:")
    print(
        f"  {'class':<10} {'n_gt':>6} {'precision':>10} "
        f"{'recall':>8} {'F1':>8}"
    )
    for k, cls in enumerate(CLS_ORDER):
        print(
            f"  {cls:<10} {m['n_true'][cls]:>6} "
            f"{m['per_class_precision'][k]:>10.4f} "
            f"{m['per_class_recall'][k]:>8.4f} "
            f"{m['per_class_f1'][k]:>8.4f}"
        )

    print("\nPredicted vs true class distribution:")
    print(
        f"  {'class':<10} {'n_pred':>7} {'pred%':>7} "
        f"{'n_true':>7} {'true%':>7}"
    )
    total = m["total"]
    for cls in CLS_ORDER:
        npred = m["n_pred"][cls]
        ntrue = m["n_true"][cls]
        print(
            f"  {cls:<10} {npred:>7} {100.0 * npred / total:>6.1f}% "
            f"{ntrue:>7} {100.0 * ntrue / total:>6.1f}%"
        )


def _print_multi_label_breakdown(
    results: List[SetMatchResult],
    title: str,
) -> None:
    """Set-match tables + per-progression / per-finding for multi groups."""
    multi = [r for r in results if r.is_multi]
    print(f"\n{'-' * 60}")
    print(title)
    print(f"{'-' * 60}")
    if not multi:
        print("  (no multi-label groups)")
        return

    n = len(multi)
    mean_r = sum(r.recall for r in multi) / n
    mean_p = sum(r.precision for r in multi) / n
    mean_j = sum(r.jaccard for r in multi) / n
    mean_e = sum(r.exact for r in multi) / n
    print(f"n                     {n:>8}")
    print(f"mean recall           {mean_r:>8.4f}")
    print(f"mean precision        {mean_p:>8.4f}")
    print(f"mean Jaccard          {mean_j:>8.4f}")
    print(f"exact set match       {mean_e:>8.4f}   (Ŷ == GT)")

    print("\nBreakdown by |GT|:")
    by_k: Dict[int, List[SetMatchResult]] = defaultdict(list)
    for r in multi:
        by_k[r.k].append(r)
    for k in sorted(by_k):
        rs = by_k[k]
        print(
            f"  |GT|={k}: n={len(rs):>4}  "
            f"recall={sum(x.recall for x in rs) / len(rs):.4f}  "
            f"jaccard={sum(x.jaccard for x in rs) / len(rs):.4f}  "
            f"exact={sum(x.exact for x in rs) / len(rs):.4f}"
        )

    _print_progression_retrieval(
        multi,
        "Per-progression label retrieval (MULTI):",
    )

    print("\nPer-disease set-match (MULTI):")
    print(
        f"  {'finding':<26} {'n':>6} {'recall':>8} "
        f"{'precision':>10} {'jaccard':>8} {'exact':>8}"
    )
    by_f: Dict[str, List[SetMatchResult]] = defaultdict(list)
    for r in multi:
        by_f[r.finding or "<unknown>"].append(r)
    for finding in sorted(by_f):
        rs = by_f[finding]
        print(
            f"  {finding:<26} {len(rs):>6} "
            f"{sum(x.recall for x in rs) / len(rs):>8.4f} "
            f"{sum(x.precision for x in rs) / len(rs):>10.4f} "
            f"{sum(x.jaccard for x in rs) / len(rs):>8.4f} "
            f"{sum(x.exact for x in rs) / len(rs):>8.4f}"
        )

    cos_sums = [0.0] * len(CLS_ORDER)
    n_cos = 0
    for r in multi:
        if r.cos_class_scores:
            n_cos += 1
            for k, s in enumerate(r.cos_class_scores):
                cos_sums[k] += s
    if n_cos:
        print("\nMean class cosine (avg over multi-label groups):")
        print(f"  {'class':<10} {'mean_cos':>10}")
        for k, cls in enumerate(CLS_ORDER):
            print(f"  {cls:<10} {cos_sums[k] / n_cos:>10.4f}")


def _print_progression_retrieval(
    results: List[SetMatchResult],
    heading: str,
) -> None:
    """Per-progression label retrieval over the given groups.

    For every group, prediction set ``Ŷ`` is top-|GT| (so |GT|=1 ⇒
    ``Ŷ = {argmax}``). Per class::

        recall    = P(cls ∈ Ŷ | cls ∈ GT)
        precision = P(cls ∈ GT | cls ∈ Ŷ)
    """
    print(f"\n{heading}")
    if not results:
        print("  (no groups)")
        return
    print(
        f"  {'class':<10} {'n_in_GT':>8} {'recall':>8} "
        f"{'n_in_Ŷ':>8} {'precision':>10}"
    )
    for cls in CLS_ORDER:
        in_gt = [r for r in results if cls in r.gt]
        in_pred = [r for r in results if cls in r.pred_set]
        rec = (
            sum(1 for r in in_gt if cls in r.pred_set) / len(in_gt)
            if in_gt else float("nan")
        )
        prec = (
            sum(1 for r in in_pred if cls in r.gt) / len(in_pred)
            if in_pred else float("nan")
        )
        print(
            f"  {cls:<10} {len(in_gt):>8} {rec:>8.4f} "
            f"{len(in_pred):>8} {prec:>10.4f}"
        )


def _print_combined_per_finding(results: List[SetMatchResult]) -> None:
    """Per-disease combined score (single 0/1 + multi Jaccard)."""
    print(
        "\nPer-disease COMBINED "
        "(group_score: single=argmax 0/1, multi=Jaccard):"
    )
    if not results:
        print("  (no groups)")
        return
    print(
        f"  {'finding':<26} {'n':>6} {'n_single':>8} {'n_multi':>8} "
        f"{'combined':>8} {'single_acc':>10} {'multi_jac':>9} "
        f"{'multi_rec':>9}"
    )
    by_f: Dict[str, List[SetMatchResult]] = defaultdict(list)
    for r in results:
        by_f[r.finding or "<unknown>"].append(r)
    for finding in sorted(by_f):
        rs = by_f[finding]
        single = [r for r in rs if not r.is_multi]
        multi = [r for r in rs if r.is_multi]
        comb = sum(r.group_score for r in rs) / len(rs)
        s_acc = (
            sum(r.group_score for r in single) / len(single)
            if single else float("nan")
        )
        m_j = (
            sum(r.jaccard for r in multi) / len(multi)
            if multi else float("nan")
        )
        m_r = (
            sum(r.recall for r in multi) / len(multi)
            if multi else float("nan")
        )
        print(
            f"  {finding:<26} {len(rs):>6} {len(single):>8} {len(multi):>8} "
            f"{comb:>8.4f} {s_acc:>10.4f} {m_j:>9.4f} {m_r:>9.4f}"
        )


def print_setmatch_report(
    results: List[SetMatchResult],
    backend: str,
    pooling_note: str = "",
) -> None:
    """Full report with per-progression + per-disease for single/multi/combined."""
    summary = summarize_setmatch(results)

    print(f"\n{'=' * 60}")
    print(f"=== Gold set-match results ({backend}{pooling_note})")
    print(f"{'=' * 60}")
    print(
        f"Groups: {int(summary['n_groups'])} total | "
        f"{int(summary['n_single'])} single-label | "
        f"{int(summary['n_multi'])} multi-label"
    )

    _print_multi_label_breakdown(
        results,
        "(A) MULTI-label groups — summary + per-progression + per-disease",
    )

    print(f"\n{'-' * 60}")
    print(
        "(B) COMBINED (all groups) — "
        "single=argmax 0/1, multi=top-|GT| Jaccard"
    )
    print(f"{'-' * 60}")
    print(f"  single-label acc {summary['single_acc']:>8.4f}   "
          f"(n={int(summary['n_single'])})")
    print(f"  multi Jaccard    {summary['multi_jaccard']:>8.4f}   "
          f"(n={int(summary['n_multi'])})")
    print(
        f"  COMBINED score   {summary['combined_score']:>8.4f}   "
        f"(mean group_score over all groups)"
    )
    print(
        f"  combined (alt)   {summary['combined_score_recall']:>8.4f}   "
        f"(single 0/1 + multi recall)"
    )
    _print_progression_retrieval(
        results,
        "Per-progression label retrieval (COMBINED — all groups):",
    )
    _print_combined_per_finding(results)

    _print_single_label_breakdown(
        results,
        "(C) SINGLE-label groups — full breakdown "
        "(argmax; per-progression + per-disease)",
    )
