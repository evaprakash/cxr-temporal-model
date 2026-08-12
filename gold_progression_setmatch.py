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

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from progression_phrases import CLS_ORDER

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


def topk_set_match(
    cos_class_scores: Sequence[float],
    gt_labels: Iterable[str],
    classes: Sequence[str] = CLS_ORDER,
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
        # Mean of per-group scores: single=0/1 exact, multi=Jaccard.
        "combined_score": _mean([r.group_score for r in results]),
        # Also mean multi recall folded into combined soft coverage view:
        # single 0/1 + multi recall (optional alternate).
        "combined_score_recall": _mean(
            [
                (r.group_score if not r.is_multi else r.recall)
                for r in results
            ]
        ),
    }


def print_setmatch_report(
    summary: Dict[str, float],
    backend: str,
    pooling_note: str = "",
) -> None:
    print(f"\n{'=' * 60}")
    print(f"=== Gold set-match results ({backend}{pooling_note})")
    print(f"{'=' * 60}")
    print(
        f"Groups: {int(summary['n_groups'])} total | "
        f"{int(summary['n_single'])} single-label | "
        f"{int(summary['n_multi'])} multi-label"
    )

    print("\n(A) Multi-progression groups only (top-|GT| set match):")
    if summary["n_multi"] == 0:
        print("  (no multi-label groups)")
    else:
        print(f"  n                {int(summary['n_multi']):>8}")
        print(f"  mean recall      {summary['multi_recall']:>8.4f}")
        print(f"  mean precision   {summary['multi_precision']:>8.4f}")
        print(f"  mean Jaccard     {summary['multi_jaccard']:>8.4f}")
        print(
            f"  exact set match  {summary['multi_exact']:>8.4f}   "
            f"(Ŷ == GT)"
        )

    print(
        "\n(B) Combined overall "
        "(single = argmax accuracy; multi = top-|GT| Jaccard):"
    )
    print(f"  single-label acc {summary['single_acc']:>8.4f}   "
          f"(n={int(summary['n_single'])})")
    print(f"  multi Jaccard    {summary['multi_jaccard']:>8.4f}   "
          f"(n={int(summary['n_multi'])})")
    print(
        f"  COMBINED score   {summary['combined_score']:>8.4f}   "
        f"(mean over all groups)"
    )
    print(
        f"  combined (alt)   {summary['combined_score_recall']:>8.4f}   "
        f"(single 0/1 + multi recall)"
    )
