"""Gold checks: five-ẑ film collapse vs consistent tiny 5-way gaps."""

from __future__ import annotations

from typing import Any, List, Sequence

import torch
import torch.nn.functional as F

from progression_phrases import CLS_ORDER

_AXIS = {"improving": "better", "resolved": "better",
         "stable": "same", "worsening": "worse", "new": "worse"}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def deltacos_class_scores(
    zhats: torch.Tensor,
    z_cur: torch.Tensor,
    z_prior: torch.Tensor,
    eps: float = 1e-8,
) -> List[float]:
    """``cos(ẑ^c − z_prior, z_cur − z_prior)`` per class.

    Same flatten as ``eval_jepa_only_gold._change_align``. ``zhats`` is
    ``(C, N, D)``; ``z_cur`` / ``z_prior`` are ``(1, N, D)`` or
    ``(N, D)``. Degenerate (near-zero) deltas score 0.
    """
    pred = zhats.float()
    prior = z_prior.float()
    if prior.dim() == 2:
        prior = prior.unsqueeze(0)
    cur = z_cur.float()
    if cur.dim() == 2:
        cur = cur.unsqueeze(0)
    dpred = (pred - prior).flatten(1)
    dtrue = (cur - prior).flatten()
    if float(dtrue.norm()) < eps:
        return [0.0] * int(pred.shape[0])
    norms = dpred.norm(dim=1)
    sim = F.cosine_similarity(dpred, dtrue.unsqueeze(0).expand_as(dpred), dim=1)
    return [
        0.0 if float(n) < eps else float(s)
        for n, s in zip(norms, sim)
    ]


def five_forecast_offdiag_cos(zhats: torch.Tensor) -> float:
    """Mean off-diagonal patch-mean cosine among C forecasts.

    ``zhats`` is ``(C, N, D)``. Same formula as ``eval_jepa_only_gold``.
    """
    z = F.normalize(zhats.float(), dim=-1)
    c = z.shape[0]
    if c < 2:
        return float("nan")
    sim = torch.einsum("cnd,knd->ck", z, z) / z.shape[1]
    return float(((sim.sum() - sim.trace()) / (c * (c - 1))).item())


def _row_scores_gt_off(row: Any):
    if isinstance(row, dict):
        scores = list(row["scores"] if "scores" in row else row["cos_class_scores"])
        gt = list(row["gt_labels"] if "gt_labels" in row else row.get("gt", ()))
        off = row.get("zhat_offdiag")
        return scores, gt, off
    scores = list(getattr(row, "cos_class_scores", ()))
    gt = list(getattr(row, "gt", ()))
    off = getattr(row, "zhat_offdiag", None)
    return scores, gt, off


def print_jepa_score_diagnostics(
    score_rows: List[Any],
    classes: Sequence[str] = CLS_ORDER,
    close_eps: float = 0.02,
) -> None:
    """Pairwise win / close-call / 3-way / ẑ-film off-diag.

    Each row is a dict (``scores`` / ``gt_labels`` / optional
    ``zhat_offdiag``) or a ``SetMatchResult``. Single-label only for
    win-rate / close-call / 3-way.
    """
    parsed = [_row_scores_gt_off(r) for r in score_rows]
    singles = [(s, g, o) for s, g, o in parsed if len(g) == 1]
    n_s = len(singles)
    print()
    print("------------------------------------------------------------")
    print("(D) 5-way hair vs film collapse (single-label gold)")
    print("------------------------------------------------------------")
    if n_s == 0:
        print("  no single-label rows")
        return

    wins: List[float] = []
    wins_vs = {c: [] for c in classes}
    margins: List[float] = []
    close_ok: List[float] = []
    axis_ok: List[float] = []
    offdiags: List[float] = []

    for scores, gt_labels, off in singles:
        gt = gt_labels[0]
        if gt not in classes:
            continue
        g = classes.index(gt)
        s_gt = scores[g]
        others = [scores[i] for i in range(len(classes)) if i != g]
        second = max(others)
        margins.append(s_gt - second)
        for i, c in enumerate(classes):
            if i == g:
                continue
            w = 1.0 if s_gt > scores[i] else 0.0
            wins.append(w)
            wins_vs[c].append(w)
        ranked = sorted(scores, reverse=True)
        if ranked[0] - ranked[1] < close_eps:
            pred = classes[max(range(len(scores)), key=lambda k: scores[k])]
            close_ok.append(1.0 if pred == gt else 0.0)
        pred = classes[max(range(len(scores)), key=lambda k: scores[k])]
        axis_ok.append(1.0 if _AXIS[pred] == _AXIS[gt] else 0.0)
        if off is not None:
            offdiags.append(float(off))

    print(f"  n_single              {n_s}")
    print(f"  pairwise win (gt>wrong)  {_mean(wins):.4f}   "
          f"(~0.5 = coin-flip hair; >>0.5 = small but consistent)")
    print("  win vs each wrong class:")
    for c in classes:
        print(f"    vs {c:<12} {_mean(wins_vs[c]):.4f}")
    print(f"  mean margin gt−2nd    {_mean(margins):.4f}")
    print(f"  close-call acc        {_mean(close_ok):.4f}   "
          f"(max−2nd < {close_eps}, n={len(close_ok)})")
    print(f"  3-way axis acc        {_mean(axis_ok):.4f}   "
          f"(better/same/worse)")
    if offdiags:
        print(f"  mean cos(ẑ^i, ẑ^j)    {_mean(offdiags):.4f}   "
              f"(~1 = five forecasts are one film)")
    print("  Text templates can differ and ẑ still be one film.")
    print("  Keep 5-way if pairwise win >> 0.5; else residual readout.")
