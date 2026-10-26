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
    python -m vljepa.check_class_signal --supervised \\
        --ckpt checkpoints_supervised_progression_unfrozen/epoch_5.pt
    python -m vljepa.check_class_signal --reuse <scores.npz>   # rerun the math only

The supervised model is scored on the same silver val films (splits_jepa.csv)
with its five logits in place of cosines.
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
BONUS_GRIDS = {
    "cosine": np.round(np.arange(-0.15, 0.3001, 0.005), 4),
    "logit": np.round(np.arange(-3.0, 6.0001, 0.05), 4),
}


def _print(msg: str = "") -> None:
    print(msg, flush=True)


# ----------------------------------------------------------------------
# Scoring (needs the model; imported lazily so --reuse runs anywhere)
# ----------------------------------------------------------------------
class VLJEPAScorer:
    """Five sentence cosines of the VL-JEPA guess."""

    kind = "cosine"

    def __init__(self, model):
        self.model = model
        self.bank = {}
        self.text_cache = {}

    def _targets(self, finding):
        import torch.nn.functional as F

        from .prompts import class_target_texts

        if finding not in self.bank:
            g, _, _ = self.model.encode_targets(class_target_texts(finding))
            self.bank[finding] = F.normalize(g.float(), dim=-1)
        return self.bank[finding]

    def batch(self, prior, current, findings, queries):
        import torch
        import torch.nn.functional as F

        img_g, img_p = self.model.encode_images(prior, current)
        pred = F.normalize(self.model.predict(img_g, img_p, queries).float(), dim=-1)
        tgt = torch.stack([self._targets(f) for f in findings])
        return torch.einsum("bd,bcd->bc", pred, tgt)

    def one(self, prior, current, finding):
        return self.model.score_classes(prior, current, finding, text_cache=self.text_cache)


class SupervisedScorer:
    """Five logits of the unfrozen supervised BioViL-T + linear head.

    Mirrors ``SupervisedProgressionModel`` in train_supervised_progression.py:
    ``Linear([norm(image pair); norm(finding text)])``.
    """

    kind = "logit"

    def __init__(self, ckpt_path, device):
        import torch
        import torch.nn as nn

        from tempcxr.modules.image_encoder_jepa import BioViLTImageEncoderJEPA
        from tempcxr.modules.text_encoder import BioViLTTextEncoder

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "image_encoder" not in ckpt or "finding_vocab" in ckpt:
            raise SystemExit(f"[check] {ckpt_path} is not an unfrozen supervised checkpoint")
        if list(ckpt.get("classes", CLS_ORDER)) != list(CLS_ORDER):
            raise SystemExit(f"[check] class order mismatch: {ckpt.get('classes')}")
        self.image_encoder = BioViLTImageEncoderJEPA(mode="biovilt")
        self.text_encoder = BioViLTTextEncoder(mode="biovilt")
        self.image_encoder.load_state_dict(ckpt["image_encoder"])
        self.text_encoder.load_state_dict(ckpt["text_encoder"])
        self.head = nn.Linear(int(ckpt.get("in_dim", 256)), N_CLS)
        self.head.load_state_dict({k.split("fc.", 1)[-1]: v for k, v in ckpt["head"].items()})
        for m in (self.image_encoder, self.text_encoder, self.head):
            m.to(device).eval()
            for p in m.parameters():
                p.requires_grad = False
        self.text_cache = {}
        _print(f"[check] supervised epoch {ckpt.get('epoch')} "
               f"(train_loss {ckpt.get('train_loss')})")

    def _text(self, finding):
        import torch.nn.functional as F

        if finding not in self.text_cache:
            t, _, _ = self.text_encoder.forward_contrastive([finding])
            self.text_cache[finding] = F.normalize(t.float(), dim=-1)
        return self.text_cache[finding]

    def batch(self, prior, current, findings, queries=None):
        import torch
        import torch.nn.functional as F

        img, _ = self.image_encoder(current, prior)
        img = F.normalize(img.float(), dim=-1)
        txt = torch.cat([self._text(f) for f in findings], dim=0)
        return self.head(torch.cat([img, txt], dim=-1))

    def one(self, prior, current, finding):
        return self.batch(prior, current, [finding]).squeeze(0)


def score_silver_val(scorer, device, batch_size, num_workers, limit):
    import torch
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
    scores, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="silver val", file=sys.stdout):
            prior = batch["prior_image"].to(device, non_blocking=True)
            current = batch["current_image"].to(device, non_blocking=True)
            s = scorer.batch(prior, current, batch["finding"], batch["query_text"])
            scores.append(s.float().cpu().numpy())
            labels.append(batch["cls_idx"].numpy())
    return np.concatenate(scores), np.concatenate(labels)


def score_gold(scorer, device, limit):
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
            s = scorer.one(
                prior.unsqueeze(0).to(device),
                current.unsqueeze(0).to(device),
                str(row["finding"]),
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
               grid: np.ndarray = BONUS_GRIDS["cosine"], rounds: int = 3) -> np.ndarray:
    """Coordinate ascent on ``objective``. Stable is held at 0."""
    bonus = np.zeros(N_CLS)
    best = objective((scores + bonus).argmax(1), labels)
    for _ in range(rounds):
        changed = False
        for c in range(N_CLS):
            if c == STABLE:
                continue
            for v in grid:
                trial = bonus.copy()
                trial[c] = v
                f = objective((scores + trial).argmax(1), labels)
                if f > best + 1e-9:
                    best, bonus, changed = f, trial, True
        if not changed:
            break
    return bonus


def fit_ce_bias(scores: np.ndarray, labels: np.ndarray, temperature: float,
                steps: int = 3000, lr: float = 0.5,
                class_weights: np.ndarray = None) -> np.ndarray:
    """Per-class bias b minimizing weighted CE of softmax(scores / T + b).

    Default weights are fully balanced (the supervised baseline's loss).
    Returned in score units (b * T) so it adds to the raw scores.
    """
    counts = np.bincount(labels, minlength=N_CLS).astype(np.float64)
    if class_weights is None:
        class_weights = counts.sum() / (N_CLS * np.maximum(counts, 1))
    w = np.asarray(class_weights, dtype=np.float64)[labels]
    onehot = np.eye(N_CLS)[labels]
    z = scores / temperature
    b = np.zeros(N_CLS)
    for _ in range(steps):
        logits = z + b
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        b -= lr * (w[:, None] * (p - onehot)).sum(0) / w.sum()
    b -= b[STABLE]
    return b * temperature


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


def fit_calibration(scores: np.ndarray, labels: np.ndarray, t0: float,
                    steps: int = 5000, lr: float = 0.5):
    """Temperature T and per-class bias b so softmax(scores / T + b) is
    calibrated on these labels (plain CE, no class weights)."""
    onehot = np.eye(N_CLS)[labels]
    log_t, b = np.log(t0), np.zeros(N_CLS)
    for _ in range(steps):
        t = np.exp(log_t)
        err = _softmax(scores / t + b) - onehot
        b -= lr * err.mean(0)
        log_t -= 0.1 * lr * (err * (-scores / t)).sum(1).mean()
    return float(np.exp(log_t)), b - b[STABLE]


def em_target_prior(p_src: np.ndarray, prior_src: np.ndarray,
                    iters: int = 2000, tol: float = 1e-10) -> np.ndarray:
    """Saerens et al. (2002): class shares of an unlabeled set, from
    posteriors calibrated under ``prior_src``."""
    prior = prior_src.copy()
    for _ in range(iters):
        q = p_src * (prior / prior_src)
        q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.abs(new - prior).max() < tol:
            return new
        prior = new
    return prior


def em_prior_bonus(val_scores, val_labels, gold_scores, t0):
    """Score-unit bonus that moves silver-calibrated posteriors to the gold
    class shares estimated without gold labels."""
    t, b = fit_calibration(val_scores, val_labels, t0)
    prior_src = np.bincount(val_labels, minlength=N_CLS) / len(val_labels)
    prior_gold = em_target_prior(_softmax(gold_scores / t + b), prior_src)
    bonus = t * (b + np.log(prior_gold / prior_src))
    return bonus - bonus[STABLE], t, prior_src, prior_gold


def per_class_f1(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros(N_CLS)
    for c in range(N_CLS):
        tp = int(((pred == c) & (labels == c)).sum())
        n_pred, n_true = int((pred == c).sum()), int((labels == c).sum())
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_true if n_true else 0.0
        out[c] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def gold_single(gold_scores, gold_gts):
    keep = np.array(["|" not in g for g in gold_gts])
    labels = np.array([CLS_ORDER.index(g) for g in gold_gts[keep]], dtype=np.int64)
    return gold_scores[keep], labels


def knob_bonus(alpha: float, prior: np.ndarray, temperature: float) -> np.ndarray:
    """One-knob balancing: alpha * T * log(1 / silver share), stable at 0.
    alpha=0 is the raw model; alpha=1 fully undoes the silver class mix."""
    b = alpha * temperature * -np.log(prior)
    return b - b[STABLE]


def knob_table(val_labels, gold_scores, gold_gts, gold_findings, kind, model_name,
               ref=None, margin=0.02):
    temperature = 0.07 if kind == "cosine" else 1.0
    prior = np.bincount(val_labels, minlength=N_CLS) / len(val_labels)
    single_scores, single_labels = gold_single(gold_scores, gold_gts)
    imp, wor = CLS_ORDER.index("improving"), CLS_ORDER.index("worsening")

    _print(f"\n=== One-knob balancing on GOLD ({model_name}, {kind}s, T={temperature})")
    _print("  bonus per class = alpha * T * log(1 / silver share); stable fixed at 0")
    _print("  silver shares: " + ", ".join(
        f"{c} {prior[i]:.3f}" for i, c in enumerate(CLS_ORDER)))
    if ref is not None:
        _print("  reference (no bonus): combined {:.4f}  macro F1 {:.4f}  F1 ".format(
            ref["combined"], ref["macro_f1"])
            + " ".join(f"{c[:4]} {ref['f1'][i]:.3f}" for i, c in enumerate(CLS_ORDER)))
        _print(f"  ok = improving and worsening F1 both within {margin:.2f} of the reference")
    header = (f"  {'alpha':>5} {'combined':>9} {'macroF1':>8} "
              + "".join(f"{c[:4] + ' F1':>9}" for c in CLS_ORDER)
              + "".join(f"{c[:4] + ' n':>7}" for c in CLS_ORDER))
    _print(header + ("    ok" if ref is not None else ""))
    for alpha in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        b = knob_bonus(alpha, prior, temperature)
        pred = (single_scores + b).argmax(1)
        f1 = per_class_f1(pred, single_labels)
        combined = summarize_setmatch(
            gold_results(gold_scores, gold_gts, gold_findings, b))["combined_score"]
        counts = np.bincount(pred, minlength=N_CLS)
        line = (f"  {alpha:>5.2f} {combined:>9.4f} {f1.mean():>8.4f} "
                + "".join(f"{x:>9.3f}" for x in f1)
                + "".join(f"{n:>7d}" for n in counts))
        if ref is not None:
            ok = f1[imp] >= ref["f1"][imp] - margin and f1[wor] >= ref["f1"][wor] - margin
            line += "   yes" if ok else "    no"
        _print(line)
    true_n = np.bincount(single_labels, minlength=N_CLS)
    _print(f"  {'true':>5} {'':>9} {'':>8} " + " " * 9 * N_CLS
           + "".join(f"{n:>7d}" for n in true_n))


def per_class_table(gold_scores, gold_gts, gold_findings, kind, model_name, ref=None):
    """Nudge one class at a time (all others fixed) and report the best F1
    that class can reach, plus what it costs the rest."""
    grid = BONUS_GRIDS[kind]
    single_scores, single_labels = gold_single(gold_scores, gold_gts)
    true_n = np.bincount(single_labels, minlength=N_CLS)

    _print(f"\n=== One class nudged at a time on GOLD ({model_name}, {kind} units)")
    _print("  best F1 = highest F1 that class reaches when only it is nudged;")
    _print("  matched = nudge where that class is called as often as it truly occurs")
    _print(f"  {'class':<10} {'row':<8} {'nudge':>7} {'own F1':>7} {'ref F1':>7} "
           f"{'calls':>6} {'true':>5} {'stab F1':>8} {'macroF1':>8} {'combined':>9}")
    for c, cls in enumerate(CLS_ORDER):
        if c == STABLE:
            continue
        rows = []
        for v in grid:
            b = np.zeros(N_CLS)
            b[c] = v
            pred = (single_scores + b).argmax(1)
            f1 = per_class_f1(pred, single_labels)
            rows.append((v, f1, int((pred == c).sum())))
        best = max(rows, key=lambda r: r[1][c])
        matched = min(rows, key=lambda r: abs(r[2] - true_n[c]))
        for tag, (v, f1, n) in (("best F1", best), ("matched", matched)):
            b = np.zeros(N_CLS)
            b[c] = v
            combined = summarize_setmatch(
                gold_results(gold_scores, gold_gts, gold_findings, b))["combined_score"]
            ref_f1 = f"{ref['f1'][c]:>7.3f}" if ref is not None else f"{'-':>7}"
            _print(f"  {cls:<10} {tag:<8} {v:>+7.3f} {f1[c]:>7.3f} {ref_f1} "
                   f"{n:>6d} {true_n[c]:>5d} {f1[STABLE]:>8.3f} {f1.mean():>8.4f} {combined:>9.4f}")


def preview_bias(val_scores, val_labels, gold_scores, gold_gts, gold_findings,
                 kind, model_name, class_weights, ref=None):
    """Bias a learned per-class term would settle at under the training loss,
    fit on silver val with the predictor held fixed, then scored on gold."""
    temperature = 0.07 if kind == "cosine" else 1.0
    single_scores, single_labels = gold_single(gold_scores, gold_gts)
    true_n = np.bincount(single_labels, minlength=N_CLS)
    runs = {
        "no bias": np.zeros(N_CLS),
        "training weights": fit_ce_bias(val_scores, val_labels, temperature,
                                        class_weights=class_weights),
        "balanced weights": fit_ce_bias(val_scores, val_labels, temperature),
    }
    _print(f"\n=== Preview: learned per-class bias, fit on silver val ({model_name}, T={temperature})")
    _print("  training weights: " + ", ".join(
        f"{c} {w:.3f}" for c, w in zip(CLS_ORDER, class_weights)))
    _print(f"  {'':<18}" + "".join(f"{c[:9]:>10}" for c in CLS_ORDER))
    for name, b in runs.items():
        if name != "no bias":
            _print(f"  {name + ' bias':<18}" + "".join(f"{x:>+10.3f}" for x in b))
    for name, b in runs.items():
        pred = (single_scores + b).argmax(1)
        f1 = per_class_f1(pred, single_labels)
        combined = summarize_setmatch(
            gold_results(gold_scores, gold_gts, gold_findings, b))["combined_score"]
        _print(f"\n  --- GOLD with {name}: combined {combined:.4f}   macro F1 {f1.mean():.4f}")
        _print(f"  {'class':<10} {'F1':>7} {'ref F1':>7} {'calls':>6} {'true':>5}")
        for c, cls in enumerate(CLS_ORDER):
            ref_f1 = f"{ref['f1'][c]:>7.3f}" if ref is not None else f"{'-':>7}"
            _print(f"  {cls:<10} {f1[c]:>7.3f} {ref_f1} {int((pred == c).sum()):>6d} {true_n[c]:>5d}")
    if ref is not None:
        _print(f"\n  reference: combined {ref['combined']:.4f}   macro F1 {ref['macro_f1']:.4f}")


def reference_metrics(path):
    data = np.load(path, allow_pickle=True)
    scores, gts, findings = data["gold_scores"], data["gold_gts"], data["gold_findings"]
    single_scores, single_labels = gold_single(scores, gts)
    f1 = per_class_f1(single_scores.argmax(1), single_labels)
    combined = summarize_setmatch(
        gold_results(scores, gts, findings, np.zeros(N_CLS)))["combined_score"]
    return {"f1": f1, "macro_f1": float(f1.mean()), "combined": combined}


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


def report(val_scores, val_labels, gold_scores, gold_gts, gold_findings, show_full,
           kind="cosine", model_name="vljepa", gold_oracle=False):
    grid = BONUS_GRIDS[kind]
    temperature = 0.07 if kind == "cosine" else 1.0
    _print(f"\n[check] model: {model_name}   scores: {kind}s")
    _print(f"[check] silver val films: {len(val_labels)}   gold groups: {len(gold_gts)}")
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
        "F1 bonus": tune_bonus(val_scores, val_labels, macro_f1, grid),
        "recall bonus": tune_bonus(val_scores, val_labels, macro_recall, grid),
        "CE bias": fit_ce_bias(val_scores, val_labels, temperature),
    }
    em_bonus, em_t, prior_src, prior_gold = em_prior_bonus(
        val_scores, val_labels, gold_scores, temperature,
    )
    bonuses["EM prior"] = em_bonus
    gold_counts = Counter(c for g in gold_gts for c in g.split("|"))
    gold_true = np.array([gold_counts.get(c, 0) for c in CLS_ORDER], dtype=float)
    gold_true /= gold_true.sum()
    _print(f"\n=== EM prior (no gold labels): temperature fit on silver val = {em_t:.4f}")
    _print(f"  {'class':<10} {'silver val':>11} {'gold est.':>10} {'gold true':>10}")
    for c, cls in enumerate(CLS_ORDER):
        _print(f"  {cls:<10} {prior_src[c]:>11.3f} {prior_gold[c]:>10.3f} {gold_true[c]:>10.3f}")
    _print("  (gold true is shown only to judge the estimate; it is not used)")
    if gold_oracle:
        # Diagnostic only: tuned on the gold labels it is scored on.
        bonuses["GOLD oracle"] = tune_bonus(
            gold_single_scores, gold_single_labels, macro_f1, grid,
        )
    _print(f"\n=== Per-class bonuses chosen on silver val (stable fixed at 0; {kind} units)")
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
        for bonus_name, b in bonuses.items():
            print_setmatch_report(
                gold_results(gold_scores, gold_gts, gold_findings, b),
                f"{model_name} + silver-val {bonus_name}",
                ", bonus=" + ",".join(f"{x:+.3f}" for x in b),
            )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", help="checkpoint to score")
    p.add_argument("--supervised", action="store_true",
                   help="--ckpt is an unfrozen supervised checkpoint "
                        "(checkpoints_supervised_progression_unfrozen/epoch_N.pt)")
    p.add_argument("--reuse", help="scores .npz from an earlier run; skips the model")
    p.add_argument("--out", help="where to save scores (default next to the checkpoint)")
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--gold-limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-full-report", action="store_true")
    p.add_argument("--gold-oracle", action="store_true",
                   help="also tune bonuses on gold itself (upper bound; never report it)")
    p.add_argument("--knob", action="store_true",
                   help="only print the one-knob balancing table (needs --reuse)")
    p.add_argument("--per-class", action="store_true",
                   help="only print the one-class-at-a-time nudge table (needs --reuse)")
    p.add_argument("--preview-bias", action="store_true",
                   help="only print the learned per-class bias preview (needs --reuse)")
    p.add_argument("--class-weights", type=float, nargs=N_CLS,
                   default=[0.366, 0.140, 0.233, 0.432, 3.829],
                   help="training loss class weights, CLS_ORDER order "
                        "(default: the vljepa CBW weights printed at startup)")
    p.add_argument("--ref", help="scores .npz of a reference model (e.g. supervised) "
                                 "to compare improving / worsening F1 against")
    p.add_argument("--margin", type=float, default=0.02)
    args = p.parse_args()

    if args.reuse:
        data = np.load(args.reuse, allow_pickle=True)
        kind = str(data["score_kind"]) if "score_kind" in data else "cosine"
        name = str(data["model_name"]) if "model_name" in data else "vljepa"
        _print(f"[check] reusing scores from {args.reuse}")
        if args.preview_bias:
            ref = reference_metrics(args.ref) if args.ref else None
            preview_bias(data["val_scores"], data["val_labels"], data["gold_scores"],
                         data["gold_gts"], data["gold_findings"], kind, name,
                         np.array(args.class_weights), ref)
            return 0
        if args.per_class:
            ref = reference_metrics(args.ref) if args.ref else None
            per_class_table(data["gold_scores"], data["gold_gts"], data["gold_findings"],
                            kind, name, ref)
            return 0
        if args.knob:
            ref = reference_metrics(args.ref) if args.ref else None
            knob_table(data["val_labels"], data["gold_scores"], data["gold_gts"],
                       data["gold_findings"], kind, name, ref, args.margin)
            return 0
        report(data["val_scores"], data["val_labels"], data["gold_scores"],
               data["gold_gts"], data["gold_findings"], not args.no_full_report,
               kind, name, args.gold_oracle)
        return 0
    if not args.ckpt:
        p.error("--ckpt or --reuse is required")

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print(f"[check] ckpt   = {args.ckpt}")
    _print(f"[check] device = {device}")
    if args.supervised:
        scorer = SupervisedScorer(args.ckpt, device)
        name = "supervised"
    else:
        from .eval_gold import load_vljepa_from_ckpt

        model = load_vljepa_from_ckpt(args.ckpt, device)
        _print(f"[check] text lora = {getattr(model, 'text_lora_path', None)}")
        scorer = VLJEPAScorer(model)
        name = "vljepa"

    val_scores, val_labels = score_silver_val(
        scorer, device, args.batch_size, args.num_workers, args.val_limit,
    )
    gold_scores, gold_gts, gold_findings = score_gold(scorer, device, args.gold_limit)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)),
        "class_signal_" + os.path.splitext(os.path.basename(args.ckpt))[0] + ".npz",
    )
    np.savez(out, val_scores=val_scores, val_labels=val_labels,
             gold_scores=gold_scores, gold_gts=gold_gts, gold_findings=gold_findings,
             score_kind=scorer.kind, model_name=name)
    _print(f"[check] saved scores to {out}")
    report(val_scores, val_labels, gold_scores, gold_gts, gold_findings,
           not args.no_full_report, scorer.kind, name, args.gold_oracle)
    return 0


if __name__ == "__main__":
    if __package__ in (None, ""):
        __package__ = "vljepa"
    raise SystemExit(main())
