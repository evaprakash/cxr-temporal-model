#!/usr/bin/env python3
"""Find class wordings whose frozen BioViL-T embeddings are far apart.

Encodes a bank of report-style sentences for each progression class with the
pretrained BioViL-T text encoder, averaged over the 11 findings. The current
target is ``{Finding} is {class}.`` Improving and stable land at cosine 0.90
under that template. This script searches one sentence per class and keeps
the set whose worst pairwise cosine is as low as possible.

Text only. No images, no Llama, no training.

    sbatch vljepa/probe_phrase_wordings.sh
"""

from __future__ import annotations

import itertools
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import _root  # noqa: F401
from .prompts import cap_finding

from progression_phrases import CLS_ORDER, PROGRESSION_PHRASES
from tempcxr.modules.text_encoder import BioViLTTextEncoder

FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged cardiomediastinum",
    "lung lesion",
    "lung opacity",
    "pleural effusion",
    "pleural other",
    "pneumonia",
    "pneumothorax",
]

# Longer comparison sentences. ``{finding}`` is sentence-cased at encode time.
# The one-word bank in PROGRESSION_PHRASES is added on top of these.
LONG_PHRASES: Dict[str, List[str]] = {
    "improving": [
        "{finding} has decreased compared with the prior.",
        "{finding} is smaller than on the prior study.",
        "{finding} shows interval improvement.",
        "{finding} has partially cleared.",
        "{finding} is better than on the prior.",
        "{finding} is less conspicuous than before.",
        "{finding} has improved since the prior study.",
    ],
    "stable": [
        "{finding} is unchanged from the prior.",
        "{finding} shows no interval change.",
        "{finding} is not significantly changed.",
        "{finding} persists without change.",
        "{finding} is similar to the prior study.",
        "{finding} is again seen and unchanged.",
        "{finding} remains the same.",
    ],
    "worsening": [
        "{finding} has increased compared with the prior.",
        "{finding} is larger than on the prior study.",
        "{finding} shows interval worsening.",
        "{finding} has progressed since the prior.",
        "{finding} is worse than on the prior.",
        "{finding} is more conspicuous than before.",
        "{finding} has worsened since the prior study.",
    ],
    "new": [
        "{finding} was not present on the prior.",
        "{finding} is not seen on the prior study.",
        "{finding} has appeared since the prior.",
        "{finding} was absent on the prior study.",
        "{finding} is a new finding.",
        "{finding} has developed since the prior.",
    ],
    "resolved": [
        "{finding} has cleared.",
        "{finding} is no longer seen.",
        "{finding} has resolved since the prior.",
        "{finding} is no longer present.",
        "{finding} has completely cleared.",
        "{finding} was present before and has cleared.",
        "{finding} is gone on the current study.",
    ],
}

CURRENT = {
    "improving": "{finding} is improving.",
    "stable": "{finding} is stable.",
    "worsening": "{finding} is worsening.",
    "new": "{finding} is new.",
    "resolved": "{finding} is resolved.",
}


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def candidate_texts() -> Dict[str, List[str]]:
    """One list of unique sentence templates per class. Current wording is first."""
    out: Dict[str, List[str]] = {}
    for cls in CLS_ORDER:
        seen = set()
        texts: List[str] = []
        for raw in [CURRENT[cls], *LONG_PHRASES[cls]]:
            text = raw.strip()
            if text not in seen:
                seen.add(text)
                texts.append(text)
        for phrase in PROGRESSION_PHRASES[cls]:
            text = "{finding} is " + phrase.strip() + "."
            if text not in seen:
                seen.add(text)
                texts.append(text)
        out[cls] = texts
    return out


def realize(template: str, finding: str) -> str:
    return template.format(finding=cap_finding(finding))


@torch.no_grad()
def embed_all(
    enc: BioViLTTextEncoder,
    texts: Sequence[str],
    batch_size: int = 32,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(texts), batch_size):
        g, _, _ = enc.forward_contrastive(list(texts[start:start + batch_size]))
        chunks.append(F.normalize(g.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def pair_tables(
    emb: Dict[str, torch.Tensor],
) -> Dict[Tuple[str, str], torch.Tensor]:
    """Mean-over-findings cosine. ``emb[cls]`` is (F, W, D). Returns (W_i, W_j)."""
    tables = {}
    for i, ci in enumerate(CLS_ORDER):
        for cj in CLS_ORDER[i + 1:]:
            # (F, Wi, Wj)
            sim = torch.einsum("fad,fbd->fab", emb[ci], emb[cj])
            tables[(ci, cj)] = sim.mean(dim=0)
    return tables


def combo_stats(
    tables: Dict[Tuple[str, str], torch.Tensor],
    choice: Sequence[int],
) -> Tuple[float, float, List[Tuple[str, str, float]]]:
    pairs = []
    for i, ci in enumerate(CLS_ORDER):
        for j, cj in enumerate(CLS_ORDER):
            if j <= i:
                continue
            cos = float(tables[(ci, cj)][choice[i], choice[j]])
            pairs.append((ci, cj, cos))
    worst = max(p[2] for p in pairs)
    mean = sum(p[2] for p in pairs) / len(pairs)
    return worst, mean, pairs


def search(
    tables: Dict[Tuple[str, str], torch.Tensor],
    sizes: Sequence[int],
) -> List[Tuple[float, float, Tuple[int, ...]]]:
    """Every one-wording-per-class combo, best worst-pair first."""
    ranked = []
    ranges = [range(n) for n in sizes]
    for choice in itertools.product(*ranges):
        worst, mean, _ = combo_stats(tables, choice)
        ranked.append((worst, mean, choice))
    ranked.sort()
    return ranked


def matrix_lines(
    tables: Dict[Tuple[str, str], torch.Tensor],
    choice: Sequence[int],
) -> List[str]:
    n = len(CLS_ORDER)
    grid = [[1.0] * n for _ in range(n)]
    for i, ci in enumerate(CLS_ORDER):
        for j, cj in enumerate(CLS_ORDER):
            if j <= i:
                continue
            cos = float(tables[(ci, cj)][choice[i], choice[j]])
            grid[i][j] = cos
            grid[j][i] = cos
    header = f"{'':12}" + "".join(f"{c[:10]:>12}" for c in CLS_ORDER)
    lines = [header]
    for i, cls in enumerate(CLS_ORDER):
        lines.append(
            f"{cls[:12]:12}" + "".join(f"{grid[i][j]:12.4f}" for j in range(n))
        )
    return lines


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cands = candidate_texts()
    _print(f"device={device}")
    _print("candidates per class: " + ", ".join(
        f"{c}={len(cands[c])}" for c in CLS_ORDER
    ))
    n_combo = 1
    for c in CLS_ORDER:
        n_combo *= len(cands[c])
    _print(f"combinations={n_combo}")

    enc = BioViLTTextEncoder(mode="biovilt").to(device).eval()
    for p in enc.parameters():
        p.requires_grad = False

    flat: List[str] = []
    index: Dict[Tuple[str, int, int], int] = {}
    for fi, finding in enumerate(FINDINGS):
        for cls in CLS_ORDER:
            for wi, template in enumerate(cands[cls]):
                index[(cls, wi, fi)] = len(flat)
                flat.append(realize(template, finding))
    _print(f"encoding {len(flat)} sentences")
    vecs = embed_all(enc, flat)
    del enc

    emb: Dict[str, torch.Tensor] = {}
    n_find = len(FINDINGS)
    dim = vecs.shape[1]
    for cls in CLS_ORDER:
        n_w = len(cands[cls])
        block = torch.empty(n_find, n_w, dim)
        for fi in range(n_find):
            for wi in range(n_w):
                block[fi, wi] = vecs[index[(cls, wi, fi)]]
        emb[cls] = block
    del vecs

    tables = pair_tables(emb)
    current = tuple(0 for _ in CLS_ORDER)
    cur_worst, cur_mean, cur_pairs = combo_stats(tables, current)
    cur_pairs_sorted = sorted(cur_pairs, key=lambda p: -p[2])

    _print("")
    _print("=" * 72)
    _print("CURRENT  {Finding} is {class}.")
    _print("=" * 72)
    for line in matrix_lines(tables, current):
        _print(line)
    _print(f"worst pair  {cur_pairs_sorted[0][0]}–{cur_pairs_sorted[0][1]}"
           f"  {cur_pairs_sorted[0][2]:.4f}")
    _print(f"mean off-diagonal  {cur_mean:.4f}")

    _print("")
    _print("searching")
    ranked = search(tables, [len(cands[c]) for c in CLS_ORDER])
    best_worst, best_mean, best = ranked[0]

    _print("")
    _print("=" * 72)
    _print("BEST  lowest worst-pair cosine, then lowest mean off-diagonal")
    _print("=" * 72)
    for cls, wi in zip(CLS_ORDER, best):
        example = realize(cands[cls][wi], "pleural effusion")
        _print(f"  {cls:12} {example}")
    _print("")
    for line in matrix_lines(tables, best):
        _print(line)
    _, _, best_pairs = combo_stats(tables, best)
    best_pairs_sorted = sorted(best_pairs, key=lambda p: -p[2])
    _print(f"worst pair  {best_pairs_sorted[0][0]}–{best_pairs_sorted[0][1]}"
           f"  {best_pairs_sorted[0][2]:.4f}"
           f"   (current {cur_worst:.4f})")
    _print(f"mean off-diagonal  {best_mean:.4f}   (current {cur_mean:.4f})")
    _print("closest pairs:")
    for a, b, cos in best_pairs_sorted[:5]:
        _print(f"  {a:12} {b:12} {cos:.4f}")

    _print("")
    _print("=" * 72)
    _print("TOP 5 SETS")
    _print("=" * 72)
    shown = 0
    last = None
    for worst, mean, choice in ranked:
        key = (round(worst, 4), round(mean, 4))
        if key == last:
            continue
        last = key
        shown += 1
        if shown > 5:
            break
        _print(f"#{shown}  worst={worst:.4f}  mean={mean:.4f}")
        for cls, wi in zip(CLS_ORDER, choice):
            _print(f"    {cls:12} {realize(cands[cls][wi], 'pleural effusion')}")


if __name__ == "__main__":
    main()
