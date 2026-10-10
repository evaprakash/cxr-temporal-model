#!/usr/bin/env python3
"""Do progression sentences cluster by class, or by finding?

Encodes the same sentences with frozen BioViL-T text and with
Qwen3-Embedding-0.6B. No images. CPU is enough.

A useful encoder has same-class / different-finding cosine higher than
same-finding / different-class cosine, and improving–stable well below 0.90.

    bash vljepa/probe_text_encoder_clusters.sh
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import _root  # noqa: F401
from .cluster_paths import hf_home
from .prompts import cap_finding

from progression_phrases import CLS_ORDER

os.environ.setdefault("HF_HOME", hf_home())
os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home(), "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

# First wording of each class is the current VL-JEPA target.
TEMPLATES: Dict[str, List[str]] = {
    "improving": [
        "{finding} is improving.",
        "{finding} has decreased compared with the prior.",
        "{finding} shows interval improvement.",
        "{finding} is better than on the prior.",
    ],
    "stable": [
        "{finding} is stable.",
        "{finding} is unchanged from the prior.",
        "{finding} shows no interval change.",
        "{finding} remains the same.",
    ],
    "worsening": [
        "{finding} is worsening.",
        "{finding} has increased compared with the prior.",
        "{finding} shows interval worsening.",
        "{finding} is worse than on the prior.",
    ],
    "new": [
        "{finding} is new.",
        "{finding} was not present on the prior.",
        "{finding} has appeared since the prior.",
        "{finding} is a new finding.",
    ],
    "resolved": [
        "{finding} is resolved.",
        "{finding} has cleared.",
        "{finding} is no longer seen.",
        "{finding} has resolved since the prior.",
    ],
}

NEGATION_PAIRS = [
    ("Pleural effusion is worsening.", "Pleural effusion is not worsening."),
    ("Pleural effusion has resolved.", "Pleural effusion has not resolved."),
    ("Pleural effusion is present.", "No pleural effusion."),
]

QWEN_ID = "Qwen/Qwen3-Embedding-0.6B"


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def realize(template: str, finding: str) -> str:
    return template.format(finding=cap_finding(finding))


def sentence_rows() -> List[Tuple[str, str, int, str]]:
    rows = []
    for finding in FINDINGS:
        for cls in CLS_ORDER:
            for wi, template in enumerate(TEMPLATES[cls]):
                rows.append((finding, cls, wi, realize(template, finding)))
    return rows


def _check_transformers() -> None:
    import transformers
    parts = []
    for piece in transformers.__version__.split("."):
        if piece.isdigit():
            parts.append(int(piece))
        else:
            break
    while len(parts) < 2:
        parts.append(0)
    if tuple(parts[:2]) < (4, 51):
        raise SystemExit(
            f"transformers {transformers.__version__} cannot load Qwen3. "
            "Need transformers>=4.51.0."
        )


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0]), lengths]


@torch.no_grad()
def embed_qwen(texts: Sequence[str], batch_size: int = 4) -> torch.Tensor:
    _check_transformers()
    from transformers import AutoModel, AutoTokenizer

    _print(f"loading {QWEN_ID} on cpu")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_ID, padding_side="left")
    model = AutoModel.from_pretrained(QWEN_ID, torch_dtype=torch.float32)
    model.eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        tokens = tokenizer(
            batch, padding=True, truncation=True, max_length=64, return_tensors="pt",
        )
        hidden = model(**tokens).last_hidden_state
        pooled = last_token_pool(hidden, tokens["attention_mask"])
        chunks.append(F.normalize(pooled.float(), dim=-1).cpu())
        _print(f"  qwen {min(start + batch_size, len(texts))}/{len(texts)}")
    del model
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def embed_biovilt(texts: Sequence[str], batch_size: int = 32) -> torch.Tensor:
    from tempcxr.modules.text_encoder import BioViLTTextEncoder

    _print("loading BioViL-T text on cpu")
    enc = BioViLTTextEncoder(mode="biovilt").eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        global_emb, _, _ = enc.forward_contrastive(batch)
        chunks.append(F.normalize(global_emb.float(), dim=-1).cpu())
    del enc
    return torch.cat(chunks, dim=0)


def _mean_pair(vecs: Sequence[torch.Tensor]) -> float:
    n = len(vecs)
    if n < 2:
        return float("nan")
    acc = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            acc += float((vecs[i] * vecs[j]).sum())
            count += 1
    return acc / count


def canonical_index(rows: Sequence[Tuple[str, str, int, str]]) -> Dict[Tuple[str, str], int]:
    return {
        (finding, cls): i
        for i, (finding, cls, wi, _text) in enumerate(rows)
        if wi == 0
    }


def report(name: str, vecs: torch.Tensor, rows: Sequence[Tuple[str, str, int, str]]) -> None:
    canon = canonical_index(rows)
    _print("")
    _print("=" * 72)
    _print(name)
    _print("=" * 72)
    _print(f"dim={vecs.shape[1]}  sentences={len(rows)}")

    within_finding = []
    for finding in FINDINGS:
        group = [vecs[canon[(finding, cls)]] for cls in CLS_ORDER]
        within_finding.append(_mean_pair(group))
    within = sum(within_finding) / len(within_finding)

    across_finding = []
    for cls in CLS_ORDER:
        group = [vecs[canon[(finding, cls)]] for finding in FINDINGS]
        across_finding.append(_mean_pair(group))
    across = sum(across_finding) / len(across_finding)

    _print(f"same finding, different class   {within:.4f}   (want this low)")
    _print(f"same class, different finding   {across:.4f}   (want this higher)")
    _print(f"class minus finding              {across - within:+.4f}   (positive = clusters by class)")

    _print("")
    _print("canonical '{Finding} is {class}.'  mean cosine over findings")
    header = f"{'':12}" + "".join(f"{c[:10]:>12}" for c in CLS_ORDER)
    _print(header)
    grid = [[1.0] * 5 for _ in range(5)]
    for i, ci in enumerate(CLS_ORDER):
        for j, cj in enumerate(CLS_ORDER):
            if j <= i:
                continue
            cos = 0.0
            for finding in FINDINGS:
                a = vecs[canon[(finding, ci)]]
                b = vecs[canon[(finding, cj)]]
                cos += float((a * b).sum())
            cos /= len(FINDINGS)
            grid[i][j] = cos
            grid[j][i] = cos
    for i, cls in enumerate(CLS_ORDER):
        _print(f"{cls[:12]:12}" + "".join(f"{grid[i][j]:12.4f}" for j in range(5)))
    _print(f"improving–stable {grid[0][1]:.4f}")

    _print("")
    _print("negation  (lower = the encoder notices 'not' / 'no')")
    base = len(rows)
    for k, (a, b) in enumerate(NEGATION_PAIRS):
        cos = float((vecs[base + 2 * k] * vecs[base + 2 * k + 1]).sum())
        _print(f"  {cos:.4f}   {a}  ||  {b}")

    _maybe_tsne(name, vecs[: len(rows)], rows)


def _maybe_tsne(name: str, vecs: torch.Tensor, rows: Sequence[Tuple[str, str, int, str]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError as exc:
        _print(f"t-SNE skipped ({exc})")
        return
    try:
        _write_tsne(name, vecs, rows, TSNE, plt)
    except Exception as exc:
        _print(f"t-SNE skipped ({type(exc).__name__}: {exc})")


def _write_tsne(name, vecs, rows, TSNE, plt) -> None:
    perplexity = min(15, max(5, len(rows) // 4))
    xy = TSNE(
        n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
        random_state=0,
    ).fit_transform(vecs.numpy())
    out_dir = os.environ.get(
        "VLJEPA_CLUSTER_DIR",
        os.path.join(hf_home(), "..", "logs", "encoder_clusters"),
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    slug = name.split()[0].lower()
    colors = {
        "improving": "#1b9e77",
        "stable": "#666666",
        "worsening": "#d95f02",
        "new": "#7570b3",
        "resolved": "#e7298a",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mode in zip(axes, ("class", "finding")):
        if mode == "class":
            labels = [row[1] for row in rows]
            keys = list(CLS_ORDER)
            cmap = colors
        else:
            labels = [row[0] for row in rows]
            keys = list(FINDINGS)
            cmap = None
        for key in keys:
            idx = [i for i, lab in enumerate(labels) if lab == key]
            ax.scatter(
                xy[idx, 0], xy[idx, 1],
                s=18, label=key,
                c=None if cmap is None else cmap[key],
            )
        ax.set_title(f"{name}: colored by {mode}")
        ax.legend(fontsize=7, loc="best", frameon=False)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{slug}_tsne.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    _print(f"t-SNE {path}")


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    rows = sentence_rows()
    texts = [row[3] for row in rows]
    neg = [t for pair in NEGATION_PAIRS for t in pair]
    all_texts = texts + neg
    _print(f"sentences={len(texts)}  negation_pairs={len(NEGATION_PAIRS)}  device=cpu")

    qwen = embed_qwen(all_texts)
    report("Qwen3-Embedding-0.6B", qwen, rows)
    del qwen

    biovilt = embed_biovilt(all_texts)
    report("BioViL-T", biovilt, rows)


if __name__ == "__main__":
    main()
