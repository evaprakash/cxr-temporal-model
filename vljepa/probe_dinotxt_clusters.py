#!/usr/bin/env python3
"""Do progression sentences cluster by class under the dino.txt text tower?

Same sentences, cosine tables, negation pairs, and t-SNE as
``probe_text_encoder_clusters.py``. This encoder is the text half of
``dinov2_vitl14_reg4_dinotxt_tet1280d20h24l``: 24 causal layers, width
1280, CLIP BPE (49,408 tokens), then a linear map to 2048-d. The vector
used here is that projected output, L2-normalized. Pooling is the CLIP
end-of-text token (argmax over token ids), which is how dino.txt reads text.

The checkpoint is the local text-encoder file, not a download:

    dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth

    python -m vljepa.probe_dinotxt_clusters
    DINOTXT_TEXT_CKPT=/path/to/that.pth python -m vljepa.probe_dinotxt_clusters
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .probe_text_encoder_clusters import NEGATION_PAIRS, report, sentence_rows

CKPT_NAME = "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth"
CONTEXT_LENGTH = 77
VOCAB_SIZE = 49408
WIDTH = 1280
HEADS = 20
LAYERS = 24
FFN_RATIO = 4
EMBED_DIM = 2048
CLIP_TOKENIZER = "openai/clip-vit-base-patch32"


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def resolve_ckpt() -> str:
    explicit = os.environ.get("DINOTXT_TEXT_CKPT")
    if explicit:
        if not os.path.isfile(explicit):
            raise SystemExit(f"DINOTXT_TEXT_CKPT is not a file: {explicit}")
        return explicit
    path = os.path.join(_repo_root(), CKPT_NAME)
    if os.path.isfile(path):
        return path
    raise SystemExit(
        f"missing {CKPT_NAME}\n"
        f"looked at {path}\n"
        "Set DINOTXT_TEXT_CKPT to the text-encoder .pth."
    )


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        bsz, length, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, length, 3, self.num_heads, dim // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        out = out.transpose(1, 2).contiguous().view(bsz, length, dim)
        return self.proj_drop(self.proj(out))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float, is_causal: bool) -> None:
        super().__init__()
        self.is_causal = is_causal
        self.ls1 = nn.Identity()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = _Attention(dim, num_heads)
        self.ffn_norm = nn.LayerNorm(dim)
        hidden = int(dim * ffn_ratio)
        self.feed_forward = nn.Module()
        self.feed_forward.fc1 = nn.Linear(dim, hidden, bias=True)
        self.feed_forward.act = nn.GELU()
        self.feed_forward.fc2 = nn.Linear(hidden, dim, bias=True)
        self.feed_forward.drop = nn.Dropout(0.0)
        self.ls2 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        hidden = self.feed_forward.fc1(self.ffn_norm(x))
        hidden = self.feed_forward.drop(self.feed_forward.act(hidden))
        hidden = self.feed_forward.drop(self.feed_forward.fc2(hidden))
        return x + self.ls2(hidden)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = WIDTH
        self.num_heads = HEADS
        self.token_embedding = nn.Embedding(VOCAB_SIZE, WIDTH)
        self.positional_embedding = nn.Parameter(torch.empty(CONTEXT_LENGTH, WIDTH))
        self.dropout = nn.Dropout(0.0)
        self.blocks = nn.ModuleList(
            [_Block(WIDTH, HEADS, FFN_RATIO, is_causal=True) for _ in range(LAYERS)]
        )
        self.ln_final = nn.LayerNorm(WIDTH)

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        length = token_indices.shape[1]
        x = self.token_embedding(token_indices) + self.positional_embedding[:length]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.ln_final(x)


class _Head(nn.Module):
    """No extra blocks. Projects 1280-d tokens to the 2048-d shared space."""

    def __init__(self) -> None:
        super().__init__()
        self.ln_final = nn.Identity()
        self.block_list = nn.ModuleList([nn.Identity()])
        self.linear_projection = nn.Linear(WIDTH, EMBED_DIM, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.block_list:
            tokens = block(tokens)
        return self.linear_projection(self.ln_final(tokens))


class DinoTxtText(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.head = _Head()

    def forward(self, token_indices: torch.Tensor) -> torch.Tensor:
        tokens = self.head(self.backbone(token_indices))
        # CLIP end-of-text is the largest id in the row. Padding is 0.
        index = token_indices.argmax(dim=-1)
        return tokens[torch.arange(tokens.shape[0], device=tokens.device), index]


def load_text_tower(path: str) -> DinoTxtText:
    _print(f"loading {path}")
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        state = blob["state_dict"]
    elif isinstance(blob, dict):
        state = blob
    else:
        raise SystemExit(f"unrecognized checkpoint type: {type(blob)}")
    if state and all(key.startswith("text_model.") for key in state):
        state = {key[len("text_model."):]: value for key, value in state.items()}
    model = DinoTxtText()
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise SystemExit(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    model.eval()
    n_param = sum(p.numel() for p in model.parameters())
    _print(f"  loaded strict  params={n_param:,}  out_dim={EMBED_DIM}")
    return model


def _tokenize(tokenizer, texts: Sequence[str]) -> torch.Tensor:
    encoded = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=CONTEXT_LENGTH,
        return_tensors="pt",
    )
    return encoded["input_ids"]


@torch.no_grad()
def embed_dinotxt(texts: Sequence[str], ckpt: str, batch_size: int = 8) -> torch.Tensor:
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(CLIP_TOKENIZER)
    model = load_text_tower(ckpt)
    chunks: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        ids = _tokenize(tokenizer, batch)
        if int(ids.max()) >= VOCAB_SIZE:
            raise SystemExit(f"token id {int(ids.max())} is outside vocab {VOCAB_SIZE}")
        vec = F.normalize(model(ids).float(), dim=-1).cpu()
        chunks.append(vec)
        _print(f"  dinotxt {min(start + batch_size, len(texts))}/{len(texts)}")
    del model
    return torch.cat(chunks, dim=0)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    ckpt = resolve_ckpt()
    rows = sentence_rows()
    texts = [row[3] for row in rows]
    neg = [t for pair in NEGATION_PAIRS for t in pair]
    all_texts = texts + neg
    _print(
        f"sentences={len(texts)}  negation_pairs={len(NEGATION_PAIRS)}  "
        f"device=cpu  ckpt={ckpt}"
    )
    vecs = embed_dinotxt(all_texts, ckpt)
    report("dino.txt text", vecs, rows)


if __name__ == "__main__":
    main()
