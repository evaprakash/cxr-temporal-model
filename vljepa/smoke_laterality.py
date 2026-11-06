#!/usr/bin/env python3
"""CPU smoke test for laterality 5-way VL-JEPA.

  1. Labels: sentence templates, label indices, and the silver side table
     (skipped if silver_laterality.parquet is missing).
  2. Data: 4 silver val finding-examples, sided and unsided mixed. Falls
     back to random images and made-up sides if silver data is missing.
  3. Model: original BioViL-T text encoder (no LoRA) and a tiny 2-layer
     Llama (``--full``: the real 8 Llama-3.2-1B layers and pretrained image
     weights). Falls back to stub encoders if BioViL-T cannot load.
  4. One training step through train_laterality.batch_step: loss equals the
     class-weighted cross-entropy over each row's own 5 or 15 candidates;
     only the predictor gets gradients.
  5. Readouts on hand-made logits: class with the side summed out, side
     with the class summed out.
  6. Gold scoring: 5 scores per finding, equal to tau * logsumexp over the
     plain and sided sentences; cardiomegaly uses the plain sentences only.
  7. Target cache: a sentence is encoded once.
  8. Text encoder probe (real BioViL-T only): how far apart left and right
     sentences are, compared with different classes.

    CUDA_VISIBLE_DEVICES="" python -m vljepa.smoke_laterality
    bash vljepa/run_ccw.sh lat_smoke [--full]
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import zlib

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _root  # noqa: F401

from progression_phrases import CLS_ORDER

from .laterality import (
    LATERALITY_PATH,
    N_CLS,
    SIDES,
    TEMPERATURE,
    LateralityVLJEPA,
    TargetBank,
    candidate_texts,
    label_index,
    laterality_collate_fn,
    load_side_map,
    readouts,
    scoring_variants,
    sided_target_text,
)
from .prompts import query_text

B = 4
FALLBACK = [("pleural effusion", "worsening", "left"), ("cardiomegaly", "stable", None),
            ("edema", "improving", "bilateral"), ("pneumothorax", "new", "right")]
FAILURES = []


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def check(ok: bool, what: str) -> None:
    _print(f"  [{'ok  ' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILURES.append(what)


def labels_check() -> None:
    _print("\n[1] labels")
    texts = candidate_texts("Pleural effusion", "left")
    _print(f"  sided candidates: {len(texts)}, e.g. {texts[0]!r}, {texts[7]!r}, {texts[-1]!r}")
    check(len(texts) == 15 and texts[0] == "Left pleural effusion is improving."
          and texts[-1] == "Bilateral pleural effusion is resolved.",
          "15 sided candidates, side-major")
    check(candidate_texts("pleural effusion", None)[2] == "Pleural effusion is worsening.",
          "unsided rows keep '{Finding} is {class}.'")
    check(all(candidate_texts("edema", s)[label_index(c, s)] ==
              (sided_target_text("edema", CLS_ORDER[c], s) if s else
               f"Edema is {CLS_ORDER[c]}.")
              for s in (None,) + SIDES for c in range(N_CLS)),
          "label index points at the right sentence for every side and class")
    check(len(scoring_variants("cardiomegaly")) == 1 and len(scoring_variants("edema")) == 4,
          "gold scoring: plain only for cardiomegaly, plain + 3 sides otherwise")
    if os.path.isfile(LATERALITY_PATH):
        side_map = load_side_map(LATERALITY_PATH)
        k = next(iter(side_map))
        _print(f"  {len(side_map):,} sided keys in {LATERALITY_PATH}; e.g. {k[0]} ... {k[-1]} -> "
               f"{side_map[k]}")
        check(set(side_map.values()) <= set(SIDES), "side table holds only left/right/bilateral")
        check(not any(k[-1] in ("cardiomegaly", "enlarged cardiomediastinum") for k in side_map),
              "no sides for cardiomegaly / enlarged cardiomediastinum")
    else:
        _print(f"  {LATERALITY_PATH} missing; skipping the side table checks")


def load_batch(use_data: bool):
    if use_data:
        try:
            from .cluster_paths import image_roots
            from .laterality import LateralityFindingDataset

            ds = LateralityFindingDataset(image_roots(), split="val", train=False)
            sided = [i for i, s in enumerate(ds.sides) if s is not None][:3]
            plain = [i for i, s in enumerate(ds.sides) if s is None][:1]
            items = [ds[i] for i in sided + plain]
            return laterality_collate_fn(items), "silver val"
        except Exception as exc:
            _print(f"  silver data unavailable ({type(exc).__name__}: {exc}); using fallback")
    items = []
    for finding, cls, side in FALLBACK:
        c = CLS_ORDER.index(cls)
        items.append({
            "prior_image": torch.randn(3, 448, 448), "current_image": torch.randn(3, 448, 448),
            "finding": finding, "cls_idx": c, "label": label_index(c, side), "side": side,
            "query_text": query_text(finding), "target_texts": candidate_texts(finding, side),
            "dataset": "fallback",
        })
    return laterality_collate_fn(items), "fallback (random images)"


class StubImage(nn.Module):
    embed_dim = 128

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 16, kernel_size=32, stride=32)
        self.proj = nn.Linear(16, 128)

    def forward(self, current, prior):
        x = self.conv(torch.cat([current, prior], dim=1))
        patches = self.proj(x.flatten(2).transpose(1, 2))
        return F.normalize(patches.mean(dim=1), dim=-1), F.normalize(patches, dim=-1)


class StubText(nn.Module):
    """Same text -> same embedding, like a frozen encoder. Counts calls."""

    proj_dim = 128

    def __init__(self):
        super().__init__()
        self.n_encoded = 0

    def forward_contrastive(self, texts):
        self.n_encoded += len(texts)
        out = []
        for t in texts:
            g = torch.Generator().manual_seed(zlib.crc32(t.encode()))
            out.append(torch.randn(128, generator=g))
        g = F.normalize(torch.stack(out), dim=-1)
        return g, g[:, None, :], torch.ones(len(texts), 1, dtype=torch.bool)


def build_model(full: bool):
    try:
        model = LateralityVLJEPA(
            image_mode="biovilt", smoke=not full, n_llama_layers=8 if full else 2,
            freeze_image_encoder=True, freeze_text_encoder=True,
            gradient_checkpointing=False, text_lora=False,
        )
        return model, "real BioViL-T text" + (" + full model" if full else " + tiny Llama"), True
    except Exception as exc:
        _print(f"  real encoders failed ({type(exc).__name__}: {exc})")
        traceback.print_exc()
        _print("  falling back to stub encoders (shapes and logic only)")
    from .model import LlamaPredictor

    model = LateralityVLJEPA.__new__(LateralityVLJEPA)
    nn.Module.__init__(model)
    model.image_encoder = StubImage()
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    model.text_encoder = StubText()
    model.text_lora_path = None
    model.predictor = LlamaPredictor(vis_dim=128, out_dim=128, n_layers=2, smoke=True,
                                     gradient_checkpointing=False)
    model.freeze_image_encoder = model.freeze_text_encoder = model.freeze_image_norm = True
    return model, "stub encoders + tiny Llama", False


def text_probe(model) -> None:
    _print("\n[8] text encoder: left vs right, compared with class")
    bank = TargetBank(model)
    for finding in ("pleural effusion", "pneumothorax", "lung opacity"):
        e = bank(candidate_texts(finding, "left"), torch.device("cpu")).view(len(SIDES), N_CLS, -1)
        lr = float((e[0] * e[1]).sum(-1).mean())
        lb = float((e[0] * e[2]).sum(-1).mean())
        cls = e[0] @ e[0].T
        off = float((cls.sum() - cls.diagonal().sum()) / (N_CLS * (N_CLS - 1)))
        _print(f"  {finding:17s} cos(left, right) {lr:.3f}  cos(left, bilateral) {lb:.3f}  "
               f"cos(class a, class b | left) {off:.3f}")
    _print("  (closer to 1 = harder to tell apart; the original encoder put improving and "
           "stable at about 0.90)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true",
                   help="real Llama-3.2-1B layers and pretrained BioViL-T image weights")
    p.add_argument("--no-data", action="store_true", help="skip building the silver dataset")
    args = p.parse_args()
    torch.manual_seed(0)
    from .train_laterality import batch_step

    _print("=" * 72)
    _print("Laterality 5-way VL-JEPA CPU smoke test")
    _print("=" * 72)
    labels_check()

    _print("\n[2] data")
    batch, source = load_batch(not args.no_data)
    _print(f"  source: {source}")
    for f, s, c, t in zip(batch["finding"], batch["side"], batch["cls_idx"], batch["target_texts"]):
        _print(f"  {f:18s} side={str(s):9s} class={CLS_ORDER[int(c)]:9s} "
               f"{len(t):2d} candidates, positive {t[label_index(int(c), s)]!r}")

    _print("\n[3] model")
    model, backend, real = build_model(args.full)
    _print(f"  backend: {backend}")
    lora = [n for n, _ in model.text_encoder.named_parameters() if "lora" in n.lower()]
    check(model.text_lora_path is None and not lora, "text encoder is the original BioViL-T (no LoRA)")
    check(not any(p.requires_grad for p in model.text_encoder.parameters()), "text encoder frozen")
    check(not any(p.requires_grad for p in model.image_encoder.parameters()), "image encoder frozen")

    _print("\n[4] one training step")
    model.train()
    bank = TargetBank(model)
    weights = torch.tensor([0.4, 0.2, 0.3, 0.5, 3.6])
    loss, stats = batch_step(model, bank, batch, torch.device("cpu"), weights)
    out = model(batch["prior_image"], batch["current_image"], batch["query_text"])
    pred = F.normalize(out["pred"].detach().float(), dim=-1)
    num = den = 0.0
    for b in range(B):
        t = bank(batch["target_texts"][b], torch.device("cpu"))
        logit = (t @ pred[b]) / TEMPERATURE
        w = float(weights[batch["cls_idx"][b]])
        num += w * float(F.cross_entropy(logit[None], batch["label"][b:b + 1]))
        den += w
    _print(f"  loss {float(loss):.4f}  manual {num / den:.4f}  stats {stats}")
    check(abs(float(loss) - num / den) < 1e-4,
          "loss = class-weighted cross-entropy over each row's own candidates")
    check(stats[1] == B and stats[4] == sum(s is not None for s in batch["side"]),
          "stats count rows and sided rows")
    loss.backward()
    pred_grad = sum(p.grad.abs().sum().item() for p in model.predictor.parameters()
                    if p.grad is not None)
    enc_grad = [n for n, p in list(model.image_encoder.named_parameters())
                + list(model.text_encoder.named_parameters()) if p.grad is not None]
    check(pred_grad > 0, f"predictor gets gradients (sum |grad| = {pred_grad:.3g})")
    check(not enc_grad, "image and text encoders get no gradients")
    model.zero_grad(set_to_none=True)

    _print("\n[5] readouts")
    logits = torch.full((2, 15), -5.0)
    logits[0, :5] = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0])
    logits[1, label_index(CLS_ORDER.index("new"), "right")] = 4.0
    cls_pred, side_pred = readouts(logits, [None, "right"])
    check(cls_pred.tolist() == [2, 3] and side_pred.tolist() == [-1, 1],
          "class with side summed out; side with class summed out")

    _print("\n[6] gold scoring")
    model.eval()
    prior, current = batch["prior_image"][:1], batch["current_image"][:1]
    cache = {}
    s = model.score_classes(prior, current, "pleural effusion", text_cache=cache)
    cos = model.variant_cosines(prior, current, "pleural effusion", text_cache=cache)
    _print("  pleural effusion " + "  ".join(f"{c} {x:+.3f}" for c, x in zip(CLS_ORDER, s.tolist())))
    check(s.shape == (5,) and cos.shape == (4, 5), "5 scores from 4 variants")
    check(torch.allclose(s, TEMPERATURE * torch.logsumexp(cos / TEMPERATURE, dim=0)),
          "class score = tau * logsumexp over plain and sided sentences")
    check(bool((s >= cos.max(dim=0).values - 1e-6).all()), "soft maximum >= best single variant")
    c = model.variant_cosines(prior, current, "cardiomegaly", text_cache=cache)
    check(c.shape == (1, 5) and torch.allclose(
        model.score_classes(prior, current, "cardiomegaly", text_cache=cache), c[0]),
        "cardiomegaly: plain sentences only")

    _print("\n[7] target cache")
    if isinstance(model.text_encoder, StubText):
        bank2 = TargetBank(model)
        before = model.text_encoder.n_encoded
        bank2(candidate_texts("edema", "left"), torch.device("cpu"))
        mid = model.text_encoder.n_encoded
        bank2(candidate_texts("edema", "left") + ["Edema is stable."], torch.device("cpu"))
        check(mid - before == 15 and model.text_encoder.n_encoded - mid == 1,
              "each sentence is encoded once")
    else:
        _print("  real text encoder: call counting only with the stub; skipped")

    if real:
        text_probe(model)

    _print("\n" + "=" * 72)
    if FAILURES:
        _print(f"SMOKE TEST FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    _print(f"SMOKE TEST PASSED ({backend}, data: {source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
