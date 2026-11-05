#!/usr/bin/env python3
"""CPU smoke test for dynamic-sentence VL-JEPA.

  1. Builds the silver val dataset (one pair per example, dynamic-sentence
     targets) and takes 4 pairs. Falls back to random images and made-up
     sentences if the silver data or images are missing.
  2. Builds VL-JEPA with the original BioViL-T text encoder (no LoRA) and a
     tiny 2-layer Llama (``--full``: the real 8 Llama-3.2-1B layers and
     pretrained BioViL-T image weights). Falls back to stub encoders if
     BioViL-T cannot load (shapes and logic only).
  3. One forward pass, the batch InfoNCE loss, one backward; checks that
     only the predictor gets gradients.
  4. Duplicate check: two rows whose sentences match ("No interval change."
     and "no interval change") must not be each other's negatives.
  5. Zero-shot scoring: per-finding phrase bank, all-findings combinations
     with the soft maximum, and the fallback above --max-k findings.
  6. Retrieval recall on a toy case where the answer is known.
  7. Second stage: --init-from restores saved weights; with the image
     encoder unfrozen it gets gradients at LR * IMAGE_LR_MULT, the text
     encoder gets none, and BatchNorm stats do not move.

    CUDA_VISIBLE_DEVICES="" python -m vljepa.smoke_dynamic
    CUDA_VISIBLE_DEVICES="" python -m vljepa.smoke_dynamic --full
    bash vljepa/run_ccw.sh dyn_smoke
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

from .dynamic import (
    DYNAMIC_QUERY,
    DynamicScorer,
    dynamic_collate_fn,
    retrieval_recall,
    same_text_mask,
)

B = 4
FALLBACK_SENTENCES = [
    "Interval increase in the left pleural effusion.",
    "No interval change.",
    "Pulmonary edema has improved.",
    "New right lower lobe consolidation.",
]
DUP_SENTENCES = [
    "No interval change.",
    "Interval increase in the left pleural effusion.",
    "no interval change",
    "Worsening pulmonary edema.",
]
FAILURES = []


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def check(ok: bool, what: str) -> None:
    _print(f"  [{'ok  ' if ok else 'FAIL'}] {what}")
    if not ok:
        FAILURES.append(what)


def shape(x) -> str:
    return f"{tuple(x.shape)} {x.dtype}" if torch.is_tensor(x) else repr(type(x))


def load_batch(use_data: bool):
    if use_data:
        try:
            from .cluster_paths import image_roots
            from .dynamic import VLJEPADynamicDataset

            ds = VLJEPADynamicDataset(image_roots(), split="val", train=False)
            items = [ds[i] for i in range(B)]
            for i in range(B):
                _print(f"  pair {i}: {len(ds.sentences[i])} dynamic sentences")
                for s in ds.sentences[i][:4]:
                    _print(f"      - {s}")
            return dynamic_collate_fn(items), "silver val"
        except Exception as exc:
            _print(f"  silver data unavailable ({type(exc).__name__}: {exc}); using fallback")
    batch = {
        "prior_image": torch.randn(B, 3, 448, 448),
        "current_image": torch.randn(B, 3, 448, 448),
        "query_text": [DYNAMIC_QUERY] * B,
        "target_text": list(FALLBACK_SENTENCES),
        "dataset": ["fallback"] * B,
    }
    return batch, "fallback (random images)"


class StubImage(nn.Module):
    """Tiny conv + BatchNorm encoder: 448px -> 14x14 = 196 patches of 128."""

    embed_dim = 128

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 16, kernel_size=32, stride=32)
        self.bn = nn.BatchNorm2d(16)
        self.proj = nn.Linear(16, 128)

    def forward(self, current, prior):
        x = self.bn(self.conv(torch.cat([current, prior], dim=1)))
        patches = self.proj(x.flatten(2).transpose(1, 2))
        return F.normalize(patches.mean(dim=1), dim=-1), F.normalize(patches, dim=-1)


class StubText(nn.Module):
    """Same text -> same embedding, like a frozen encoder."""

    proj_dim = 128

    def forward_contrastive(self, texts):
        out = []
        for t in texts:
            g = torch.Generator().manual_seed(zlib.crc32(t.encode()))
            out.append(torch.randn(128, generator=g))
        g = F.normalize(torch.stack(out), dim=-1)
        return g, g[:, None, :], torch.ones(len(texts), 1, dtype=torch.bool)


def build_model(full: bool):
    from .model import VLJEPA

    try:
        model = VLJEPA(
            image_mode="biovilt",
            smoke=not full,
            n_llama_layers=8 if full else 2,
            freeze_image_encoder=True,
            freeze_text_encoder=True,
            gradient_checkpointing=False,
            text_lora=False,
        )
        return model, "real BioViL-T text" + (" + full model" if full else " + tiny Llama")
    except Exception as exc:
        _print(f"  real encoders failed ({type(exc).__name__}: {exc})")
        traceback.print_exc()
        _print("  falling back to stub encoders (shapes and logic only)")

    model = VLJEPA.__new__(VLJEPA)
    nn.Module.__init__(model)
    from .model import LlamaPredictor

    model.image_encoder = StubImage()
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    model.image_encoder.eval()
    model.text_encoder = StubText()
    model.text_lora_path = None
    model.predictor = LlamaPredictor(vis_dim=128, out_dim=128, n_layers=2, smoke=True,
                                     gradient_checkpointing=False)
    model.freeze_image_encoder = model.freeze_text_encoder = True
    model.freeze_image_norm = True
    return model, "stub encoders + tiny Llama"


def second_stage(model, batch) -> None:
    import tempfile

    from .model import pair_infonce_loss
    from .train_dynamic import IMAGE_LR_MULT, LR, build_optimizer, load_init_weights

    _print("\n[7] second stage: --init-from + --unfreeze-image")
    state = model.predictor.state_dict()
    keys = list(state)[-2:]
    saved = {k: state[k].clone() for k in keys}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "epoch_1.pt")
        torch.save({"epoch": 1, "model": {f"predictor.{k}": v for k, v in saved.items()},
                    "vljepa_cfg": {"objective": "dynamic", "target_mode": "sentence"}}, path)
        with torch.no_grad():
            for k in keys:
                state[k].add_(1.0)
        load_init_weights(model, path, "sentence", 0)
    now = model.predictor.state_dict()
    check(all(torch.equal(saved[k], now[k]) for k in keys),
          "--init-from restores the saved predictor weights")

    img_params = list(model.image_encoder.parameters())
    if not img_params:
        _print("  stub image encoder has no weights; skipping the unfreeze checks")
        return
    model.freeze_image_encoder = False
    for p in img_params:
        p.requires_grad = True
    model.train()
    check(not any(m.training for m in model.image_encoder.modules()),
          "image encoder stays in eval mode (BatchNorm stats + dropout frozen)")
    buffers = {n: b.clone() for n, b in model.image_encoder.named_buffers()}
    opt = build_optimizer(model)
    lrs = [g["lr"] for g in opt.param_groups]
    n_img = sum(p.numel() for p in opt.param_groups[-1]["params"])
    _print(f"  optimizer groups: lr {lrs}, image params {n_img:,}")
    check(lrs == [LR, LR * IMAGE_LR_MULT], "image encoder trains at LR * IMAGE_LR_MULT")

    out = model(batch["prior_image"], batch["current_image"], batch["query_text"],
                target_texts=batch["target_text"])
    same = same_text_mask(batch["target_text"], batch["target_text"])
    loss, _ = pair_infonce_loss(out["pred"], out["target_global"], same, torch.arange(B))
    loss.backward()
    img_grad = sum(p.grad.abs().sum().item() for p in img_params if p.grad is not None)
    text_grad = [n for n, p in model.text_encoder.named_parameters() if p.grad is not None]
    check(img_grad > 0, f"image encoder gets gradients (sum |grad| = {img_grad:.3g})")
    check(not text_grad, "text encoder still gets no gradients")
    opt.step()
    check(all(torch.equal(buffers[n], b) for n, b in model.image_encoder.named_buffers()),
          "BatchNorm running stats unchanged after a training step")
    model.zero_grad(set_to_none=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true",
                   help="real Llama-3.2-1B layers and pretrained BioViL-T image weights")
    p.add_argument("--no-data", action="store_true", help="skip building the silver dataset")
    args = p.parse_args()
    torch.manual_seed(0)
    from .model import pair_infonce_loss

    _print("=" * 72)
    _print("Dynamic-sentence VL-JEPA CPU smoke test")
    _print(f"  query: {DYNAMIC_QUERY}")
    _print("=" * 72)

    _print("\n[1] data")
    batch, source = load_batch(not args.no_data)
    _print(f"  source: {source}")
    for i, t in enumerate(batch["target_text"]):
        _print(f"  target {i}: {t}")

    _print("\n[2] model")
    model, backend = build_model(args.full)
    _print(f"  backend: {backend}")
    lora_params = [n for n, _ in model.text_encoder.named_parameters() if "lora" in n.lower()]
    check(model.text_lora_path is None and not lora_params,
          "text encoder is the original BioViL-T (no LoRA)")
    check(not any(p.requires_grad for p in model.text_encoder.parameters()), "text encoder frozen")
    check(not any(p.requires_grad for p in model.image_encoder.parameters()), "image encoder frozen")
    _print(f"  predictor: {model.predictor.init_source}, tokenizer {model.predictor.tokenizer_source}")

    _print("\n[3] forward, loss, backward")
    model.train()
    out = model(batch["prior_image"], batch["current_image"], batch["query_text"],
                target_texts=batch["target_text"])
    for k in ("img_global", "img_patches", "query_ids", "query_llama", "pred", "target_global"):
        _print(f"  {k:14s} {shape(out[k])}")
    check(out["pred"].shape == (B, 128), "Ŝ is (B, 128)")
    check(out["target_global"].shape == (B, 128), "one target embedding per pair")
    same = same_text_mask(batch["target_text"], batch["target_text"])
    pos = torch.arange(B)
    loss, logits = pair_infonce_loss(out["pred"], out["target_global"], same, pos)
    _print(f"  same-text mask:\n{same.int()}")
    _print(f"  logits (row = Ŝ, column = sentence):\n{logits.detach().numpy().round(2)}")
    _print(f"  loss {float(loss):.4f}  (chance is about ln {B} = {torch.log(torch.tensor(float(B))):.4f})")
    check(torch.isfinite(loss).item(), "loss is finite")
    if not (same.sum() > B):
        plain = F.cross_entropy(logits, pos)
        check(torch.allclose(loss, plain), "no duplicates: loss equals plain cross-entropy")
    loss.backward()
    pred_grad = sum(p.grad.abs().sum().item() for p in model.predictor.parameters() if p.grad is not None)
    enc_grad = [n for n, p in list(model.image_encoder.named_parameters())
                + list(model.text_encoder.named_parameters()) if p.grad is not None]
    check(pred_grad > 0, f"predictor gets gradients (sum |grad| = {pred_grad:.3g})")
    check(not enc_grad, "image and text encoders get no gradients")
    model.zero_grad(set_to_none=True)

    _print("\n[4] duplicate sentences")
    with torch.no_grad():
        g, _, _ = model.encode_targets(DUP_SENTENCES)
    same = same_text_mask(DUP_SENTENCES, DUP_SENTENCES)
    _print(f"  sentences: {DUP_SENTENCES}")
    _print(f"  same-text mask:\n{same.int()}")
    check(bool(same[0, 2]) and bool(same[2, 0]), "'No interval change.' == 'no interval change'")
    loss_d, logits_d = pair_infonce_loss(out["pred"].detach(), g, same, pos)
    manual = []
    for i in range(B):
        keep = ~same[i] | (torch.arange(B) == i)
        manual.append(F.cross_entropy(logits_d[i, keep][None], keep[:i + 1].sum()[None] - 1))
    check(torch.allclose(loss_d, torch.stack(manual).mean(), atol=1e-6),
          f"duplicates dropped from negatives (loss {float(loss_d):.4f})")

    _print("\n[5] zero-shot scoring")
    model.eval()
    prior, current = batch["prior_image"][0], batch["current_image"][0]
    per = DynamicScorer(model, mode="perfinding")
    s = per.score_pair(prior, current, ["edema", "pleural effusion"])
    for f, v in s.items():
        _print(f"  per-finding {f:18s} " + "  ".join(f"{c} {x:+.3f}" for c, x in zip(CLS_ORDER, v)))
    check(all(v.shape == (5,) for v in s.values()), "per-finding: 5 scores per finding")
    comb = DynamicScorer(model, mode="combined", max_k=4)
    s2 = comb.score_pair(prior, current, ["pleural effusion", "edema"])
    texts, ids, emb = comb.combos(("edema", "pleural effusion"))
    _print(f"  combined: {len(texts)} sentences, e.g. {texts[0]!r} ... {texts[-1]!r}")
    pred = comb.predict(prior, current)
    top = (emb @ pred).topk(3)
    for v, j in zip(top.values, top.indices):
        _print(f"    {float(v):+.3f}  {texts[int(j)]}")
    for f, v in s2.items():
        _print(f"  combined    {f:18s} " + "  ".join(f"{c} {x:+.3f}" for c, x in zip(CLS_ORDER, v)))
    check(len(texts) == 25 and set(s2) == {"edema", "pleural effusion"},
          "combined: 25 combinations for 2 findings, scores for both")
    one = comb.score_combined(pred, ["edema"])["edema"]
    direct = (comb.class_sentences("edema") @ pred).numpy()
    check(abs(one - direct).max() < 1e-5, "combined with 1 finding equals '{Finding} is {class}.' cosine")
    small = DynamicScorer(model, mode="combined", max_k=1)
    small.score_pair(prior, current, ["edema", "pleural effusion"])
    check(small.n_fallback == 1, "pairs above --max-k fall back to per-finding sentences")

    _print("\n[6] retrieval")
    toy = F.normalize(torch.randn(5, 128), dim=-1)
    toy_texts = ["A.", "B.", "a", "C.", "D."]
    toy_tgt = toy.clone()
    toy_tgt[2] = toy_tgt[0]
    r = retrieval_recall(toy_tgt, toy_tgt, toy_texts)
    _print(f"  {r}")
    check(r["recall@1"] == 1.0 and r["gallery"] == 4, "exact predictions give R@1 = 1, duplicates counted once")

    second_stage(model, batch)

    _print("\n" + "=" * 72)
    if FAILURES:
        _print(f"SMOKE TEST FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
        return 1
    _print(f"SMOKE TEST PASSED ({backend}, data: {source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
