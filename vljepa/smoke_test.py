#!/usr/bin/env python3
"""CPU smoke test for option-2 VL-JEPA.

Loads (or fabricates) 5 silver-style finding-examples, runs one forward
+ InfoNCE step on CPU with the tiny Llama predictor, and prints every
example's texts and every relevant tensor shape.

    python -m vljepa.smoke_test
    python vljepa/smoke_test.py
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _root  # noqa: F401
from .cluster_paths import image_roots as cluster_image_roots
from .dataset import flatten_target_texts
from .model import VLJEPA, class_infonce_loss
from .prompts import QUERY_TEMPLATE, TARGET_TEMPLATE, class_target_texts, query_text
from progression_phrases import CLS_ORDER

N_SMOKE = 5
IMAGE_HW = 448
# Scan this many silver rows to find N_SMOKE pairs whose CXRs exist on disk.
_SILVER_SCAN_CAP = 4000

_FALLBACK_SILVER = [
    ("edema", "worsening"),
    ("atelectasis", "stable"),
    ("cardiomegaly", "improving"),
    ("consolidation", "new"),
    ("pleural effusion", "resolved"),
]


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _shape(x) -> str:
    if torch.is_tensor(x):
        return f"{tuple(x.shape)} dtype={x.dtype} device={x.device}"
    return repr(type(x))


def _image_roots():
    """Same silver image roots as ``vljepa.train``."""
    return cluster_image_roots()


def _try_load_cxr_pair(dataset, prev_rel, curr_rel, roots):
    """Return (prior_t, current_t, prev_path, curr_path) or None."""
    from PIL import Image

    from dataset_combined import BASE_TRANSFORM, apply_augmentation
    from dataset_combined_jepa import _resolve_image_path

    if not dataset or dataset not in roots:
        return None
    if prev_rel is None or curr_rel is None:
        return None
    try:
        prev_p = _resolve_image_path(str(dataset), str(prev_rel), roots)
        curr_p = _resolve_image_path(str(dataset), str(curr_rel), roots)
    except Exception:
        return None
    if not prev_p.is_file() or not curr_p.is_file():
        return None
    try:
        prior = apply_augmentation(
            BASE_TRANSFORM(Image.open(prev_p).convert("RGB")), None,
        )
        current = apply_augmentation(
            BASE_TRANSFORM(Image.open(curr_p).convert("RGB")), None,
        )
    except Exception:
        return None
    return prior, current, str(prev_p), str(curr_p)


def _load_silver_rows(n: int):
    from dataset_combined_jepa import DEFAULT_FINDINGS
    from progression_phrases import SILVER_TO_CLS

    path = os.environ.get("VLJEPA_SMOKE_FINDINGS", DEFAULT_FINDINGS)
    roots = _image_roots()
    _print(f"[smoke] image roots: {roots}")
    if not os.path.isfile(path):
        _print(f"[smoke] silver parquet not found ({path}); using fallback labels")
        return [
            {"finding": f, "cls": c, "source": "fallback"}
            for f, c in _FALLBACK_SILVER[:n]
        ]
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        rows = []
        n_tried = 0
        n_missing_img = 0
        for _, r in df.iterrows():
            if n_tried >= _SILVER_SCAN_CAP:
                break
            finding = str(r.get("finding", "")).strip().lower()
            raw = str(r.get("progression", "")).strip()
            cls = SILVER_TO_CLS.get(raw, SILVER_TO_CLS.get(raw.title(), None))
            if not finding or cls not in CLS_ORDER:
                continue
            n_tried += 1
            loaded = _try_load_cxr_pair(
                r.get("dataset"),
                r.get("parent_image_prev"),
                r.get("parent_image_curr"),
                roots,
            )
            row = {
                "finding": finding,
                "cls": cls,
                "source": "silver",
                "dataset": str(r.get("dataset", "")),
                "parent_image_prev": r.get("parent_image_prev"),
                "parent_image_curr": r.get("parent_image_curr"),
            }
            if loaded is None:
                n_missing_img += 1
                continue
            prior, current, prev_p, curr_p = loaded
            row["prior_image"] = prior
            row["current_image"] = current
            row["prior_path"] = prev_p
            row["current_path"] = curr_p
            row["image_source"] = "silver-cxr"
            rows.append(row)
            if len(rows) >= n:
                break
        _print(
            f"[smoke] silver parquet={path}  scanned={n_tried}  "
            f"missing/unreadable images={n_missing_img}  "
            f"loaded CXRs={len(rows)}"
        )
        if len(rows) < n:
            _print(
                f"[smoke] only {len(rows)} pairs with real CXRs; "
                f"padding {n - len(rows)} with random tensors"
            )
            for f, c in _FALLBACK_SILVER:
                if len(rows) >= n:
                    break
                rows.append({"finding": f, "cls": c, "source": "fallback"})
        return rows[:n]
    except Exception as exc:
        _print(f"[smoke] failed to read silver ({exc}); using fallback")
        return [
            {"finding": f, "cls": c, "source": "fallback"}
            for f, c in _FALLBACK_SILVER[:n]
        ]


class _StubImage(nn.Module):
    embed_dim = 128

    def forward(self, curr_imgs, prev_imgs=None):
        b = curr_imgs.shape[0]
        g = F.normalize(torch.randn(b, 128), dim=-1)
        p = F.normalize(torch.randn(b, 196, 128), dim=-1)
        return g, p


class _StubText(nn.Module):
    proj_dim = 128

    def forward_contrastive(self, texts):
        b = len(texts)
        t = 16
        g = F.normalize(torch.randn(b, 128), dim=-1)
        loc = F.normalize(torch.randn(b, t, 128), dim=-1)
        mask = torch.ones(b, t, dtype=torch.bool)
        mask[:, -2:] = False
        return g, loc, mask


def _try_build_model():
    _print("[smoke] building VLJEPA(smoke=True) — tiny Llama, no image ckpt fetch")
    model = VLJEPA(
        smoke=True,
        n_llama_layers=2,
        freeze_image_encoder=True,
        freeze_text_encoder=False,
        gradient_checkpointing=False,
    )
    model.eval()
    return model, "real-biovilt+smoke-llama"


def _build_stub_model():
    """Last-resort: skip BioViL-T entirely, keep Llama predictor."""
    from .model import LlamaPredictor

    class StubVLJEPA(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = _StubImage()
            self.text_encoder = _StubText()
            self.predictor = LlamaPredictor(
                vis_dim=128, out_dim=128,
                n_layers=2, smoke=True, gradient_checkpointing=False,
            )
            self.freeze_image_encoder = True
            self.freeze_text_encoder = False

        def encode_images(self, prior, current):
            return self.image_encoder(current, prior)

        def encode_targets(self, texts):
            return self.text_encoder.forward_contrastive(texts)

        def forward(self, prior, current, query_texts, target_texts=None, return_aux=False):
            img_g, img_p = self.encode_images(prior, current)
            q_h, q_mask, q_ids = self.predictor.embed_query(query_texts, prior.device)
            vis = torch.cat([img_g.unsqueeze(1), img_p], dim=1)
            pred = self.predictor(vis, query_texts, return_aux=return_aux)
            aux = None
            if return_aux:
                pred, aux = pred
            out = {
                "pred": pred,
                "img_global": img_g,
                "img_patches": img_p,
                "query_llama": q_h,
                "query_mask": q_mask,
                "query_ids": q_ids,
            }
            if target_texts is not None:
                t_g, t_loc, t_mask = self.encode_targets(target_texts)
                out["target_global"] = t_g
                out["target_local"] = t_loc
                out["target_mask"] = t_mask
            if aux is not None:
                out["aux"] = aux
            return out

    return StubVLJEPA()


def main() -> int:
    torch.manual_seed(0)
    device = torch.device("cpu")
    _print("=" * 72)
    _print("VL-JEPA CPU smoke test (option 2)")
    _print(f"  query  template: {QUERY_TEMPLATE}")
    _print(f"  target template: {TARGET_TEMPLATE}")
    _print(f"  classes: {CLS_ORDER}")
    _print(f"  device: {device}")
    _print("=" * 72)

    rows = _load_silver_rows(N_SMOKE)
    examples = []
    for i, row in enumerate(rows):
        finding = row["finding"]
        cls = row["cls"]
        cls_idx = CLS_ORDER.index(cls)
        prior = row.get("prior_image")
        current = row.get("current_image")
        if prior is None or current is None:
            prior = torch.randn(3, IMAGE_HW, IMAGE_HW)
            current = torch.randn(3, IMAGE_HW, IMAGE_HW)
            image_source = "randn"
        else:
            image_source = row.get("image_source", "silver-cxr")
        examples.append(
            {
                "idx": i,
                "finding": finding,
                "cls": cls,
                "cls_idx": cls_idx,
                "source": row.get("source", "?"),
                "image_source": image_source,
                "prior_path": row.get("prior_path", ""),
                "current_path": row.get("current_path", ""),
                "query_text": query_text(finding),
                "target_texts": class_target_texts(finding),
                "prior_image": prior,
                "current_image": current,
            }
        )

    try:
        model, backend = _try_build_model()
    except Exception as exc:
        _print(f"[smoke] real encoders failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        _print("[smoke] falling back to stub image/text encoders (shapes only)")
        model = _build_stub_model()
        backend = "stub-encoders+smoke-llama"

    model = model.to(device)
    _print(f"[smoke] backend = {backend}")
    if hasattr(model, "predictor"):
        _print(
            f"[smoke] predictor init={model.predictor.init_source} "
            f"hidden={model.predictor.hidden_size} "
            f"layers={model.predictor.n_layers} "
            f"tok={model.predictor.tokenizer_source}"
        )

    # ---- per-example forward (batch size 1) so shapes are readable ----
    all_ok = True
    for ex in examples:
        _print("")
        _print("-" * 72)
        _print(f"EXAMPLE {ex['idx']}  source={ex['source']}")
        _print(f"  finding     : {ex['finding']}")
        _print(f"  gold class  : {ex['cls']}  (idx={ex['cls_idx']})")
        _print(f"  query       : {ex['query_text']}")
        _print(f"  image_source: {ex['image_source']}")
        if ex.get("prior_path"):
            _print(f"  prior_path  : {ex['prior_path']}")
            _print(f"  current_path: {ex['current_path']}")
        _print("  targets     :")
        for c_i, t in enumerate(ex["target_texts"]):
            mark = "  <-- POS" if c_i == ex["cls_idx"] else "  (neg)"
            _print(f"    [{c_i}] {t}{mark}")
        _print(f"  prior_image : {_shape(ex['prior_image'])}")
        _print(f"  current_image: {_shape(ex['current_image'])}")

        prior = ex["prior_image"].unsqueeze(0)
        current = ex["current_image"].unsqueeze(0)
        try:
            out = model(
                prior,
                current,
                [ex["query_text"]],
                target_texts=ex["target_texts"],
                return_aux=True,
            )
        except Exception as exc:
            all_ok = False
            _print(f"  FORWARD FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue

        _print("  -- encoder / predictor shapes --")
        for key in (
            "img_global",
            "img_patches",
            "query_ids",
            "query_llama",
            "query_mask",
            "pred",
            "target_global",
            "target_local",
            "target_mask",
        ):
            if key in out:
                _print(f"  {key:16s} {_shape(out[key])}")
        if "aux" in out:
            for k, v in out["aux"].items():
                _print(f"  aux.{k:16s} {v}")

        labels = torch.tensor([ex["cls_idx"]], dtype=torch.long)
        loss, logits = class_infonce_loss(
            out["pred"], out["target_global"], labels, temperature=0.07,
        )
        _print(f"  logits          {_shape(logits)}  values={logits.detach().tolist()}")
        _print(f"  loss            {_shape(loss)}  value={float(loss.item()):.6f}")
        _print(f"  pred_class      {int(logits.argmax(dim=-1).item())}  "
              f"(gold={ex['cls_idx']})")

        # one backward on the last example to prove grads flow
        if ex["idx"] == len(examples) - 1:
            model.train()
            out_b = model(
                prior, current, [ex["query_text"]],
                target_texts=ex["target_texts"],
            )
            loss_b, _ = class_infonce_loss(
                out_b["pred"], out_b["target_global"], labels, temperature=0.07,
            )
            loss_b.backward()
            n_grad = sum(
                p.grad.numel()
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            )
            _print(f"  backward        ok  grad_elems={n_grad:,}")
            model.eval()

    # ---- batched forward of all 5 ----
    _print("")
    _print("-" * 72)
    _print("BATCHED FORWARD (all 5 examples)")
    prior = torch.stack([e["prior_image"] for e in examples])
    current = torch.stack([e["current_image"] for e in examples])
    queries = [e["query_text"] for e in examples]
    targets = flatten_target_texts([e["target_texts"] for e in examples])
    labels = torch.tensor([e["cls_idx"] for e in examples], dtype=torch.long)
    _print(f"  prior_image     {_shape(prior)}")
    _print(f"  current_image   {_shape(current)}")
    _print(f"  query_text      list[{len(queries)}]")
    _print(f"  target_texts    list[{len(targets)}]  (B*C={len(examples)*len(CLS_ORDER)})")
    _print(f"  labels          {_shape(labels)}  {labels.tolist()}")
    try:
        out = model(prior, current, queries, target_texts=targets, return_aux=True)
        loss, logits = class_infonce_loss(
            out["pred"], out["target_global"], labels, temperature=0.07,
        )
        _print(f"  pred            {_shape(out['pred'])}")
        _print(f"  target_global   {_shape(out['target_global'])}")
        _print(f"  logits          {_shape(logits)}")
        _print(f"  loss            {float(loss.item()):.6f}")
        _print(f"  pred_classes    {logits.argmax(dim=-1).tolist()}")
        if "aux" in out:
            for k, v in out["aux"].items():
                _print(f"  aux.{k:16s} {v}")
    except Exception as exc:
        all_ok = False
        _print(f"  BATCH FORWARD FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    _print("")
    _print("=" * 72)
    if all_ok:
        _print("SMOKE TEST PASSED")
        return 0
    _print("SMOKE TEST FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
