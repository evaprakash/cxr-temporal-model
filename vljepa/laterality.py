"""Laterality 5-way VL-JEPA: the side goes in the target, not the query.

Same as the 5-way run (query ``What is the progression of {finding}?``,
class-weighted InfoNCE over target sentences), except:

  * targets come from the frozen BioViL-T text encoder, with the 5-way run's
    text LoRA by default or as released (``--text-encoder original``);
  * a finding with an agreed silver side (MedGemma and Qwen3-32B-AWQ, see
    build_laterality.py) is scored against all 15 sentences
    ``{Side} {finding} is {class}.`` for left / right / bilateral x the five
    classes, so the side has to come from the images;
  * findings without a side keep the five ``{Finding} is {class}.`` sentences.

Gold and other benchmarks have no side, so a class score is the soft maximum
(tau * logsumexp) over the plain and the three sided sentences of that class.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from . import _root  # noqa: F401

from dataset_combined_jepa import DEFAULT_FINDINGS
from progression_phrases import CLS_ORDER

from .dataset import VLJEPAFindingDataset
from .model import VLJEPA
from .prompts import class_target_texts, query_text

SIDES = ("left", "right", "bilateral")
NO_SIDE_FINDINGS = {"cardiomegaly", "enlarged cardiomediastinum"}
N_CLS = len(CLS_ORDER)
TEMPERATURE = 0.07
KEY = ["dataset", "patient_id", "study_id_curr", "study_id_prev", "finding"]
LATERALITY_PATH = os.environ.get(
    "VLJEPA_LATERALITY",
    os.path.join(os.path.dirname(DEFAULT_FINDINGS), "silver_laterality.parquet"),
)


def sided_target_text(finding: str, cls: str, side: str) -> str:
    return f"{side.capitalize()} {str(finding).strip().lower()} is {cls}."


def candidate_texts(finding: str, side: Optional[str]) -> List[str]:
    """Plain five sentences, or the 15 sided ones in side-major order."""
    if side is None:
        return class_target_texts(finding)
    return [sided_target_text(finding, c, s) for s in SIDES for c in CLS_ORDER]


def label_index(cls_idx: int, side: Optional[str]) -> int:
    return cls_idx if side is None else SIDES.index(side) * N_CLS + cls_idx


def scoring_variants(finding: str) -> List[List[str]]:
    """Per variant, the five class sentences: plain first, then each side."""
    plain = class_target_texts(finding)
    if str(finding).strip().lower() in NO_SIDE_FINDINGS:
        return [plain]
    return [plain] + [[sided_target_text(finding, c, s) for c in CLS_ORDER] for s in SIDES]


def load_side_map(path: str) -> Dict[tuple, str]:
    import pandas as pd

    df = pd.read_parquet(path, columns=KEY + ["side"])
    keys = zip(*(df[c].astype(str).str.strip() for c in KEY[:-1]),
               df["finding"].astype(str).str.strip().str.lower())
    return {k: s for k, s in zip(keys, df["side"]) if s in SIDES}


class LateralityFindingDataset(VLJEPAFindingDataset):
    """VLJEPAFindingDataset plus each finding's agreed side (or None)."""

    def __init__(self, image_roots, laterality_path: Optional[str] = None, **kwargs):
        super().__init__(image_roots, **kwargs)
        path = laterality_path or LATERALITY_PATH
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found; build it with python -m vljepa.build_laterality "
                "or set VLJEPA_LATERALITY")
        side_map = load_side_map(path)
        df = self.pairs.df
        ids = list(zip(*(df[c].astype(str).str.strip() for c in KEY[:-1])))
        self.sides = [side_map.get(ids[p] + (f,)) for p, f, _ in self.index]
        counts = {s: self.sides.count(s) for s in SIDES}
        n = max(len(self.sides), 1)
        print(f"[laterality dataset] {sum(counts.values()):,} of {len(self.sides):,} finding-examples "
              f"sided ({sum(counts.values()) / n:.1%}): "
              + ", ".join(f"{s} {c:,}" for s, c in counts.items()) + f"  <- {path}")

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        side = self.sides[idx]
        item["side"] = side
        item["target_texts"] = candidate_texts(item["finding"], side)
        item["label"] = label_index(item["cls_idx"], side)
        return item


def laterality_collate_fn(batch: List[dict]) -> dict:
    return {
        "prior_image": torch.stack([b["prior_image"] for b in batch]),
        "current_image": torch.stack([b["current_image"] for b in batch]),
        "finding": [b["finding"] for b in batch],
        "cls_idx": torch.tensor([b["cls_idx"] for b in batch], dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "side": [b["side"] for b in batch],
        "query_text": [b["query_text"] for b in batch],
        "target_texts": [b["target_texts"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }


class TargetBank:
    """Embeddings of target sentences, encoded once. Needs a frozen text encoder."""

    def __init__(self, model: VLJEPA):
        if not model.freeze_text_encoder:
            raise ValueError("TargetBank caches embeddings; the text encoder must be frozen")
        self.model = model
        self.cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def __call__(self, texts: List[str], device: torch.device) -> torch.Tensor:
        new = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if new:
            g, _, _ = self.model.encode_targets(new)
            for t, e in zip(new, F.normalize(g.float(), dim=-1)):
                self.cache[t] = e.to(device)
        return torch.stack([self.cache[t] for t in texts])


def laterality_infonce_loss(pred, target_texts, labels, cls_idx, bank, temperature=TEMPERATURE,
                            class_weights=None):
    """InfoNCE over each row's own candidates (5 or 15), weighted by the
    progression class like F.cross_entropy(weight=...). Padded logits are -inf."""
    device = pred.device
    sizes = [len(t) for t in target_texts]
    flat = bank([t for row in target_texts for t in row], device)
    k = max(sizes)
    tgt = flat.new_zeros(len(sizes), k, flat.shape[-1])
    valid = torch.zeros(len(sizes), k, dtype=torch.bool, device=device)
    start = 0
    for b, n in enumerate(sizes):
        tgt[b, :n] = flat[start:start + n]
        valid[b, :n] = True
        start += n
    pred_n = F.normalize(pred.float(), dim=-1)
    logits = torch.einsum("bd,bkd->bk", pred_n, tgt) / float(temperature)
    logits = logits.masked_fill(~valid, float("-inf"))
    labels = labels.to(device)
    per_row = F.cross_entropy(logits, labels, reduction="none")
    if class_weights is None:
        return per_row.mean(), logits
    w = class_weights.to(device)[cls_idx.to(device)]
    return (per_row * w).sum() / w.sum(), logits


@torch.no_grad()
def readouts(logits: torch.Tensor, sides: List[Optional[str]]):
    """Predicted class per row (sides summed out for sided rows) and, for
    sided rows, predicted side index (classes summed out); -1 elsewhere."""
    cls_pred, side_pred = [], []
    for row, side in zip(logits, sides):
        if side is None:
            cls_pred.append(int(row[:N_CLS].argmax()))
            side_pred.append(-1)
        else:
            grid = row[:len(SIDES) * N_CLS].view(len(SIDES), N_CLS)
            cls_pred.append(int(torch.logsumexp(grid, dim=0).argmax()))
            side_pred.append(int(torch.logsumexp(grid, dim=1).argmax()))
    return torch.tensor(cls_pred), torch.tensor(side_pred)


class LateralityVLJEPA(VLJEPA):
    """VLJEPA whose ``score_classes`` sums the side out, so eval_gold_setmatch
    and other 5-class scorers work unchanged."""

    temperature = TEMPERATURE

    def _texts(self, texts: List[str], cache: Dict[str, torch.Tensor]) -> torch.Tensor:
        new = [t for t in texts if t not in cache]
        if new:
            g, _, _ = self.encode_targets(new)
            for t, e in zip(new, g):
                cache[t] = e.detach()
        return F.normalize(torch.stack([cache[t] for t in texts]).float(), dim=-1)

    @torch.no_grad()
    def variant_cosines(self, prior, current, finding, text_cache=None) -> torch.Tensor:
        """``(V, 5)`` cosines: plain, then left / right / bilateral where the finding has sides."""
        cache = text_cache if text_cache is not None else {}
        variants = scoring_variants(finding)
        img_g, img_p = self.encode_images(prior, current)
        pred = F.normalize(self.predict(img_g, img_p, [query_text(finding)]).float(), dim=-1)
        tgt = self._texts([t for v in variants for t in v], cache)
        return (tgt @ pred.squeeze(0)).view(len(variants), N_CLS)

    @torch.no_grad()
    def score_classes(self, prior, current, finding, text_cache=None) -> torch.Tensor:
        cos = self.variant_cosines(prior, current, finding, text_cache)
        return self.temperature * torch.logsumexp(cos / self.temperature, dim=0)


def load_laterality_from_ckpt(ckpt_path: str, device: torch.device,
                              shift_json: Optional[str] = None) -> LateralityVLJEPA:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("vljepa_cfg", {})
    if cfg.get("objective") != "laterality":
        raise SystemExit(f"[laterality] {ckpt_path} is not a laterality checkpoint "
                         f"(objective {cfg.get('objective')})")
    model = LateralityVLJEPA(
        image_mode="biovilt_no_pretrained",
        smoke=False,
        n_llama_layers=int(cfg.get("n_llama_layers", 8)),
        llama_name=cfg.get("llama_name", "meta-llama/Llama-3.2-1B"),
        llama_local=cfg.get("llama_local"),
        freeze_image_encoder=True,
        freeze_text_encoder=True,
        gradient_checkpointing=False,
        text_lora=bool(cfg.get("text_lora")),
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[laterality] missing {len(missing)} / unexpected {len(unexpected)} keys")
    model.class_shift = None
    if shift_json:
        import json

        with open(shift_json) as f:
            info = json.load(f)
        if info.get("class_order") != list(CLS_ORDER):
            raise ValueError(f"{shift_json}: class order {info.get('class_order')}")
        model.class_shift = torch.tensor(info["class_shift"], dtype=torch.float32)
        print(f"[laterality] class shift {info['class_shift']} from {shift_json}")
    print(f"[laterality] loaded {ckpt_path} (epoch {ckpt.get('epoch')})")
    return model.to(device).eval()


def main() -> int:
    import argparse

    from progression_classify import DEFAULT_GOLD_PARQUET, discover_gold_image_roots

    from .cluster_paths import gold_image_dirs, image_roots
    from .eval_gold import eval_gold_setmatch, load_gold_groups

    p = argparse.ArgumentParser(description="Gold set-match for a laterality checkpoint")
    p.add_argument("command", choices=["gold"])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--shift-json", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_laterality_from_ckpt(args.ckpt, device, args.shift_json)
    groups = load_gold_groups()
    if args.limit:
        groups = groups.iloc[: args.limit].reset_index(drop=True)
    roots = {**image_roots(),
             **discover_gold_image_roots(os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))),
             **gold_image_dirs()}
    eval_gold_setmatch(model, groups, roots, epoch=-1, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
