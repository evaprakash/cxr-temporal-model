"""Phrase-bank scorers for zero-shot baselines on CheXTemporal gold.

Each baseline lives under ``baselines_ext/`` in this repo (copied from
the sibling baselines checkout, isolated so nothing overwrites JEPA
``tempcxr/`` or ``infer_jepa.py``). This module adds that folder to
``sys.path`` and reuses ``infer_*.make_classify_fn``.

Scoring matches BioViL-T gold set-match:

  1. Build the 37-phrase bank ``"{finding} is {phrase}"``.
  2. ``classify_fn(prev_path, curr_path, prompts)`` → phrase cosines.
  3. Per-class score = max phrase cosine in that class.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from eval_progression_biovilt import PROMPT_TEMPLATE, build_phrase_bank
from progression_classify import _resolve_with_fallbacks
from progression_phrases import CLS_ORDER

_HERE = Path(__file__).resolve().parent
DEFAULT_BASELINES_ROOT = os.environ.get(
    "BASELINES_ROOT",
    str(_HERE / "baselines_ext"),
)

PATH_BACKENDS = ("tila", "biovil", "biomedclip", "alta", "tempavlp")


def add_baselines_root(baselines_root: str) -> str:
    root = str(Path(baselines_root).expanduser().resolve())
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"baselines root not found: {root}. Pass --baselines-root "
            "or set BASELINES_ROOT to the checkout that has infer_tila.py etc."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def phrase_sims_to_class_scores(
    sims: torch.Tensor,
    phrase_to_class: List[str],
) -> List[float]:
    """Max phrase cosine within each class (same as BioViL-T gold eval)."""
    scores: List[float] = []
    for cls in CLS_ORDER:
        idxs = [i for i, c in enumerate(phrase_to_class) if c == cls]
        if not idxs:
            scores.append(float("-inf"))
        else:
            scores.append(float(sims[idxs].max().item()))
    return scores


def resolve_pair_paths(row, image_roots: Dict[str, str]) -> Tuple[str, str]:
    prev = _resolve_with_fallbacks(
        str(row["dataset"]), str(row["parent_image_prev"]), image_roots,
    )
    curr = _resolve_with_fallbacks(
        str(row["dataset"]), str(row["parent_image_curr"]), image_roots,
    )
    return str(prev), str(curr)


def make_phrasebank_score_fn(
    classify_fn: Callable,
    device: torch.device,
    template: str = PROMPT_TEMPLATE,
) -> Callable:
    """Wrap a path-based ``classify_fn`` into the set-match score_fn."""

    def score_fn(row, text_cache: Optional[Dict[str, Tuple]], image_roots):
        finding = str(row["finding"])
        cache_key = f"{finding}||{template}"
        if text_cache is not None and cache_key in text_cache:
            prompts, phrase_to_class = text_cache[cache_key]
        else:
            prompts, phrase_to_class = build_phrase_bank(finding, template)
            if text_cache is not None:
                text_cache[cache_key] = (prompts, phrase_to_class)

        prev_path, curr_path = resolve_pair_paths(row, image_roots)
        sims = classify_fn(prev_path, curr_path, prompts)
        if sims.ndim != 1 or sims.numel() != len(prompts):
            raise RuntimeError(
                f"classify_fn returned shape {tuple(sims.shape)}, "
                f"expected ({len(prompts)},)"
            )
        scores = phrase_sims_to_class_scores(sims, phrase_to_class)
        pred_class = max(range(len(scores)), key=lambda k: scores[k])
        return {
            "cos_class_scores": scores,
            "pred_class": pred_class,
            "prompts": prompts,
        }

    return score_fn


def phrasebank_class_scores(
    classify_fn: Callable,
    prev_path: str,
    curr_path: str,
    finding: str,
    text_cache: Dict[str, Tuple],
    template: str = PROMPT_TEMPLATE,
) -> List[float]:
    """Five class scores (CLS_ORDER) for one pair given absolute image paths."""
    key = f"{finding}||{template}"
    if key not in text_cache:
        text_cache[key] = build_phrase_bank(finding, template)
    prompts, phrase_to_class = text_cache[key]
    return phrase_sims_to_class_scores(classify_fn(prev_path, curr_path, prompts), phrase_to_class)


def load_path_backend(
    backend: str,
    device: torch.device,
    baselines_root: str,
    ckpt: Optional[str] = None,
    image_mode: str = "diff",
    dtype_arg: str = "auto",
):
    """Return ``(score_fn_factory, note)``.

    ``score_fn_factory(image_roots)`` → ``score_fn(row, text_cache)``.
    """
    classify, note = load_classify_fn(backend, device, baselines_root, ckpt, image_mode, dtype_arg)
    wrapped = make_phrasebank_score_fn(classify, device)

    def score_fn(row, text_cache, image_roots=None):
        if image_roots is None:
            raise RuntimeError("image_roots required for path backends")
        return wrapped(row, text_cache, image_roots)

    return score_fn, note


def load_classify_fn(
    backend: str,
    device: torch.device,
    baselines_root: str,
    ckpt: Optional[str] = None,
    image_mode: str = "diff",
    dtype_arg: str = "auto",
):
    """Return ``(classify_fn(prev_path, curr_path, prompts) -> sims, note)``."""
    add_baselines_root(baselines_root)

    if backend == "tila":
        from infer_tila import (  # type: ignore
            TILA_REPO,
            load_model,
            make_classify_fn,
            resolve_dtype,
        )
        dtype = resolve_dtype(dtype_arg, device)
        model = load_model(device, dtype, checkpoint=ckpt or TILA_REPO)
        classify = make_classify_fn(model, device, dtype)
        note = f", ckpt={ckpt or TILA_REPO}"

    elif backend == "biovil":
        from infer_biovil import (  # type: ignore
            DEFAULT_BIOVIL_IMAGE_CKPT,
            DEFAULT_CXR_BERT_PATH,
            load_model,
            make_classify_fn,
        )
        image_ckpt = ckpt or DEFAULT_BIOVIL_IMAGE_CKPT
        image_model, tokenizer, text_model = load_model(
            device, image_ckpt=image_ckpt, cxr_bert_path=DEFAULT_CXR_BERT_PATH,
        )
        classify = make_classify_fn(
            image_model, tokenizer, text_model, device, image_mode=image_mode,
        )
        note = f", image_mode={image_mode}, ckpt={image_ckpt}"

    elif backend == "biomedclip":
        from infer_biomedclip import (  # type: ignore
            DEFAULT_MODEL_ID,
            load_model,
            make_classify_fn,
        )
        model_id = ckpt or DEFAULT_MODEL_ID
        model, tokenizer = load_model(device, model_id=model_id)
        classify = make_classify_fn(
            model, tokenizer, device, image_mode=image_mode,
        )
        note = f", image_mode={image_mode}, model={model_id}"

    elif backend == "alta":
        from infer_alta import (  # type: ignore
            DEFAULT_CHECKPOINT_PATH,
            load_model,
            make_classify_fn,
        )
        alta_ckpt = ckpt or DEFAULT_CHECKPOINT_PATH
        model = load_model(device, checkpoint_path=alta_ckpt)
        classify = make_classify_fn(model, device)
        note = f", ckpt={alta_ckpt}"

    elif backend == "tempavlp":
        from infer_tempavlp import (  # type: ignore
            DEFAULT_CHECKPOINT,
            load_model,
            make_classify_fn,
            resolve_dtype,
        )
        dtype = resolve_dtype(dtype_arg, device)
        tav_ckpt = ckpt or DEFAULT_CHECKPOINT
        model = load_model(device, dtype, checkpoint=tav_ckpt)
        classify = make_classify_fn(model, device, dtype)
        note = f", ckpt={tav_ckpt}"

    else:
        raise ValueError(f"unknown path backend {backend!r}")

    return classify, note
