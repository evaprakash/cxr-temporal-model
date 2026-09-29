#!/usr/bin/env python3
"""Cluster smoke: new m000081 layout + real Llama load/run.

Checks all_data, hi-ml, CheXTemporal, gold images, and Llama weights,
then tokenizes a query and runs embed_tokens on CPU. No smoke fallback.

    cd /scratch/m000081/eprakash/temporal/final/cxr-temporal-model
    python -m vljepa.smoke_cluster
"""

from __future__ import annotations

import os
import sys
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

import torch

from .cluster_paths import (
    GOLD_DATASETS,
    chextemporal_dir,
    gold_image_dirs,
    hf_home,
    himl_src,
    image_roots,
    image_roots_dir,
    inventory,
    llama_dir,
    project_dir,
    repo_root,
    scratch_base,
    silver_findings_parquet,
)
from .model import load_llama_model, resolve_llama_local
from .prompts import QUERY_TEMPLATE


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _check_paths() -> bool:
    _print("=" * 72)
    _print("VL-JEPA cluster path check (cycle-6 /scratch/m000081)")
    _print(f"  SCRATCH_BASE = {scratch_base()}")
    _print(f"  PROJECT_DIR  = {project_dir()}")
    _print(f"  repo_root    = {repo_root()}")
    _print(f"  CheXTemporal = {chextemporal_dir()}")
    _print(f"  all_data     = {image_roots_dir()}")
    _print(f"  hi-ml        = {himl_src()}")
    _print(f"  VLJEPA_HF    = {hf_home()}")
    _print(f"  Llama dir    = {llama_dir()}")
    _print(f"  gold dirs    = {gold_image_dirs() or '<none>'}")
    _print("=" * 72)

    ok = True
    for name, path, good, detail in inventory():
        mark = "OK  " if good else "FAIL"
        extra = f"  ({detail})" if detail else ""
        _print(f"  [{mark}] {name:32s} {path}{extra}")
        if not good:
            ok = False
    if not ok:
        _print("")
        _print("PATH CHECK FAILED — fix missing dirs/files before sbatch.")
        _print("Llama should be at $SCRATCH_BASE/hf/Llama-3.2-1B (mv from pm06 if needed).")
    else:
        _print("")
        _print("PATH CHECK PASSED")
    return ok


def _check_himl_import() -> bool:
    src = himl_src()
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import health_multimodal  # noqa: F401

        _print(f"[himl] import health_multimodal OK  ({health_multimodal.__file__})")
        return True
    except Exception as exc:
        _print(f"[himl] import FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def _check_silver_row() -> bool:
    path = silver_findings_parquet()
    roots = image_roots()
    try:
        import pandas as pd

        from dataset_combined_jepa import _resolve_image_path

        df = pd.read_parquet(path)
        n_ok = 0
        n_try = 0
        for _, r in df.iterrows():
            if n_try >= 200:
                break
            ds = str(r.get("dataset", "")).strip()
            prev = r.get("parent_image_prev")
            curr = r.get("parent_image_curr")
            if ds not in roots or prev is None or curr is None:
                continue
            n_try += 1
            try:
                prev_p = _resolve_image_path(ds, str(prev), roots)
                curr_p = _resolve_image_path(ds, str(curr), roots)
            except Exception:
                continue
            if prev_p.is_file() and curr_p.is_file():
                n_ok += 1
                _print(f"[silver] readable pair  dataset={ds}")
                _print(f"         prior  = {prev_p}")
                _print(f"         current= {curr_p}")
                return True
        _print(f"[silver] no readable CXR pair in first {n_try} candidate rows")
        return False
    except Exception as exc:
        _print(f"[silver] FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def _check_gold_row() -> bool:
    try:
        from progression_classify import (
            DEFAULT_GOLD_PARQUET,
            discover_gold_image_roots,
            load_gold_pairs,
            load_image_tensor,
        )
        from dataset_combined_jepa import DEFAULT_FINDINGS

        gold = load_gold_pairs(DEFAULT_GOLD_PARQUET, DEFAULT_FINDINGS)
        pq_dir = os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))
        roots = {**image_roots(), **discover_gold_image_roots(pq_dir), **gold_image_dirs()}
        _print(f"[gold] parquet={DEFAULT_GOLD_PARQUET}  rows={len(gold)}")
        for d in GOLD_DATASETS:
            _print(f"[gold] root {d} = {roots.get(d, '<missing>')}")
        for _, r in gold.iterrows():
            ds = str(r.get("dataset", "")).strip()
            if ds not in roots:
                continue
            try:
                t = load_image_tensor(ds, r["parent_image_curr"], roots)
            except Exception:
                continue
            _print(
                f"[gold] loaded tensor {tuple(t.shape)}  "
                f"dataset={ds} finding={r.get('finding')}"
            )
            return True
        _print("[gold] no gold image resolved")
        return False
    except Exception as exc:
        _print(f"[gold] FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def _check_llama() -> bool:
    local = resolve_llama_local()
    dest = llama_dir()
    _print(f"[llama] resolve_llama_local = {local}")
    _print(f"[llama] expected dir        = {dest}")
    if not local:
        _print("[llama] FAIL  no on-disk snapshot (would random-init in train)")
        return False
    if os.path.abspath(local) != os.path.abspath(dest) and not dest.startswith(
        scratch_base()
    ):
        _print(f"[llama] WARN  using {local} (not the cycle-6 default)")

    query = QUERY_TEMPLATE.format(finding="edema")
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(local, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        batch = tok(
            [query],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        _print(f"[llama] tokenizer OK  vocab={tok.vocab_size}  query={query!r}")
        _print(f"[llama] input_ids     {tuple(batch['input_ids'].shape)}")
    except Exception as exc:
        _print(f"[llama] tokenizer FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False

    try:
        llama = load_llama_model(local, local_files_only=True, torch_dtype=torch.float32)
        llama.eval()
        emb = llama.get_input_embeddings()
        with torch.no_grad():
            hidden = emb(batch["input_ids"])
        _print(
            f"[llama] weights OK   hidden={llama.config.hidden_size} "
            f"layers={llama.config.num_hidden_layers}"
        )
        _print(f"[llama] embed_tokens {tuple(hidden.shape)}  "
              f"mean={float(hidden.mean()):.6f}")
        if hidden.shape[-1] < 64 or hidden.abs().sum().item() == 0:
            _print("[llama] FAIL  embedding looks empty")
            return False
        _print("[llama] load/run PASSED (CPU embed_tokens)")
        return True
    except Exception as exc:
        _print(f"[llama] weights/run FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def main() -> int:
    os.environ.setdefault("CHEXTEMPORAL_DIR", chextemporal_dir())
    os.environ.setdefault("JEPA_IMAGE_ROOTS_DIR", image_roots_dir())
    os.environ.setdefault("VLJEPA_HF_HOME", hf_home())
    if os.path.isfile(os.path.join(llama_dir(), "config.json")):
        os.environ.setdefault("VLJEPA_LLAMA_LOCAL", llama_dir())

    checks = [
        ("paths", _check_paths),
        ("hi-ml import", _check_himl_import),
        ("silver CXR", _check_silver_row),
        ("gold CXR", _check_gold_row),
        ("Llama load/run", _check_llama),
    ]
    failed = []
    for name, fn in checks:
        _print("")
        _print("-" * 72)
        _print(name)
        try:
            if not fn():
                failed.append(name)
        except Exception as exc:
            failed.append(name)
            _print(f"{name} crashed: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    _print("")
    _print("=" * 72)
    if failed:
        _print("CLUSTER SMOKE FAILED: " + ", ".join(failed))
        return 1
    _print("CLUSTER SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
