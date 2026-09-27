#!/usr/bin/env python3
"""Llama-only write + load probe on persistent /scratch/m000081 (not pm06).

Copy this file anywhere and run it from the m000081 tree. It does not import
vljepa and does not touch BioViL-T or CXRs.

    cd /scratch/m000081/eprakash/cxr-temporal-model
    python probe_llama_m000081.py --write-only
    python probe_llama_m000081.py --download
    python probe_llama_m000081.py --copy-from /scratch/m000081-pm06/eprakash/hf/Llama-3.2-1B
    python probe_llama_m000081.py --load-only
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import traceback

BASE = os.environ.get("VLJEPA_M000081", "/scratch/m000081/eprakash")
HF_HOME = os.path.join(BASE, "hf")
DEST_DEFAULT = os.path.join(HF_HOME, "Llama-3.2-1B")
CACHE_DEFAULT = os.path.join(HF_HOME, "hub")
NAME_DEFAULT = os.environ.get("VLJEPA_LLAMA_NAME", "meta-llama/Llama-3.2-1B")


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def try_write(dirpath: str) -> bool:
    os.makedirs(dirpath, exist_ok=True)
    probe = os.path.join(dirpath, ".llama_write_probe")
    try:
        with open(probe, "w") as f:
            f.write("ok\n")
        os.remove(probe)
        _print(f"WRITE OK   {dirpath}")
        return True
    except OSError as exc:
        _print(f"WRITE FAIL {dirpath}: {type(exc).__name__}: {exc}")
        return False


def is_llama_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "config.json"))


def relax_rope(raw: dict) -> dict:
    raw = dict(raw)
    rs = raw.get("rope_scaling")
    if isinstance(rs, dict) and "type" not in rs:
        raw["rope_scaling"] = {"type": "linear", "factor": float(rs.get("factor", 1.0))}
    return raw


def llama_config_from_dir(src: str):
    from transformers import LlamaConfig

    with open(os.path.join(src, "config.json")) as f:
        raw = relax_rope(json.load(f))
    try:
        return LlamaConfig.from_dict(raw)
    except (TypeError, ValueError):
        pass
    allowed = set(inspect.signature(LlamaConfig.__init__).parameters) - {"self"}
    slim = relax_rope({k: v for k, v in raw.items() if k in allowed})
    try:
        return LlamaConfig(**slim)
    except ValueError:
        slim["rope_scaling"] = None
        return LlamaConfig(**slim)


def load_llama(src: str, cache_dir: str, local_only: bool):
    import torch
    from transformers import LlamaModel

    kwargs = {"torch_dtype": torch.float32, "cache_dir": cache_dir}
    if local_only:
        kwargs["local_files_only"] = True
    try:
        return LlamaModel.from_pretrained(src, attn_implementation="eager", **kwargs)
    except TypeError:
        pass
    except Exception as exc:
        _print(f"(first from_pretrained: {type(exc).__name__}: {exc})")

    if not os.path.isdir(src):
        raise FileNotFoundError(src)
    cfg = llama_config_from_dir(src)
    cfg._attn_implementation = "eager"
    try:
        return LlamaModel.from_pretrained(src, config=cfg, **kwargs)
    except Exception as exc:
        _print(f"(config= from_pretrained: {type(exc).__name__}: {exc})")
        from safetensors.torch import load_file

        llama = LlamaModel(cfg)
        tensors = [
            p
            for p in os.listdir(src)
            if p.endswith(".safetensors") and "index" not in p
        ]
        sd = {}
        for name in tensors:
            sd.update(load_file(os.path.join(src, name)))
        if any(k.startswith("model.") for k in sd):
            sd = {
                (k[6:] if k.startswith("model.") else k): v
                for k, v in sd.items()
                if not k.startswith("lm_head")
            }
        llama.load_state_dict(sd, strict=False)
        return llama


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", default=DEST_DEFAULT)
    p.add_argument("--cache", default=CACHE_DEFAULT)
    p.add_argument("--name", default=NAME_DEFAULT)
    p.add_argument("--write-only", action="store_true")
    p.add_argument("--load-only", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--copy-from", default=None)
    args = p.parse_args()

    os.environ["HF_HUB_CACHE"] = args.cache
    os.environ["TRANSFORMERS_CACHE"] = args.cache
    os.environ.setdefault("HF_HOME", HF_HOME)

    _print(f"base  = {BASE}")
    _print(f"dest  = {args.dest}")
    _print(f"cache = {args.cache}")
    _print(f"name  = {args.name}")
    _print()

    if not args.load_only:
        if not try_write(args.dest):
            return 2
        if not try_write(args.cache):
            return 2
        if args.write_only:
            return 0

        if args.copy_from:
            if not is_llama_dir(args.copy_from):
                _print(f"COPY FAIL  no config.json under {args.copy_from}")
                return 1
            _print(f"copy {args.copy_from} → {args.dest}")
            try:
                os.makedirs(args.dest, exist_ok=True)
                shutil.copytree(args.copy_from, args.dest, dirs_exist_ok=True)
            except OSError as exc:
                _print(f"COPY FAIL  {type(exc).__name__}: {exc}")
                return 2

        if args.download:
            from huggingface_hub import snapshot_download

            try:
                path = snapshot_download(
                    args.name,
                    local_dir=args.dest,
                    cache_dir=args.cache,
                    token=os.environ.get("HF_TOKEN") or True,
                )
                _print(f"DOWNLOAD OK {path}")
            except Exception as exc:
                _print(f"DOWNLOAD FAIL {type(exc).__name__}: {exc}")
                traceback.print_exc()
                return 2

    if not is_llama_dir(args.dest):
        _print(f"LOAD SKIP  no snapshot at {args.dest} (use --download or --copy-from)")
        return 1 if args.load_only else 0

    try:
        llama = load_llama(args.dest, args.cache, local_only=True)
    except Exception as exc:
        _print(f"LOAD FAIL  {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    _print(
        f"LOAD OK    hidden={llama.config.hidden_size} "
        f"layers={llama.config.num_hidden_layers}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
