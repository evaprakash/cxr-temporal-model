#!/usr/bin/env python3
"""Llama-only write + load probe. No BioViL-T, no data, no GPU required.

    # load the snapshot you already have (pm06 / VLJEPA_LLAMA_LOCAL)
    python -m vljepa.probe_llama

    # can this dest create files? (tiny touch, no 2.5G download)
    python -m vljepa.probe_llama --dest /scratch/m000081/eprakash/hf/Llama-3.2-1B --write-only

    # copy existing snapshot then load
    python -m vljepa.probe_llama \\
        --dest /scratch/m000081/eprakash/hf/Llama-3.2-1B \\
        --copy-from /scratch/m000081-pm06/eprakash/hf/Llama-3.2-1B

    # hub download into dest (uses dest/../hub as cache)
    python -m vljepa.probe_llama --dest /scratch/m000081/eprakash/hf/Llama-3.2-1B --download
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

from .model import (
    LLAMA_NAME_DEFAULT,
    default_llama_dir,
    is_llama_dir,
    load_llama_model,
    resolve_llama_local,
)


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _try_write(dirpath: str) -> bool:
    os.makedirs(dirpath, exist_ok=True)
    probe = os.path.join(dirpath, ".vljepa_write_probe")
    try:
        with open(probe, "w") as f:
            f.write("ok\n")
        os.remove(probe)
        _print(f"WRITE OK   {dirpath}")
        return True
    except OSError as exc:
        _print(f"WRITE FAIL {dirpath}: {type(exc).__name__}: {exc}")
        return False


def _copy_tree(src: str, dest: str) -> None:
    _print(f"copy {src} → {dest}")
    os.makedirs(dest, exist_ok=True)
    shutil.copytree(src, dest, dirs_exist_ok=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dest",
        default=None,
        help="Directory to write/copy/download into (default: existing snapshot or pm06).",
    )
    p.add_argument(
        "--copy-from",
        default=None,
        help="Existing Llama-3.2-1B folder to rsync-style copy into --dest.",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="snapshot_download from the hub into --dest (a few GB).",
    )
    p.add_argument(
        "--write-only",
        action="store_true",
        help="Only test creating a file under --dest; do not load weights.",
    )
    p.add_argument(
        "--name",
        default=os.environ.get("VLJEPA_LLAMA_NAME", LLAMA_NAME_DEFAULT),
    )
    args = p.parse_args()

    existing = resolve_llama_local()
    dest = args.dest or existing or default_llama_dir()
    cache = os.path.join(os.path.dirname(os.path.abspath(dest)), "hub")
    os.environ["VLJEPA_HF_HOME"] = os.path.dirname(os.path.abspath(dest))
    os.environ["HF_HUB_CACHE"] = cache

    _print(f"name       = {args.name}")
    _print(f"dest       = {dest}")
    _print(f"cache/hub  = {cache}")
    _print(f"existing   = {existing}")
    _print()

    if not _try_write(dest):
        return 2
    if not _try_write(cache):
        return 2
    if args.write_only:
        return 0

    if args.copy_from:
        if not is_llama_dir(args.copy_from):
            _print(f"COPY FAIL  no config.json under {args.copy_from}")
            return 1
        try:
            _copy_tree(args.copy_from, dest)
        except OSError as exc:
            _print(f"COPY FAIL  {type(exc).__name__}: {exc}")
            return 2

    if args.download:
        from huggingface_hub import snapshot_download

        try:
            path = snapshot_download(
                args.name,
                local_dir=dest,
                cache_dir=cache,
                token=os.environ.get("HF_TOKEN") or True,
            )
            _print(f"DOWNLOAD OK {path}")
        except OSError as exc:
            _print(f"DOWNLOAD FAIL {type(exc).__name__}: {exc}")
            return 2
        except Exception as exc:
            _print(f"DOWNLOAD FAIL {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 1

    src = dest if is_llama_dir(dest) else (existing or dest)
    local_only = is_llama_dir(src)
    _print(f"load src   = {src}  local_files_only={local_only}")
    try:
        llama = load_llama_model(src, local_files_only=local_only)
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
