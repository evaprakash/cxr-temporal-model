#!/usr/bin/env python3
"""Snapshot Llama-3.2-1B onto pm06 scratch (not the quota-full /scratch/m000081 cache).

    python -m vljepa.download_llama
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

from .model import (
    LLAMA_NAME_DEFAULT,
    default_llama_dir,
    llama_hub_cache,
)


def main() -> int:
    dest = os.environ.get("VLJEPA_LLAMA_LOCAL") or default_llama_dir()
    name = os.environ.get("VLJEPA_LLAMA_NAME", LLAMA_NAME_DEFAULT)
    cache = llama_hub_cache()
    os.makedirs(dest, exist_ok=True)
    print("repo       =", name)
    print("local_dir  =", dest)
    print("cache_dir  =", cache)

    from huggingface_hub import snapshot_download

    path = snapshot_download(
        name,
        local_dir=dest,
        cache_dir=cache,
        token=os.environ.get("HF_TOKEN") or True,
    )
    print("downloaded =", path)
    cfg = os.path.join(dest, "config.json")
    if not os.path.isfile(cfg):
        print("WARNING: no config.json under", dest, file=sys.stderr)
        return 1
    print("OK  export VLJEPA_LLAMA_LOCAL=" + dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
