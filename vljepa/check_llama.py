#!/usr/bin/env python3
"""Show what train will do for Llama tok / weights. No GPU, no data.

    python -m vljepa.check_llama
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"


def main() -> int:
    from .model import (
        default_llama_dir,
        llama_hf_home,
        llama_hub_cache,
        load_llama_model,
        resolve_llama_local,
    )

    local = resolve_llama_local()
    extra = os.environ.get("VLJEPA_TOKENIZER") or None
    name = os.environ.get("VLJEPA_LLAMA_NAME", "meta-llama/Llama-3.2-1B")
    bert = os.environ.get("VLJEPA_TOKENIZER_FALLBACK", "bert-base-uncased")
    print("VLJEPA_LLAMA_LOCAL         =", os.environ.get("VLJEPA_LLAMA_LOCAL"))
    print("resolved local snapshot    =", local)
    print("default local dir          =", default_llama_dir())
    print("VLJEPA_HF_HOME             =", llama_hf_home())
    print("Llama hub cache            =", llama_hub_cache())
    print("VLJEPA_TOKENIZER           =", extra)
    print("VLJEPA_LLAMA_NAME          =", name)
    print("VLJEPA_TOKENIZER_FALLBACK  =", bert)
    print()

    tok_ok = False
    tok_src = None
    from transformers import AutoTokenizer

    for src in (local, extra, name, bert):
        if not src:
            continue
        try:
            kw = {"use_fast": True}
            if not os.path.isdir(src):
                kw["cache_dir"] = llama_hub_cache()
            tok = AutoTokenizer.from_pretrained(src, **kw)
            print(f"TOKENIZER OK  {src}  vocab={tok.vocab_size}")
            print("  sample:", tok("What is the progression of pleural effusion?"))
            tok_ok = True
            tok_src = src
            break
        except Exception as exc:
            print(f"TOKENIZER FAIL {src}: {type(exc).__name__}: {exc}")

    if not tok_ok:
        print("TRAIN WOULD LOG  tok=smoke-charhash-fallback")
    elif tok_src == bert and tok_src not in (local, extra, name):
        print(f"TRAIN WOULD LOG  tok=pretrained:{bert}  (BERT word pieces, not a QA encoder)")
    else:
        print("TRAIN WOULD LOG  tok=pretrained:...")

    print()
    wsrc = local or name
    try:
        llama = load_llama_model(wsrc, local_files_only=bool(local))
        print(
            f"WEIGHTS OK     {wsrc}  "
            f"hidden={llama.config.hidden_size} layers={llama.config.num_hidden_layers}"
        )
        print("TRAIN WOULD LOG  predictor init=pretrained:...:last8")
    except Exception as exc:
        print(f"WEIGHTS FAIL    {wsrc}: {type(exc).__name__}: {exc}")
        print("TRAIN WOULD LOG  predictor init=random:llama3.2-1b:8L")
        print("(layers + word table start random; that is the paper-minus-init run)")
        print("If this was a disk quota error, run: python -m vljepa.download_llama")
    return 0


if __name__ == "__main__":
    sys.exit(main())
