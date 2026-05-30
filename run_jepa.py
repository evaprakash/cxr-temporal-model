#!/usr/bin/env python3
"""Direct launcher for ``resume_train_jepa.py``.

For environments without SLURM (e.g., a single GCP VM). Auto-detects
the visible GPUs and spawns torchrun with the right ``--nproc_per_node``.

Examples
--------

    # auto-detect GPUs
    python run_jepa.py

    # force a specific process count
    python run_jepa.py --nproc 2

    # resume from a checkpoint
    python run_jepa.py --resume checkpoints_jepa/epoch_5.pt

Any flags after ``--`` are forwarded verbatim to
``resume_train_jepa.py``::

    python run_jepa.py -- --resume checkpoints_jepa/best.pt

Override checkpoint and log directories via env vars (also honored
inside ``resume_train_jepa.py``)::

    JEPA_CHECKPOINT_DIR=/data/ckpts JEPA_LOG_DIR=/data/logs \
        python run_jepa.py
"""

import argparse
import os
import subprocess
import sys

import torch


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="number of processes (default: torch.cuda.device_count())",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint path to resume from",
    )
    args, extra = parser.parse_known_args()

    nproc = args.nproc if args.nproc is not None else torch.cuda.device_count()
    if nproc < 1:
        print(
            "No CUDA devices visible. Pass --nproc explicitly to override "
            "(e.g., --nproc 1 for a CPU sanity run).",
            file=sys.stderr,
        )
        return 1

    here = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(here, "resume_train_jepa.py")

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        train_script,
    ]
    if args.resume is not None:
        cmd += ["--resume", args.resume]
    cmd += extra

    print(f"Launching: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
