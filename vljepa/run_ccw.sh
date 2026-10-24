#!/bin/bash
# Run this on ccw-gpu-1, inside a shell from srun. /nvme is not on the
# login node, and it is wiped if that machine reboots.
#
#   bash vljepa/run_ccw.sh smoke
#   bash vljepa/run_ccw.sh train
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export SCRATCH_BASE="${SCRATCH_BASE:-/nvme/people/eprakash}"
export PROJECT_DIR="${PROJECT_DIR:-$ROOT}"
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-/nvme/data/temporal/all_data}"
export VLJEPA_HF_HOME="${VLJEPA_HF_HOME:-$SCRATCH_BASE/hf}"
export HF_HOME="$VLJEPA_HF_HOME"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VLJEPA_HF_HOME/hub}"
export VLJEPA_LLAMA_LOCAL="${VLJEPA_LLAMA_LOCAL:-$VLJEPA_HF_HOME/Llama-3.2-1B}"
export VLJEPA_CHECKPOINT_DIR="${VLJEPA_CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints_vljepa_lora}"
export VLJEPA_LOG_DIR="${VLJEPA_LOG_DIR:-$SCRATCH_BASE/logs/vljepa_lora}"
mkdir -p "$VLJEPA_HF_HOME/hub" "$VLJEPA_LOG_DIR" "$VLJEPA_CHECKPOINT_DIR"

HI_ML="$ROOT/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
export PYTHONPATH="${ROOT}:${HI_ML}${PYTHONPATH:+:$PYTHONPATH}"

echo "[ccw] PROJECT_DIR=$PROJECT_DIR"
echo "[ccw] all_data=$JEPA_IMAGE_ROOTS_DIR"
echo "[ccw] hf=$VLJEPA_HF_HOME"
echo "[ccw] ckpt=$VLJEPA_CHECKPOINT_DIR"

MODE="${1:-smoke}"
if [ "$MODE" = "smoke" ]; then
    export CUDA_VISIBLE_DEVICES=""
    python -m vljepa.smoke_ccw_cpu
elif [ "$MODE" = "train" ]; then
    NGPU="${SLURM_GPUS_ON_NODE:-4}"
    torchrun --nproc_per_node="$NGPU" -m vljepa.train
else
    echo "usage: bash vljepa/run_ccw.sh smoke|train" >&2
    exit 1
fi
