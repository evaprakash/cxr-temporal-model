#!/bin/bash
# Run this on ccw-gpu-1, inside a shell from srun. /nvme is not on the
# login node, and it is wiped if that machine reboots.
#
#   bash vljepa/run_ccw.sh smoke
#   bash vljepa/run_ccw.sh train
#   bash vljepa/run_ccw.sh check --ckpt <checkpoint dir>/epoch_N.pt
#   bash vljepa/run_ccw.sh shift --ckpt <checkpoint dir>/epoch_N.pt
#   bash vljepa/run_ccw.sh gold  --ckpt <checkpoint dir>/epoch_N_shifted.pt
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
export VLJEPA_CHECKPOINT_DIR="${VLJEPA_CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints_vljepa_lora_5way}"
export VLJEPA_LOG_DIR="${VLJEPA_LOG_DIR:-$SCRATCH_BASE/logs/vljepa_lora_5way}"
mkdir -p "$VLJEPA_HF_HOME/hub" "$VLJEPA_LOG_DIR" "$VLJEPA_CHECKPOINT_DIR"

HI_ML="$ROOT/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
export PYTHONPATH="${ROOT}:${HI_ML}${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-smoke}"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    LOG="$VLJEPA_LOG_DIR/vljepa_lora_${MODE}_${SLURM_JOB_ID}.log"
else
    LOG="$VLJEPA_LOG_DIR/vljepa_lora_${MODE}_$(date +%Y%m%d_%H%M%S).log"
fi
export PYTHONUNBUFFERED=1

# A fresh srun shell does not activate conda. The packages for this run
# (torch, timm, pyarrow) live in the roentgen env on /nvme.
ROENTGEN_PY="/nvme/people/eprakash/miniconda3/envs/roentgen/bin/python"
if [ -x "$ROENTGEN_PY" ]; then
    PY="$ROENTGEN_PY"
elif command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo "[ccw] no python found. Expected $ROENTGEN_PY" >&2
    exit 1
fi

# Screen still shows the run. The file keeps the full gold report.
run() {
    echo "[ccw] PROJECT_DIR=$PROJECT_DIR"
    echo "[ccw] all_data=$JEPA_IMAGE_ROOTS_DIR"
    echo "[ccw] hf=$VLJEPA_HF_HOME"
    echo "[ccw] ckpt=$VLJEPA_CHECKPOINT_DIR"
    echo "[ccw] log=$LOG"
    echo "[ccw] python=$PY"
    if [ "$MODE" = "smoke" ]; then
        export CUDA_VISIBLE_DEVICES=""
        "$PY" -m vljepa.smoke_ccw_cpu
    elif [ "$MODE" = "train" ]; then
        NGPU="${SLURM_GPUS_ON_NODE:-4}"
        echo "[ccw] nproc_per_node=$NGPU"
        "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" -m vljepa.train
    elif [ "$MODE" = "check" ]; then
        shift
        "$PY" -m vljepa.check_class_signal "$@"
    elif [ "$MODE" = "shift" ]; then
        shift
        "$PY" -m vljepa.save_class_shift "$@"
    elif [ "$MODE" = "gold" ]; then
        shift
        "$PY" -m vljepa.eval_gold "$@"
    else
        echo "usage: bash vljepa/run_ccw.sh smoke|train|check|shift|gold --ckpt <epoch_N.pt>" >&2
        return 1
    fi
}

run "$@" 2>&1 | tee -a "$LOG"
