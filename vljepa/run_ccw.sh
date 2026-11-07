#!/bin/bash
# Run this on ccw-gpu-1, inside a shell from srun. /nvme is not on the
# login node, and it is wiped if that machine reboots.
#
#   bash vljepa/run_ccw.sh smoke
#   bash vljepa/run_ccw.sh train
#   bash vljepa/run_ccw.sh train_img      # epoch_5.pt weights, image encoder unfrozen, 5 epochs
#   bash vljepa/run_ccw.sh check --ckpt <checkpoint dir>/epoch_N.pt
#   bash vljepa/run_ccw.sh shift --ckpt <checkpoint dir>/epoch_N.pt
#   bash vljepa/run_ccw.sh gold  --ckpt <checkpoint dir>/epoch_N_shifted.pt
#   bash vljepa/run_ccw.sh rules --vljepa <npz> --supervised <npz>   # no GPU needed
#   bash vljepa/run_ccw.sh mscxrt prepare | score --model ... | compare
#   bash vljepa/run_ccw.sh dyn_smoke [--full]      # dynamic-sentence CPU smoke test
#   bash vljepa/run_ccw.sh dyn_train [--epochs N]  # dynamic-sentence training
#   bash vljepa/run_ccw.sh dyn_train_img           # dynamic epoch_9.pt weights, image encoder unfrozen, 5 epochs
#   bash vljepa/run_ccw.sh dyn_eval gold|csv|retrieval --ckpt ...
#   bash vljepa/run_ccw.sh lat_smoke [--full]      # laterality 5-way CPU smoke test
#   bash vljepa/run_ccw.sh lat_train [--epochs N]  # laterality 5-way training, text LoRA
#       (--text-encoder original: set VLJEPA_CHECKPOINT_DIR to a new folder)
#   bash vljepa/run_ccw.sh lat_train_img           # laterality epoch_5.pt weights, image encoder unfrozen, 5 epochs
#   bash vljepa/run_ccw.sh lat_eval gold --ckpt ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

case "${1:-}" in
    dyn_train_img)
        export VLJEPA_CHECKPOINT_DIR="${VLJEPA_CHECKPOINT_DIR:-$ROOT/checkpoints_vljepa_dynamic_img}"
        export VLJEPA_LOG_DIR="${VLJEPA_LOG_DIR:-/nvme/people/eprakash/logs/vljepa_dynamic_img}"
        VLJEPA_INIT_FROM="${VLJEPA_INIT_FROM:-$ROOT/checkpoints_vljepa_dynamic/epoch_9.pt}"
        ;;
    dyn_*)
        export VLJEPA_CHECKPOINT_DIR="${VLJEPA_CHECKPOINT_DIR:-$ROOT/checkpoints_vljepa_dynamic}"
        export VLJEPA_LOG_DIR="${VLJEPA_LOG_DIR:-/nvme/people/eprakash/logs/vljepa_dynamic}"
        ;;
    lat_train_img)
        export VLJEPA_CHECKPOINT_DIR="${VLJEPA_CHECKPOINT_DIR:-$ROOT/checkpoints_vljepa_lora_laterality_imgft}"
        export VLJEPA_LOG_DIR="${VLJEPA_LOG_DIR:-/nvme/people/eprakash/logs/vljepa_lora_laterality_imgft}"
        VLJEPA_INIT_FROM="${VLJEPA_INIT_FROM:-$ROOT/checkpoints_vljepa_lora_laterality/epoch_5.pt}"
        ;;
    lat_*)
        export VLJEPA_CHECKPOINT_DIR="${VLJEPA_CHECKPOINT_DIR:-$ROOT/checkpoints_vljepa_lora_laterality}"
        export VLJEPA_LOG_DIR="${VLJEPA_LOG_DIR:-/nvme/people/eprakash/logs/vljepa_lora_laterality}"
        ;;
esac

if [ "${1:-}" = "train_img" ]; then
    export VLJEPA_CHECKPOINT_DIR="${VLJEPA_CHECKPOINT_DIR:-$ROOT/checkpoints_vljepa_lora_5way_imgft}"
    export VLJEPA_LOG_DIR="${VLJEPA_LOG_DIR:-/nvme/people/eprakash/logs/vljepa_lora_5way_imgft}"
    VLJEPA_INIT_FROM="${VLJEPA_INIT_FROM:-$ROOT/checkpoints_vljepa_lora_5way/epoch_5.pt}"
fi

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
    elif [ "$MODE" = "train_img" ]; then
        shift
        NGPU="${SLURM_GPUS_ON_NODE:-4}"
        echo "[ccw] nproc_per_node=$NGPU init_from=$VLJEPA_INIT_FROM"
        "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" -m vljepa.train \
            --init-from "$VLJEPA_INIT_FROM" --unfreeze-image --epochs 5 "$@"
    elif [ "$MODE" = "check" ]; then
        shift
        "$PY" -m vljepa.check_class_signal "$@"
    elif [ "$MODE" = "shift" ]; then
        shift
        "$PY" -m vljepa.save_class_shift "$@"
    elif [ "$MODE" = "gold" ]; then
        shift
        "$PY" -m vljepa.eval_gold "$@"
    elif [ "$MODE" = "worsening" ]; then
        shift
        "$PY" -m vljepa.check_worsening "$@"
    elif [ "$MODE" = "rules" ]; then
        shift
        "$PY" -m vljepa.compare_shift_rules "$@"
    elif [ "$MODE" = "mscxrt" ]; then
        shift
        "$PY" -m vljepa.eval_mscxrt "$@"
    elif [ "$MODE" = "dyn_smoke" ]; then
        shift
        export CUDA_VISIBLE_DEVICES=""
        "$PY" -m vljepa.smoke_dynamic "$@"
    elif [ "$MODE" = "dyn_train" ]; then
        shift
        NGPU="${SLURM_GPUS_ON_NODE:-4}"
        echo "[ccw] nproc_per_node=$NGPU"
        "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" -m vljepa.train_dynamic "$@"
    elif [ "$MODE" = "dyn_train_img" ]; then
        shift
        NGPU="${SLURM_GPUS_ON_NODE:-4}"
        echo "[ccw] nproc_per_node=$NGPU init_from=$VLJEPA_INIT_FROM"
        "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" -m vljepa.train_dynamic \
            --init-from "$VLJEPA_INIT_FROM" --unfreeze-image --epochs 5 "$@"
    elif [ "$MODE" = "dyn_eval" ]; then
        shift
        "$PY" -m vljepa.eval_dynamic "$@"
    elif [ "$MODE" = "lat_smoke" ]; then
        shift
        export CUDA_VISIBLE_DEVICES=""
        "$PY" -m vljepa.smoke_laterality "$@"
    elif [ "$MODE" = "lat_train" ]; then
        shift
        NGPU="${SLURM_GPUS_ON_NODE:-4}"
        echo "[ccw] nproc_per_node=$NGPU"
        "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" -m vljepa.train_laterality "$@"
    elif [ "$MODE" = "lat_train_img" ]; then
        shift
        NGPU="${SLURM_GPUS_ON_NODE:-4}"
        echo "[ccw] nproc_per_node=$NGPU init_from=$VLJEPA_INIT_FROM"
        "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" -m vljepa.train_laterality \
            --init-from "$VLJEPA_INIT_FROM" --unfreeze-image --epochs 5 "$@"
    elif [ "$MODE" = "lat_eval" ]; then
        shift
        "$PY" -m vljepa.laterality "$@"
    else
        echo "usage: bash vljepa/run_ccw.sh smoke|train|train_img|check|shift|gold|worsening|rules|mscxrt|dyn_smoke|dyn_train|dyn_train_img|dyn_eval|lat_smoke|lat_train|lat_train_img|lat_eval ..." >&2
        return 1
    fi
}

run "$@" 2>&1 | tee -a "$LOG"
