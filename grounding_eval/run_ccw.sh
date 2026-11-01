#!/bin/bash
# Grounding on gold_bboxes for BioViL-T, supervised, VL-JEPA and BAAP. Run on ccw-gpu-1
# inside an srun shell.
#
#   bash grounding_eval/run_ccw.sh maps --model biovilt --name biovilt
#   bash grounding_eval/run_ccw.sh maps --model supervised --name supervised \
#       --ckpt checkpoints_supervised_progression_unfrozen/epoch_5.pt
#   bash grounding_eval/run_ccw.sh maps --model vljepa --name vljepa \
#       --ckpt checkpoints_vljepa_lora_5way/epoch_5_shifted.pt
#   bash grounding_eval/run_ccw.sh maps --model baap --name baap
#   bash grounding_eval/run_ccw.sh compare --names biovilt supervised vljepa baap
# Add --method swap to maps and compare for the image-change maps.
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
HI_ML="$ROOT/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
export PYTHONPATH="${ROOT}:${HI_ML}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
LOG_DIR="${GROUNDING_LOG_DIR:-$SCRATCH_BASE/logs/grounding}"
mkdir -p "$LOG_DIR"

ROENTGEN_PY="/nvme/people/eprakash/miniconda3/envs/roentgen/bin/python"
if [ -x "$ROENTGEN_PY" ]; then
    PY="$ROENTGEN_PY"
else
    PY="$(command -v python || command -v python3)"
fi

MODE="${1:-}"
[ -n "$MODE" ] && shift
case "$MODE" in
    maps|compare) ;;
    *) echo "usage: bash grounding_eval/run_ccw.sh maps|compare ..." >&2; exit 1 ;;
esac

LOG="$LOG_DIR/grounding_${MODE}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}.log"
{
    echo "[grounding] python=$PY"
    echo "[grounding] log=$LOG"
    "$PY" -m grounding_eval.eval_grounding "$MODE" "$@"
} 2>&1 | tee -a "$LOG"
