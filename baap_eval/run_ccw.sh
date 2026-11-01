#!/bin/bash
# BAAP baseline on gold and MS-CXR-T. Run on ccw-gpu-1 inside an srun shell.
#
#   bash baap_eval/run_ccw.sh download                 # checkpoint (login node is fine)
#   bash baap_eval/run_ccw.sh boxes --set gold         # CXAS anatomy boxes (GPU)
#   bash baap_eval/run_ccw.sh boxes --set mscxrt
#   bash baap_eval/run_ccw.sh qa --set gold            # PNGs in baap_eval/qa/ to eyeball boxes
#   bash baap_eval/run_ccw.sh gold
#   bash baap_eval/run_ccw.sh mscxrt
#   bash baap_eval/run_ccw.sh compare-gold --model NAME=NPZ[:SHIFT_JSON] ...
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export SCRATCH_BASE="${SCRATCH_BASE:-/nvme/people/eprakash}"
export PROJECT_DIR="${PROJECT_DIR:-$ROOT}"
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-/nvme/data/temporal/all_data}"
HI_ML="$ROOT/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
export PYTHONPATH="${ROOT}:${HI_ML}${PYTHONPATH:+:$PYTHONPATH}"
# Optional: a ChestXRayAnatomySegmentation checkout, if cxas is not pip-installed.
[ -n "${CXAS_DIR:-}" ] && export PYTHONPATH="$CXAS_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
LOG_DIR="${BAAP_LOG_DIR:-$SCRATCH_BASE/logs/baap}"
mkdir -p "$LOG_DIR"

ROENTGEN_PY="/nvme/people/eprakash/miniconda3/envs/roentgen/bin/python"
if [ -x "$ROENTGEN_PY" ]; then
    PY="$ROENTGEN_PY"
else
    PY="$(command -v python || command -v python3)"
fi

MODE="${1:-}"
[ -n "$MODE" ] && shift

if [ "$MODE" = "download" ]; then
    DIR="$ROOT/baap_eval/ckpt"
    mkdir -p "$DIR"
    URL="https://github.com/chenyuming052/BAAP/releases/download/baap-v1.0"
    curl -L "$URL/baap-v1-paperbest.ckpt" -o "$DIR/baap-v1-paperbest.ckpt"
    echo "761f6781401e866f53561aaac3b704a27e97c531e2bd9ce089678b96098c8f69  baap-v1-paperbest.ckpt" \
        > "$DIR/SHA256SUMS"
    (cd "$DIR" && sha256sum -c SHA256SUMS)
    exit 0
fi

case "$MODE" in
    boxes|gold|mscxrt|compare-gold|qa) ;;
    *) echo "usage: bash baap_eval/run_ccw.sh download|boxes|qa|gold|mscxrt|compare-gold ..." >&2; exit 1 ;;
esac

LOG="$LOG_DIR/baap_${MODE}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}.log"
{
    echo "[baap] python=$PY"
    echo "[baap] log=$LOG"
    "$PY" -m baap_eval.eval_baap "$MODE" "$@"
} 2>&1 | tee -a "$LOG"
