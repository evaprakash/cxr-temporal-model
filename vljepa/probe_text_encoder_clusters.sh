#!/bin/bash
# CPU only. No GPU. Run on the login node:
#
#   cd /scratch/m000081/eprakash/temporal/final/cxr-temporal-model
#   git pull
#   bash vljepa/probe_text_encoder_clusters.sh
#
# Qwen3-Embedding-0.6B needs transformers>=4.51. The first run downloads
# the weights into $SCRATCH_BASE/hf.

set -euo pipefail

source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/m000081/eprakash}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH_BASE/temporal/final/cxr-temporal-model}"
cd "$PROJECT_DIR"

HI_ML_SRC="$PROJECT_DIR/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
export PYTHONPATH="${PROJECT_DIR}:${HI_ML_SRC}${PYTHONPATH:+:$PYTHONPATH}"
export VLJEPA_HF_HOME="${VLJEPA_HF_HOME:-$SCRATCH_BASE/hf}"
export HF_HOME="$VLJEPA_HF_HOME"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VLJEPA_HF_HOME/hub}"
export VLJEPA_CLUSTER_DIR="${VLJEPA_CLUSTER_DIR:-$SCRATCH_BASE/logs/encoder_clusters}"
mkdir -p "$VLJEPA_CLUSTER_DIR" "$SCRATCH_BASE/logs"

echo "[cpu] PROJECT_DIR = $PROJECT_DIR"
echo "[cpu] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "[cpu] HF_HOME     = $HF_HOME"
python -c "import transformers; print('[cpu] transformers', transformers.__version__)"

python -m vljepa.probe_text_encoder_clusters
