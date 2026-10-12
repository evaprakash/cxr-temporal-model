#!/bin/bash
# CPU only. No GPU. Same sentence probe as probe_text_encoder_clusters.sh,
# for the dino.txt text tower.
#
# Put the text-encoder weights in the repo root, or set DINOTXT_TEXT_CKPT:
#   dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth
#
#   cd /scratch/m000081/eprakash/temporal/final/cxr-temporal-model
#   bash vljepa/probe_dinotxt_clusters.sh

set -euo pipefail

source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/m000081/eprakash}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH_BASE/temporal/final/cxr-temporal-model}"
cd "$PROJECT_DIR"

export PYTHONPATH="${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export VLJEPA_HF_HOME="${VLJEPA_HF_HOME:-$SCRATCH_BASE/hf}"
export HF_HOME="$VLJEPA_HF_HOME"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VLJEPA_HF_HOME/hub}"
export VLJEPA_CLUSTER_DIR="${VLJEPA_CLUSTER_DIR:-$SCRATCH_BASE/logs/encoder_clusters}"
mkdir -p "$VLJEPA_CLUSTER_DIR" "$SCRATCH_BASE/logs"

echo "[cpu] PROJECT_DIR = $PROJECT_DIR"
echo "[cpu] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "[cpu] HF_HOME     = $HF_HOME"
python -c "import transformers; print('[cpu] transformers', transformers.__version__)"

python -m vljepa.probe_dinotxt_clusters
