#!/bin/bash
#SBATCH --job-name=jepa_cbw99999_anatjepaonly100
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/m000081/eprakash/temporal/logs/jepa_cbw99999_anatjepaonly100_%j.out
#SBATCH --error=/scratch/m000081/eprakash/temporal/logs/jepa_cbw99999_anatjepaonly100_%j.err

# ============================================================
# SLURM launcher: anatomy-only dual-mask JEPA (``cbw99999_anatjepaonly100``).
#
#   * W_JEPA = 0  (no full-grid JEPA)
#   * W_ANAT_JEPA = 1.0 — 22 fixed CXAS anatomies, prior mask→ẑ,
#     current mask→z_cur, mean over anatomies
#   * Train/val filtered to pairs with full 22/22 inventory on BOTH
#     prior and current under filtered_masks_anatomy/
#   * No change-localization
#   * Progression CE + report contrastive unchanged
#
#     export PROJECT_DIR=/scratch/m000081/eprakash/temporal/final/cxr-temporal-model
#     export CHEXTEMPORAL_DIR=$PROJECT_DIR/CheXTemporal
#     export JEPA_IMAGE_ROOTS_DIR=/scratch/m000081/eprakash/all_data
#     sbatch resume_train_jepa.sh
# ============================================================

module load slurm
module load nvhpc

source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTHONFAULTHANDLER=1

PROJECT_DIR="${PROJECT_DIR:-/scratch/m000081/eprakash/temporal/final/cxr-temporal-model}"
cd "$PROJECT_DIR" || {
    echo "[slurm] ERROR: PROJECT_DIR not found: $PROJECT_DIR" >&2
    exit 1
}
echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] branch      = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<not a git checkout>')"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"

HI_ML_SRC="$PROJECT_DIR/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
if [ ! -d "$HI_ML_SRC/health_multimodal" ]; then
    echo "[slurm] ERROR: health_multimodal not found at $HI_ML_SRC" >&2
    exit 1
fi
echo "[slurm] hi-ml OK: $HI_ML_SRC"
export PYTHONPATH="${HI_ML_SRC}${PYTHONPATH:+:$PYTHONPATH}"

export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-/scratch/m000081/eprakash/all_data}"
echo "[slurm] JEPA_IMAGE_ROOTS_DIR = $JEPA_IMAGE_ROOTS_DIR"
for d in \
    "$JEPA_IMAGE_ROOTS_DIR/mimic" \
    "$JEPA_IMAGE_ROOTS_DIR/chexpert/train" \
    "$JEPA_IMAGE_ROOTS_DIR/rexgradient/deid_png"
do
    if [ ! -d "$d" ]; then
        echo "[slurm] WARNING: missing image root: $d" >&2
    fi
done

_CHEX="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
if [ ! -d "$_CHEX/filtered_masks_anatomy" ]; then
    echo "[slurm] ERROR: filtered_masks_anatomy not found under $_CHEX" >&2
    exit 1
fi
_N_ANAT=$(find "$_CHEX/filtered_masks_anatomy" -name '*.json' 2>/dev/null | head -5 | wc -l | tr -d ' ')
if [ "${_N_ANAT}" -lt 1 ]; then
    echo "[slurm] ERROR: no anatomy mask JSONs under $_CHEX/filtered_masks_anatomy" >&2
    exit 1
fi
echo "[slurm] filtered_masks_anatomy OK under $_CHEX"

if ! python -c "import pycocotools.mask" >/dev/null 2>&1; then
    echo "[slurm] ERROR: pycocotools not importable. pip install pycocotools" >&2
    exit 1
fi
echo "[slurm] pycocotools OK"

torchrun --nproc_per_node=4 resume_train_jepa.py "$@"
