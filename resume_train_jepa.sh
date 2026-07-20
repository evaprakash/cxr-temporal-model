#!/bin/bash
#SBATCH --job-name=jepa_cbw99999_dw05
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/m000081/eprakash/temporal/logs/jepa_cbw99999_dw05_%j.out
#SBATCH --error=/scratch/m000081/eprakash/temporal/logs/jepa_cbw99999_dw05_%j.err

# ============================================================
# SLURM launcher: from-scratch β=0.99999 progression CBW +
# disease multi-label image–text BCE (prior + pred ẑ_cur).
#
# What this run does:
#   * Progression CBW β = 0.99999 (frozen).
#   * W_REPORT_PRIOR = W_REPORT_PRED = 0.10 (baseline).
#   * Disease multi-label BCE on prior_patches and on dynamic-
#     conditioned pred_current_patches (NOT progression templates),
#     with all findings on a pair as multi-hot positives.
#   * Disease CBW β = 0.9999; W_DISEASE_PRIOR = W_DISEASE_PRED = 0.05.
#   * ckpt/log tag: cbw99999_dw05_dcw9999
#
# Same 50-epoch, save-every-5 schedule. ``best.pt`` overwritten when
# val_total improves.
#
#     export CHEXTEMPORAL_DIR=/path/to/CheXTemporal
#     export JEPA_IMAGE_ROOTS_DIR=/path/to/all_data
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

PROJECT_DIR="${PROJECT_DIR:-/scratch/m000081/eprakash/temporal/final/cxr-temporal-model-cbw/cxr-temporal-model}"
cd "$PROJECT_DIR"
echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] branch      = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<not a git checkout>')"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"

export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-/scratch/m000081/eprakash/all_data}"
echo "[slurm] JEPA_IMAGE_ROOTS_DIR = $JEPA_IMAGE_ROOTS_DIR"
for d in "$JEPA_IMAGE_ROOTS_DIR/mimic" \
         "$JEPA_IMAGE_ROOTS_DIR/chexpert/train" \
         "$JEPA_IMAGE_ROOTS_DIR/rexgradient/deid_png"; do
  if [ ! -d "$d" ]; then
    echo "[slurm] ERROR: image root not found: $d" >&2
    exit 1
  fi
done
echo "[slurm] image roots OK"

export CHEXTEMPORAL_DIR="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
if [ ! -f "$CHEXTEMPORAL_DIR/silver_findings.parquet" ]; then
  echo "[slurm] ERROR: silver_findings.parquet not found under $CHEXTEMPORAL_DIR" >&2
  exit 1
fi
echo "[slurm] CheXTemporal OK ($CHEXTEMPORAL_DIR)"

python run_jepa.py
