#!/bin/bash
#SBATCH --job-name=jepa_cbw99999_featstd
#SBATCH -p batch
#SBATCH -A marlowe-m000081-pm06
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/jepa_cbw99999_featstd_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/jepa_cbw99999_featstd_%j.err

# ============================================================
# SLURM launcher: reported per-patch JEPA (0.452 recipe) + feat-std
# monitoring. Does NOT overwrite checkpoints_jepa_dynamic_cbw99999/.
#
#   * W_JEPA = 1.0 — mean_p (1 - cos(ẑ_dyn[p], z_cur[p]))
#   * W_PROG = 0.1 — per-patch-mean cosine 5-way CE
#   * Dynamic sentence condition for the JEPA loss
#   * Report contrastive unchanged (W_REPORT_* = 0.1)
#   * EPOCHS = 5
#   * Writes to checkpoints_jepa_dynamic_cbw99999_featstd/
#   * Patch-token std every WANDB_LOG_EVERY train steps (CSV + W&B)
#   * Rank-0 gold set-match after each epoch (--pooling perpatch).
#     Skip with: sbatch resume_train_jepa.sh --skip-gold
#
#     mkdir -p /scratch/m000081-pm06/eprakash/logs
#     cd /scratch/m000081-pm06/eprakash/cxr-temporal-model
#     git pull
#     sbatch resume_train_jepa.sh
# ============================================================

module load slurm
module load nvhpc

source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# W&B: offline by default so the job does not need a login. Sync later
# with: wandb sync logs_dynamic_cbw99999_featstd/wandb/offline-run-*
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-cxr-jepa}"
export WANDB_LOG_EVERY="${WANDB_LOG_EVERY:-20}"

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/m000081-pm06/eprakash}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH_BASE/cxr-temporal-model}"
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

export CHEXTEMPORAL_DIR="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-$SCRATCH_BASE/all_data}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/logs_dynamic_cbw99999_featstd}"
echo "[slurm] CHEXTEMPORAL_DIR     = $CHEXTEMPORAL_DIR"
echo "[slurm] JEPA_IMAGE_ROOTS_DIR = $JEPA_IMAGE_ROOTS_DIR"
echo "[slurm] WANDB_MODE           = $WANDB_MODE"
echo "[slurm] WANDB_PROJECT        = $WANDB_PROJECT"
for d in \
    "$JEPA_IMAGE_ROOTS_DIR/mimic" \
    "$JEPA_IMAGE_ROOTS_DIR/chexpert/train" \
    "$JEPA_IMAGE_ROOTS_DIR/rexgradient/deid_png"
do
    if [ ! -d "$d" ]; then
        echo "[slurm] WARNING: missing image root: $d" >&2
    fi
done

mkdir -p "$SCRATCH_BASE/logs" "$WANDB_DIR"

torchrun --nproc_per_node=4 resume_train_jepa.py "$@"
