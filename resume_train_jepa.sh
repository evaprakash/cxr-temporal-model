#!/bin/bash
#SBATCH --job-name=jepa_cbw99999_globalpool
#SBATCH -p batch
#SBATCH -A marlowe-m000081-pm06
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/jepa_cbw99999_globalpool_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/jepa_cbw99999_globalpool_%j.err

# ============================================================
# SLURM launcher: global-pool JEPA + progression CE
# (``cbw99999_globalpool``) on Marlowe project m000081-pm06.
#
#   * W_JEPA = 1.0 — cos(mean-pool(ẑ), mean-pool(z_cur))
#   * W_ANAT_JEPA = 0 — no anatomy masks / filtering
#   * Progression CE uses the same global-pool cosine logits
#   * Report contrastive unchanged (W_REPORT_* = 0.1)
#
# Layout expected under scratch:
#   /scratch/m000081-pm06/eprakash/
#     all_data/
#     cxr-temporal-model/   (+ CheXTemporal inside)
#     logs/
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
echo "[slurm] CHEXTEMPORAL_DIR     = $CHEXTEMPORAL_DIR"
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

mkdir -p "$SCRATCH_BASE/logs"

torchrun --nproc_per_node=4 resume_train_jepa.py "$@"
