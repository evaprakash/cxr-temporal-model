#!/bin/bash
#SBATCH --job-name=jepa_tiles
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/jepa_tiles_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/jepa_tiles_%j.err

# ============================================================
# Eval-only: paper epoch_5 decision tiles (which tiles made the 5-way).
# Does not train. Does not touch checkpoints.
#
#     mkdir -p /scratch/m000081-pm06/eprakash/logs
#     cd /scratch/m000081-pm06/eprakash/cxr-temporal-model
#     git pull
#     sbatch eval_jepa_decision_tiles.sh
# ============================================================

module load slurm
module load nvhpc

source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/m000081-pm06/eprakash}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH_BASE/cxr-temporal-model}"
cd "$PROJECT_DIR" || {
    echo "[slurm] ERROR: PROJECT_DIR not found: $PROJECT_DIR" >&2
    exit 1
}
echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] branch      = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<n/a>')"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"
echo "[slurm] partition   = ${SLURM_JOB_PARTITION:-unknown}"

HI_ML_SRC="$PROJECT_DIR/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
if [ ! -d "$HI_ML_SRC/health_multimodal" ]; then
    echo "[slurm] ERROR: health_multimodal not found at $HI_ML_SRC" >&2
    exit 1
fi
export PYTHONPATH="${HI_ML_SRC}${PYTHONPATH:+:$PYTHONPATH}"

export CHEXTEMPORAL_DIR="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-$SCRATCH_BASE/all_data}"
JEPA_CKPT="${JEPA_CKPT:-$PROJECT_DIR/checkpoints_jepa_dynamic_cbw99999/epoch_5.pt}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/decision_tiles_jepa_paper}"
echo "[slurm] CHEXTEMPORAL_DIR     = $CHEXTEMPORAL_DIR"
echo "[slurm] JEPA_IMAGE_ROOTS_DIR = $JEPA_IMAGE_ROOTS_DIR"
echo "[slurm] JEPA_CKPT            = $JEPA_CKPT"
echo "[slurm] OUT_DIR              = $OUT_DIR"
if [ ! -f "$JEPA_CKPT" ]; then
    echo "[slurm] ERROR: missing paper ckpt: $JEPA_CKPT" >&2
    exit 1
fi

mkdir -p "$SCRATCH_BASE/logs" "$OUT_DIR"

echo
echo "===== paper epoch_5 / decision tiles (perpatch 5-way) ====="
python eval_jepa_decision_tiles.py --eval \
    --ckpt "$JEPA_CKPT" \
    --out-dir "$OUT_DIR" \
    "$@"

echo
echo "[slurm] done."
