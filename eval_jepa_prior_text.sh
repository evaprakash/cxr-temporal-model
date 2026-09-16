#!/bin/bash
#SBATCH --job-name=jepa_prior_text
#SBATCH -p batch
#SBATCH -A marlowe-m000081-pm06
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/jepa_prior_text_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/jepa_prior_text_%j.err

# ============================================================
# Prior + text JEPA-only gold eval (class-conditional residual on
# that finding's prior boxes, text ablations, film energy, optional
# next-film target). Supervised cannot run this.
#
#     mkdir -p /scratch/m000081-pm06/eprakash/logs
#     cd /scratch/m000081-pm06/eprakash/cxr-temporal-model
#     git pull
#     sbatch eval_jepa_prior_text.sh
#
# Loc ep8 (same 5-way, slightly better maps):
#     JEPA_CKPT=$PWD/checkpoints_jepa_dynamic_cbw99999_loc10_lr2e6/epoch_8.pt \
#       OUT_TAG=loc_ep8 sbatch eval_jepa_prior_text.sh
#
# Extra python flags after the script name:
#     sbatch eval_jepa_prior_text.sh --limit 40
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

HI_ML_SRC="$PROJECT_DIR/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
if [ ! -d "$HI_ML_SRC/health_multimodal" ]; then
    echo "[slurm] ERROR: health_multimodal not found at $HI_ML_SRC" >&2
    exit 1
fi
export PYTHONPATH="${HI_ML_SRC}${PYTHONPATH:+:$PYTHONPATH}"
export CHEXTEMPORAL_DIR="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-$SCRATCH_BASE/all_data}"

JEPA_CKPT="${JEPA_CKPT:-$PROJECT_DIR/checkpoints_jepa_dynamic_cbw99999/epoch_5.pt}"
BBOX_PARQUET="${BBOX_PARQUET:-$CHEXTEMPORAL_DIR/gold_bboxes.parquet}"
OUT_TAG="${OUT_TAG:-paper_ep5}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/logs_jepa_prior_text}"
CSV="$OUT_DIR/jepa_prior_text_${OUT_TAG}.csv"

echo "[slurm] CHEXTEMPORAL_DIR     = $CHEXTEMPORAL_DIR"
echo "[slurm] JEPA_IMAGE_ROOTS_DIR = $JEPA_IMAGE_ROOTS_DIR"
echo "[slurm] JEPA_CKPT            = $JEPA_CKPT"
echo "[slurm] BBOX_PARQUET         = $BBOX_PARQUET"
echo "[slurm] CSV                  = $CSV"

if [ ! -f "$JEPA_CKPT" ]; then
    echo "[slurm] ERROR: missing ckpt: $JEPA_CKPT" >&2
    exit 1
fi
if [ ! -f "$BBOX_PARQUET" ]; then
    echo "[slurm] ERROR: missing $BBOX_PARQUET" >&2
    exit 1
fi

mkdir -p "$SCRATCH_BASE/logs" "$OUT_DIR"

echo
echo "===== prior + text JEPA-only ($OUT_TAG) ====="
python eval_jepa_prior_text.py --eval \
    --jepa-ckpt "$JEPA_CKPT" \
    --gold-parquet "$BBOX_PARQUET" \
    --csv "$CSV" \
    "$@"

echo
echo "[slurm] done."
