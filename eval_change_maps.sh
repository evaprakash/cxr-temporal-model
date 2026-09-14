#!/bin/bash
#SBATCH --job-name=change_maps
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/change_maps_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/change_maps_%j.err

# ============================================================
# Rerun change-map CNR / PG / mIoU / pixel-AUROC on gold_bboxes.
# Metrics only (--no-render). Three models, same pairs:
#   1. official BioViL-T   role-swap 1-cos          (no ckpt)
#   2. supervised unfrozen role-swap 1-cos          (head unused)
#   3. paper JEPA          predictor-delta ||ẑ-z_prior||
#
#     mkdir -p /scratch/m000081-pm06/eprakash/logs
#     cd /scratch/m000081-pm06/eprakash/cxr-temporal-model
#     git pull
#     sbatch eval_change_maps.sh
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
BBOX_PARQUET="${BBOX_PARQUET:-$CHEXTEMPORAL_DIR/gold_bboxes.parquet}"
JEPA_CKPT="${JEPA_CKPT:-$PROJECT_DIR/checkpoints_jepa_dynamic_cbw99999/epoch_5.pt}"
SUP_CKPT="${SUP_CKPT:-$PROJECT_DIR/checkpoints_supervised_progression_unfrozen/epoch_5.pt}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/change_maps_region}"

echo "[slurm] CHEXTEMPORAL_DIR     = $CHEXTEMPORAL_DIR"
echo "[slurm] JEPA_IMAGE_ROOTS_DIR = $JEPA_IMAGE_ROOTS_DIR"
echo "[slurm] BBOX_PARQUET         = $BBOX_PARQUET"
echo "[slurm] JEPA_CKPT            = $JEPA_CKPT"
echo "[slurm] SUP_CKPT             = $SUP_CKPT"
echo "[slurm] OUT_DIR              = $OUT_DIR"

if [ ! -f "$BBOX_PARQUET" ]; then
    echo "[slurm] ERROR: missing $BBOX_PARQUET" >&2
    exit 1
fi
if [ ! -f "$JEPA_CKPT" ]; then
    echo "[slurm] ERROR: missing paper ckpt: $JEPA_CKPT" >&2
    exit 1
fi
if [ ! -f "$SUP_CKPT" ]; then
    echo "[slurm] ERROR: missing supervised ckpt: $SUP_CKPT" >&2
    exit 1
fi

mkdir -p "$SCRATCH_BASE/logs" "$OUT_DIR"

echo
echo "===== 1/3 official BioViL-T / role-swap 1-cos ====="
python biovilt_change_map_pairs.py --no-render \
    --gold-parquet "$BBOX_PARQUET" \
    --out-dir "$OUT_DIR/biovilt" \
    --cnr-csv "$OUT_DIR/biovilt_cnr.csv" \
    "$@"

echo
echo "===== 2/3 supervised unfrozen / role-swap 1-cos (head unused) ====="
python biovilt_change_map_pairs.py --no-render \
    --ckpt "$SUP_CKPT" \
    --gold-parquet "$BBOX_PARQUET" \
    --out-dir "$OUT_DIR/supervised" \
    --cnr-csv "$OUT_DIR/supervised_cnr.csv" \
    "$@"

echo
echo "===== 3/3 paper JEPA / predictor-delta ====="
python jepa_change_map_pairs.py --no-render \
    --ckpt "$JEPA_CKPT" \
    --gold-parquet "$BBOX_PARQUET" \
    --out-dir "$OUT_DIR/jepa" \
    --cnr-csv "$OUT_DIR/jepa_cnr.csv" \
    "$@"

echo
echo "[slurm] done. CSVs in $OUT_DIR"
