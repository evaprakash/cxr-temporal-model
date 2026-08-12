#!/bin/bash
#SBATCH --job-name=jepa_silver_prog_eval
#SBATCH -p batch
#SBATCH -A marlowe-m000081-pm06
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/jepa_silver_prog_eval_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/jepa_silver_prog_eval_%j.err

# ============================================================
# SLURM launcher: JEPA 5-way progression eval on CheXTemporal silver.
#
# Default = train-split stratified subsample (overfitting check) with
# per-patch scoring. Override via env or trailing sbatch args:
#
#   # Train check (default)
#   sbatch eval_progression_jepa_silver.sh
#
#   # Matching val check
#   JEPA_SILVER_SPLIT=val sbatch eval_progression_jepa_silver.sh
#
#   # Or pass flags through:
#   sbatch eval_progression_jepa_silver.sh --split val --limit 5000
#
#   # Global-pool checkpoint
#   JEPA_CKPT=checkpoints_jepa_dynamic_cbw99999_globalpool/best.pt \
#     sbatch eval_progression_jepa_silver.sh --pooling global
#
# Layout expected under scratch (same as resume_train_jepa.sh):
#   /scratch/m000081-pm06/eprakash/
#     all_data/
#     cxr-temporal-model/   (+ CheXTemporal, splits_jepa.csv, ckpts)
#     logs/
# ============================================================

module load slurm
module load nvhpc

source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
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

if [ ! -f "$CHEXTEMPORAL_DIR/silver_findings.parquet" ]; then
    echo "[slurm] ERROR: missing $CHEXTEMPORAL_DIR/silver_findings.parquet" >&2
    exit 1
fi
echo "[slurm] silver_findings.parquet OK"

CKPT="${JEPA_CKPT:-$PROJECT_DIR/checkpoints_jepa_dynamic_cbw99999/best.pt}"
SPLIT="${JEPA_SILVER_SPLIT:-train}"
LIMIT="${JEPA_SILVER_LIMIT:-5000}"
POOLING="${JEPA_SILVER_POOLING:-perpatch}"

if [ ! -f "$CKPT" ]; then
    echo "[slurm] ERROR: checkpoint not found: $CKPT" >&2
    echo "[slurm] Set JEPA_CKPT to your best.pt path." >&2
    exit 1
fi
echo "[slurm] ckpt    = $CKPT"
echo "[slurm] split   = $SPLIT"
echo "[slurm] limit   = $LIMIT"
echo "[slurm] pooling = $POOLING"

mkdir -p "$SCRATCH_BASE/logs"

# Trailing sbatch args override the defaults above (e.g. --split val).
python eval_progression_jepa_silver.py --eval \
    --ckpt "$CKPT" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --pooling "$POOLING" \
    "$@"
