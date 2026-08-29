#!/bin/bash
#SBATCH --job-name=ensemble_gold
#SBATCH -p batch
#SBATCH -A marlowe-m000081-pm06
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/ensemble_gold_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/ensemble_gold_%j.err

# ============================================================
# Gold set-match ensemble: JEPA per-patch cosine + supervised logits.
#
#   sbatch eval_ensemble_gold_setmatch.sh
#
# Override ckpts:
#   JEPA_CKPT=... SUPERVISED_CKPT=... sbatch eval_ensemble_gold_setmatch.sh
#
# Extra flags after sbatch are forwarded, e.g.:
#   sbatch eval_ensemble_gold_setmatch.sh --combine zscore
#   sbatch eval_ensemble_gold_setmatch.sh --limit 50
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
export PYTHONPATH="${HI_ML_SRC}${PYTHONPATH:+:$PYTHONPATH}"

export CHEXTEMPORAL_DIR="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-$SCRATCH_BASE/all_data}"

JEPA_CKPT="${JEPA_CKPT:-$PROJECT_DIR/checkpoints_jepa_dynamic_cbw99999/epoch_5.pt}"
SUP_CKPT="${SUPERVISED_CKPT:-$PROJECT_DIR/checkpoints_supervised_progression_unfrozen/epoch_5.pt}"

echo "[slurm] JEPA_CKPT       = $JEPA_CKPT"
echo "[slurm] SUPERVISED_CKPT = $SUP_CKPT"
if [ ! -f "$JEPA_CKPT" ]; then
    echo "[slurm] ERROR: missing JEPA ckpt: $JEPA_CKPT" >&2
    exit 1
fi
if [ ! -f "$SUP_CKPT" ]; then
    echo "[slurm] ERROR: missing supervised ckpt: $SUP_CKPT" >&2
    exit 1
fi

mkdir -p "$SCRATCH_BASE/logs"

python eval_ensemble_gold_setmatch.py --eval \
    --jepa-ckpt "$JEPA_CKPT" \
    --supervised-ckpt "$SUP_CKPT" \
    --pooling perpatch \
    --combine softmax \
    "$@"
