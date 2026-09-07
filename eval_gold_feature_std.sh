#!/bin/bash
#SBATCH --job-name=gold_featstd
#SBATCH -p batch
#SBATCH -A marlowe-m000081-pm06
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/gold_featstd_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/gold_featstd_%j.err

# Gold-set feature std: official BioViL-T (single + pair) vs supervised
# vs JEPA. Patch-tile std and global-across-film std.
# Defaults: paper JEPA epoch_5.pt and unfrozen supervised epoch_5.pt.
#
#   sbatch eval_gold_feature_std.sh
#   sbatch eval_gold_feature_std.sh --limit 40
#
# Override ckpts with flags after the script name (argparse last-wins):
#   sbatch eval_gold_feature_std.sh \
#     --jepa-ckpt /path/to/epoch_3.pt

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
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"

HI_ML_SRC="$PROJECT_DIR/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
if [ ! -d "$HI_ML_SRC/health_multimodal" ]; then
    echo "[slurm] ERROR: health_multimodal not found at $HI_ML_SRC" >&2
    exit 1
fi
export PYTHONPATH="${HI_ML_SRC}${PYTHONPATH:+:$PYTHONPATH}"
export CHEXTEMPORAL_DIR="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"

JEPA_CKPT="${JEPA_CKPT:-$PROJECT_DIR/checkpoints_jepa_dynamic_cbw99999/epoch_5.pt}"
SUP_CKPT="${SUPERVISED_CKPT:-$PROJECT_DIR/checkpoints_supervised_progression_unfrozen/epoch_5.pt}"

echo "[slurm] JEPA_CKPT       = $JEPA_CKPT"
echo "[slurm] SUPERVISED_CKPT = $SUP_CKPT"

mkdir -p "$SCRATCH_BASE/logs" "$PROJECT_DIR/logs_gold_feature_std"

python eval_gold_feature_std.py --eval \
    --jepa-ckpt "$JEPA_CKPT" \
    --supervised-ckpt "$SUP_CKPT" \
    --csv "$PROJECT_DIR/logs_gold_feature_std/gold_feature_std.csv" \
    "$@"
