#!/bin/bash
#SBATCH --job-name=jepa_reps
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/jepa_reps_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/jepa_reps_%j.err

# ============================================================
# Paper JEPA vs findq epoch_4 — representation checks (not argmax).
# Same gold pairs, both ckpts, same two readouts.
#
#   1. eval_jepa_only_gold  — next-film, Δ align, five-ẑ collapse, energy
#   2. gold set-match --pooling perpatch   (paper train rule + section D)
#   3. gold set-match --pooling findquery  (findq train rule + section D)
#   4. change-map CNR/PG (--no-render) if gold_bboxes.parquet exists
#
# Ignore combined / kappa for "better reps." Compare (D) pairwise win,
# cos(ẑ^i, ẑ^j), next-film, change align, CNR.
#
# Five-film cosine and energy do not use z_cur (fairest vs paper, which
# was trained toward single-image current; this code scores joint z_pair).
#
#     mkdir -p /scratch/m000081-pm06/eprakash/logs
#     cd /scratch/m000081-pm06/eprakash/cxr-temporal-model
#     git pull
#     sbatch compare_jepa_reps.sh
#     sbatch compare_jepa_reps.sh --limit 40
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
echo "[slurm] CHEXTEMPORAL_DIR     = $CHEXTEMPORAL_DIR"
echo "[slurm] JEPA_IMAGE_ROOTS_DIR = $JEPA_IMAGE_ROOTS_DIR"

PAPER_CKPT="${PAPER_CKPT:-$PROJECT_DIR/checkpoints_jepa_dynamic_cbw99999/epoch_5.pt}"
FINDQ_CKPT="${FINDQ_CKPT:-$PROJECT_DIR/checkpoints_jepa_dynamic_cbw99999_txtfrzcls_jointtgt_findq/epoch_4.pt}"
BBOX_PARQUET="${BBOX_PARQUET:-$CHEXTEMPORAL_DIR/gold_bboxes.parquet}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/logs_jepa_rep_compare}"

echo "[slurm] PAPER_CKPT  = $PAPER_CKPT"
echo "[slurm] FINDQ_CKPT  = $FINDQ_CKPT"
echo "[slurm] OUT_DIR     = $OUT_DIR"

for f in "$PAPER_CKPT" "$FINDQ_CKPT"; do
    if [ ! -f "$f" ]; then
        echo "[slurm] ERROR: missing ckpt: $f" >&2
        exit 1
    fi
done

mkdir -p "$SCRATCH_BASE/logs" "$OUT_DIR"

run_one() {
    local name="$1"
    local ckpt="$2"
    shift 2
    echo
    echo "############################################################"
    echo "# $name"
    echo "# $ckpt"
    echo "############################################################"

    echo
    echo "===== $name / JEPA-only gold ====="
    python eval_jepa_only_gold.py --eval \
        --jepa-ckpt "$ckpt" \
        --csv "$OUT_DIR/${name}_jepa_only.csv" \
        "$@"

    echo
    echo "===== $name / set-match pooling=perpatch ====="
    python eval_progression_gold_setmatch.py --backend jepa --eval \
        --ckpt "$ckpt" \
        --pooling perpatch \
        "$@"

    echo
    echo "===== $name / set-match pooling=findquery ====="
    python eval_progression_gold_setmatch.py --backend jepa --eval \
        --ckpt "$ckpt" \
        --pooling findquery \
        "$@"

    if [ -f "$BBOX_PARQUET" ]; then
        echo
        echo "===== $name / change-map CNR (no PNGs) ====="
        python jepa_change_map_pairs.py \
            --ckpt "$ckpt" \
            --gold-parquet "$BBOX_PARQUET" \
            --out-dir "$OUT_DIR/${name}_change_maps" \
            --cnr-csv "$OUT_DIR/${name}_cnr.csv" \
            --no-render
    else
        echo "[slurm] skip change-maps (missing $BBOX_PARQUET)"
    fi
}

run_one paper "$PAPER_CKPT" "$@"
run_one findq "$FINDQ_CKPT" "$@"

echo
echo "[slurm] done. CSVs in $OUT_DIR"
