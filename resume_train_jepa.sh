#!/bin/bash
#SBATCH --job-name=jepa_eqw1_preempt
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/jepa_eqw1_preempt_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/jepa_eqw1_preempt_%j.err

# ============================================================
# SLURM launcher: preempt queue (batch is down for the hero run).
# Marlowe: preempt uses the bare project account (no -pm06) and
# max wall time 4h. -pm06 is a batch QOS and sbatch rejects it here.
# GLoRIA on, full text frozen, official CLS proj, EMA 0.996 → 1.0,
# all four live losses at weight 1.0.
#
#   * W_JEPA = 1.0 — mean_p (1 - cos(ẑ_dyn[p], z_cur[p]))
#   * W_PROG = 1.0 — per-patch-mean cosine 5-way CE
#   * W_REPORT_PRIOR = W_REPORT_PRED = 1.0  (GLoRIA; was 0.1)
#   * Text frozen in full: CXR-BERT + local 768→128
#     (= official cls_projection_head, not a random head)
#   * EMA 0.996 → 1.0
#   * Writes to checkpoints_jepa_dynamic_cbw99999_rp100_wprog100_txtfrzcls/
#     (does NOT resume _txtfrzcls, _wjepa50_wprog50, or _ema999)
#   * Auto-resumes latest epoch_N.pt in that dir if preempted.
#   * Rank-0 gold set-match after each epoch (--pooling perpatch).
#
#     mkdir -p /scratch/m000081-pm06/eprakash/logs
#     cd /scratch/m000081-pm06/eprakash/cxr-temporal-model
#     git pull
#     scancel <old-batch-jobid>   # if the 0.5/0.5 batch job is still PD
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

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/m000081-pm06/eprakash}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH_BASE/cxr-temporal-model}"
cd "$PROJECT_DIR" || {
    echo "[slurm] ERROR: PROJECT_DIR not found: $PROJECT_DIR" >&2
    exit 1
}
echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] branch      = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<not a git checkout>')"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"
echo "[slurm] partition   = ${SLURM_JOB_PARTITION:-unknown}"

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
