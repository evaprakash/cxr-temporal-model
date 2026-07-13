#!/bin/bash
#SBATCH --job-name=jepa_cbw
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/m000081/eprakash/temporal/logs/jepa_cbw_%j.out
#SBATCH --error=/scratch/m000081/eprakash/temporal/logs/jepa_cbw_%j.err

# ============================================================
# SLURM launcher for the ``main`` branch's class-balanced-weighting
# variant of the 4th (progression) loss.
#
# What this run does that ``main`` did NOT do before:
#   * ``progression_classification_loss`` now takes a ``weight=``
#     tensor and forwards it to ``F.cross_entropy``. The tensor is
#     computed at trainer startup from the actual silver-train split
#     using the Cui et al. 2019 effective-number-of-samples formula
#     with β = 0.99999 (``CBW_BETA`` in ``resume_train_jepa.py``).
#   * Checkpoints / logs are namespaced with a ``cbw<β_digits>``
#     suffix so this ablation never clobbers the earlier unweighted-CE
#     ``checkpoints_jepa_dynamic/`` / ``logs_dynamic/`` runs.
#
# Same 50-epoch, save-every-5 schedule as before; ``best.pt`` is
# still overwritten whenever ``val_total`` improves. Everything else
# (LR, batch size, warmup, JEPA + report-contrastive weights, EMA
# schedule) is untouched, so this is a clean single-variable ablation
# vs the unweighted-CE baseline.
#
# All data paths (CheXTemporal parquets, image roots) resolve relative
# to ``$PROJECT_DIR`` by default, so each branch clone can own its own
# ``all_data/`` and ``CheXTemporal/`` symlinks. If the bulk data lives
# somewhere non-standard, override with:
#
#     export CHEXTEMPORAL_DIR=/path/to/CheXTemporal
#     export JEPA_IMAGE_ROOTS_DIR=/path/to/all_data
#     sbatch resume_train_jepa.sh
#
# Auto-resume from the latest epoch checkpoint kicks in automatically
# if the run is preempted and re-queued, so ``sbatch resume_train_jepa.sh``
# just picks up where it left off.
# ============================================================

# ============================================================
# Modules
# ============================================================
module load slurm
module load nvhpc

# ============================================================
# Conda
# ============================================================
source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

# ============================================================
# Environment variables for PyTorch DDP / NCCL
# ============================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTHONFAULTHANDLER=1

# ============================================================
# Project directory (override by exporting PROJECT_DIR before
# ``sbatch``). Defaults to a ``main``-branch clone dedicated to
# the class-balanced-weighting ablation so its checkpoint /log dirs
# don't collide with any other ``main``-based experiment.
# ============================================================
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000081/eprakash/temporal/final/cxr-temporal-model-cbw}"
cd "$PROJECT_DIR"
echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] branch      = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<not a git checkout>')"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"

# ============================================================
# CheXTemporal silver/gold parquets.
#
# ``dataset_combined_jepa.py`` defaults ``CHEXTEMPORAL_DIR`` to
# ``$PROJECT_DIR/CheXTemporal``. Symlink it into the project root
# (or set the env var below) so the loader finds
# {silver,gold}_*.parquet without any code edits.
# ============================================================
# export CHEXTEMPORAL_DIR=/path/to/CheXTemporal

# ============================================================
# Bulk image data (mimic / chexpert / rexgradient).
#
# ``resume_train_jepa.py`` resolves image roots relative to
# ``$JEPA_IMAGE_ROOTS_DIR`` (default ``$PROJECT_DIR/all_data``), so
# the recommended layout is:
#
#     $PROJECT_DIR/
#         all_data/               (symlink or directory of symlinks)
#             mimic         -> /path/to/real/mimic
#             chexpert/train -> /path/to/real/chexpert/train
#             rexgradient/deid_png -> /path/to/real/rexgradient/deid_png
#
# Sanity-check that everything is reachable from the compute node
# before burning time launching DDP.
# ============================================================
IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-$PROJECT_DIR/all_data}"
echo "[slurm] IMAGE_ROOTS_DIR = $IMAGE_ROOTS_DIR"
for d in "$IMAGE_ROOTS_DIR/mimic" \
         "$IMAGE_ROOTS_DIR/chexpert/train" \
         "$IMAGE_ROOTS_DIR/rexgradient/deid_png"; do
  if [ ! -d "$d" ]; then
    echo "[slurm] ERROR: image root not found: $d" >&2
    echo "[slurm]   symlink or populate it before submitting." >&2
    exit 1
  fi
done
echo "[slurm] image roots OK"

# ============================================================
# CheXTemporal reachability check (same idea as image roots).
# ============================================================
CHEXTEMPORAL_DIR_RESOLVED="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
if [ ! -f "$CHEXTEMPORAL_DIR_RESOLVED/silver_findings.parquet" ]; then
  echo "[slurm] ERROR: silver_findings.parquet not found under $CHEXTEMPORAL_DIR_RESOLVED" >&2
  echo "[slurm]   symlink CheXTemporal into the project root or set CHEXTEMPORAL_DIR." >&2
  exit 1
fi
echo "[slurm] CheXTemporal OK ($CHEXTEMPORAL_DIR_RESOLVED)"

# ============================================================
# Launch training (4 GPUs, DDP). run_jepa.py auto-detects the
# visible GPU count via torch.cuda.device_count(), so we don't
# have to keep --nproc_per_node in sync with --gres.
# ============================================================
python run_jepa.py
