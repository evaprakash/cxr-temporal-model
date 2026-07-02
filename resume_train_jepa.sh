#!/bin/bash
#SBATCH --job-name=jepa_allfind
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/m000081/eprakash/temporal/logs/jepa_allfind_%j.out
#SBATCH --error=/scratch/m000081/eprakash/temporal/logs/jepa_allfind_%j.err

# ============================================================
# SLURM launcher for the prog-loss-all-findings branch.
#
# Same template as the baseline-jepa-dynamic-noprog SLURM script,
# with two changes:
#   * job / log names namespaced to ``allfind`` so this branch's
#     runs don't collide with other branches' log files.
#   * ``--time=6:00:00`` (up from 2h): this branch runs the
#     progression loss over ALL findings of every pair per step, so
#     wall-clock per epoch is meaningfully higher than the noprog
#     variant. On preempt this just means fewer requeues.
#
# All data paths (CheXTemporal parquets, image roots) now resolve
# relative to ``$PROJECT_DIR`` by default, so each branch clone can
# own its own ``all_data/`` and ``CheXTemporal/`` symlinks without
# fighting over ``/home/evaprakash/...``. If the bulk data lives
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
# ``sbatch``). Defaults to the conventional scratch location for
# this branch's clone.
# ============================================================
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000081/eprakash/temporal/final/cxr-temporal-model-allfindings}"
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
# ``resume_train_jepa.py`` now resolves image roots relative to
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
