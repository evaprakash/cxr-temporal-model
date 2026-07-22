#!/bin/bash
#SBATCH --job-name=jepa_cbw99999_chgloc05
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/m000081/eprakash/temporal/logs/jepa_cbw99999_chgloc05_%j.out
#SBATCH --error=/scratch/m000081/eprakash/temporal/logs/jepa_cbw99999_chgloc05_%j.err

# ============================================================
# SLURM launcher for the ``main`` branch's from-scratch β=0.99999
# + change-localization grounding add-on (``cbw99999_chgloc05``).
#
# What this run does:
#   * Progression CBW β = 0.99999 (frozen after the β sweep —
#     best gold per-class balance; MS-CXR-T stable known to be weak).
#   * W_REPORT_PRIOR = W_REPORT_PRED = 0.10 (baseline).
#   * Progression CE + full-grid JEPA unchanged (global patch mean).
#   * Change-localization add-on (W_CHANGE_LOC=0.05): soft-pool the
#     prior-grid change map 1-cos(ẑ, z_prior) inside vs outside the
#     prior finding-mask union; maximize s_in - s_out. Active only for
#     improving / worsening / new when a prior mask exists.
#   * Checkpoints / logs under ``cbw99999_chgloc05``.
#
# Same 50-epoch, save-every-5 schedule. ``best.pt`` is overwritten
# whenever ``val_total`` improves.
#
# Bulk image data defaults to the shared cluster path
# ``/scratch/m000081/eprakash/all_data`` (not per-project) so multiple
# branch clones can point at the same DICOM store without duplicating
# it. Override per-run by exporting ``JEPA_IMAGE_ROOTS_DIR`` before
# ``sbatch``; both this script's reachability check and the Python
# trainer read the same env var.
#
# ``CheXTemporal/`` still defaults to ``$PROJECT_DIR/CheXTemporal`` so
# each branch can pin its own silver / gold parquets snapshot; override
# via ``CHEXTEMPORAL_DIR`` if the parquets also live in a shared path.
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
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000081/eprakash/temporal/final/cxr-temporal-model-cbw/cxr-temporal-model}"
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
# ``$JEPA_IMAGE_ROOTS_DIR``. Default is the shared cluster location
# ``/scratch/m000081/eprakash/all_data`` so this doesn't need to be
# duplicated / symlinked into every branch clone. Layout under that
# directory must be:
#   $JEPA_IMAGE_ROOTS_DIR/mimic/
#   $JEPA_IMAGE_ROOTS_DIR/chexpert/train/
#   $JEPA_IMAGE_ROOTS_DIR/rexgradient/deid_png/
# ============================================================
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-/scratch/m000081/eprakash/all_data}"
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

# filtered_masks for change-localization (prior finding masks)
if [ -d "${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}/filtered_masks" ]; then
    echo "[slurm] filtered_masks OK under ${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
else
    echo "[slurm] WARNING: filtered_masks not found — change-localization will be inactive" >&2
fi

# ============================================================
# Launch
# ============================================================
torchrun --nproc_per_node=4 resume_train_jepa.py "$@"
