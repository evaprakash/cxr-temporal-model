#!/bin/bash
#SBATCH --job-name=jepa_cbw99999_chglocdir05
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/m000081/eprakash/temporal/logs/jepa_cbw99999_chglocdir05_%j.out
#SBATCH --error=/scratch/m000081/eprakash/temporal/logs/jepa_cbw99999_chglocdir05_%j.err

# ============================================================
# SLURM launcher for the ``main`` branch's from-scratch β=0.99999
# + where+which-way change-localization (``cbw99999_chglocdir05``).
#
# What this run does:
#   * Progression CBW β = 0.99999 (frozen after the β sweep —
#     best gold per-class balance; MS-CXR-T stable known to be weak).
#   * W_REPORT_PRIOR = W_REPORT_PRED = 0.10 (baseline).
#   * Progression CE + full-grid JEPA unchanged (W_JEPA=1.0).
#   * Change-localization add-on (W_CHANGE_LOC=0.05):
#       (1) Where — soft-pool 1-cos(ẑ, z_prior) inside vs outside the
#           sampled prog_finding prior mask; maximize s_in - s_out.
#       (2) Which way — soft-pool region summaries; push ẑ toward the
#           progression-appropriate side of (z_prior, z_cur).
#     Active for improving / worsening / new / resolved when a finding
#     mask exists (stable excluded entirely).
#   * Checkpoints / logs under ``cbw99999_chglocdir05`` (new tag — does
#     not resume into the older ``chgloc05`` dirs).
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
#     export PROJECT_DIR=/scratch/m000081/eprakash/temporal/final/cxr-temporal-model
#     export CHEXTEMPORAL_DIR=$PROJECT_DIR/CheXTemporal
#     export JEPA_IMAGE_ROOTS_DIR=/scratch/m000081/eprakash/all_data
#     git checkout main && git pull
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
# ``sbatch``). Defaults to the main cluster checkout.
# ============================================================
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000081/eprakash/temporal/final/cxr-temporal-model}"
cd "$PROJECT_DIR" || {
    echo "[slurm] ERROR: PROJECT_DIR not found: $PROJECT_DIR" >&2
    exit 1
}
echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] branch      = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<not a git checkout>')"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"

# ============================================================
# hi-ml-multimodal (provides ``health_multimodal``). Expected at:
#   $PROJECT_DIR/tempcxr/modules/hi-ml/hi-ml-multimodal/src
# Clone once if missing:
#   cd $PROJECT_DIR/tempcxr/modules
#   git clone https://github.com/microsoft/hi-ml.git
# ============================================================
HI_ML_SRC="$PROJECT_DIR/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
if [ ! -d "$HI_ML_SRC/health_multimodal" ]; then
    echo "[slurm] ERROR: health_multimodal not found at $HI_ML_SRC" >&2
    echo "[slurm]        Clone hi-ml under tempcxr/modules/ (see README)." >&2
    exit 1
fi
echo "[slurm] hi-ml OK: $HI_ML_SRC"
export PYTHONPATH="${HI_ML_SRC}${PYTHONPATH:+:$PYTHONPATH}"

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

# filtered_masks for change-localization (finding masks, dual prior/curr)
_CHEX="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
if [ ! -d "$_CHEX/filtered_masks" ]; then
    echo "[slurm] ERROR: filtered_masks not found under $_CHEX" >&2
    exit 1
fi
_N_FINDING_JSON=$(find "$_CHEX/filtered_masks" -name '*__*.json' 2>/dev/null | head -5 | wc -l | tr -d ' ')
if [ "${_N_FINDING_JSON}" -lt 1 ]; then
    echo "[slurm] ERROR: no finding-mask JSONs (*__*.json) under $_CHEX/filtered_masks" >&2
    exit 1
fi
echo "[slurm] filtered_masks OK under $_CHEX (sample *__*.json present)"

# pycocotools required to decode compressed COCO RLEs
if ! python -c "import pycocotools.mask" >/dev/null 2>&1; then
    echo "[slurm] ERROR: pycocotools not importable in this env." >&2
    echo "[slurm]        conda activate roentgen && pip install pycocotools" >&2
    exit 1
fi
echo "[slurm] pycocotools OK"

# ============================================================
# Launch
# ============================================================
torchrun --nproc_per_node=4 resume_train_jepa.py "$@"
