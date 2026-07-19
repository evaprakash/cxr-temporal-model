#!/bin/bash
#SBATCH --job-name=jepa_cbw9999to99999
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/m000081/eprakash/temporal/logs/jepa_cbw9999to99999_%j.out
#SBATCH --error=/scratch/m000081/eprakash/temporal/logs/jepa_cbw9999to99999_%j.err

# ============================================================
# SLURM launcher for the ``main`` branch's two-stage class-balanced
# variant of the 4th (progression) loss. NOT β-annealing — β is a
# hard 0.99999 for every step of this run; only the starting point
# is different from a from-scratch β=0.99999 launch.
#
# What this run does that ``main`` did NOT do before:
#   * ``progression_classification_loss`` takes a ``weight=`` tensor
#     forwarded to ``F.cross_entropy``. The tensor is computed at
#     trainer startup from the actual silver-train split using the
#     Cui et al. 2019 effective-number-of-samples formula.
#   * Two stages, hard β at each:
#       - Stage 1 (already trained separately):
#           β = 0.9999. Best single-stage headline (resolved boost
#           ~4.3× stable, directional-class boost ~1.05×), preserves
#           MS-CXR-T stable recall. Checkpoint we resume FROM:
#           ``checkpoints_jepa_dynamic_cbw9999/epoch_5.pt``.
#       - Stage 2 (THIS run):
#           β = 0.99999 (``CBW_BETA`` in ``resume_train_jepa.py``).
#           From scratch, β=0.99999 collapses MS-CXR-T stable because
#           the 2.5× directional-class boost fires while features are
#           still immature and the LR is at its peak. Here the model
#           starts from stage 1's well-shaped features and the LR
#           schedule is inherited from stage 1 (already past warmup,
#           decaying), so the same β=0.99999 objective lands on a
#           different starting condition than a from-scratch launch.
#   * GLoRIA report-contrastive weights are back at the 0.10 baseline
#     (``W_REPORT_PRIOR`` / ``W_REPORT_PRED``). We briefly tried 0.15
#     to lift disease-class alignment; it slightly helped disease but
#     shaved gold minority-class F1, so we reset it for this β sweep.
#   * Checkpoints / logs are namespaced with
#     ``cbw{stage1}to{cur}`` for two-stage runs (this run resolves
#     to ``cbw9999to99999``), so they never clobber the single-stage
#     β-sweep variants (``cbw9999`` / ``cbw99997`` / ``cbw99999`` from
#     scratch) or the earlier report-reweighted runs
#     (``cbw9999_rp15``) or the unweighted-CE archives
#     (``checkpoints_jepa_dynamic/`` / ``logs_dynamic/``).
#   * ``SAVE_EVERY_N_EPOCHS`` is 1 in the trainer for this config so
#     each stage-2 epoch (6, 7, 8, 9, 10, ...) is captured for the
#     per-benchmark best-checkpoint pick.
#
# Everything else (LR, batch size, warmup, W_JEPA, W_PROG, EMA
# schedule) is untouched, so the ONLY variable versus stage 1 is the
# CBW β. ``best.pt`` is still overwritten whenever ``val_total`` improves.
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
#     export JEPA_RESUME_CKPT=/path/to/checkpoints_jepa_dynamic_cbw9999/epoch_5.pt
#     sbatch resume_train_jepa.sh
#
# Auto-resume from the latest stage-2 epoch checkpoint kicks in
# automatically if the run is preempted and re-queued (as long as
# the stage-2 CHECKPOINT_DIR already has an ``epoch_*.pt``), so
# ``sbatch resume_train_jepa.sh`` just picks up where it left off
# without re-winding to the stage-1 starting point.
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
# Stage-1 checkpoint we ``--resume`` from.
#
# The trainer's auto-resume looks inside the ``main``-configured
# stage-2 ``CHECKPOINT_DIR`` first (currently
# ``checkpoints_jepa_dynamic_cbw9999to99999``). If that dir is empty
# (fresh two-stage launch), we point ``--resume`` at the stage-1
# checkpoint via ``JEPA_RESUME_CKPT`` so weights + optimizer + LR
# scheduler state transfer over. On preemption re-queues, the
# stage-2 dir now contains ``epoch_*.pt`` so auto-resume kicks in
# and we do NOT pass ``--resume`` (otherwise we'd repeatedly wind
# training back to the stage-1 starting point).
# ============================================================
JEPA_RESUME_CKPT="${JEPA_RESUME_CKPT:-/scratch/m000081/eprakash/temporal/final/cxr-temporal-model-cbw/cxr-temporal-model/checkpoints_jepa_dynamic_cbw9999/epoch_5.pt}"
STAGE2_CKPT_DIR="${JEPA_CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints_jepa_dynamic_cbw9999to99999}"
if compgen -G "$STAGE2_CKPT_DIR/epoch_*.pt" > /dev/null; then
  echo "[slurm] stage-2 checkpoints already exist under $STAGE2_CKPT_DIR"
  echo "[slurm]   → auto-resume from latest stage-2 epoch (no --resume passed)"
  RESUME_ARG=""
else
  if [ ! -f "$JEPA_RESUME_CKPT" ]; then
    echo "[slurm] ERROR: stage-1 checkpoint not found: $JEPA_RESUME_CKPT" >&2
    echo "[slurm]   set JEPA_RESUME_CKPT to a valid β=0.9999 epoch_N.pt path." >&2
    exit 1
  fi
  echo "[slurm] initializing stage 2 from $JEPA_RESUME_CKPT"
  RESUME_ARG="--resume $JEPA_RESUME_CKPT"
fi

# ============================================================
# Bulk image data (mimic / chexpert / rexgradient).
#
# ``resume_train_jepa.py`` resolves image roots relative to
# ``$JEPA_IMAGE_ROOTS_DIR``. Default is the shared cluster location
# ``/scratch/m000081/eprakash/all_data`` so this doesn't need to be
# duplicated / symlinked into every branch clone. Layout under that
# root must match:
#
#     $JEPA_IMAGE_ROOTS_DIR/
#         mimic/                 (or symlink to real mimic root)
#         chexpert/train/        (or symlink)
#         rexgradient/deid_png/  (or symlink)
#
# ``export`` (not just assign) so the Python trainer sees the same
# value the shell just verified.
# ============================================================
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-/scratch/m000081/eprakash/all_data}"
echo "[slurm] JEPA_IMAGE_ROOTS_DIR = $JEPA_IMAGE_ROOTS_DIR"
for d in "$JEPA_IMAGE_ROOTS_DIR/mimic" \
         "$JEPA_IMAGE_ROOTS_DIR/chexpert/train" \
         "$JEPA_IMAGE_ROOTS_DIR/rexgradient/deid_png"; do
  if [ ! -d "$d" ]; then
    echo "[slurm] ERROR: image root not found: $d" >&2
    echo "[slurm]   symlink or populate it before submitting." >&2
    exit 1
  fi
done
echo "[slurm] image roots OK"

# ============================================================
# CheXTemporal reachability check.
#
# Same ``export`` pattern as the image roots so the Python loader
# sees the value the shell just verified. Default is
# ``$PROJECT_DIR/CheXTemporal`` since each branch clone typically
# pins its own parquet snapshot; override with ``CHEXTEMPORAL_DIR``
# if you keep the parquets in a shared cluster path.
# ============================================================
export CHEXTEMPORAL_DIR="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
if [ ! -f "$CHEXTEMPORAL_DIR/silver_findings.parquet" ]; then
  echo "[slurm] ERROR: silver_findings.parquet not found under $CHEXTEMPORAL_DIR" >&2
  echo "[slurm]   symlink CheXTemporal into the project root or set CHEXTEMPORAL_DIR." >&2
  exit 1
fi
echo "[slurm] CheXTemporal OK ($CHEXTEMPORAL_DIR)"

# ============================================================
# Launch training (4 GPUs, DDP). run_jepa.py auto-detects the
# visible GPU count via torch.cuda.device_count(), so we don't
# have to keep --nproc_per_node in sync with --gres.
#
# ``$RESUME_ARG`` is empty on preemption re-queues (auto-resume
# from stage-2 dir), and ``--resume /path/to/stage-1/epoch_5.pt``
# on the initial launch.
# ============================================================
python run_jepa.py $RESUME_ARG
