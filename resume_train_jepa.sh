#!/bin/bash
#SBATCH --job-name=jepa_cr_np
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/m000081/eprakash/temporal/logs/jepa_cr_np_%j.out
#SBATCH --error=/scratch/m000081/eprakash/temporal/logs/jepa_cr_np_%j.err

# ============================================================
# SLURM launcher for the baseline-jepa-current-report-noprog
# branch.
#
# Mirrors the existing biovilt SLURM template (preempt partition,
# 4 GPUs, 32 CPUs, 400G mem, 2h wall clock) and just swaps the
# project directory + entry point.
#
# This branch trains a 3-loss JEPA model (JEPA + 2 local CL, no
# 5-way progression CE) with the current-report condition (raw
# impression + findings of the current study, no silver labels
# or silver sentences) for 15 epochs. With the 4-GPU template
# that's well under the 2h wall clock; bump --time if your run
# is longer.
#
# Override PROJECT_DIR if you cloned somewhere other than the
# default below. Auto-resume from the latest epoch checkpoint
# kicks in automatically if the run is preempted and re-queued,
# so just `sbatch resume_train_jepa.sh` will pick up where it
# left off.
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
# `sbatch`). Defaults to the conventional scratch location for
# this branch's clone.
# ============================================================
PROJECT_DIR="${PROJECT_DIR:-/scratch/m000081/eprakash/temporal/final/cxr-temporal-model-current-report-noprog}"
cd "$PROJECT_DIR"
echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] branch      = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<not a git checkout>')"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"

# ============================================================
# CheXTemporal silver/gold parquets.
#
# `dataset_combined_jepa.py` defaults DEFAULT_DATASET_DIR to
# `<script_dir>/CheXTemporal`, so if you symlinked
# `<PROJECT_DIR>/CheXTemporal -> /home/evaprakash/CheXTemporal`
# you don't need to set anything here. Otherwise uncomment and
# point at the parent dir of {silver,gold}_*.parquet.
# ============================================================
# export CHEXTEMPORAL_DIR=/home/evaprakash/CheXTemporal

# ============================================================
# Sanity-check that the IMAGE_ROOTS hardcoded in
# resume_train_jepa.py are actually reachable from the compute
# node before we burn time launching DDP. (Symlinks at
# /home/evaprakash/all_data/{mimic,chexpert/train,rexgradient/deid_png}
# pointing at the real scratch location work fine — the loader
# just resolves them.)
# ============================================================
for d in /home/evaprakash/all_data/mimic \
         /home/evaprakash/all_data/chexpert/train \
         /home/evaprakash/all_data/rexgradient/deid_png; do
  if [ ! -d "$d" ]; then
    echo "[slurm] ERROR: image root not found: $d" >&2
    exit 1
  fi
done
echo "[slurm] image roots OK"

# ============================================================
# Launch training (4 GPUs, DDP). run_jepa.py auto-detects the
# visible GPU count via torch.cuda.device_count(), so we don't
# have to keep --nproc_per_node in sync with --gres.
# ============================================================
python run_jepa.py
