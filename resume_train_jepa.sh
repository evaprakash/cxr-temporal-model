#!/bin/bash
#SBATCH --job-name=jepa
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/m000081/eprakash/temporal/logs/jepa_%j.out
#SBATCH --error=/scratch/m000081/eprakash/temporal/logs/jepa_%j.err

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
# Go to project directory
# ============================================================
cd /scratch/m000081/eprakash/temporal/model/biovilt

# ============================================================
# Launch training (4 GPUs, DDP)
# ============================================================
torchrun \
  --nproc_per_node=4 \
  resume_train_jepa.py \
