#!/bin/bash
#SBATCH --job-name=vljepa_drift
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/m000081/eprakash/logs/vljepa_drift_%j.out
#SBATCH --error=/scratch/m000081/eprakash/logs/vljepa_drift_%j.err

# Answer-drift probe on 1 GPU.
#   Part A   sentence drift + pretrained 5×5
#   Part C   zero-shot image vs original sentences
#   Part B   epoch-1 and epoch-5 Ŝ vs epoch sentences and original sentences
#
#   mkdir -p /scratch/m000081/eprakash/logs
#   cd /scratch/m000081/eprakash/temporal/final/cxr-temporal-model
#   git pull
#   sbatch vljepa/probe_answer_drift.sh

module load slurm
module load nvhpc

source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

if [ -x /usr/bin/gcc ]; then
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    export TRITON_CC=/usr/bin/gcc
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/m000081/eprakash}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH_BASE/temporal/final/cxr-temporal-model}"
cd "$PROJECT_DIR" || {
    echo "[slurm] ERROR: PROJECT_DIR not found: $PROJECT_DIR" >&2
    exit 1
}

HI_ML_SRC="$PROJECT_DIR/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
export PYTHONPATH="${PROJECT_DIR}:${HI_ML_SRC}${PYTHONPATH:+:$PYTHONPATH}"
export CHEXTEMPORAL_DIR="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-$SCRATCH_BASE/all_data}"
export VLJEPA_HF_HOME="${VLJEPA_HF_HOME:-$SCRATCH_BASE/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VLJEPA_HF_HOME/hub}"
export VLJEPA_LLAMA_LOCAL="${VLJEPA_LLAMA_LOCAL:-$VLJEPA_HF_HOME/Llama-3.2-1B}"

echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "[slurm] Llama       = $VLJEPA_LLAMA_LOCAL"
mkdir -p "$SCRATCH_BASE/logs"

python -m vljepa.probe_answer_drift --pred-limit 0 --pred-epochs 1,5
