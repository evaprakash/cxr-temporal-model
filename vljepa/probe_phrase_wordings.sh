#!/bin/bash
#SBATCH --job-name=vljepa_word
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/m000081/eprakash/logs/vljepa_word_%j.out
#SBATCH --error=/scratch/m000081/eprakash/logs/vljepa_word_%j.err

# Frozen BioViL-T text only. Tries alternate class sentences and prints
# the set with the lowest worst-pair cosine.
#
#   mkdir -p /scratch/m000081/eprakash/logs
#   cd /scratch/m000081/eprakash/temporal/final/cxr-temporal-model
#   git pull
#   sbatch vljepa/probe_phrase_wordings.sh

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
export VLJEPA_HF_HOME="${VLJEPA_HF_HOME:-$SCRATCH_BASE/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VLJEPA_HF_HOME/hub}"

echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
mkdir -p "$SCRATCH_BASE/logs"

python -m vljepa.probe_phrase_wordings
