#!/bin/bash
#SBATCH --job-name=vljepa_nudge
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/m000081/eprakash/logs/vljepa_nudge_%j.out
#SBATCH --error=/scratch/m000081/eprakash/logs/vljepa_nudge_%j.err

# ============================================================
# Option-2 VL-JEPA (Chen et al. arXiv:2512.10942 predictor).
#
#   image  : BioViL-T pair encoder (prior + current)
#   query  : "What is the progression of {finding}?"  (Llama tok+embed)
#   target : "{Finding} is {class}."  BioViL-T Y-encoder; pos=gold, neg=other 4
#   pred   : last 8 Llama-3.2-1B layers, bidirectional
#   loss   : 5-way InfoNCE; all five phrases live (no stop-grad)
#   freeze : image frozen, BioViL-T text frozen, Llama predictor unfrozen
#   nudge  : learned offset on improving and worsening only, hinged off stable
#   ckpt   : checkpoints_vljepa_nudge/  (fresh; does not resume frzy)
#   grain  : one silver (pair, finding) per example
#   eval   : CheXTemporal gold set-match after every epoch
#
#     mkdir -p /scratch/m000081/eprakash/logs
#     cd /scratch/m000081/eprakash/temporal/final/cxr-temporal-model
#     git pull
#     python -m vljepa.smoke_cluster       # paths + real Llama load/run
#     python -m vljepa.smoke_test          # CPU shape check (tiny Llama)
#     sbatch vljepa/train.sh
#
# Optional:
#     export VLJEPA_HF_HOME=$SCRATCH_BASE/hf
#     export VLJEPA_LLAMA_LOCAL=$SCRATCH_BASE/hf/Llama-3.2-1B
#     export HF_TOKEN=...                 # if loading from the hub
#     export VLJEPA_CHECKPOINT_DIR=...
# ============================================================

module load slurm
module load nvhpc

source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

# nvhpc puts nvc on PATH. Llama RoPE hits a Triton compile that passes
# gcc-only -Wno-psabi; nvc then aborts the first train step.
if [ -x /usr/bin/gcc ]; then
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    export TRITON_CC=/usr/bin/gcc
fi
export CFLAGS="${CFLAGS:-}"
export CXXFLAGS="${CXXFLAGS:-}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/m000081/eprakash}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH_BASE/temporal/final/cxr-temporal-model}"
cd "$PROJECT_DIR" || {
    echo "[slurm] ERROR: PROJECT_DIR not found: $PROJECT_DIR" >&2
    exit 1
}
echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] branch      = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<not a git checkout>')"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"
echo "[slurm] partition   = ${SLURM_JOB_PARTITION:-unknown}"

python - <<'PY'
import pathlib
import sys

root = pathlib.Path("vljepa")
need = [
    "train.py",
    "model.py",
    "dataset.py",
    "eval_gold.py",
    "prompts.py",
    "smoke_test.py",
    "smoke_cluster.py",
    "cluster_paths.py",
]
bad = [f for f in need if not (root / f).is_file()]
if bad:
    print("[abort-check] FAILED missing", bad)
    sys.exit(1)

train = (root / "train.py").read_text()
model = (root / "model.py").read_text()
prompts = (root / "prompts.py").read_text()
checks = [
    ('QUERY_TEMPLATE = "What is the progression of {finding}?"', prompts),
    ('TARGET_TEMPLATE = "{finding} is {cls}."', prompts),
    ("class_infonce_loss", model),
    ("stopgrad_negatives", model),
    ("STOPGRAD_NEG_PHRASES = False", train),
    ("FREEZE_TEXT_ENCODER = True", train),
    ("checkpoints_vljepa_nudge", train),
    ("class_nudge", model),
    ("stable_separation_loss", model),
    ("NUDGE_SEP_MARGIN = 0.5", train),
    ("SCRATCH_BASE_DEFAULT = \"/scratch/m000081/eprakash\"", (root / "cluster_paths.py").read_text()),
    ("LlamaPredictor", model),
    ("embed_query", model),
    ("embed_tokens", model),
    ("eval_gold_setmatch", train),
    ("FREEZE_IMAGE_ENCODER = True", train),
    ("TEMPERATURE = 0.07", train),
    ('IMAGE_MODE = "biovilt"', train),
]
bad = []
for needle, src in checks:
    if needle not in src:
        bad.append(f"  missing {needle!r}")
if "target_image_encoder" in train:
    bad.append("  train.py still looks like image-JEPA (target_image_encoder)")
if bad:
    print("[abort-check] FAILED")
    print("\n".join(bad))
    sys.exit(1)
print("[abort-check] OK  option-2 VL-JEPA")
print("[abort-check] OK  query=What is the progression of {finding}?")
print("[abort-check] OK  target={Finding} is {class}.  + 5-way InfoNCE")
print("[abort-check] OK  text frozen; improving/worsening nudge; fresh ckpt dir nudge")
print("[abort-check] OK  cycle-6 paths /scratch/m000081/eprakash")
print("[abort-check] OK  BioViL-T pair image + Llama query + BioViL-T Y-encoder")
print("[abort-check] OK  gold set-match after every epoch")
PY

HI_ML_SRC="$PROJECT_DIR/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
if [ ! -d "$HI_ML_SRC/health_multimodal" ]; then
    echo "[slurm] ERROR: health_multimodal not found at $HI_ML_SRC" >&2
    exit 1
fi
echo "[slurm] hi-ml OK: $HI_ML_SRC"
export PYTHONPATH="${PROJECT_DIR}:${HI_ML_SRC}${PYTHONPATH:+:$PYTHONPATH}"

export CHEXTEMPORAL_DIR="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
if [ ! -d "$CHEXTEMPORAL_DIR" ] && [ -d "$SCRATCH_BASE/temporal/final/CheXTemporal" ]; then
    export CHEXTEMPORAL_DIR="$SCRATCH_BASE/temporal/final/CheXTemporal"
fi
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-$SCRATCH_BASE/all_data}"
echo "[slurm] CHEXTEMPORAL_DIR     = $CHEXTEMPORAL_DIR"
echo "[slurm] JEPA_IMAGE_ROOTS_DIR = $JEPA_IMAGE_ROOTS_DIR"
export VLJEPA_HF_HOME="${VLJEPA_HF_HOME:-$SCRATCH_BASE/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$VLJEPA_HF_HOME/hub}"
LLAMA_DEST="${VLJEPA_LLAMA_LOCAL:-$VLJEPA_HF_HOME/Llama-3.2-1B}"
mkdir -p "$VLJEPA_HF_HOME/hub"
if [ -f "$LLAMA_DEST/config.json" ]; then
    export VLJEPA_LLAMA_LOCAL="$LLAMA_DEST"
else
    echo "[slurm] ERROR: Llama snapshot missing at $LLAMA_DEST" >&2
    echo "[slurm]        mv it from pm06 or run: python -m vljepa.download_llama" >&2
    exit 1
fi
echo "[slurm] VLJEPA_HF_HOME       = $VLJEPA_HF_HOME"
echo "[slurm] VLJEPA_LLAMA_NAME    = ${VLJEPA_LLAMA_NAME:-meta-llama/Llama-3.2-1B}"
echo "[slurm] VLJEPA_LLAMA_LOCAL   = $VLJEPA_LLAMA_LOCAL"
echo "[slurm] Llama dest           = $LLAMA_DEST"

python - <<'PY'
from vljepa.cluster_paths import inventory
bad = []
for name, path, ok, detail in inventory(require_sample=False):
    mark = "OK  " if ok else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"[slurm] [{mark}] {name:32s} {path}{extra}")
    if not ok:
        bad.append(name)
if bad:
    print("[slurm] ERROR: missing cluster paths:", ", ".join(bad), flush=True)
    raise SystemExit(1)
print("[slurm] cluster paths OK")
PY

mkdir -p "$SCRATCH_BASE/logs"

torchrun --nproc_per_node=4 -m vljepa.train "$@"
