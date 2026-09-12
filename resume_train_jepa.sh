#!/bin/bash
#SBATCH --job-name=jepa_loc
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/jepa_loc_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/jepa_loc_%j.err

# ============================================================
# Paper JEPA + change-localization add-on (from scratch).
# 5-way stays per-patch mean cosine. Does NOT use finding-query.
# Writes to checkpoints_jepa_dynamic_cbw99999_loc10/
# (does not touch paper cbw99999/, _findq/, or _findqwmean/).
#
#   * W_JEPA = 1.0, W_PROG = 0.1, W_REPORT_* = 0.1, W_LOC = 0.1
#   * PROG_POOLING = perpatch
#   * s = 1-cos(ẑ_dynamic, z_prior); raise s in prior finding mask
#   * Skip silver-stable and rows with no prior finding mask
#   * FREEZE_TEXT_ENCODER = False
#   * JOINT_CURRENT_TARGET = False
#   * From scratch (do not --resume paper or findq)
#   * Rank-0 gold after each epoch (paper perpatch)
#
# Decision-tile eval (frozen paper, 1 hr): sbatch eval_jepa_decision_tiles.sh
#
#     mkdir -p /scratch/m000081-pm06/eprakash/logs
#     cd /scratch/m000081-pm06/eprakash/cxr-temporal-model
#     git pull
#     sbatch resume_train_jepa.sh
# ============================================================

module load slurm
module load nvhpc

source /users/eprakash/miniconda3/etc/profile.d/conda.sh
conda activate roentgen

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

SCRATCH_BASE="${SCRATCH_BASE:-/scratch/m000081-pm06/eprakash}"
PROJECT_DIR="${PROJECT_DIR:-$SCRATCH_BASE/cxr-temporal-model}"
cd "$PROJECT_DIR" || {
    echo "[slurm] ERROR: PROJECT_DIR not found: $PROJECT_DIR" >&2
    exit 1
}
echo "[slurm] PROJECT_DIR = $PROJECT_DIR"
echo "[slurm] branch      = $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<not a git checkout>')"
echo "[slurm] HEAD        = $(git rev-parse --short HEAD 2>/dev/null || echo '<n/a>')"
echo "[slurm] partition   = ${SLURM_JOB_PARTITION:-unknown}"

# Abort-check: paper perpatch + change-loc (not findq).
python - <<'PY'
import pathlib
import re
import sys

src = pathlib.Path("resume_train_jepa.py").read_text()
jepa = pathlib.Path("tempcxr/modules/jepa.py").read_text()
text = pathlib.Path("tempcxr/modules/text_encoder.py").read_text()

def assign(name):
    m = re.search(rf"^{name} = (.+)$", src, re.M)
    if not m:
        raise SystemExit(f"ABORT: missing {name}")
    return m.group(1).strip()

checks = {
    "W_JEPA": "1.0",
    "W_PROG": "0.1",
    "W_REPORT_PRIOR": "0.1",
    "W_REPORT_PRED": "0.1",
    "W_LOC": "0.1",
    "PROG_POOLING": '"perpatch"',
    "JOINT_CURRENT_TARGET": "False",
    "FREEZE_TEXT_ENCODER": "False",
    "USE_CHANGE_LOC": "True",
}
bad = []
for k, want in checks.items():
    got = assign(k)
    if got != want:
        bad.append(f"  {k}={got}  (want {want})")
if re.search(
    r"target_image_encoder\(\s*current_imgs\s*,\s*prior_imgs",
    jepa,
):
    bad.append("  jepa.py still encodes current with prior (want single-image)")
if re.search(
    r"target_image_encoder\(\s*current\s*,\s*prior\s*\)",
    src,
):
    bad.append("  trainer gold still encodes current with prior")
if "target_image_encoder(\n                current_imgs,\n            )" not in jepa:
    bad.append("  jepa.py missing single-image target_image_encoder(current_imgs)")
if not re.search(
    r"self\.text_projection = BertProjectionHead\([^\n]+\)\s*\n\s*self\.init_local_proj_from_official_cls",
    text,
):
    bad.append("  text_encoder does not copy official CLS into local proj at init")
if "assert_local_proj_matches_official_cls()" not in src:
    bad.append("  trainer does not verify official-CLS local-proj init")
if "change_localization_loss" not in src:
    bad.append("  trainer missing change_localization_loss")
if 'PROG_POOLING = "findquery_wmean"' in src or 'PROG_POOLING = "findquery"' in src:
    bad.append("  trainer still on finding-query pooling")
if bad:
    print("[abort-check] FAILED")
    print("\n".join(bad))
    sys.exit(1)
print("[abort-check] OK  paper perpatch + loc  1/0.1/0.1/0.1 + W_LOC=0.1")
print("[abort-check] OK  text trainable, single-image current")
print("[abort-check] OK  local text proj = official CLS init, unfrozen")
print("[abort-check] OK  dir tag should be ..._cbw99999_loc10")
PY

HI_ML_SRC="$PROJECT_DIR/tempcxr/modules/hi-ml/hi-ml-multimodal/src"
if [ ! -d "$HI_ML_SRC/health_multimodal" ]; then
    echo "[slurm] ERROR: health_multimodal not found at $HI_ML_SRC" >&2
    exit 1
fi
echo "[slurm] hi-ml OK: $HI_ML_SRC"
export PYTHONPATH="${HI_ML_SRC}${PYTHONPATH:+:$PYTHONPATH}"

export CHEXTEMPORAL_DIR="${CHEXTEMPORAL_DIR:-$PROJECT_DIR/CheXTemporal}"
export JEPA_IMAGE_ROOTS_DIR="${JEPA_IMAGE_ROOTS_DIR:-$SCRATCH_BASE/all_data}"
echo "[slurm] CHEXTEMPORAL_DIR     = $CHEXTEMPORAL_DIR"
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

mkdir -p "$SCRATCH_BASE/logs"

torchrun --nproc_per_node=4 resume_train_jepa.py "$@"
