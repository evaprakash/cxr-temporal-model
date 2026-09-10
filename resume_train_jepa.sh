#!/bin/bash
#SBATCH --job-name=jepa_findq
#SBATCH -p preempt
#SBATCH -A marlowe-m000081
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/m000081-pm06/eprakash/logs/jepa_findq_%j.out
#SBATCH --error=/scratch/m000081-pm06/eprakash/logs/jepa_findq_%j.err

# ============================================================
# SLURM launcher: preempt queue (batch is down).
# Marlowe: preempt uses the bare project account (no -pm06) and
# max wall time 4h. -pm06 is a batch QOS and sbatch rejects it here.
# GLoRIA on, full text frozen, official CLS proj, EMA 0.996 → 1.0,
# joint current target, paper weights 1 / 0.1 / 0.1 / 0.1,
# frozen BioViL-T finding query: attn from z_pair only, same
# weights pool ẑ^c and z_pair.
#
#   * W_JEPA = 1.0 — mean_p (1 - cos(ẑ_dyn[p], z_pair[p]))
#   * W_PROG = 0.1 — cos(pool_a(ẑ^c), pool_a(z_pair)), a = softmax(z_pair·Q)
#   * Q = frozen official CLS of the finding name
#   * z_pair = EMA encoder(current, prior); prior input stays single-image
#   * W_REPORT_PRIOR = W_REPORT_PRED = 0.1  (GLoRIA quiet)
#   * Text frozen in full: CXR-BERT + local 768→128
#     (= official cls_projection_head, not a random head)
#   * EMA 0.996 → 1.0
#   * Writes to
#     checkpoints_jepa_dynamic_cbw99999_txtfrzcls_jointtgt_findq/
#     (does NOT resume _wprog100_..._jointtgt or paper _txtfrzcls)
#   * Auto-resumes latest epoch_N.pt in that dir if preempted.
#   * Rank-0 gold set-match after each epoch (--pooling findquery)
#     plus pairwise-win / ẑ-film diagnostics.
#
#     mkdir -p /scratch/m000081-pm06/eprakash/logs
#     cd /scratch/m000081-pm06/eprakash/cxr-temporal-model
#     git pull
#     # cancel the old W_PROG=1 job if it is still queued/running
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

# Abort-check: refuse to launch if this checkout is not the find-query run.
python - <<'PY'
import pathlib
import re
import sys

src = pathlib.Path("resume_train_jepa.py").read_text()

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
    "PROG_POOLING": '"findquery"',
    "JOINT_CURRENT_TARGET": "True",
    "FREEZE_TEXT_ENCODER": "True",
}
bad = []
for k, want in checks.items():
    got = assign(k)
    if got != want:
        bad.append(f"  {k}={got}  (want {want})")
if bad:
    print("[abort-check] FAILED")
    print("\n".join(bad))
    sys.exit(1)
if "findq" not in src or "jointtgt" not in src:
    print("[abort-check] FAILED: expected findq + jointtgt in trainer")
    sys.exit(1)
print("[abort-check] OK  W_JEPA=1.0 W_PROG=0.1 W_REPORT=0.1/0.1")
print("[abort-check] OK  PROG_POOLING=findquery JOINT_CURRENT_TARGET=True")
print("[abort-check] OK  dir tag should be ..._txtfrzcls_jointtgt_findq")
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
