# resume_train_jepa.py
#
# DDP training entry point for the JEPA-style temporal CXR model.
#
#   - Dataset:  JEPACombinedDataset (silver corpus, paired only)
#   - Model:    TempCXRJEPA (online + EMA + predictor) — unit-sphere
#   - Losses:   JEPA per-patch cosine (1 - cos(ẑ, z_cur) mean over patches)
#               + GLoRIA local contrastive (z_prior)
#               + GLoRIA local contrastive (ẑ_cur)
#               + Progression 5-way image-image CE, class-balanced
#                 (Cui et al. 2019, β=0.99999) — see ``CBW_*`` below.
#   - EMA:      momentum scheduler, target encoder updated after
#               optimizer.step() each iteration
#   - Text condition (predictor input for JEPA loss): ``dynamic`` by
#               default — joined ``label=="dynamic"`` sentences from
#               ``silver_sentences.parquet``. Override via
#               ``CONDITION_MODE=templated`` for the per-finding
#               ``"{Finding} is {progression}."`` template.
#
# Current run: reported per-patch JEPA (dynamic sentences, W_PROG=0.1
# cosine 5-way, gold ``--pooling perpatch``) plus patch-token std
# monitoring (``_featstd`` dir tag). Same recipe as the 0.452 run;
# writes a new dir so ``checkpoints_jepa_dynamic_cbw99999/`` is not
# overwritten. Rank-0 gold set-match after every epoch.
#
# Progression loss (the "4th loss"):
#   For each pair the dataset surfaces one randomly-picked
#   ``(prog_finding, prog_cls_idx)`` per epoch. The model produces
#   ``ẑ_cur^c`` for each of the 5 class prompts
#   ``"{prog_finding} is {class}."`` and applies
#   ``F.cross_entropy(cos(ẑ_cur^c, z_cur) / τ, silver_label,
#                     weight=class_weights)`` (mean-over-patches cosine).

import os
import glob
import random
import argparse
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset_combined_jepa import (
    DEFAULT_FINDINGS,
    JEPACombinedDataset,
    jepa_collate_fn,
)
from eval_progression_jepa import PROMPT_TEMPLATE as JEPA_PROMPT_TEMPLATE
from eval_progression_jepa import _encode_prompts
from gold_progression_setmatch import (
    format_running_setmatch,
    group_gold_by_pair_finding,
    print_setmatch_report,
    summarize_setmatch,
    topk_set_match,
)
from progression_classify import (
    DEFAULT_GOLD_PARQUET,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER
from tempcxr.modules.jepa import (
    TempCXRJEPA,
    make_momentum_scheduler,
    EMA_START,
    EMA_END,
)
from losses import local_contrastive_loss
from losses_jepa import (
    anatomy_masked_pool_jepa_loss,
    global_pool_normalize,
    jepa_cosine_loss,
    patch_token_feature_stats,
    progression_classification_loss,
)
from silver_masks import N_ANATOMY_MASKS, default_anatomy_masks_root


# ============================================================
# DATALOADER WORKER SEEDING
# ============================================================
def seed_dataloader_worker(worker_id):
    """Seed Python ``random`` + numpy from the per-worker, per-epoch torch seed.

    The 4th progression-classification loss reads one randomly-picked
    finding per pair per epoch via ``random.choice`` inside
    ``JEPACombinedDataset.__getitem__``. PyTorch's DataLoader sets a
    fresh per-worker, per-epoch ``torch`` seed automatically, but does
    NOT propagate that seed to Python's ``random`` module or to numpy
    by default. Without this hook, on fork-based multiprocessing the
    workers can inherit the parent's ``random`` state — and if the
    parent process never calls ``random.X(...)`` between epochs, that
    state is identical at the start of each epoch, which means
    ``random.choice`` could (in pathological cases) replay the same
    sequence across epochs.

    Reseeding both libraries from ``torch.initial_seed()`` here makes
    the per-worker, per-epoch variation explicit. The 4th-loss finding
    pick is then guaranteed to:
      * differ across workers within the same epoch, and
      * differ across epochs within the same worker,
    without depending on fork timing or other process-level state.
    """
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# PATHS
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))

# Which text condition the predictor sees for the JEPA loss. ``dynamic``
# (the default) uses the joined ``label=="dynamic"`` sentences from
# ``silver_sentences.parquet`` — free-form report text describing the
# change between prior and current. ``templated`` uses capitalized
# per-finding ``"{Finding} is {progression}."`` clauses joined with a
# space and shuffled per-sample, built from ``silver_findings.parquet``,
# and can be selected via ``CONDITION_MODE=templated``.
#
# The 4th (progression-classification) loss always builds its own
# templated prompts ``"{prog_finding} is {class}."`` regardless of
# ``CONDITION_MODE`` — it needs all 5 candidate-class prompts at every
# step to score the image-image cosine logits.
#
# Patch-token std (Chong's collapse check) is written to
# ``feat_std_jepa.csv`` every ``FEAT_LOG_EVERY`` train steps, and to
# ``val_metrics_jepa.csv`` at the end of each val epoch. No W&B.
FEAT_LOG_EVERY = int(os.environ.get("FEAT_LOG_EVERY", "20"))
CONDITION_MODE = os.environ.get("CONDITION_MODE", "dynamic")

# Cui et al. 2019 "Class-Balanced Loss" hyperparameter for the 4th
# (progression) loss. Beta close to 1 approaches inverse-frequency
# weighting; closer to 0 approaches uniform weights.
#
# Sweep history and per-benchmark behavior we observed:
#
#   β = 0.9999  → resolved boost ~4.3×, middle-class boost ~1.05×
#       Gold overall ~0.41 but stable-magnet (stable ~0.84,
#       improving/new/resolved ~0.11–0.14). Best MS-CXR-T
#       (stable ~0.62).
#
#   β = 0.99997 → resolved boost ~12.7×, middle-class boost ~1.2–1.8×
#       Gold a bit more balanced; MS-CXR-T stable ~0.46.
#
#   β = 0.99998 → between 0.99997 and 0.99999; already on the
#       MS-CXR-T cliff (stable ~0.15) without matching 0.99999's
#       gold minorities. Not used going forward.
#
#   β = 0.99999 (from scratch) → resolved boost ~25×, middle ~2.5×
#       Best gold per-class balance vs lit baselines; MS-CXR-T
#       stable collapses (~0.06). FROZEN as the progression β for
#       this run; we now sweep W_REPORT_* on top.
#
# The two failure modes β alone runs into (from-scratch):
#   * Gold "resolved collapse":   fires when resolved weight  < ~4× stable.
#   * MS-CXR-T "stable collapse": fires when middle-class weight > ~2× stable.
CBW_BETA = 0.99999

# Purely a dir-naming annotation, NOT a training knob. Set this to
# the β value of the checkpoint you plan to ``--resume`` from; the
# ckpt / log dir tag will become ``cbw{stage1_beta}to{CBW_BETA}`` so
# a from-a-different-β restart doesn't clobber the from-scratch
# β=CBW_BETA run's dirs. Set to ``None`` when launching from scratch
# (or when resuming from a same-β checkpoint into the same dir).
# Nothing in training reads this variable — the loss only sees
# ``CBW_BETA`` above.
CBW_BETA_STAGE1 = None

# Image roots resolve relative to this script's directory by default,
# so ``all_data/`` is a peer of ``CheXTemporal/`` inside the project
# clone (both are typically symlinks into scratch storage on the
# cluster). Override the base directory via ``JEPA_IMAGE_ROOTS_DIR`` if
# you keep the bulk data somewhere else — same pattern as
# ``CHEXTEMPORAL_DIR`` in ``dataset_combined_jepa.py``.
_IMAGE_ROOTS_DIR = os.environ.get(
    "JEPA_IMAGE_ROOTS_DIR",
    os.path.join(_HERE, "all_data"),
)
IMAGE_ROOTS = {
    "mimic":       os.path.join(_IMAGE_ROOTS_DIR, "mimic"),
    "chexpert":    os.path.join(_IMAGE_ROOTS_DIR, "chexpert", "train"),
    "rexgradient": os.path.join(_IMAGE_ROOTS_DIR, "rexgradient", "deid_png"),
}


# ============================================================
# ARGUMENTS
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--resume", type=str, default=None)
parser.add_argument(
    "--skip-gold",
    action="store_true",
    help="Skip CheXTemporal gold set-match after each epoch.",
)
args = parser.parse_args()


# ============================================================
# DDP SETUP
# ============================================================
def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, torch.device(f"cuda:{local_rank}")


local_rank, DEVICE = setup_ddp()
WORLD_SIZE = dist.get_world_size()


def ddp_reduce(value):
    tensor = torch.tensor(value, device=DEVICE)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= WORLD_SIZE
    return tensor.item()


# ============================================================
# GRADIENT-PRESERVING ALL-GATHER
# ============================================================
class GatherWithGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        tensor = tensor.contiguous()
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        outputs = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(outputs, tensor)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        batch = grad_output.size(0) // ctx.world_size
        start = ctx.rank * batch
        end = start + batch
        return grad_output[start:end]


def gather_with_grad(tensor):
    return GatherWithGrad.apply(tensor)


# ============================================================
# HYPERPARAMETERS
# ============================================================
LR = 2e-5
WEIGHT_DECAY = 0.01
# Batch size was 32 before; dropped to 24 to fit under the A100-40GB
# memory ceiling with ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.
# All 4 losses scale the same way, so this doesn't change the loss
# balance — only the number of pairs per gradient step.
BATCH_SIZE = 24
EPOCHS = 5
WARMUP_RATIO = 0.03

# Checkpoint schedule: save epoch_N.pt every SAVE_EVERY_N_EPOCHS epochs
# (1 = every epoch), plus best.pt whenever val total improves.
SAVE_EVERY_N_EPOCHS = 1

# Loss weights (baseline report contrastive = 0.10).
# Per-patch JEPA: mean over patches of 1 - cos(ẑ[p], z_cur[p]).
W_JEPA = 1.0
W_REPORT_PRIOR = 0.1
W_REPORT_PRED = 0.1
# 4th loss: per-patch-mean cosine 5-way (same as the 0.452 run).
W_PROG = 0.1
PROG_TEMP = 0.1
PROG_TEMPLATE = "{} is {}."
# Gold / in-training scores: mean_p cos(ẑ^c[p], z_cur[p]).
PROG_POOLING = "perpatch"
N_CLS = len(CLS_ORDER)

# Anatomy dual-mask JEPA off for this run (per-patch full-grid only).
W_ANAT_JEPA = 0.0
USE_ANATOMY_JEPA = False
REQUIRE_FULL_ANATOMY_MASKS = False

# Stratified train/val split when the studies parquet has no 'split'
# column. Both datasets read/write the same cached splits CSV
# (DEFAULT_SPLITS_FILE inside dataset_combined_jepa.py), so the val set
# is identical across train/val DataLoaders and across re-runs.
VAL_FRACTION = 0.1
SPLIT_SEED = 42


# ============================================================
# CHECKPOINT / LOG DIR NAMING
# ============================================================
# Encode CBW β and report reweighting in the ckpt / log dir names so
# ablations never clobber each other. Main-line is per-patch (no extra
# pooling tag). Global-pool runs used a ``_globalpool`` suffix and are
# left as archives:
#   * ``cbw{beta_tag}``               — from-scratch β run
#                                       (W_REPORT_PRIOR = W_REPORT_PRED = 0.1)
#   * ``cbw{stage1}to{cur}``          — hard-β=``{cur}`` run that
#                                       ``--resume``-s from a checkpoint
#                                       previously trained at β=``{stage1}``
#                                       (see ``CBW_BETA_STAGE1``). Only
#                                       affects the dir name — training
#                                       runs at a hard ``CBW_BETA`` the
#                                       whole time, no β schedule.
#   * ``cbw{beta_tag}_rp{ww}``        — both report weights bumped to
#                                       the same non-default value
#                                       (e.g. rp15 = 0.15)
#   * ``cbw{beta_tag}_rpri{aa}_rpred{bb}`` — asymmetric report reweighting
#   * ``..._globalpool``              — archived both-losses global-pool
#   * ``..._progglobal``              — archived per-patch JEPA + global prog CE
#   * ``..._proghead``                — per-patch JEPA + [ẑ; z_cur; finding] head
#   * ``..._wprog{ww}``               — W_PROG != 0.1 (e.g. wprog50 = 0.5)
#   * ``..._featstd``                 — same recipe as the 0.452 run,
#                                       plus patch-token std logging
#   * ``..._anatjepa{ww}``            — anatomy JEPA add-on (full-grid on)
#   * ``..._anatjepaonly{ww}``        — anatomy JEPA only (W_JEPA=0)
# Legacy ``checkpoints_jepa/`` and ``logs/`` dirs from older
# (pre-4-loss / pre-CBW / mask-JEPA) runs are left untouched as archives.
def _cbw_beta_tag(beta: float) -> str:
    """``0.9999`` → ``"9999"``, ``0.99999`` → ``"99999"``, etc."""
    return str(beta).replace("0.", "").replace(".", "")


_beta_tag = _cbw_beta_tag(CBW_BETA)
if CBW_BETA_STAGE1 is not None:
    _stage1_tag = _cbw_beta_tag(CBW_BETA_STAGE1)
    _SETTING_TAG = f"cbw{_stage1_tag}to{_beta_tag}"
else:
    _SETTING_TAG = f"cbw{_beta_tag}"


def _report_weight_tag(w: float) -> str:
    """Format a report contrastive weight as a zero-padded percent tag.

    ``0.10`` → ``"10"``, ``0.15`` → ``"15"``, ``0.2`` → ``"20"``.
    Falls back to a stripped ``str`` for weights that don't land on
    integer-percent so we never silently collapse two distinct sweep
    points into the same directory.
    """
    scaled = w * 100.0
    if abs(scaled - round(scaled)) < 1e-9:
        return f"{int(round(scaled)):02d}"
    return str(w).replace("0.", "").replace(".", "")


if W_REPORT_PRIOR != 0.1 or W_REPORT_PRED != 0.1:
    _rprior_tag = _report_weight_tag(W_REPORT_PRIOR)
    _rpred_tag = _report_weight_tag(W_REPORT_PRED)
    if _rprior_tag == _rpred_tag:
        _SETTING_TAG = f"{_SETTING_TAG}_rp{_rprior_tag}"
    else:
        _SETTING_TAG = f"{_SETTING_TAG}_rpri{_rprior_tag}_rpred{_rpred_tag}"

if USE_ANATOMY_JEPA and W_ANAT_JEPA > 0:
    _anat_tag = _report_weight_tag(W_ANAT_JEPA)
    if W_JEPA == 0.0:
        _SETTING_TAG = f"{_SETTING_TAG}_anatjepaonly{_anat_tag}"
    else:
        _SETTING_TAG = f"{_SETTING_TAG}_anatjepa{_anat_tag}"

if PROG_POOLING == "global":
    _SETTING_TAG = f"{_SETTING_TAG}_progglobal"
elif PROG_POOLING == "head":
    _SETTING_TAG = f"{_SETTING_TAG}_proghead"
if W_PROG != 0.1:
    _SETTING_TAG = f"{_SETTING_TAG}_wprog{_report_weight_tag(W_PROG)}"
# Monitored retrain of the reported per-patch recipe. Do not drop this
# suffix or ``sbatch`` will write into the archived 0.452 checkpoint dir.
_SETTING_TAG = f"{_SETTING_TAG}_featstd"

_DEFAULT_CKPT_DIR = os.path.join(
    _HERE, f"checkpoints_jepa_{CONDITION_MODE}_{_SETTING_TAG}"
)
_DEFAULT_LOG_DIR = os.path.join(
    _HERE, f"logs_{CONDITION_MODE}_{_SETTING_TAG}"
)
CHECKPOINT_DIR = os.environ.get("JEPA_CHECKPOINT_DIR", _DEFAULT_CKPT_DIR)
LOG_DIR = os.environ.get("JEPA_LOG_DIR", _DEFAULT_LOG_DIR)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_LOG = os.path.join(LOG_DIR, "val_metrics_jepa.csv")


# ============================================================
# CLASS-BALANCED WEIGHTS (Cui et al. 2019)
# ============================================================
def _compute_cui_class_weights(dataset, beta: float) -> torch.Tensor:
    """Effective-number-of-samples class weights (Cui et al. 2019).

    For each class ``c`` with ``n_c`` per-finding rows in the training
    split::

        E_c = (1 - β^n_c) / (1 - β)
        w_c ∝ 1 / E_c

    then normalized so the K weights average to 1 (i.e. sum to K),
    which keeps the CE magnitude comparable to the unweighted version
    and thus keeps ``W_PROG`` on the same scale as before.

    Counts are taken from the *actual training split* (not the raw
    silver totals from the HF dataset card), so if the train/val
    split shifts the class distribution slightly the weights follow.
    Every rank computes the same weights from the deterministic split,
    so no broadcast is needed.
    """
    # Flatten the per-pair ``progression_cls`` lists into a single
    # class-name column, then value_counts by class.
    exploded = dataset.df["progression_cls"].explode()
    counts_by_name = exploded.value_counts()

    counts = torch.zeros(N_CLS, dtype=torch.float64)
    for name, c in counts_by_name.items():
        if name in CLS_ORDER:
            counts[CLS_ORDER.index(name)] = float(c)

    beta_t = torch.tensor(beta, dtype=torch.float64)
    effective_num = (1.0 - torch.pow(beta_t, counts)) / (1.0 - beta_t)
    weights = 1.0 / effective_num
    weights = weights * N_CLS / weights.sum()
    return weights.float(), counts.long()


# ============================================================
# DATASETS
# ============================================================
if USE_ANATOMY_JEPA and W_ANAT_JEPA > 0:
    try:
        import pycocotools.mask  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Anatomy JEPA requires pycocotools "
            "(pip install pycocotools)."
        ) from exc
    _anat_root = default_anatomy_masks_root()
    if not os.path.isdir(_anat_root):
        raise RuntimeError(
            f"filtered_masks_anatomy not found at {_anat_root}"
        )

_LOAD_ANATOMY_MASKS = bool(USE_ANATOMY_JEPA and W_ANAT_JEPA > 0)

train_dataset = JEPACombinedDataset(
    image_roots=IMAGE_ROOTS,
    split="train",
    train=True,
    val_fraction=VAL_FRACTION,
    split_seed=SPLIT_SEED,
    condition_mode=CONDITION_MODE,
    require_full_anatomy_masks=REQUIRE_FULL_ANATOMY_MASKS,
    load_anatomy_masks=_LOAD_ANATOMY_MASKS,
)

val_dataset = JEPACombinedDataset(
    image_roots=IMAGE_ROOTS,
    split="val",
    train=False,
    val_fraction=VAL_FRACTION,
    split_seed=SPLIT_SEED,
    condition_mode=CONDITION_MODE,
    require_full_anatomy_masks=REQUIRE_FULL_ANATOMY_MASKS,
    load_anatomy_masks=_LOAD_ANATOMY_MASKS,
)

# Compute Cui et al. class-balanced weights from the ACTUAL training
# split (same numbers on every rank because the split is deterministic).
# Weights are pushed to ``DEVICE`` once so the loss doesn't move them
# every step.
_prog_class_weights_cpu, _prog_class_counts = _compute_cui_class_weights(
    train_dataset, beta=CBW_BETA
)
PROG_CLASS_WEIGHTS = _prog_class_weights_cpu.to(DEVICE)

if local_rank == 0:
    print(f"[train] condition_mode={CONDITION_MODE}")
    print(f"[train] checkpoint dir: {CHECKPOINT_DIR}")
    print(f"[train] log dir:        {LOG_DIR}")
    print(
        f"[train] per-patch JEPA + {PROG_POOLING} prog CE: "
        f"W_JEPA={W_JEPA} W_PROG={W_PROG} "
        f"anatomy_jepa={USE_ANATOMY_JEPA} W_ANAT_JEPA={W_ANAT_JEPA} "
        f"require_full_anatomy_masks={REQUIRE_FULL_ANATOMY_MASKS} "
        f"load_anatomy_masks={_LOAD_ANATOMY_MASKS} "
        f"(JEPA = mean_p (1-cos(ẑ_dyn[p], z_cur[p])); "
        f"prog = per-patch-mean cos 5-way CE)"
    )
    print(
        f"[train] progression-class CBW: β={CBW_BETA} "
        f"(Cui et al. 2019, effective-number-of-samples)"
    )
    for cls, n, w in zip(
        CLS_ORDER,
        _prog_class_counts.tolist(),
        _prog_class_weights_cpu.tolist(),
    ):
        print(
            f"[train]   {cls:<10} n_train={n:>7d}  weight={w:.4f}"
        )

train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=WORLD_SIZE,
    rank=local_rank,
    shuffle=True,
    drop_last=True,
)

val_sampler = DistributedSampler(
    val_dataset,
    num_replicas=WORLD_SIZE,
    rank=local_rank,
    shuffle=False,
    drop_last=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=jepa_collate_fn,
    drop_last=True,
    worker_init_fn=seed_dataloader_worker,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=jepa_collate_fn,
    drop_last=True,
    worker_init_fn=seed_dataloader_worker,
)


# ============================================================
# MODEL
# ============================================================
model = TempCXRJEPA().to(DEVICE)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

num_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * num_steps),
    num_training_steps=num_steps,
)

momentum_scheduler = make_momentum_scheduler(
    m_start=EMA_START,
    m_end=EMA_END,
    total_iters=num_steps,
)

scaler = torch.amp.GradScaler("cuda")

start_epoch = 1
best_val_loss = float("inf")


# ============================================================
# RESUME
# ============================================================
def _ckpt_epoch_num(path: str) -> int:
    """Extract the integer epoch from an ``epoch_N.pt`` filename."""
    name = os.path.basename(path)
    return int(name[len("epoch_"):-len(".pt")])


if args.resume is None:
    checkpoints = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")),
        key=_ckpt_epoch_num,
    )
    if checkpoints:
        args.resume = checkpoints[-1]

if args.resume is not None:
    checkpoint = torch.load(args.resume, map_location=DEVICE)
    model.module.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    # Fast-forward the momentum scheduler to match the resumed step count.
    steps_so_far = (start_epoch - 1) * len(train_loader)
    for _ in range(steps_so_far):
        try:
            next(momentum_scheduler)
        except StopIteration:
            break

    if local_rank == 0:
        print(f"Resumed from {args.resume}")


# ============================================================
# CSV HEADER
# ============================================================
FEAT_CSV_LOG = os.path.join(LOG_DIR, "feat_std_jepa.csv")

if local_rank == 0 and not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w") as f:
        f.write(
            "epoch,val_total,val_jepa,val_report_prior,val_report_pred,"
            "val_prog,val_anatjepa,"
            "val_zhat_std,val_zcur_std,val_zprior_std,"
            "val_zhat_offdiag_cos,"
            "gold_combined,gold_single,gold_multi\n"
        )
if local_rank == 0 and not os.path.exists(FEAT_CSV_LOG):
    with open(FEAT_CSV_LOG, "w") as f:
        f.write(
            "step,epoch,zhat_std_over_patches,zcur_std_over_patches,"
            "zprior_std_over_patches,zhat_mean_offdiag_cos\n"
        )
    print(f"[train] feat-std CSV: {FEAT_CSV_LOG} (every {FEAT_LOG_EVERY} steps)")


# ============================================================
# GOLD SET-MATCH (rank 0, after each epoch)
# ============================================================
gold_groups = None
gold_roots = None
if not args.skip_gold:
    if local_rank == 0:
        gold_df = load_gold_pairs(DEFAULT_GOLD_PARQUET, DEFAULT_FINDINGS)
        gold_groups = group_gold_by_pair_finding(gold_df)
        gold_parquet_dir = os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))
        gold_roots = {
            **IMAGE_ROOTS,
            **discover_gold_image_roots(gold_parquet_dir),
        }
        print(
            f"[gold] set-match after every epoch "
            f"({len(gold_groups)} groups, pooling={PROG_POOLING}, rank-0 only)"
        )
        print("[gold] image roots:")
        for d in ("mimic", "chexpert", "rexgradient"):
            print(f"  {d}: {gold_roots.get(d, '<missing>')}")


# ============================================================
# PROGRESSION-PROMPT HELPER
# ============================================================
def build_progression_prompts(prog_findings):
    """Flatten per-pair findings into a B*N_CLS prompt list.

    Pair-major, class-minor order — ``TempCXRJEPA.forward`` assumes the
    first ``N_CLS`` entries are pair 0's class prompts, the next
    ``N_CLS`` are pair 1's, etc. The model then uses
    ``z_prior.repeat_interleave(N_CLS, dim=0)`` to align text to image.

    The finding string is capitalized to match the templated training
    convention (``"{Finding} is {class}."``); empty findings (defensive
    edge case) yield empty-prefix prompts that still tokenize cleanly.
    """
    prompts = []
    for finding in prog_findings:
        if finding:
            f_cap = finding[:1].upper() + finding[1:]
        else:
            f_cap = ""
        for cls in CLS_ORDER:
            prompts.append(PROG_TEMPLATE.format(f_cap, cls))
    return prompts


@torch.no_grad()
def _score_gold_pair_head(raw_model, prior_img, current_img, finding, text_cache):
    """5-way head logits: [pool(ẑ_finding); pool(z_cur); finding]."""
    key = finding.strip().lower()
    if key in text_cache:
        txt_global, txt_local, token_mask = text_cache[key]
        txt_global = txt_global.to(DEVICE)
        txt_local = txt_local.to(DEVICE)
        token_mask = token_mask.to(DEVICE)
    else:
        txt_global, txt_local, token_mask = (
            raw_model.text_encoder.forward_contrastive([key])
        )
        text_cache[key] = (
            txt_global.detach().cpu(),
            txt_local.detach().cpu(),
            token_mask.detach().cpu(),
        )
    prior = prior_img.unsqueeze(0).to(DEVICE)
    current = current_img.unsqueeze(0).to(DEVICE)
    _, z_prior = raw_model.image_encoder(prior)
    _, z_cur = raw_model.target_image_encoder(current)
    zhat = raw_model.predictor(z_prior, txt_local, token_mask)
    logits = raw_model.progression_logits(
        zhat, z_cur.detach(), txt_global,
    )
    return logits[0].float().tolist()


@torch.no_grad()
def _score_gold_pair(raw_model, prior_img, current_img, finding, text_cache):
    """5-way scores matching the train progression rule (PROG_POOLING)."""
    if PROG_POOLING == "head":
        return _score_gold_pair_head(
            raw_model, prior_img, current_img, finding, text_cache,
        )
    prompts, txt_local, token_mask = _encode_prompts(
        raw_model, finding, JEPA_PROMPT_TEMPLATE, DEVICE, text_cache,
    )
    n_prompts = len(prompts)
    prior = prior_img.unsqueeze(0).to(DEVICE)
    current = current_img.unsqueeze(0).to(DEVICE)
    _, z_prior = raw_model.image_encoder(prior)
    _, z_cur = raw_model.target_image_encoder(current)
    z_cur = z_cur.detach()
    z_prior_b = z_prior.expand(n_prompts, -1, -1).contiguous()
    preds = raw_model.predictor(z_prior_b, txt_local, token_mask)
    pred_f = preds.float()
    target_f = z_cur.float()
    if PROG_POOLING == "global":
        pred_g = global_pool_normalize(pred_f)
        target_g = global_pool_normalize(target_f)
        scores = F.cosine_similarity(
            pred_g, target_g.expand_as(pred_g), dim=-1,
        ).tolist()
    else:
        cos_per_patch = F.cosine_similarity(
            pred_f, target_f.expand_as(pred_f), dim=-1,
        )
        scores = cos_per_patch.mean(dim=1).tolist()
    return scores


@torch.no_grad()
def eval_gold_setmatch(raw_model, groups, image_roots, epoch):
    """Rank-0 CheXTemporal gold set-match (same tables as standalone eval)."""
    raw_model.eval()
    results = []
    skipped = 0
    text_cache = {}
    pbar = tqdm(
        range(len(groups)),
        desc=f"gold set-match ep{epoch}",
        dynamic_ncols=True,
        file=sys.stdout,
    )
    for i in pbar:
        row = groups.iloc[i]
        try:
            prior = load_image_tensor(
                row["dataset"], row["parent_image_prev"], image_roots,
            )
            current = load_image_tensor(
                row["dataset"], row["parent_image_curr"], image_roots,
            )
        except (FileNotFoundError, OSError):
            skipped += 1
            pbar.set_postfix(
                skipped=skipped,
                metrics=format_running_setmatch(results),
            )
            continue
        scores = _score_gold_pair(
            raw_model, prior, current, str(row["finding"]), text_cache,
        )
        results.append(
            topk_set_match(
                scores,
                list(row["gt_labels"]),
                CLS_ORDER,
                finding=str(row["finding"]),
            )
        )
        pbar.set_postfix(
            skipped=skipped,
            metrics=format_running_setmatch(results),
        )
    pbar.close()
    if skipped:
        print(f"[gold] skipped missing images: {skipped}")
    if not results:
        print("[gold] set-match: no groups evaluated")
        return None
    print_setmatch_report(
        results,
        "jepa",
        f", pooling={PROG_POOLING}, epoch={epoch} (in-training)",
    )
    return summarize_setmatch(results)


# ============================================================
# LOSS COMPUTATION (shared by train + val)
# ============================================================
def compute_jepa_losses(
    out,
    prog_cls_idx,
    gather: bool,
    mask_patch_weights_prior=None,
    mask_patch_weights_curr=None,
    mask_pool_active=None,
):
    """
    out                      : dict returned by TempCXRJEPA.forward
    prog_cls_idx             : (B,) long tensor — silver class for 4th loss
    gather                   : if True, gather contrastive features across
                               ranks for cross-rank negatives (training).
                               If False, use local features only (val).
    mask_patch_weights_prior : optional (B, A, N) prior anatomy soft weights
    mask_patch_weights_curr  : optional (B, A, N) current anatomy soft weights
    mask_pool_active         : optional (B,) bool — full 22-mask inventory

    Returns: (total, jepa, prior, pred, prog, anatjepa) as scalar tensors.
    """

    # JEPA loss is per-patch cosine; cross-rank gathering doesn't add
    # useful negatives, so we always compute it on local features. Cast
    # to fp32 so bf16's low precision on small (1 - cos) residuals doesn't
    # round to zero late in training.
    jepa = jepa_cosine_loss(
        out["pred_current_patches"].float(),
        out["current_patches_target"].float(),
    )

    if gather:
        prior_patches = gather_with_grad(out["prior_patches"])
        prior_txt_local = gather_with_grad(out["prior_txt_local"])
        prior_token_mask = gather_with_grad(
            out["prior_token_mask"].float()
        ).bool()

        pred_patches = gather_with_grad(out["pred_current_patches"])
        current_txt_local = gather_with_grad(out["current_txt_local"])
        current_token_mask = gather_with_grad(
            out["current_token_mask"].float()
        ).bool()
    else:
        prior_patches = out["prior_patches"]
        prior_txt_local = out["prior_txt_local"]
        prior_token_mask = out["prior_token_mask"]
        pred_patches = out["pred_current_patches"]
        current_txt_local = out["current_txt_local"]
        current_token_mask = out["current_token_mask"]

    prior = local_contrastive_loss(
        prior_patches,
        prior_txt_local,
        prior_token_mask,
    )
    pred = local_contrastive_loss(
        pred_patches,
        current_txt_local,
        current_token_mask,
    )

    # 5-way CE on the linear head (or archived cosine 5-way).
    # ``weight=`` uses Cui CBW.
    if "prog_logits" in out:
        prog = F.cross_entropy(
            out["prog_logits"].float(),
            prog_cls_idx,
            weight=PROG_CLASS_WEIGHTS,
        )
    else:
        prog = progression_classification_loss(
            out["pred_progression_patches"].float(),
            out["current_patches_target"].float(),
            prog_cls_idx,
            temperature=PROG_TEMP,
            class_weights=PROG_CLASS_WEIGHTS,
        )

    # Anatomy dual-mask JEPA: prior anatomy → ẑ, current anatomy → z_cur.
    # When W_JEPA=0 this is the sole JEPA term.
    if (
        USE_ANATOMY_JEPA
        and W_ANAT_JEPA > 0
        and mask_patch_weights_prior is not None
        and mask_patch_weights_curr is not None
        and mask_pool_active is not None
    ):
        if (
            mask_patch_weights_prior.shape[1] != N_ANATOMY_MASKS
            or mask_patch_weights_curr.shape[1] != N_ANATOMY_MASKS
        ):
            raise ValueError(
                f"expected A={N_ANATOMY_MASKS} anatomy masks; got "
                f"prior A={mask_patch_weights_prior.shape[1]} "
                f"curr A={mask_patch_weights_curr.shape[1]}"
            )
        anatjepa = anatomy_masked_pool_jepa_loss(
            out["pred_current_patches"].float(),
            out["current_patches_target"].float(),
            mask_patch_weights_prior.float(),
            mask_patch_weights_curr.float(),
            mask_pool_active,
        )
    else:
        anatjepa = out["pred_current_patches"].new_zeros(())

    total = (
        W_JEPA * jepa
        + W_REPORT_PRIOR * prior
        + W_REPORT_PRED * pred
        + W_PROG * prog
        + W_ANAT_JEPA * anatjepa
    )
    return total, jepa, prior, pred, prog, anatjepa


# ============================================================
# TRAIN LOOP
# ============================================================
for epoch in range(start_epoch, EPOCHS + 1):

    train_sampler.set_epoch(epoch)
    val_sampler.set_epoch(epoch)

    model.train()
    running_total = 0.0
    running_batches = 0

    if local_rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=120)
    else:
        pbar = train_loader

    for batch_idx, batch in enumerate(pbar):

        prior = batch["prior_image"].to(DEVICE)
        curr = batch["current_image"].to(DEVICE)

        prior_reports = batch["prior_report"]
        current_reports = batch["current_report"]
        condition_texts = batch["condition_text"]

        prog_cls_idx = batch["prog_cls_idx"].to(DEVICE)
        mask_w_prior = batch["mask_patch_weights_prior"].to(DEVICE)
        mask_w_curr = batch["mask_patch_weights_curr"].to(DEVICE)
        mask_active = batch["mask_pool_active"].to(DEVICE)

        if (
            epoch == start_epoch
            and batch_idx == 0
            and local_rank == 0
            and USE_ANATOMY_JEPA
        ):
            frac = mask_active.float().mean().item()
            print(
                f"[train] first-batch mask_active_frac={frac:.3f} "
                f"(n_active={int(mask_active.sum().item())}/"
                f"{mask_active.numel()})"
            )
            if REQUIRE_FULL_ANATOMY_MASKS and frac < 1.0 - 1e-6:
                raise RuntimeError(
                    "require_full_anatomy_masks=True but first batch "
                    f"mask_active_frac={frac:.3f} < 1.0"
                )

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(
                prior,
                curr,
                prior_reports,
                current_reports,
                condition_texts,
                progression_prompts_flat=build_progression_prompts(
                    batch["prog_finding"]
                ),
            )

            loss, jepa_l, prior_l, pred_l, prog_l, anat_l = (
                compute_jepa_losses(
                    out,
                    prog_cls_idx,
                    gather=True,
                    mask_patch_weights_prior=mask_w_prior,
                    mask_patch_weights_curr=mask_w_curr,
                    mask_pool_active=mask_active,
                )
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ---- EMA update of target encoder (after optimizer step) ----
        try:
            m = next(momentum_scheduler)
        except StopIteration:
            m = EMA_END
        model.module.update_ema(momentum=m)

        running_total += loss.item()
        running_batches += 1
        global_step = (epoch - 1) * len(train_loader) + batch_idx + 1

        zhat_stats = patch_token_feature_stats(out["pred_current_patches"])
        zcur_stats = patch_token_feature_stats(out["current_patches_target"])
        zprior_stats = patch_token_feature_stats(out["prior_patches"])
        zhat_std = float(zhat_stats["std_over_patches"])
        zcur_std = float(zcur_stats["std_over_patches"])
        zprior_std = float(zprior_stats["std_over_patches"])
        zhat_cos = float(zhat_stats["mean_offdiag_cos"])

        if local_rank == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "jepa": f"{jepa_l.item():.4f}",
                "prog": f"{prog_l.item():.4f}",
                "zhat_std": f"{zhat_std:.3f}",
                "ema_m": f"{m:.4f}",
                "avg": f"{running_total / running_batches:.4f}",
            })
            if global_step % FEAT_LOG_EVERY == 0:
                with open(FEAT_CSV_LOG, "a") as f:
                    f.write(
                        f"{global_step},{epoch},{zhat_std},{zcur_std},"
                        f"{zprior_std},{zhat_cos}\n"
                    )

    if local_rank == 0:
        print(
            f"Train Epoch {epoch} | "
            f"Avg Loss: {running_total / running_batches:.4f}"
        )


    # ============================================================
    # VALIDATION
    # ============================================================
    model.eval()
    val_total = val_jepa = val_prior = val_pred = val_prog = val_anatjepa = 0.0
    val_zhat_std = val_zcur_std = val_zprior_std = val_zhat_cos = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:

            prior = batch["prior_image"].to(DEVICE)
            curr = batch["current_image"].to(DEVICE)

            prior_reports = batch["prior_report"]
            current_reports = batch["current_report"]
            condition_texts = batch["condition_text"]

            prog_cls_idx = batch["prog_cls_idx"].to(DEVICE)
            mask_w_prior = batch["mask_patch_weights_prior"].to(DEVICE)
            mask_w_curr = batch["mask_patch_weights_curr"].to(DEVICE)
            mask_active = batch["mask_pool_active"].to(DEVICE)

            with torch.amp.autocast("cuda"):
                out = model(
                    prior,
                    curr,
                    prior_reports,
                    current_reports,
                    condition_texts,
                    progression_prompts_flat=build_progression_prompts(
                        batch["prog_finding"]
                    ),
                )

                total, jepa_l, prior_l, pred_l, prog_l, anat_l = (
                    compute_jepa_losses(
                        out,
                        prog_cls_idx,
                        gather=False,
                        mask_patch_weights_prior=mask_w_prior,
                        mask_patch_weights_curr=mask_w_curr,
                        mask_pool_active=mask_active,
                    )
                )

            val_total += total.item()
            val_jepa += jepa_l.item()
            val_prior += prior_l.item()
            val_pred += pred_l.item()
            val_prog += prog_l.item()
            val_anatjepa += anat_l.item()
            zs = patch_token_feature_stats(out["pred_current_patches"])
            zc = patch_token_feature_stats(out["current_patches_target"])
            zp = patch_token_feature_stats(out["prior_patches"])
            val_zhat_std += float(zs["std_over_patches"])
            val_zcur_std += float(zc["std_over_patches"])
            val_zprior_std += float(zp["std_over_patches"])
            val_zhat_cos += float(zs["mean_offdiag_cos"])
            val_batches += 1

    val_total /= max(val_batches, 1)
    val_jepa /= max(val_batches, 1)
    val_prior /= max(val_batches, 1)
    val_pred /= max(val_batches, 1)
    val_prog /= max(val_batches, 1)
    val_anatjepa /= max(val_batches, 1)
    val_zhat_std /= max(val_batches, 1)
    val_zcur_std /= max(val_batches, 1)
    val_zprior_std /= max(val_batches, 1)
    val_zhat_cos /= max(val_batches, 1)

    val_total = ddp_reduce(val_total)
    val_jepa = ddp_reduce(val_jepa)
    val_prior = ddp_reduce(val_prior)
    val_pred = ddp_reduce(val_pred)
    val_prog = ddp_reduce(val_prog)
    val_anatjepa = ddp_reduce(val_anatjepa)
    val_zhat_std = ddp_reduce(val_zhat_std)
    val_zcur_std = ddp_reduce(val_zcur_std)
    val_zprior_std = ddp_reduce(val_zprior_std)
    val_zhat_cos = ddp_reduce(val_zhat_cos)

    if local_rank == 0:

        print(
            f"Val Epoch {epoch} | "
            f"Total={val_total:.4f} | "
            f"JEPA={val_jepa:.4f} | "
            f"PriorReport={val_prior:.4f} | "
            f"PredReport={val_pred:.4f} | "
            f"Prog={val_prog:.4f} | "
            f"AnatJEPA={val_anatjepa:.4f} | "
            f"zhat_std={val_zhat_std:.4f} | "
            f"zhat_offdiag_cos={val_zhat_cos:.4f}"
        )

        ckpt = {
            "epoch": epoch,
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_loss": best_val_loss,
        }

        # Periodic snapshot: epoch 1 + every SAVE_EVERY_N_EPOCHS thereafter.
        if epoch == 1 or (epoch % SAVE_EVERY_N_EPOCHS == 0):
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Always overwrite best.pt when val improves (regardless of epoch).
        if val_total < best_val_loss:
            best_val_loss = val_total
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print("Saved new BEST checkpoint")

        gold_combined = gold_single = gold_multi = ""
        if gold_groups is not None:
            gold_sum = eval_gold_setmatch(
                model.module, gold_groups, gold_roots, epoch,
            )
            if gold_sum is not None:
                gold_combined = f"{gold_sum['combined_score']:.6f}"
                gold_single = f"{gold_sum['single_acc']:.6f}"
                gold_multi = f"{gold_sum['multi_jaccard']:.6f}"

        with open(CSV_LOG, "a") as f:
            f.write(
                f"{epoch},{val_total},{val_jepa},{val_prior},{val_pred},"
                f"{val_prog},{val_anatjepa},"
                f"{val_zhat_std},{val_zcur_std},{val_zprior_std},"
                f"{val_zhat_cos},"
                f"{gold_combined},{gold_single},{gold_multi}\n"
            )

    if WORLD_SIZE > 1:
        dist.barrier()

dist.destroy_process_group()
