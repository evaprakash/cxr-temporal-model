# resume_train_jepa.py
#
# DDP training entry point for the JEPA-style temporal CXR model.
#
#   - Dataset:  JEPACombinedDataset (silver corpus, paired only)
#   - Model:    TempCXRJEPA (online + EMA + predictor) — unit-sphere
#   - Losses:   JEPA cosine (1 - cos(ẑ_cur, z_cur))
#               + GLoRIA local contrastive (z_prior)
#               + GLoRIA local contrastive (ẑ_cur)
#               + Progression 5-way image-image CE, class-balanced
#                 (Cui et al. 2019, β=0.99999 in this run) — see the
#                 ``CBW_*`` constants below.
#   - EMA:      momentum scheduler, target encoder updated after
#               optimizer.step() each iteration
#   - Text condition (predictor input for JEPA loss): ``dynamic`` by
#               default — joined ``label=="dynamic"`` sentences from
#               ``silver_sentences.parquet``. Override via
#               ``CONDITION_MODE=templated`` for the per-finding
#               ``"{Finding} is {progression}."`` template.
#
# Current run: from-scratch β=0.99999 progression CBW, W_REPORT_*=0.10,
# plus soft-mask pooled JEPA (``_mskjepa05`` dir tag).
#
# Progression loss (the "4th loss"):
#   For each pair the dataset surfaces one randomly-picked
#   ``(prog_finding, prog_cls_idx)`` per epoch. The trainer builds 5
#   templated prompts ``"{prog_finding} is {class}."`` (one per
#   progression class), passes them to ``TempCXRJEPA.forward`` as
#   ``progression_prompts_flat`` (length B*5, pair-major class-minor),
#   and the model runs the predictor a second time on
#   ``z_prior.repeat_interleave(5, dim=0)`` with these 5 text conditions
#   to produce ``ẑ_cur^c`` for each class. The loss is
#   ``F.cross_entropy(mean_patches(cos(ẑ_cur^c, z_cur)) / τ, silver_label,
#                     weight=class_weights)`` — still a *global* patch mean.
#
# Soft-mask pooled JEPA add-on (5th loss, optional per sample):
#   When ``filtered_masks`` has any usable RLEs for the pair's findings
#   on the current image, build the union soft 14×14 float mask, pool
#   dynamic-conditioned ``ẑ_cur`` and EMA ``z_cur``, and apply
#   ``1 - cos(u, v)``. Full-grid JEPA is unchanged. No-mask samples are
#   omitted from this term's batch average (not zero-filled).

import os
import glob
import random
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset_combined_jepa import JEPACombinedDataset, jepa_collate_fn
from progression_phrases import CLS_ORDER
from tempcxr.modules.jepa import (
    TempCXRJEPA,
    make_momentum_scheduler,
    EMA_START,
    EMA_END,
)
from losses import local_contrastive_loss
from losses_jepa import (
    jepa_cosine_loss,
    progression_classification_loss,
    masked_pool_jepa_loss,
)


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
EPOCHS = 50
WARMUP_RATIO = 0.03

# Checkpoint schedule: save epoch_N.pt every SAVE_EVERY_N_EPOCHS epochs
# (1 = every epoch), plus best.pt whenever val total improves.
SAVE_EVERY_N_EPOCHS = 1

# Loss weights (baseline report contrastive = 0.10).
W_JEPA = 1.0
W_REPORT_PRIOR = 0.1
W_REPORT_PRED = 0.1
# 4th loss: 5-way image-image CE on the predictor's class-conditioned
# ẑ_cur. Same magnitude bracket as the two contrastive heads; sweep if
# it dominates or under-shoots at later epochs.
W_PROG = 0.1
PROG_TEMP = 0.1
PROG_TEMPLATE = "{} is {}."
N_CLS = len(CLS_ORDER)

# Soft-mask pooled JEPA add-on (union of all finding masks). Full-grid
# JEPA and global progression CE are unchanged.
W_MASK_JEPA = 0.05
USE_MASKED_POOL_JEPA = True

# Stratified train/val split when the studies parquet has no 'split'
# column. Both datasets read/write the same cached splits CSV
# (DEFAULT_SPLITS_FILE inside dataset_combined_jepa.py), so the val set
# is identical across train/val DataLoaders and across re-runs.
VAL_FRACTION = 0.1
SPLIT_SEED = 42


# ============================================================
# CHECKPOINT / LOG DIR NAMING
# ============================================================
# Encode CBW β, report reweighting, and masked-pool JEPA in the ckpt /
# log dir names so ablations never clobber each other:
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
#   * ``..._mskjepa{ww}``             — soft-mask pooled JEPA weight
# Legacy ``checkpoints_jepa/`` and ``logs/`` dirs from older
# (pre-4-loss / pre-CBW) runs are left untouched as archives.
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

if USE_MASKED_POOL_JEPA and W_MASK_JEPA > 0:
    _SETTING_TAG = (
        f"{_SETTING_TAG}_mskjepa{_report_weight_tag(W_MASK_JEPA)}"
    )

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
train_dataset = JEPACombinedDataset(
    image_roots=IMAGE_ROOTS,
    split="train",
    train=True,
    val_fraction=VAL_FRACTION,
    split_seed=SPLIT_SEED,
    condition_mode=CONDITION_MODE,
)

val_dataset = JEPACombinedDataset(
    image_roots=IMAGE_ROOTS,
    split="val",
    train=False,
    val_fraction=VAL_FRACTION,
    split_seed=SPLIT_SEED,
    condition_mode=CONDITION_MODE,
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
        f"[train] masked-pool JEPA: enabled={USE_MASKED_POOL_JEPA} "
        f"W_MASK_JEPA={W_MASK_JEPA} "
        f"(union soft masks of all findings; full-grid JEPA unchanged)"
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
if local_rank == 0 and not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w") as f:
        f.write(
            "epoch,val_total,val_jepa,val_report_prior,val_report_pred,"
            "val_prog,val_mskjepa\n"
        )


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


# ============================================================
# LOSS COMPUTATION (shared by train + val)
# ============================================================
def compute_jepa_losses(
    out,
    prog_cls_idx,
    gather: bool,
    mask_patch_weights=None,
    mask_pool_active=None,
):
    """
    out                : dict returned by TempCXRJEPA.forward
    prog_cls_idx       : (B,) long tensor — silver class for the 4th loss
    gather             : if True, gather contrastive features across ranks
                         for cross-rank negatives (training). If False, use
                         local features only (validation).
    mask_patch_weights : optional (B, N) soft union-mask weights
    mask_pool_active   : optional (B,) bool gate for masked-pool JEPA

    Returns: (total, jepa, prior, pred, prog, msk_jepa) as scalar tensors.
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

    # 5-way image-image CE on the predictor's class-conditioned ẑ_cur.
    # Global mean over patches (not masked). ``weight=`` uses Cui CBW.
    prog = progression_classification_loss(
        out["pred_progression_patches"].float(),
        out["current_patches_target"].float(),
        prog_cls_idx,
        temperature=PROG_TEMP,
        class_weights=PROG_CLASS_WEIGHTS,
    )

    # Soft-mask pooled JEPA: 1 - cos(pooled ẑ, pooled z) using the union
    # of all findings' filtered masks. Full-grid JEPA above is unchanged.
    if (
        USE_MASKED_POOL_JEPA
        and W_MASK_JEPA > 0
        and mask_patch_weights is not None
        and mask_pool_active is not None
    ):
        msk_jepa = masked_pool_jepa_loss(
            out["pred_current_patches"].float(),
            out["current_patches_target"].float(),
            mask_patch_weights.float(),
            mask_pool_active,
        )
    else:
        msk_jepa = out["pred_current_patches"].new_zeros(())

    total = (
        W_JEPA * jepa
        + W_REPORT_PRIOR * prior
        + W_REPORT_PRED * pred
        + W_PROG * prog
        + W_MASK_JEPA * msk_jepa
    )
    return total, jepa, prior, pred, prog, msk_jepa


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

    for batch in pbar:

        prior = batch["prior_image"].to(DEVICE)
        curr = batch["current_image"].to(DEVICE)

        prior_reports = batch["prior_report"]
        current_reports = batch["current_report"]
        condition_texts = batch["condition_text"]

        prog_prompts = build_progression_prompts(batch["prog_finding"])
        prog_cls_idx = batch["prog_cls_idx"].to(DEVICE)
        mask_w = batch["mask_patch_weights"].to(DEVICE)
        mask_active = batch["mask_pool_active"].to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(
                prior,
                curr,
                prior_reports,
                current_reports,
                condition_texts,
                progression_prompts_flat=prog_prompts,
            )

            loss, jepa_l, prior_l, pred_l, prog_l, msk_l = (
                compute_jepa_losses(
                    out,
                    prog_cls_idx,
                    gather=True,
                    mask_patch_weights=mask_w,
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

        if local_rank == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "jepa": f"{jepa_l.item():.4f}",
                "prog": f"{prog_l.item():.4f}",
                "mskj": f"{msk_l.item():.4f}",
                "mask": f"{mask_active.float().mean().item():.2f}",
                "ema_m": f"{m:.4f}",
                "avg": f"{running_total / running_batches:.4f}",
            })

    if local_rank == 0:
        print(
            f"Train Epoch {epoch} | "
            f"Avg Loss: {running_total / running_batches:.4f}"
        )


    # ============================================================
    # VALIDATION
    # ============================================================
    model.eval()
    val_total = val_jepa = val_prior = val_pred = val_prog = val_msk = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:

            prior = batch["prior_image"].to(DEVICE)
            curr = batch["current_image"].to(DEVICE)

            prior_reports = batch["prior_report"]
            current_reports = batch["current_report"]
            condition_texts = batch["condition_text"]

            prog_prompts = build_progression_prompts(batch["prog_finding"])
            prog_cls_idx = batch["prog_cls_idx"].to(DEVICE)
            mask_w = batch["mask_patch_weights"].to(DEVICE)
            mask_active = batch["mask_pool_active"].to(DEVICE)

            with torch.amp.autocast("cuda"):
                out = model(
                    prior,
                    curr,
                    prior_reports,
                    current_reports,
                    condition_texts,
                    progression_prompts_flat=prog_prompts,
                )

                total, jepa_l, prior_l, pred_l, prog_l, msk_l = (
                    compute_jepa_losses(
                        out,
                        prog_cls_idx,
                        gather=False,
                        mask_patch_weights=mask_w,
                        mask_pool_active=mask_active,
                    )
                )

            val_total += total.item()
            val_jepa += jepa_l.item()
            val_prior += prior_l.item()
            val_pred += pred_l.item()
            val_prog += prog_l.item()
            val_msk += msk_l.item()
            val_batches += 1

    val_total /= max(val_batches, 1)
    val_jepa /= max(val_batches, 1)
    val_prior /= max(val_batches, 1)
    val_pred /= max(val_batches, 1)
    val_prog /= max(val_batches, 1)
    val_msk /= max(val_batches, 1)

    val_total = ddp_reduce(val_total)
    val_jepa = ddp_reduce(val_jepa)
    val_prior = ddp_reduce(val_prior)
    val_pred = ddp_reduce(val_pred)
    val_prog = ddp_reduce(val_prog)
    val_msk = ddp_reduce(val_msk)

    if local_rank == 0:

        print(
            f"Val Epoch {epoch} | "
            f"Total={val_total:.4f} | "
            f"JEPA={val_jepa:.4f} | "
            f"PriorReport={val_prior:.4f} | "
            f"PredReport={val_pred:.4f} | "
            f"Prog={val_prog:.4f} | "
            f"MskJEPA={val_msk:.4f}"
        )

        with open(CSV_LOG, "a") as f:
            f.write(
                f"{epoch},{val_total},{val_jepa},{val_prior},{val_pred},"
                f"{val_prog},{val_msk}\n"
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

dist.destroy_process_group()
