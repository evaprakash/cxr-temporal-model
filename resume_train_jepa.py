# resume_train_jepa.py
#
# DDP training entry point for the JEPA-style temporal CXR model.
#
# Branch defaults (``raddino-image-encoder``):
#   - Image encoder swapped from BioViL-T (ResNet50 + temporal
#     transformer, 128-d joint space) to RAD-DINO-MAIRA-2 (DINOv2-base
#     ViT finetuned on chest X-rays, 768-d hidden). The paired text
#     encoder (BioViL-T CXR-BERT) is instantiated with
#     ``use_projection=False`` so text tokens are also raw 768-d and
#     match the image-patch dim without any projection heads.
#   - Predictor sized to ``d_model=768`` and ``num_patches=1369``
#     (RAD-DINO's 37x37 patch grid at 518x518).
#   - 15 epochs (rather than 50) with per-epoch checkpoint snapshots so
#     the RAD-DINO baseline is directly comparable to the noprog
#     branches.
#   - Batch size lowered to 16 by default because RAD-DINO's much larger
#     feature/patch footprint (768-d vs 128-d, 1369 patches vs 196)
#     roughly 40x's the size of the tensors flowing through the
#     predictor and the local contrastive loss. Bump back to 32 (or
#     higher) if it fits on your GPU.
#
#   - Dataset:  JEPACombinedDataset (silver corpus, paired only)
#   - Model:    TempCXRJEPA (online + EMA + predictor) — unit-sphere
#   - Losses:   JEPA cosine (1 - cos(ẑ_cur, z_cur))
#               + GLoRIA local contrastive (z_prior)
#               + GLoRIA local contrastive (ẑ_cur)
#               + Progression 5-way image-image CE
#   - EMA:      momentum scheduler, target encoder updated after
#               optimizer.step() each iteration
#   - Text condition (predictor input for JEPA loss): ``dynamic`` by
#               default — joined ``label=="dynamic"`` sentences from
#               ``silver_sentences.parquet``. Override via
#               ``CONDITION_MODE=templated`` for the per-finding
#               ``"{Finding} is {progression}."`` template.
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
#   ``F.cross_entropy(mean_patches(cos(ẑ_cur^c, z_cur)) / τ, silver_label)``
#   — the train-time analog of the image-image eval rule in
#   ``eval_progression_jepa.py``.

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
from losses_jepa import jepa_cosine_loss, progression_classification_loss


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

# Which image encoder backs the JEPA model. ``raddino`` (this branch's
# default) uses ``microsoft/rad-dino-maira-2`` — a DINOv2-base ViT
# finetuned on chest X-rays, 768-d hidden, 37x37 patches at 518x518.
# ``biovilt`` reverts to the original BioViL-T ResNet50 + temporal
# transformer with the 128-d joint-space head. Any BioViL-T variant
# (biovil / biovilt / biovilt_finetuned) or RAD-DINO variant
# (raddino / raddino_finetuned) accepted by ``TempCXRJEPA`` works here.
MODEL_MODE = os.environ.get("MODEL_MODE", "raddino")

# Namespace checkpoints / logs by (model_mode, condition_mode) so the
# BioViL-T and RAD-DINO variants never clobber each other's checkpoints
# and so legacy ``checkpoints_jepa/`` / ``logs/`` dirs from older runs
# are left untouched as an archive.
_RUN_SUFFIX = f"{MODEL_MODE}_{CONDITION_MODE}"
_DEFAULT_CKPT_DIR = os.path.join(_HERE, f"checkpoints_jepa_{_RUN_SUFFIX}")
_DEFAULT_LOG_DIR = os.path.join(_HERE, f"logs_{_RUN_SUFFIX}")
CHECKPOINT_DIR = os.environ.get("JEPA_CHECKPOINT_DIR", _DEFAULT_CKPT_DIR)
LOG_DIR = os.environ.get("JEPA_LOG_DIR", _DEFAULT_LOG_DIR)

IMAGE_ROOTS = {
    "mimic": "/home/evaprakash/all_data/mimic",
    "chexpert": "/home/evaprakash/all_data/chexpert/train",
    "rexgradient": "/home/evaprakash/all_data/rexgradient/deid_png",
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_LOG = os.path.join(LOG_DIR, "val_metrics_jepa.csv")


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
# RAD-DINO's 768-d, 1369-patch features (vs BioViL-T's 128-d, 196)
# roughly 40x the size of the tensors flowing through the predictor
# and the local contrastive loss, so BATCH_SIZE=16 (per GPU) is the
# safe default on 80 GB A100s. Bump if it fits — 32 does fit on some
# 80 GB configurations but leaves less headroom for the 4th-loss
# second predictor pass (which scales as B*C=5B in the batch dim).
BATCH_SIZE = 16
EPOCHS = 15
WARMUP_RATIO = 0.03

# Checkpoint schedule: save every epoch so we get 15 comparable
# snapshots for the RAD-DINO baseline (matches "save every epoch"
# request for this branch); best.pt still gets written whenever val
# total improves.
SAVE_EVERY_N_EPOCHS = 1

# Loss weights (matching the smoke-test defaults in the old jepa.py)
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

# Stratified train/val split when the studies parquet has no 'split'
# column. Both datasets read/write the same cached splits CSV
# (DEFAULT_SPLITS_FILE inside dataset_combined_jepa.py), so the val set
# is identical across train/val DataLoaders and across re-runs.
VAL_FRACTION = 0.1
SPLIT_SEED = 42


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

if local_rank == 0:
    print(f"[train] model_mode    ={MODEL_MODE}")
    print(f"[train] condition_mode={CONDITION_MODE}")
    print(f"[train] batch_size    ={BATCH_SIZE}  epochs={EPOCHS}")
    print(f"[train] save every N  ={SAVE_EVERY_N_EPOCHS} epoch(s)")
    print(f"[train] checkpoint dir: {CHECKPOINT_DIR}")
    print(f"[train] log dir:        {LOG_DIR}")

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
model = TempCXRJEPA(mode=MODEL_MODE).to(DEVICE)
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
            "val_prog\n"
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
def compute_jepa_losses(out, prog_cls_idx, gather: bool):
    """
    out          : dict returned by TempCXRJEPA.forward
    prog_cls_idx : (B,) long tensor — silver class for the 4th loss
    gather       : if True, gather contrastive features across ranks
                   for cross-rank negatives (training). If False, use
                   local features only (validation).

    Returns: (total, jepa, prior, pred, prog) as scalar tensors.
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
    # No cross-rank gather here: the candidates and the target are both
    # per-pair quantities, so other ranks' samples wouldn't add useful
    # negatives — the "negative classes" are already inside each pair's
    # own 5-way softmax. Cast to fp32 so the CE on the cosine logits
    # doesn't lose precision at small (1 - cos) residuals.
    prog = progression_classification_loss(
        out["pred_progression_patches"].float(),
        out["current_patches_target"].float(),
        prog_cls_idx,
        temperature=PROG_TEMP,
    )

    total = (
        W_JEPA * jepa
        + W_REPORT_PRIOR * prior
        + W_REPORT_PRED * pred
        + W_PROG * prog
    )
    return total, jepa, prior, pred, prog


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

            loss, jepa_l, prior_l, pred_l, prog_l = compute_jepa_losses(
                out, prog_cls_idx, gather=True
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
    val_total = val_jepa = val_prior = val_pred = val_prog = 0.0
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

            with torch.amp.autocast("cuda"):
                out = model(
                    prior,
                    curr,
                    prior_reports,
                    current_reports,
                    condition_texts,
                    progression_prompts_flat=prog_prompts,
                )

                total, jepa_l, prior_l, pred_l, prog_l = (
                    compute_jepa_losses(
                        out, prog_cls_idx, gather=False
                    )
                )

            val_total += total.item()
            val_jepa += jepa_l.item()
            val_prior += prior_l.item()
            val_pred += pred_l.item()
            val_prog += prog_l.item()
            val_batches += 1

    val_total /= max(val_batches, 1)
    val_jepa /= max(val_batches, 1)
    val_prior /= max(val_batches, 1)
    val_pred /= max(val_batches, 1)
    val_prog /= max(val_batches, 1)

    val_total = ddp_reduce(val_total)
    val_jepa = ddp_reduce(val_jepa)
    val_prior = ddp_reduce(val_prior)
    val_pred = ddp_reduce(val_pred)
    val_prog = ddp_reduce(val_prog)

    if local_rank == 0:

        print(
            f"Val Epoch {epoch} | "
            f"Total={val_total:.4f} | "
            f"JEPA={val_jepa:.4f} | "
            f"PriorReport={val_prior:.4f} | "
            f"PredReport={val_pred:.4f} | "
            f"Prog={val_prog:.4f}"
        )

        with open(CSV_LOG, "a") as f:
            f.write(
                f"{epoch},{val_total},{val_jepa},{val_prior},{val_pred},"
                f"{val_prog}\n"
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
