# resume_train_jepa.py
#
# DDP training entry point for the JEPA-style temporal CXR model.
#
#   - Dataset:  JEPACombinedDataset (silver corpus, paired only)
#   - Model:    TempCXRJEPA (online + EMA + predictor)
#   - Losses:   JEPA Smooth L1
#               + GLoRIA local contrastive (z_prior)
#               + GLoRIA local contrastive (ẑ_cur)
#               + per-finding 5-way progression CE on ẑ_cur ↔ class prompts
#   - EMA:      momentum scheduler, target encoder updated after
#               optimizer.step() each iteration

import os
import glob
import argparse

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
from losses import local_contrastive_loss, progression_classification_loss
from losses_jepa import jepa_smooth_l1_loss


# ============================================================
# PATHS
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))

# Which text condition the predictor sees. ``dynamic`` (the default)
# uses the joined ``label=="dynamic"`` sentences from
# ``silver_sentences.parquet`` — free-form report text. ``templated``
# builds the condition from ``silver_findings.parquet`` as capitalized
# ``"{Finding} is {progression}."`` clauses joined with a space and
# shuffled per-sample at train time.  The 4th progression loss
# (introduced alongside dynamic-as-default) builds its own per-finding
# class prompts independently of this knob, so the per-finding silver
# supervision is available in both modes.
CONDITION_MODE = os.environ.get("CONDITION_MODE", "dynamic")

# Namespace checkpoints / logs by condition mode in the default paths
# so the two modes never clobber each other's checkpoints. The
# ``dynamic`` default stays at ``checkpoints_jepa/`` / ``logs/`` for
# backwards compatibility with prior runs; other modes get their own
# dirs (e.g. ``checkpoints_jepa_templated/`` / ``logs_templated/``).
_DEFAULT_CKPT_DIR = (
    os.path.join(_HERE, "checkpoints_jepa")
    if CONDITION_MODE == "dynamic"
    else os.path.join(_HERE, f"checkpoints_jepa_{CONDITION_MODE}")
)
_DEFAULT_LOG_DIR = (
    os.path.join(_HERE, "logs")
    if CONDITION_MODE == "dynamic"
    else os.path.join(_HERE, f"logs_{CONDITION_MODE}")
)
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
BATCH_SIZE = 32
EPOCHS = 50
WARMUP_RATIO = 0.03

# Checkpoint schedule: always save epoch 1 + every SAVE_EVERY_N_EPOCHS,
# plus best.pt whenever val total improves.
SAVE_EVERY_N_EPOCHS = 5

# Loss weights (matching the smoke-test defaults in the old jepa.py)
W_JEPA = 1.0
W_REPORT_PRIOR = 0.1
W_REPORT_PRED = 0.1

# Progression loss: per-finding 5-way softmax-CE on cosine(ẑ_cur, class
# prompt). τ = 0.1 maps the cosine range [-1, 1] onto logits in [-10,
# 10] which is peaky enough to drive gradients but not so peaky that
# wrong-class logits vanish. Start the weight at the same scale as the
# two contrastive heads; sweep if it dominates or under-shoots.
W_PROG = 0.1
PROG_TEMP = 0.1
N_CLS = len(CLS_ORDER)
PROG_TEMPLATE = "{} is {}."

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
    print(f"[train] condition_mode={CONDITION_MODE}")
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
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=jepa_collate_fn,
    drop_last=True,
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
            "epoch,val_total,val_jepa,val_report_prior,val_report_pred,val_prog\n"
        )


# ============================================================
# PROGRESSION-LOSS BATCH BUILDER
# ============================================================
def build_progression_inputs(findings_per_pair, silver_per_pair, device):
    """Flatten per-pair (findings, silver_cls_idx) lists into the
    finding-major prompt order the progression loss consumes.

    Returns
    -------
    class_prompts : list[str]
        Length = sum_pair (n_findings * N_CLS). Order is
        ``[pair_0_finding_0_cls_0 … cls_{N_CLS-1}, pair_0_finding_1_cls_0, …]``
        so the loss can reshape to ``(F_total, N_CLS)`` directly.
    pair_idx_per_finding : LongTensor (F_total,)
    silver_per_finding   : LongTensor (F_total,)

    F_total can be 0 if no pair in the batch has any usable finding;
    the caller should short-circuit in that case.
    """
    class_prompts = []
    pair_idx_per_finding = []
    silver_per_finding = []
    for pair_i, (findings, silvers) in enumerate(
        zip(findings_per_pair, silver_per_pair)
    ):
        for finding, silver in zip(findings, silvers):
            f_cap = finding[:1].upper() + finding[1:] if finding else finding
            for cls in CLS_ORDER:
                class_prompts.append(PROG_TEMPLATE.format(f_cap, cls))
            pair_idx_per_finding.append(pair_i)
            silver_per_finding.append(int(silver))

    pair_idx_t = torch.tensor(
        pair_idx_per_finding, device=device, dtype=torch.long
    )
    silver_t = torch.tensor(
        silver_per_finding, device=device, dtype=torch.long
    )
    return class_prompts, pair_idx_t, silver_t


# ============================================================
# LOSS COMPUTATION (shared by train + val)
# ============================================================
def compute_jepa_losses(out, gather: bool, prog_inputs):
    """
    out         : dict returned by TempCXRJEPA.forward
    gather      : if True, gather contrastive features across ranks
                  for cross-rank negatives (training). If False, use
                  local features only (validation).
    prog_inputs : tuple (pair_idx_per_finding, silver_per_finding,
                  batch_size) — the index tensors produced by
                  ``build_progression_inputs``. Used here so the
                  progression loss can be computed on the *local*
                  ẑ_cur (no cross-rank gather — the loss is per-pair).

    Returns: (total, jepa, prior, pred, prog) as scalar tensors.
    """

    # JEPA loss is per-element MSE-style; cross-rank gathering doesn't
    # add useful negatives, so we always compute it on local features.
    # Cast to fp32 so bf16's low precision on small residuals doesn't
    # bite once both pred and target are in LayerNorm scale.
    jepa = jepa_smooth_l1_loss(
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

    # Progression loss is per-pair (no cross-rank info needed) so we
    # always use the LOCAL ẑ_cur and class prompts. This also keeps the
    # softmax-CE numerically simple (no need to track which rank a
    # finding belongs to after a gather).
    pair_idx, silver, B_local = prog_inputs
    class_local = out["class_prompts_local"]
    class_mask = out["class_prompts_mask"]
    if class_local is None or class_local.numel() == 0:
        prog = out["pred_current_patches"].new_zeros(())
    else:
        prog = progression_classification_loss(
            pred_patches=out["pred_current_patches"],
            class_prompts_local=class_local,
            class_prompts_mask=class_mask,
            pair_idx_per_finding=pair_idx,
            silver_per_finding=silver,
            batch_size=B_local,
            n_classes=N_CLS,
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

        # Flatten per-pair (findings, silver) into the finding-major prompt
        # order the prog loss expects. Done once on the host before the
        # forward so the model can encode the prompts inside its forward
        # (necessary for DDP gradient sync on the text encoder).
        class_prompts, pair_idx, silver = build_progression_inputs(
            batch["findings"],
            batch["progression_cls_idx"],
            DEVICE,
        )
        prog_inputs = (pair_idx, silver, prior.size(0))

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(
                prior,
                curr,
                prior_reports,
                current_reports,
                condition_texts,
                class_prompts=class_prompts,
            )

            loss, jepa_l, prior_l, pred_l, prog_l = compute_jepa_losses(
                out, gather=True, prog_inputs=prog_inputs
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

            class_prompts, pair_idx, silver = build_progression_inputs(
                batch["findings"],
                batch["progression_cls_idx"],
                DEVICE,
            )
            prog_inputs = (pair_idx, silver, prior.size(0))

            with torch.amp.autocast("cuda"):
                out = model(
                    prior,
                    curr,
                    prior_reports,
                    current_reports,
                    condition_texts,
                    class_prompts=class_prompts,
                )

                total, jepa_l, prior_l, pred_l, prog_l = compute_jepa_losses(
                    out, gather=False, prog_inputs=prog_inputs
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
                f"{epoch},{val_total},{val_jepa},{val_prior},{val_pred},{val_prog}\n"
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
