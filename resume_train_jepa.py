# resume_train_jepa.py
#
# DDP training entry point for the JEPA-style temporal CXR model.
#
# Branch defaults (``prog-loss-all-findings``):
#   Same setup as main, with one change to the 4th (progression-
#   classification) loss: every optimizer step processes *all* silver
#   findings for *all* pairs in the batch, not just one randomly
#   sampled finding per pair per epoch. Concretely, if pair ``p`` has
#   ``F_p`` silver findings the prog pass builds ``F_p * 5`` prompts
#   for it (all findings x all 5 progression classes), and the total
#   prog batch across the whole mini-batch is
#   ``F_total = sum_p F_p`` findings x ``5`` classes. The predictor
#   runs once over that whole flattened batch, one CE per finding is
#   averaged into ``prog_loss``, then a single ``optimizer.step()``
#   fires. Result: no finding gets ``.step()``-ed on without every
#   other finding for the same pair having contributed to the same
#   gradient update.
#
#   - Dataset:  JEPACombinedDataset (silver corpus, paired only)
#   - Model:    TempCXRJEPA (online + EMA + predictor) — unit-sphere
#   - Losses:   JEPA cosine (1 - cos(ẑ_cur, z_cur))
#               + GLoRIA local contrastive (z_prior)
#               + GLoRIA local contrastive (ẑ_cur)
#               + Progression 5-way image-image CE (over ALL findings
#                 for every pair, not a per-epoch random pick)
#   - EMA:      momentum scheduler, target encoder updated after
#               optimizer.step() each iteration
#   - Text condition (predictor input for JEPA loss): ``dynamic`` by
#               default — joined ``label=="dynamic"`` sentences from
#               ``silver_sentences.parquet``. Override via
#               ``CONDITION_MODE=templated`` for the per-finding
#               ``"{Finding} is {progression}."`` template.
#
# Progression loss (the "4th loss") — all-findings variant:
#   The trainer walks every ``(pair, finding)`` tuple in the batch and
#   emits 5 templated prompts ``"{finding} is {class}."`` per tuple,
#   plus a per-finding pair index (which source pair it came from) and
#   a per-finding silver label. The model runs the predictor on
#   ``prior_patches[pair_idx].repeat_interleave(5, dim=0)`` conditioned
#   on the ``F_total * 5`` prompt tokens, producing
#   ``pred_progression_patches`` of shape ``(F_total, 5, N, D)``. The
#   trainer gathers the EMA target the same way
#   (``current_patches_target[pair_idx]`` -> ``(F_total, N, D)``) and
#   calls ``progression_classification_loss`` with the ``F_total``
#   silver labels, giving one mean-over-findings CE that lands in
#   ``val_prog`` / ``prog_l``.
#
#   This differs from ``main`` in two places only: the trainer builds
#   an all-findings prompt list + pair-idx tensor instead of a
#   per-epoch random pick, and ``TempCXRJEPA.forward`` gathers
#   ``z_prior`` per finding when ``progression_prior_pair_idx`` is
#   provided. Everything else (loss weights, LR, EMA schedule, JEPA
#   and local-CL branches, dataset filtering) matches main.

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

# Always namespace checkpoints / logs by condition mode so the two
# modes never clobber each other's checkpoints and so legacy
# ``checkpoints_jepa/`` / ``logs/`` dirs from older (pre-3-loss, pre-
# softmax-fix) runs are left untouched as an archive.
_DEFAULT_CKPT_DIR = os.path.join(_HERE, f"checkpoints_jepa_{CONDITION_MODE}")
_DEFAULT_LOG_DIR = os.path.join(_HERE, f"logs_{CONDITION_MODE}")
CHECKPOINT_DIR = os.environ.get("JEPA_CHECKPOINT_DIR", _DEFAULT_CKPT_DIR)
LOG_DIR = os.environ.get("JEPA_LOG_DIR", _DEFAULT_LOG_DIR)

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
            "val_prog\n"
        )


# ============================================================
# PROGRESSION-PROMPT HELPER (all-findings variant)
# ============================================================
def build_progression_prompts_all(batch_findings, batch_prog_cls_idx):
    """Flatten every ``(pair, finding)`` in the batch into a prog prompt list.

    Layout is **finding-major, class-minor**: for each pair ``p`` (in
    input order) and each of pair ``p``'s findings (in the order they
    appear in ``batch_findings[p]``), emit ``N_CLS`` templated prompts
    ``"{Finding} is {class}."``. Also emit a per-finding pair index
    (which source pair index the finding came from) and the silver
    progression class for that finding.

    Returns
    -------
    prompts : list[str]
        Length ``F_total * N_CLS`` where ``F_total = sum_p F_p``.
    pair_idx : list[int]
        Length ``F_total``. ``pair_idx[i]`` is the batch position of the
        pair that finding ``i`` belongs to. Consumed by
        ``TempCXRJEPA.forward`` via ``progression_prior_pair_idx`` to
        gather ``z_prior`` and (in the trainer) ``z_cur`` per finding.
    silver_labels : list[int]
        Length ``F_total``. ``silver_labels[i]`` is the silver
        progression class index (into ``CLS_ORDER``) for finding ``i``,
        used as the CE target in
        ``progression_classification_loss``.

    Pairs with no findings are skipped silently. The dataset already
    drops those rows in ``__init__``, so in practice every pair
    contributes at least one finding.
    """
    prompts = []
    pair_idx = []
    silver_labels = []
    for p, (findings, cls_ids) in enumerate(
        zip(batch_findings, batch_prog_cls_idx)
    ):
        for finding, cls in zip(findings, cls_ids):
            if not finding:
                # Defensive: dataset filter should have removed these,
                # but skip rather than emit an empty-prefix prompt that
                # would train the model to associate "" with a real
                # silver label.
                continue
            f_cap = finding[:1].upper() + finding[1:]
            for prog_cls in CLS_ORDER:
                prompts.append(PROG_TEMPLATE.format(f_cap, prog_cls))
            pair_idx.append(p)
            silver_labels.append(int(cls))
    return prompts, pair_idx, silver_labels


# ============================================================
# LOSS COMPUTATION (shared by train + val)
# ============================================================
def compute_jepa_losses(out, prog_pair_idx, prog_silver_labels, gather: bool):
    """
    out                : dict returned by TempCXRJEPA.forward
    prog_pair_idx      : (F_total,) long tensor — for each finding
                         contributing to the prog loss, the batch
                         position of the pair it came from. Used to
                         gather the EMA target patches per finding so
                         they line up with ``pred_progression_patches``.
    prog_silver_labels : (F_total,) long tensor — silver progression
                         class index per finding.
    gather             : if True, gather contrastive features across
                         ranks for cross-rank negatives (training). If
                         False, use local features only (validation).

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
    #
    # All-findings variant: the model returns
    # ``pred_progression_patches`` of shape ``(F_total, C, N, D)`` (one
    # ẑ_cur^c per (pair, finding) pair, not per pair). We gather the
    # EMA target the same way — ``current_patches_target[pair_idx]``
    # duplicates the target across findings sharing a pair — so
    # ``progression_classification_loss`` sees F_total-shaped tensors
    # everywhere and reduces to a plain mean-CE across findings.
    if prog_pair_idx.numel() == 0:
        # Extreme edge case: the batch had zero valid findings. Skip
        # the prog term so we don't blow up on an empty batch dim.
        prog = out["pred_current_patches"].new_zeros(())
    else:
        target_per_finding = out["current_patches_target"].index_select(
            0, prog_pair_idx
        )
        prog = progression_classification_loss(
            out["pred_progression_patches"].float(),
            target_per_finding.float(),
            prog_silver_labels,
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

        prog_prompts, prog_pair_idx_list, prog_silver_list = (
            build_progression_prompts_all(
                batch["findings"], batch["progression_cls_idx"]
            )
        )
        prog_pair_idx = torch.tensor(
            prog_pair_idx_list, dtype=torch.long, device=DEVICE
        )
        prog_silver_labels = torch.tensor(
            prog_silver_list, dtype=torch.long, device=DEVICE
        )

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            out = model(
                prior,
                curr,
                prior_reports,
                current_reports,
                condition_texts,
                progression_prompts_flat=prog_prompts,
                progression_prior_pair_idx=prog_pair_idx,
            )

            loss, jepa_l, prior_l, pred_l, prog_l = compute_jepa_losses(
                out, prog_pair_idx, prog_silver_labels, gather=True
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
                "F_tot": int(prog_pair_idx.numel()),
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

            prog_prompts, prog_pair_idx_list, prog_silver_list = (
                build_progression_prompts_all(
                    batch["findings"], batch["progression_cls_idx"]
                )
            )
            prog_pair_idx = torch.tensor(
                prog_pair_idx_list, dtype=torch.long, device=DEVICE
            )
            prog_silver_labels = torch.tensor(
                prog_silver_list, dtype=torch.long, device=DEVICE
            )

            with torch.amp.autocast("cuda"):
                out = model(
                    prior,
                    curr,
                    prior_reports,
                    current_reports,
                    condition_texts,
                    progression_prompts_flat=prog_prompts,
                    progression_prior_pair_idx=prog_pair_idx,
                )

                total, jepa_l, prior_l, pred_l, prog_l = (
                    compute_jepa_losses(
                        out, prog_pair_idx, prog_silver_labels,
                        gather=False,
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
