# resume_train.py

import os
import glob
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset_combined import BioViLTCombinedDataset, biovilt_collate_fn
from tempcxr.modules.tempcxr_model import TempCXR
from losses import global_contrastive_loss, local_contrastive_loss, mlm_loss


# ============================================================
# PATHS
# ============================================================
CHECKPOINT_DIR = "/scratch/m000081/eprakash/temporal/checkpoints"
CSV_DIR = "/scratch/m000081/eprakash/temporal/model/biovilt"
LOG_DIR = "/scratch/m000081/eprakash/temporal/logs"

IMAGE_ROOTS = {
    "mimic": "/scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0/files",
    "chexpert": "/scratch/m000081/yabin/datasets/chestxpert/CheXpert-v1.0",
    "rexgradient": "/scratch/m000081/chong/data/ReXGradient-160K",
}

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CSV_LOG = os.path.join(LOG_DIR, "val_metrics.csv")


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
# PAPER-FAITHFUL MIXED DISTRIBUTED BATCH SAMPLER
# ============================================================
class DistributedMixedBatchSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, batch_size, shuffle=True):
        self.single = dataset.single_indices
        self.multi = dataset.multi_indices
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.global_batch = batch_size * self.world_size
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        total = len(self.single) + len(self.multi)
        return total // self.global_batch

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.epoch)

        single = torch.tensor(self.single)
        multi = torch.tensor(self.multi)

        if self.shuffle:
            single = single[torch.randperm(len(single), generator=g)]
            multi = multi[torch.randperm(len(multi), generator=g)]

        single = single.tolist()
        multi = multi.tolist()

        sp = 0
        mp = 0
        total = len(single) + len(multi)
        p_single = len(single) / total

        while True:

            if sp + self.global_batch > len(single) and \
               mp + self.global_batch > len(multi):
                break

            if sp + self.global_batch > len(single):
                choose_single = False
            elif mp + self.global_batch > len(multi):
                choose_single = True
            else:
                choose_single = torch.rand(1, generator=g).item() < p_single

            if choose_single:
                batch = single[sp: sp + self.global_batch]
                sp += self.global_batch
            else:
                batch = multi[mp: mp + self.global_batch]
                mp += self.global_batch

            start = self.rank * self.batch_size
            end = start + self.batch_size
            yield batch[start:end]


# ============================================================
# HYPERPARAMETERS
# ============================================================
LR = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
EPOCHS = 50
WARMUP_RATIO = 0.03

W_GLOBAL = 1.0
W_LOCAL = 0.5
W_MLM = 1.0


# ============================================================
# DATASETS
# ============================================================
train_dataset = BioViLTCombinedDataset(
    csv_path=os.path.join(CSV_DIR, "biovilt_pretrain_alldata_train_imagelevel.csv"),
    image_roots=IMAGE_ROOTS,
    split="train",
    train=True,
)

val_dataset = BioViLTCombinedDataset(
    csv_path=os.path.join(CSV_DIR, "biovilt_pretrain_alldata_valid_imagelevel.csv"),
    image_roots=IMAGE_ROOTS,
    split="val",
    train=False,
)

train_sampler = DistributedMixedBatchSampler(train_dataset, BATCH_SIZE, shuffle=True)
val_sampler = DistributedMixedBatchSampler(val_dataset, BATCH_SIZE, shuffle=False)

train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=biovilt_collate_fn,
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    num_workers=8,
    pin_memory=True,
    collate_fn=biovilt_collate_fn,
)


# ============================================================
# MODEL
# ============================================================
model = TempCXR().to(DEVICE)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

num_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(WARMUP_RATIO * num_steps),
    num_training_steps=num_steps,
)

scaler = torch.amp.GradScaler("cuda")

start_epoch = 1
best_val_loss = float("inf")


# ============================================================
# RESUME
# ============================================================
if args.resume is None:
    checkpoints = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "epoch_*.pt")))
    if checkpoints:
        args.resume = checkpoints[-1]

if args.resume is not None:
    checkpoint = torch.load(args.resume, map_location=DEVICE)
    model.module.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    if local_rank == 0:
        print(f"Resumed from {args.resume}")


# ============================================================
# CSV HEADER
# ============================================================
if local_rank == 0 and not os.path.exists(CSV_LOG):
    with open(CSV_LOG, "w") as f:
        f.write("epoch,val_total,val_global,val_local,val_mlm\n")


# ============================================================
# TRAIN LOOP
# ============================================================
for epoch in range(start_epoch, EPOCHS + 1):

    train_sampler.set_epoch(epoch)
    val_sampler.set_epoch(epoch)

    model.train()
    running_total = 0
    running_batches = 0

    if local_rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=120)
    else:
        pbar = train_loader

    for batch in pbar:

        curr = batch["current_image"].to(DEVICE)
        prev = batch["prior_image"]
        texts = batch["text"]

        if prev is not None:
            prev = prev.to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(curr, prev, texts)

            img_global_all = gather_with_grad(outputs["img_global"])
            txt_global_all = gather_with_grad(outputs["txt_global"])
            img_patches_all = gather_with_grad(outputs["img_patches"])
            txt_local_all = gather_with_grad(outputs["txt_local"])
            token_mask_all = gather_with_grad(outputs["token_mask"].float()).bool()

            loss_g = global_contrastive_loss(img_global_all, txt_global_all)
            loss_l = local_contrastive_loss(img_patches_all, txt_local_all, token_mask_all)
            loss_m = mlm_loss(outputs["mlm_logits"], outputs["mlm_labels"])

            loss = W_GLOBAL*loss_g + W_LOCAL*loss_l + W_MLM*loss_m

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_total += loss.item()
        running_batches += 1

        if local_rank == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{running_total/running_batches:.4f}"
            })

    if local_rank == 0:
        print(f"Train Epoch {epoch} | Avg Loss: {running_total/running_batches:.4f}")


    # ============================================================
    # VALIDATION
    # ============================================================
    model.eval()
    val_total = val_g = val_l = val_m = 0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:

            curr = batch["current_image"].to(DEVICE)
            prev = batch["prior_image"]
            texts = batch["text"]

            if prev is not None:
                prev = prev.to(DEVICE)

            with torch.amp.autocast("cuda"):
                outputs = model(curr, prev, texts)

                loss_g = global_contrastive_loss(outputs["img_global"], outputs["txt_global"])
                loss_l = local_contrastive_loss(outputs["img_patches"], outputs["txt_local"], outputs["token_mask"])
                loss_m = mlm_loss(outputs["mlm_logits"], outputs["mlm_labels"])

                total = W_GLOBAL*loss_g + W_LOCAL*loss_l + W_MLM*loss_m

            val_total += total.item()
            val_g += loss_g.item()
            val_l += loss_l.item()
            val_m += loss_m.item()
            val_batches += 1

    val_total /= val_batches
    val_g /= val_batches
    val_l /= val_batches
    val_m /= val_batches

    val_total = ddp_reduce(val_total)
    val_g = ddp_reduce(val_g)
    val_l = ddp_reduce(val_l)
    val_m = ddp_reduce(val_m)

    if local_rank == 0:

        print(
            f"Val Epoch {epoch} | "
            f"Total={val_total:.4f} | "
            f"Global={val_g:.4f} | "
            f"Local={val_l:.4f} | "
            f"MLM={val_m:.4f}"
        )

        with open(CSV_LOG, "a") as f:
            f.write(f"{epoch},{val_total},{val_g},{val_l},{val_m}\n")

        ckpt = {
            "epoch": epoch,
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
        }

        torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt"))

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
            print("🔥 Saved new BEST checkpoint")

dist.destroy_process_group()

