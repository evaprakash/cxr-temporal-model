"""DDP trainer for laterality 5-way VL-JEPA (side in the target only).

Query ``What is the progression of {finding}?`` as in the 5-way run. Sided
findings are scored against the 15 ``{Side} {finding} is {class}.``
sentences, unsided ones against the five ``{Finding} is {class}.``; targets
come from the original BioViL-T text encoder (frozen, no LoRA). Loss is
class-weighted InfoNCE over each row's own candidates (see laterality.py).

After every epoch rank 0 reports silver val progression accuracy (side
summed out), side accuracy on sided findings (class summed out), and gold
set-match with the side summed out.

    torchrun --nproc_per_node=4 -m vljepa.train_laterality
    bash vljepa/run_ccw.sh lat_train
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from . import _root  # noqa: F401

from progression_classify import DEFAULT_GOLD_PARQUET, discover_gold_image_roots

from .cluster_paths import gold_image_dirs, image_roots as cluster_image_roots
from .laterality import (
    LATERALITY_PATH,
    SIDES,
    LateralityFindingDataset,
    LateralityVLJEPA,
    TargetBank,
    laterality_collate_fn,
    laterality_infonce_loss,
    readouts,
)
from .model import llama_hub_cache, resolve_llama_local
from .prompts import QUERY_TEMPLATE

# ============================================================
# HYPERPARAMETERS (same as the 5-way run)
# ============================================================
LR = 2e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
EPOCHS = 50
WARMUP_RATIO = 0.03
TEMPERATURE = 0.07
CBW_BETA = 0.99999
VAL_FRACTION = 0.1
SPLIT_SEED = 42
N_LLAMA_LAYERS = 8
LLAMA_NAME = os.environ.get("VLJEPA_LLAMA_NAME", "meta-llama/Llama-3.2-1B")
llama_hub_cache()
LLAMA_LOCAL = resolve_llama_local()
IMAGE_MODE = "biovilt"

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IMAGE_ROOTS = cluster_image_roots()
CHECKPOINT_DIR = os.environ.get(
    "VLJEPA_CHECKPOINT_DIR", os.path.join(_ROOT, "checkpoints_vljepa_laterality"))
LOG_DIR = os.environ.get("VLJEPA_LOG_DIR", os.path.join(_ROOT, "logs_vljepa_laterality"))
CSV_LOG = os.path.join(LOG_DIR, "val_metrics_vljepa_laterality.csv")


def seed_dataloader_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, torch.device(f"cuda:{local_rank}"), dist.get_world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return 0, device, 1


def ddp_sum(values, device, world_size):
    t = torch.tensor(values, device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


def cui_class_weights(counts: torch.Tensor, beta: float) -> torch.Tensor:
    counts = counts.to(dtype=torch.float64)
    effective = (1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64),
                                 counts.clamp_min(1))) / (1.0 - beta)
    weights = 1.0 / effective
    return (weights * len(counts) / weights.sum()).float()


def batch_step(net, bank, batch, device, class_weights):
    """``net`` must be the DDP wrapper when training, or gradients are not
    averaged across GPUs. Returns the loss and summed counts for logging."""
    prior = batch["prior_image"].to(device, non_blocking=True)
    current = batch["current_image"].to(device, non_blocking=True)
    out = net(prior, current, batch["query_text"])
    loss, logits = laterality_infonce_loss(
        out["pred"], batch["target_texts"], batch["label"], batch["cls_idx"], bank,
        temperature=TEMPERATURE, class_weights=class_weights)
    cls_pred, side_pred = readouts(logits.detach().cpu(), batch["side"])
    sided = torch.tensor([s is not None for s in batch["side"]])
    true_side = torch.tensor([SIDES.index(s) if s else -1 for s in batch["side"]])
    b = len(batch["side"])
    stats = [
        float(loss.item()) * b,
        b,
        int((cls_pred == batch["cls_idx"]).sum()),
        int((logits.detach().argmax(dim=1).cpu() == batch["label"]).sum()),
        int(sided.sum()),
        int(((side_pred == true_side) & sided).sum()),
    ]
    return loss, stats


def summarize(stats):
    loss, n, cls_ok, exact_ok, n_sided, side_ok = stats
    return {"loss": loss / max(n, 1), "cls_acc": cls_ok / max(n, 1),
            "exact_acc": exact_ok / max(n, 1), "side_acc": side_ok / max(n_sided, 1),
            "n": int(n), "n_sided": int(n_sided)}


def train_one_epoch(model, bank, loader, optimizer, scheduler, device, class_weights, epoch, rank):
    model.train()
    totals = [0.0] * 6
    pbar = tqdm(loader, desc=f"laterality train ep{epoch}", disable=rank != 0)
    for batch in pbar:
        loss, stats = batch_step(model, bank, batch, device, class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        totals = [a + b for a, b in zip(totals, stats)]
        if rank == 0:
            s = summarize(totals)
            pbar.set_postfix(loss=f"{s['loss']:.4f}", cls=f"{s['cls_acc']:.3f}",
                             side=f"{s['side_acc']:.3f}")
    return totals


@torch.no_grad()
def run_val(raw_model, bank, loader, device, class_weights, epoch, rank):
    raw_model.eval()
    totals = [0.0] * 6
    for batch in tqdm(loader, desc=f"laterality val ep{epoch}", disable=rank != 0):
        _, stats = batch_step(raw_model, bank, batch, device, class_weights)
        totals = [a + b for a, b in zip(totals, stats)]
    return totals


def latest_ckpt(ckpt_dir: str):
    paths = glob.glob(os.path.join(ckpt_dir, "epoch_*.pt"))
    if not paths:
        return None
    return max(paths, key=lambda p: int(os.path.basename(p)[6:-3]) if
               os.path.basename(p)[6:-3].isdigit() else -1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", default=None)
    p.add_argument("--skip-gold", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    local_rank, device, world_size = setup_ddp()
    rank = dist.get_rank() if world_size > 1 else 0
    epochs = args.epochs or EPOCHS
    batch_size = args.batch_size or BATCH_SIZE

    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        print("[laterality] 5-way VL-JEPA with the side in the target")
        print(f"[laterality]   query  = {QUERY_TEMPLATE} (Llama tok+embed)")
        print("[laterality]   target = '{Side} {finding} is {class}.' x 15 for sided findings, "
              "'{Finding} is {class}.' x 5 otherwise")
        print("[laterality]   text   = original BioViL-T, frozen, no LoRA")
        print(f"[laterality]   sides  = {LATERALITY_PATH}")
        print(f"[laterality]   loss   = class-weighted InfoNCE τ={TEMPERATURE}, CBW β={CBW_BETA}")
        print(f"[laterality]   image  = BioViL-T pair, frozen; pred = Llama last {N_LLAMA_LAYERS}")
        print(f"[laterality]   ckpt   = {CHECKPOINT_DIR}")
        print(f"[laterality]   logs   = {LOG_DIR}")
        print(f"[laterality]   images = {IMAGE_ROOTS}")

    ds_kw = dict(val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED)
    train_ds = LateralityFindingDataset(IMAGE_ROOTS, split="train", train=True, **ds_kw)
    val_ds = LateralityFindingDataset(IMAGE_ROOTS, split="val", train=False, **ds_kw)
    counts = train_ds.class_counts()
    class_weights = cui_class_weights(counts, CBW_BETA).to(device)
    val_sided = [s for s in val_ds.sides if s is not None]
    majority = max(SIDES, key=val_sided.count) if val_sided else None
    side_baseline = val_sided.count(majority) / max(len(val_sided), 1)
    if rank == 0:
        print(f"[laterality] train class counts: {counts.tolist()}")
        print(f"[laterality] CBW weights: {[round(w, 4) for w in class_weights.tolist()]}")
        print(f"[laterality] val side accuracy from always guessing {majority}: {side_baseline:.3f}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    loader_kw = dict(batch_size=batch_size, num_workers=args.num_workers, pin_memory=True,
                     collate_fn=laterality_collate_fn, worker_init_fn=seed_dataloader_worker)
    train_loader = DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **loader_kw)

    model = LateralityVLJEPA(
        image_mode=IMAGE_MODE,
        smoke=False,
        n_llama_layers=N_LLAMA_LAYERS,
        llama_name=LLAMA_NAME,
        llama_local=LLAMA_LOCAL,
        freeze_image_encoder=True,
        freeze_text_encoder=True,
        gradient_checkpointing=True,
        text_lora=False,
    ).to(device)
    if rank == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[laterality] predictor init={model.predictor.init_source} "
              f"tok={model.predictor.tokenizer_source} text lora={model.text_lora_path}")
        print(f"[laterality] params trainable={n_train:,}")
    bank = TargetBank(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model

    optimizer = AdamW([p for p in raw_model.predictor.parameters() if p.requires_grad],
                      lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(len(train_loader), 1) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps)

    cfg = {
        "objective": "laterality",
        "sides": list(SIDES),
        "laterality_path": LATERALITY_PATH,
        "query_template": QUERY_TEMPLATE,
        "image_mode": IMAGE_MODE,
        "n_llama_layers": N_LLAMA_LAYERS,
        "llama_name": LLAMA_NAME,
        "llama_local": LLAMA_LOCAL,
        "freeze_image_encoder": True,
        "freeze_text_encoder": True,
        "text_lora": False,
        "temperature": TEMPERATURE,
        "cbw_beta": CBW_BETA,
    }
    start_epoch, best_val = 1, float("inf")
    resume_path = args.resume or latest_ckpt(CHECKPOINT_DIR)
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        if ckpt.get("vljepa_cfg", {}).get("objective") != "laterality":
            raise SystemExit(f"[laterality] refusing {resume_path}: not a laterality checkpoint. "
                             f"Use an empty {CHECKPOINT_DIR}.")
        raw_model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        if rank == 0:
            print(f"[laterality] resumed {resume_path} -> next epoch {start_epoch}")
        del ckpt

    gold_groups = gold_roots = None
    if rank == 0 and not args.skip_gold:
        try:
            from .eval_gold import load_gold_groups

            gold_groups = load_gold_groups()
            gold_roots = {
                **IMAGE_ROOTS,
                **discover_gold_image_roots(os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))),
                **gold_image_dirs(),
            }
            print(f"[laterality] gold groups: {len(gold_groups)}")
        except Exception as exc:
            print(f"[laterality] gold load failed ({exc}); skipping gold eval")

    if rank == 0 and not os.path.isfile(CSV_LOG):
        with open(CSV_LOG, "w") as f:
            f.write("epoch,train_loss,train_cls_acc,train_side_acc,val_loss,val_cls_acc,"
                    "val_exact_acc,val_side_acc,val_n_sided,gold_combined,gold_single,"
                    "gold_multi\n")

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr = summarize(ddp_sum(train_one_epoch(
            model, bank, train_loader, optimizer, scheduler, device, class_weights, epoch, rank),
            device, world_size))
        va = summarize(ddp_sum(run_val(
            raw_model, bank, val_loader, device, class_weights, epoch, rank), device, world_size))
        if rank == 0:
            print(f"[laterality] epoch {epoch}: train loss={tr['loss']:.4f} "
                  f"cls={tr['cls_acc']:.3f} side={tr['side_acc']:.3f} | val loss={va['loss']:.4f} "
                  f"cls={va['cls_acc']:.3f} exact={va['exact_acc']:.3f} "
                  f"side={va['side_acc']:.3f} (majority {side_baseline:.3f}; "
                  f"sided n={va['n_sided']} of {va['n']})")
            ckpt = {
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": min(best_val, va["loss"]),
                "vljepa_cfg": cfg,
            }
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt"))
            print(f"[laterality] saved epoch_{epoch}.pt")
            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
                print("[laterality] saved new BEST (silver val loss)")
            del ckpt
            gold = {}
            if gold_groups is not None:
                from .eval_gold import eval_gold_setmatch

                gold = eval_gold_setmatch(raw_model, gold_groups, gold_roots, epoch, device) or {}
            with open(CSV_LOG, "a") as f:
                f.write(f"{epoch},{tr['loss']:.6f},{tr['cls_acc']:.6f},{tr['side_acc']:.6f},"
                        f"{va['loss']:.6f},{va['cls_acc']:.6f},{va['exact_acc']:.6f},"
                        f"{va['side_acc']:.6f},{va['n_sided']},"
                        f"{gold.get('combined_score', '')},{gold.get('single_acc', '')},"
                        f"{gold.get('multi_jaccard', '')}\n")
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
