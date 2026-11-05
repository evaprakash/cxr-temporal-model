"""DDP trainer for dynamic-sentence VL-JEPA (option 1).

Each silver study pair is one example. The predictor sees the pair and
"Describe what has changed between these two chest X-rays." and predicts
Ŝ. The target is one of the current report's dynamic sentences, encoded by
the original frozen BioViL-T text encoder (no LoRA). Loss is InfoNCE over
the sentences of the whole global batch (all GPUs); sentences with the same
text as the positive are left out of the negatives.

After every epoch, rank 0 reports silver val retrieval (does Ŝ rank its own
sentence first among all distinct val sentences?) and zero-shot gold scores
(phrase-bank cosine per finding).

Second stage: ``--init-from <frozen run's epoch_N.pt> --unfreeze-image`` also
trains the BioViL-T image encoder at LR * IMAGE_LR_MULT (BatchNorm stats and
dropout stay frozen), into a new checkpoint dir.

    torchrun --nproc_per_node=4 -m vljepa.train_dynamic
    bash vljepa/run_ccw.sh dyn_train
    bash vljepa/run_ccw.sh dyn_train_img
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
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from . import _root  # noqa: F401

from progression_classify import DEFAULT_GOLD_PARQUET, discover_gold_image_roots

from .cluster_paths import gold_image_dirs, image_roots as cluster_image_roots
from .dynamic import (
    DYNAMIC_QUERY,
    TARGET_MODES,
    DynamicScorer,
    VLJEPADynamicDataset,
    dynamic_collate_fn,
    retrieval_recall,
    same_text_mask,
    score_gold_groups,
)
from .model import VLJEPA, llama_hub_cache, pair_infonce_loss, resolve_llama_local

# ============================================================
# HYPERPARAMETERS (same as the 5-way run unless noted)
# ============================================================
LR = 2e-5
IMAGE_LR_MULT = 0.1
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
EPOCHS = 50
WARMUP_RATIO = 0.03
TEMPERATURE = 0.07
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
    "VLJEPA_CHECKPOINT_DIR", os.path.join(_ROOT, "checkpoints_vljepa_dynamic"))
LOG_DIR = os.environ.get("VLJEPA_LOG_DIR", os.path.join(_ROOT, "logs_vljepa_dynamic"))
CSV_LOG = os.path.join(LOG_DIR, "val_metrics_vljepa_dynamic.csv")


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


def ddp_mean(value, device, world_size):
    if world_size <= 1:
        return float(value)
    t = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size)


def gather_targets(t_g, texts, world_size):
    """Targets and their texts from every GPU, plus where this GPU's rows start.
    The text encoder is frozen, so the gathered targets need no gradient."""
    if world_size <= 1:
        return t_g, list(texts), 0
    parts = [torch.zeros_like(t_g) for _ in range(world_size)]
    dist.all_gather(parts, t_g.detach().contiguous())
    text_parts = [None] * world_size
    dist.all_gather_object(text_parts, list(texts))
    offset = sum(len(t) for t in text_parts[:dist.get_rank()])
    return torch.cat(parts), [t for part in text_parts for t in part], offset


def batch_loss(net, batch, device, world_size):
    """``net`` must be the DDP wrapper when training, or gradients are not
    averaged across GPUs."""
    prior = batch["prior_image"].to(device, non_blocking=True)
    current = batch["current_image"].to(device, non_blocking=True)
    texts = batch["target_text"]
    out = net(prior, current, batch["query_text"], target_texts=texts)
    targets, all_texts, offset = gather_targets(out["target_global"], texts, world_size)
    same = same_text_mask(texts, all_texts).to(device)
    pos = torch.arange(len(texts), device=device) + offset
    loss, logits = pair_infonce_loss(out["pred"], targets, same, pos, temperature=TEMPERATURE)
    rows = torch.arange(len(texts), device=device)
    correct = int(same[rows, logits.argmax(dim=1)].sum().item())
    n_dup = int(same.sum().item()) - len(texts)
    return loss, correct, n_dup, out


def latest_ckpt(ckpt_dir: str):
    paths = glob.glob(os.path.join(ckpt_dir, "epoch_*.pt"))
    if not paths:
        return None
    return max(paths, key=lambda p: int(os.path.basename(p)[6:-3]) if
               os.path.basename(p)[6:-3].isdigit() else -1)


def train_one_epoch(model, loader, optimizer, scheduler, device, world_size, epoch, rank):
    model.train()
    running, correct, n, n_dup, steps = 0.0, 0, 0, 0, 0
    pbar = tqdm(loader, desc=f"dynamic train ep{epoch}", disable=rank != 0)
    for batch in pbar:
        loss, c, d, _ = batch_loss(model, batch, device, world_size)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        b = len(batch["target_text"])
        running += float(loss.item()) * b
        correct += c
        n += b
        n_dup += d
        steps += 1
        if rank == 0:
            pbar.set_postfix(loss=f"{running / n:.4f}", acc=f"{correct / n:.3f}",
                             dup_masked=f"{n_dup / steps:.2f}")
    return running / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def run_val(raw_model, loader, device, world_size, epoch, rank):
    raw_model.eval()
    total, correct, n = 0.0, 0, 0
    preds, tgts, texts = [], [], []
    for batch in tqdm(loader, desc=f"dynamic val ep{epoch}", disable=rank != 0):
        loss, c, _, out = batch_loss(raw_model, batch, device, world_size)
        b = len(batch["target_text"])
        total += float(loss.item()) * b
        correct += c
        n += b
        preds.append(F.normalize(out["pred"].float(), dim=-1).cpu())
        tgts.append(F.normalize(out["target_global"].float(), dim=-1).cpu())
        texts.extend(batch["target_text"])
    local = (torch.cat(preds), torch.cat(tgts), texts)
    if world_size > 1:
        parts = [None] * world_size
        dist.all_gather_object(parts, local)
    else:
        parts = [local]
    retrieval = None
    if rank == 0:
        pred = torch.cat([p[0] for p in parts]).to(device)
        tgt = torch.cat([p[1] for p in parts]).to(device)
        all_texts = [t for p in parts for t in p[2]]
        retrieval = retrieval_recall(pred, tgt, all_texts)
        retrieval["pred_std"] = float(pred.std(dim=0).mean())
    return total / max(n, 1), correct / max(n, 1), retrieval


def build_optimizer(model):
    groups = [{"params": [p for p in model.predictor.parameters() if p.requires_grad],
               "lr": LR}]
    image_params = [p for p in model.image_encoder.parameters() if p.requires_grad]
    if image_params:
        groups.append({"params": image_params, "lr": LR * IMAGE_LR_MULT})
    return AdamW(groups, weight_decay=WEIGHT_DECAY)


def load_init_weights(raw_model, path, target_mode, rank):
    ckpt = torch.load(path, map_location="cpu")
    old = ckpt.get("vljepa_cfg", {})
    if old.get("objective") != "dynamic" or old.get("target_mode") != target_mode:
        raise SystemExit(f"[dynamic] refusing --init-from {path}: objective "
                         f"{old.get('objective')}, target {old.get('target_mode')}")
    missing, unexpected = raw_model.load_state_dict(ckpt["model"], strict=False)
    if rank == 0:
        print(f"[dynamic] initialised weights from {path} (epoch {ckpt.get('epoch')}); "
              f"fresh optimizer and schedule; missing={len(missing)} "
              f"unexpected={len(unexpected)}")


def gold_eval(raw_model, groups, roots, device, epoch):
    from baap_eval.eval_baap import print_gold_metrics

    raw_model.eval()
    scorer = DynamicScorer(raw_model, mode="perfinding")
    scores, gts, findings, _, datasets = score_gold_groups(
        scorer, groups, roots, device, desc=f"dynamic gold ep{epoch}")
    return print_gold_metrics(f"vljepa_dynamic ep{epoch} (per-finding phrase bank)",
                              scores, gts, findings, datasets)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", default=None)
    p.add_argument("--init-from", default=None,
                   help="start from this dynamic checkpoint's weights only (fresh optimizer "
                        "and schedule); must be outside the checkpoint dir")
    p.add_argument("--unfreeze-image", action="store_true",
                   help="train the BioViL-T image encoder at LR * IMAGE_LR_MULT; "
                        "its BatchNorm stats and dropout stay frozen")
    p.add_argument("--target-mode", default="sentence", choices=TARGET_MODES,
                   help="sentence: one dynamic sentence per pair, resampled each epoch; "
                        "report: all of them joined (truncated at 112 tokens)")
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
    freeze_image = not args.unfreeze_image
    if args.init_from and (os.path.dirname(os.path.abspath(args.init_from))
                           == os.path.abspath(CHECKPOINT_DIR)):
        raise SystemExit(f"[dynamic] --init-from {args.init_from} is inside {CHECKPOINT_DIR}; "
                         "set VLJEPA_CHECKPOINT_DIR to a new folder so the source run "
                         "stays intact")

    if rank == 0:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        print("[dynamic] option 1: dynamic sentences")
        print(f"[dynamic]   query  = {DYNAMIC_QUERY!r} (Llama tok+embed)")
        print(f"[dynamic]   target = silver dynamic sentence ({args.target_mode}), "
              "original BioViL-T text encoder, frozen, no LoRA")
        print(f"[dynamic]   loss   = InfoNCE over {batch_size * world_size} sentences per "
              f"global batch, τ={TEMPERATURE}, same-text negatives masked")
        if freeze_image:
            print(f"[dynamic]   image  = BioViL-T pair, frozen; "
                  f"pred = Llama last {N_LLAMA_LAYERS}")
        else:
            print(f"[dynamic]   image  = BioViL-T pair, TRAINED at lr {LR * IMAGE_LR_MULT:g} "
                  f"(BatchNorm stats + dropout frozen); pred = Llama last {N_LLAMA_LAYERS}")
        if args.init_from:
            print(f"[dynamic]   init from = {args.init_from} (weights only)")
        print(f"[dynamic]   ckpt   = {CHECKPOINT_DIR}")
        print(f"[dynamic]   logs   = {LOG_DIR}")
        print(f"[dynamic]   images = {IMAGE_ROOTS}")

    train_ds = VLJEPADynamicDataset(IMAGE_ROOTS, split="train", train=True,
                                    val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
                                    target_mode=args.target_mode)
    val_ds = VLJEPADynamicDataset(IMAGE_ROOTS, split="val", train=False,
                                  val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
                                  target_mode=args.target_mode)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    loader_kw = dict(batch_size=batch_size, num_workers=args.num_workers, pin_memory=True,
                     collate_fn=dynamic_collate_fn, worker_init_fn=seed_dataloader_worker)
    train_loader = DataLoader(train_ds, shuffle=train_sampler is None, sampler=train_sampler,
                              drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **loader_kw)

    model = VLJEPA(
        image_mode=IMAGE_MODE,
        smoke=False,
        n_llama_layers=N_LLAMA_LAYERS,
        llama_name=LLAMA_NAME,
        llama_local=LLAMA_LOCAL,
        freeze_image_encoder=freeze_image,
        freeze_text_encoder=True,
        gradient_checkpointing=True,
        text_lora=False,
    ).to(device)
    if rank == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[dynamic] predictor init={model.predictor.init_source} "
              f"tok={model.predictor.tokenizer_source} text lora={model.text_lora_path}")
        print(f"[dynamic] params trainable={n_train:,}")
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model

    optimizer = build_optimizer(raw_model)
    total_steps = max(len(train_loader), 1) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps)

    cfg = {
        "objective": "dynamic",
        "target_mode": args.target_mode,
        "query": DYNAMIC_QUERY,
        "image_mode": IMAGE_MODE,
        "n_llama_layers": N_LLAMA_LAYERS,
        "llama_name": LLAMA_NAME,
        "llama_local": LLAMA_LOCAL,
        "freeze_image_encoder": freeze_image,
        "image_lr_mult": IMAGE_LR_MULT,
        "init_from": args.init_from,
        "freeze_text_encoder": True,
        "text_lora": False,
        "temperature": TEMPERATURE,
        "global_batch": batch_size * world_size,
    }
    start_epoch, best_val = 1, float("inf")
    resume_path = args.resume or latest_ckpt(CHECKPOINT_DIR)
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        old = ckpt.get("vljepa_cfg", {})
        if old.get("objective") != "dynamic" or old.get("target_mode") != args.target_mode:
            raise SystemExit(f"[dynamic] refusing {resume_path}: objective "
                             f"{old.get('objective')}, target {old.get('target_mode')}. "
                             f"Use an empty {CHECKPOINT_DIR}.")
        if bool(old.get("freeze_image_encoder", True)) != freeze_image:
            raise SystemExit(f"[dynamic] refusing {resume_path}: it was trained with "
                             f"freeze_image_encoder={old.get('freeze_image_encoder', True)}, "
                             f"this run has {freeze_image}. Use a new checkpoint dir.")
        raw_model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        if rank == 0:
            print(f"[dynamic] resumed {resume_path} -> next epoch {start_epoch}")
            if args.init_from:
                print(f"[dynamic] {CHECKPOINT_DIR} already has epochs; "
                      f"ignoring --init-from {args.init_from}")
        del ckpt
    elif args.init_from:
        load_init_weights(raw_model, args.init_from, args.target_mode, rank)

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
            print(f"[dynamic] gold groups: {len(gold_groups)}")
        except Exception as exc:
            print(f"[dynamic] gold load failed ({exc}); skipping gold eval")

    if rank == 0 and not os.path.isfile(CSV_LOG):
        with open(CSV_LOG, "w") as f:
            f.write("epoch,train_loss,train_acc,val_loss,val_acc,recall_at_1,recall_at_5,"
                    "recall_at_10,median_rank,gallery,pred_std,gold_combined,gold_macro_f1\n")

    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, world_size, epoch, rank)
        val_loss, val_acc, ret = run_val(raw_model, val_loader, device, world_size, epoch, rank)
        train_loss = ddp_mean(train_loss, device, world_size)
        train_acc = ddp_mean(train_acc, device, world_size)
        val_loss = ddp_mean(val_loss, device, world_size)
        val_acc = ddp_mean(val_acc, device, world_size)
        if rank == 0:
            print(f"[dynamic] epoch {epoch}: train loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val loss={val_loss:.4f} acc={val_acc:.3f} | retrieval over "
                  f"{ret['gallery']} val sentences: R@1={ret['recall@1']:.3f} "
                  f"R@5={ret['recall@5']:.3f} R@10={ret['recall@10']:.3f} "
                  f"median rank={ret['median_rank']:.0f} | pred_std={ret['pred_std']:.4f}")
            ckpt = {
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": min(best_val, val_loss),
                "vljepa_cfg": cfg,
            }
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt"))
            print(f"[dynamic] saved epoch_{epoch}.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
                print("[dynamic] saved new BEST (silver val loss)")
            del ckpt
            gold = {}
            if gold_groups is not None:
                gold = gold_eval(raw_model, gold_groups, gold_roots, device, epoch)
            with open(CSV_LOG, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},"
                        f"{ret['recall@1']:.6f},{ret['recall@5']:.6f},{ret['recall@10']:.6f},"
                        f"{ret['median_rank']:.0f},{ret['gallery']},{ret['pred_std']:.6f},"
                        f"{gold.get('combined', '')},{gold.get('macro_f1', '')}\n")
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
