"""DDP trainer for option-2 VL-JEPA.

Loss is 5-way InfoNCE: predicted Ŝ vs frozen BioViL-T embeddings of
``{Finding} is {class}.`` (gold = positive, other four = negatives).
Text encoder is frozen. Improving and worsening each get a learned nudge
added after the encoder, with a hinge that keeps them away from stable.
Fresh run (does not resume checkpoints_vljepa_frzy).
Rank-0 runs CheXTemporal gold set-match after every epoch.

    torchrun --nproc_per_node=4 -m vljepa.train
    # or
    sbatch vljepa/train.sh
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

from gold_progression_setmatch import group_gold_by_pair_finding
from progression_classify import (
    DEFAULT_GOLD_PARQUET,
    discover_gold_image_roots,
    load_gold_pairs,
)
from progression_phrases import CLS_ORDER
from dataset_combined_jepa import DEFAULT_FINDINGS
from losses_jepa import patch_token_feature_stats

from .cluster_paths import gold_image_dirs, image_roots as cluster_image_roots
from .dataset import VLJEPAFindingDataset, flatten_target_texts, vljepa_collate_fn
from .eval_gold import eval_gold_setmatch
from .model import (
    VLJEPA,
    class_infonce_loss,
    llama_hub_cache,
    resolve_llama_local,
    stable_separation_loss,
)
from .prompts import QUERY_TEMPLATE, TARGET_TEMPLATE

N_CLS = len(CLS_ORDER)

# ============================================================
# HYPERPARAMETERS
# ============================================================
LR = 2e-5
TEXT_LR_MULT = 0.05
IMAGE_LR_MULT = 0.1
WEIGHT_DECAY = 0.01
BATCH_SIZE = 8
EPOCHS = 50
WARMUP_RATIO = 0.03
TEMPERATURE = 0.07
CBW_BETA = 0.99999
VAL_FRACTION = 0.1
SPLIT_SEED = 42
SAVE_EVERY_N_EPOCHS = 1

FREEZE_IMAGE_ENCODER = True
FREEZE_TEXT_ENCODER = True
STOPGRAD_NEG_PHRASES = False
# Learned offset on the frozen improving / worsening sentences only.
# Hinge: cosine to the frozen stable sentence should fall to this or below.
NUDGE_LR = 1e-4
NUDGE_SEP_MARGIN = 0.5
NUDGE_SEP_WEIGHT = 1.0
N_LLAMA_LAYERS = 8
LLAMA_NAME = os.environ.get("VLJEPA_LLAMA_NAME", "meta-llama/Llama-3.2-1B")
llama_hub_cache()
LLAMA_LOCAL = resolve_llama_local()
IMAGE_MODE = "biovilt"

# ============================================================
# PATHS
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

IMAGE_ROOTS = cluster_image_roots()

CHECKPOINT_DIR = os.environ.get(
    "VLJEPA_CHECKPOINT_DIR",
    os.path.join(_ROOT, "checkpoints_vljepa_nudge"),
)
LOG_DIR = os.environ.get(
    "VLJEPA_LOG_DIR",
    os.path.join(_ROOT, "logs_vljepa_nudge"),
)
CSV_LOG = os.path.join(LOG_DIR, "val_metrics_vljepa.csv")
FEAT_CSV_LOG = os.path.join(LOG_DIR, "feat_std_vljepa.csv")
FEAT_LOG_EVERY = int(os.environ.get("FEAT_LOG_EVERY", "20"))


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


def ddp_reduce(value, device, world_size):
    if world_size <= 1:
        return float(value)
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return float(tensor.item())


def target_class_offdiag(target_global: torch.Tensor, bsz: int, n_cls: int = N_CLS) -> float:
    """Mean off-diagonal cosine among the 5 class-phrase embeds (template collapse)."""
    import torch.nn.functional as F

    t = target_global.detach().float()
    if t.dim() == 2:
        t = t.view(bsz, n_cls, -1)
    t = F.normalize(t, dim=-1)
    sim = torch.bmm(t, t.transpose(1, 2))
    off = sim.sum(dim=(1, 2)) - sim.diagonal(dim1=-2, dim2=-1).sum(-1)
    return float((off / float(n_cls * (n_cls - 1))).mean().item())


def feat_stats_from_out(out: dict, bsz: int) -> dict:
    img = patch_token_feature_stats(out["img_patches"])
    q = out.get("query_llama")
    q_stats = (
        patch_token_feature_stats(q)
        if q is not None and q.dim() == 3
        else {"std_over_patches": out["pred"].new_zeros(()),
              "mean_offdiag_cos": out["pred"].new_zeros(())}
    )
    return {
        "img_patch_std": float(img["std_over_patches"]),
        "img_patch_offdiag": float(img["mean_offdiag_cos"]),
        "query_tok_std": float(q_stats["std_over_patches"]),
        "target_offdiag": target_class_offdiag(out["target_global"], bsz),
        "pred_batch_std": float(out["pred"].detach().float().std(unbiased=False).item()),
    }


def cui_class_weights(counts: torch.Tensor, beta: float) -> torch.Tensor:
    counts = counts.to(dtype=torch.float64)
    beta_t = torch.tensor(beta, dtype=torch.float64)
    effective = (1.0 - torch.pow(beta_t, counts.clamp_min(1))) / (1.0 - beta_t)
    weights = 1.0 / effective
    weights = weights * N_CLS / weights.sum()
    return weights.float()


def latest_ckpt(ckpt_dir: str):
    paths = glob.glob(os.path.join(ckpt_dir, "epoch_*.pt"))
    if not paths:
        return None
    def _ep(p):
        name = os.path.basename(p)
        try:
            return int(name.replace("epoch_", "").replace(".pt", ""))
        except ValueError:
            return -1
    paths.sort(key=_ep)
    return paths[-1]


def build_optimizer(model: VLJEPA):
    pred_params = [p for p in model.predictor.parameters() if p.requires_grad]
    text_params = [
        p for p in model.text_encoder.parameters() if p.requires_grad
    ]
    image_params = [
        p for p in model.image_encoder.parameters() if p.requires_grad
    ]
    groups = [{"params": pred_params, "lr": LR}]
    if text_params:
        groups.append({"params": text_params, "lr": LR * TEXT_LR_MULT})
    if image_params:
        groups.append({"params": image_params, "lr": LR * IMAGE_LR_MULT})
    # No weight decay: decay would pull the nudge back onto the frozen sentence.
    groups.append({
        "params": [model.class_nudge],
        "lr": NUDGE_LR,
        "weight_decay": 0.0,
    })
    return AdamW(groups, weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def run_val(raw_model, loader, device, class_weights, desc):
    raw_model.eval()
    total = 0.0
    correct = 0
    n = 0
    feat_sum = None
    n_feat = 0
    for batch in tqdm(loader, desc=desc, disable=device.type == "cpu" and False):
        prior = batch["prior_image"].to(device, non_blocking=True)
        current = batch["current_image"].to(device, non_blocking=True)
        labels = batch["cls_idx"].to(device)
        targets = flatten_target_texts(batch["target_texts"])
        out = raw_model(
            prior, current, batch["query_text"], target_texts=targets,
        )
        loss_nce, logits = class_infonce_loss(
            out["pred"], out["target_global"], labels,
            temperature=TEMPERATURE, class_weights=class_weights,
            stopgrad_negatives=STOPGRAD_NEG_PHRASES,
        )
        loss_sep, _, _ = stable_separation_loss(
            out["target_global"], margin=NUDGE_SEP_MARGIN,
        )
        loss = loss_nce + NUDGE_SEP_WEIGHT * loss_sep
        pred_cls = logits.argmax(dim=-1)
        total += float(loss.item()) * labels.shape[0]
        correct += int((pred_cls == labels).sum().item())
        n += int(labels.shape[0])
        stats = feat_stats_from_out(out, labels.shape[0])
        if feat_sum is None:
            feat_sum = {k: 0.0 for k in stats}
        for k, v in stats.items():
            feat_sum[k] += v
        n_feat += 1
    feat_avg = {k: v / max(n_feat, 1) for k, v in (feat_sum or {}).items()}
    return total / max(n, 1), correct / max(n, 1), n, feat_avg


def train_one_epoch(
    model, raw_model, loader, optimizer, scheduler, device, class_weights, epoch,
    rank, global_step=0, feat_csv=None,
):
    model.train()
    running = 0.0
    correct = 0
    n = 0
    sep_sum = 0.0
    cos_imp_sum = 0.0
    cos_wor_sum = 0.0
    n_batches = 0
    pbar = tqdm(loader, desc=f"vljepa train ep{epoch}", disable=rank != 0)
    last_stats = {}
    for batch in pbar:
        prior = batch["prior_image"].to(device, non_blocking=True)
        current = batch["current_image"].to(device, non_blocking=True)
        labels = batch["cls_idx"].to(device)
        targets = flatten_target_texts(batch["target_texts"])
        out = raw_model(
            prior, current, batch["query_text"], target_texts=targets,
        )
        loss_nce, logits = class_infonce_loss(
            out["pred"], out["target_global"], labels,
            temperature=TEMPERATURE, class_weights=class_weights,
            stopgrad_negatives=STOPGRAD_NEG_PHRASES,
        )
        loss_sep, cos_imp, cos_wor = stable_separation_loss(
            out["target_global"], margin=NUDGE_SEP_MARGIN,
        )
        loss = loss_nce + NUDGE_SEP_WEIGHT * loss_sep
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        pred_cls = logits.argmax(dim=-1)
        running += float(loss.item()) * labels.shape[0]
        correct += int((pred_cls == labels).sum().item())
        n += int(labels.shape[0])
        sep_sum += float(loss_sep.detach().item())
        cos_imp_sum += float(cos_imp.detach().item())
        cos_wor_sum += float(cos_wor.detach().item())
        n_batches += 1
        global_step += 1
        if rank == 0:
            last_stats = feat_stats_from_out(out, labels.shape[0])
            if feat_csv is not None and global_step % FEAT_LOG_EVERY == 0:
                feat_csv.write(
                    f"{global_step},{epoch},"
                    f"{last_stats['img_patch_std']:.6f},"
                    f"{last_stats['img_patch_offdiag']:.6f},"
                    f"{last_stats['query_tok_std']:.6f},"
                    f"{last_stats['target_offdiag']:.6f},"
                    f"{last_stats['pred_batch_std']:.6f}\n"
                )
                feat_csv.flush()
            pbar.set_postfix(
                loss=f"{running / max(n, 1):.4f}",
                acc=f"{correct / max(n, 1):.3f}",
                tgt_off=f"{last_stats.get('target_offdiag', 0):.3f}",
                sep=f"{cos_imp_sum / max(n_batches, 1):.3f}/{cos_wor_sum / max(n_batches, 1):.3f}",
            )
    nb = max(n_batches, 1)
    return (
        running / max(n, 1),
        correct / max(n, 1),
        global_step,
        cos_imp_sum / nb,
        cos_wor_sum / nb,
        sep_sum / nb,
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--skip-gold", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
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
        print("[vljepa] option 2")
        print(f"[vljepa]   query  = {QUERY_TEMPLATE}  (Llama tok+embed)")
        print(f"[vljepa]   target = {TARGET_TEMPLATE}  (BioViL-T Y-encoder)")
        print(f"[vljepa]   image  = BioViL-T pair (prior+current)")
        print(f"[vljepa]   pred   = Llama last {N_LLAMA_LAYERS} layers")
        print(
            f"[vljepa]   loss   = 5-way InfoNCE τ={TEMPERATURE} "
            f"stopgrad_neg={STOPGRAD_NEG_PHRASES}"
        )
        print(
            f"[vljepa]   nudge  = improving+worsening  lr={NUDGE_LR} "
            f"sep_margin={NUDGE_SEP_MARGIN} sep_weight={NUDGE_SEP_WEIGHT}"
        )
        print(f"[vljepa]   freeze image={FREEZE_IMAGE_ENCODER} "
              f"text={FREEZE_TEXT_ENCODER}")
        print(f"[vljepa]   ckpt   = {CHECKPOINT_DIR}")
        print(f"[vljepa]   logs   = {LOG_DIR}")
        print(f"[vljepa]   images = {IMAGE_ROOTS}")

    train_ds = VLJEPAFindingDataset(
        IMAGE_ROOTS, split="train", train=True,
        val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
    )
    val_ds = VLJEPAFindingDataset(
        IMAGE_ROOTS, split="val", train=False,
        val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED,
    )
    counts = train_ds.class_counts()
    class_weights = cui_class_weights(counts, CBW_BETA).to(device)
    if rank == 0:
        print(f"[vljepa] train class counts: {counts.tolist()}")
        print(f"[vljepa] CBW β={CBW_BETA} weights: {class_weights.tolist()}")

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    )
    val_sampler = (
        DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=vljepa_collate_fn,
        worker_init_fn=seed_dataloader_worker,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=vljepa_collate_fn,
        worker_init_fn=seed_dataloader_worker,
    )

    model = VLJEPA(
        image_mode=IMAGE_MODE,
        smoke=False,
        n_llama_layers=N_LLAMA_LAYERS,
        llama_name=LLAMA_NAME,
        llama_local=LLAMA_LOCAL,
        freeze_image_encoder=FREEZE_IMAGE_ENCODER,
        freeze_text_encoder=FREEZE_TEXT_ENCODER,
        gradient_checkpointing=True,
    ).to(device)
    if rank == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(
            f"[vljepa] predictor init={model.predictor.init_source} "
            f"hidden={model.predictor.hidden_size} layers={model.predictor.n_layers} "
            f"tok={model.predictor.tokenizer_source}"
        )
        print(f"[vljepa] params trainable={n_train:,} / total={n_all:,}")

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        raw_model = model.module
    else:
        raw_model = model

    optimizer = build_optimizer(raw_model)
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = steps_per_epoch * epochs
    warmup = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps,
    )

    start_epoch = 1
    best_val = float("inf")
    resume_path = args.resume or latest_ckpt(CHECKPOINT_DIR)
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        raw_model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", best_val))
        if rank == 0:
            print(f"[vljepa] resumed {resume_path} → next epoch {start_epoch}")

    gold_groups = None
    gold_roots = None
    if rank == 0 and not args.skip_gold:
        try:
            gold = load_gold_pairs(DEFAULT_GOLD_PARQUET, DEFAULT_FINDINGS)
            gold_groups = group_gold_by_pair_finding(gold)
            gold_parquet_dir = os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))
            gold_roots = {
                **IMAGE_ROOTS,
                **discover_gold_image_roots(gold_parquet_dir),
                **gold_image_dirs(),
            }
            print(f"[vljepa] gold groups: {len(gold_groups)}")
            print("[vljepa] gold image roots:")
            for d in ("mimic", "chexpert", "rexgradient"):
                print(f"  {d}: {gold_roots.get(d, '<missing>')}")
        except Exception as exc:
            print(f"[vljepa] gold load failed ({exc}); skipping gold eval")

    if rank == 0 and not os.path.isfile(CSV_LOG):
        with open(CSV_LOG, "w") as f:
            f.write(
                "epoch,train_loss,train_acc,val_loss,val_acc,"
                "val_img_patch_std,val_target_offdiag,val_pred_batch_std,"
                "gold_combined,gold_single,gold_multi\n"
            )
    feat_f = None
    if rank == 0:
        new_feat = not os.path.isfile(FEAT_CSV_LOG)
        feat_f = open(FEAT_CSV_LOG, "a")
        if new_feat:
            feat_f.write(
                "step,epoch,img_patch_std,img_patch_offdiag,"
                "query_tok_std,target_offdiag,pred_batch_std\n"
            )
        print(f"[vljepa] feat-std CSV: {FEAT_CSV_LOG} (every {FEAT_LOG_EVERY} steps)")
        print("[vljepa] gold set-match after every epoch (rank 0)")

    global_step = 0
    for epoch in range(start_epoch, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss, train_acc, global_step, cos_imp, cos_wor, sep = train_one_epoch(
            model, raw_model, train_loader, optimizer, scheduler,
            device, class_weights, epoch, rank,
            global_step=global_step, feat_csv=feat_f,
        )
        val_loss, val_acc, val_n, val_feat = run_val(
            raw_model, val_loader, device, class_weights, f"vljepa val ep{epoch}",
        )
        train_loss = ddp_reduce(train_loss, device, world_size)
        train_acc = ddp_reduce(train_acc, device, world_size)
        val_loss = ddp_reduce(val_loss, device, world_size)
        val_acc = ddp_reduce(val_acc, device, world_size)
        cos_imp = ddp_reduce(cos_imp, device, world_size)
        cos_wor = ddp_reduce(cos_wor, device, world_size)
        sep = ddp_reduce(sep, device, world_size)

        if rank == 0:
            print(
                f"[vljepa] epoch {epoch}: "
                f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"val loss={val_loss:.4f} acc={val_acc:.3f} (n={val_n}) | "
                f"img_std={val_feat.get('img_patch_std', 0):.4f} "
                f"tgt_offdiag={val_feat.get('target_offdiag', 0):.4f} "
                f"sep={sep:.4f} cos_imp_sta={cos_imp:.4f} cos_wor_sta={cos_wor:.4f}"
            )
            ckpt = {
                "epoch": epoch,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_loss": best_val,
                "vljepa_cfg": {
                    "image_mode": IMAGE_MODE,
                    "n_llama_layers": N_LLAMA_LAYERS,
                    "llama_name": LLAMA_NAME,
                    "llama_local": LLAMA_LOCAL,
                    "freeze_image_encoder": FREEZE_IMAGE_ENCODER,
                    "freeze_text_encoder": FREEZE_TEXT_ENCODER,
                    "class_nudge": True,
                    "nudge_sep_margin": NUDGE_SEP_MARGIN,
                    "temperature": TEMPERATURE,
                    "query_template": QUERY_TEMPLATE,
                    "target_template": TARGET_TEMPLATE,
                },
            }
            if epoch == 1 or (epoch % SAVE_EVERY_N_EPOCHS == 0):
                path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
                torch.save(ckpt, path)
                print(f"[vljepa] saved {path}")
            if val_loss < best_val:
                best_val = val_loss
                ckpt["best_val_loss"] = best_val
                torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
                print("[vljepa] saved new BEST")

            gold_combined = gold_single = gold_multi = ""
            if gold_groups is not None:
                gold_sum = eval_gold_setmatch(
                    raw_model, gold_groups, gold_roots, epoch, device,
                )
                if gold_sum is not None:
                    gold_combined = f"{gold_sum['combined_score']:.6f}"
                    gold_single = f"{gold_sum['single_acc']:.6f}"
                    gold_multi = f"{gold_sum['multi_jaccard']:.6f}"
            with open(CSV_LOG, "a") as f:
                f.write(
                    f"{epoch},{train_loss:.6f},{train_acc:.6f},"
                    f"{val_loss:.6f},{val_acc:.6f},"
                    f"{val_feat.get('img_patch_std', float('nan')):.6f},"
                    f"{val_feat.get('target_offdiag', float('nan')):.6f},"
                    f"{val_feat.get('pred_batch_std', float('nan')):.6f},"
                    f"{gold_combined},{gold_single},{gold_multi}\n"
                )

        if world_size > 1:
            dist.barrier()

    if feat_f is not None:
        feat_f.close()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
