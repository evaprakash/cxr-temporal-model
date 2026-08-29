#!/usr/bin/env python3
"""Supervised 5-way progression baseline (unfrozen BioViL-T + linear head).

A linear classifier is trained on CheXTemporal **silver** labels while
**both** official BioViL-T encoders stay trainable:

  * BioViL-T pair image encoder → 128-d ``(current, prior)`` embedding.
  * BioViL-T text encoder → 128-d finding embedding (so the same image
    pair can predict different classes for different findings).
  * ``Linear(256 → 5)`` with class-balanced cross-entropy.

Optimizer / schedule match JEPA (``resume_train_jepa.py``): AdamW at
``2e-5``, weight decay ``0.01``, linear warmup for 3% of steps then
linear decay.

Gold eval (``eval_progression_gold_setmatch.py --backend supervised``)
uses the 5 logits as class scores, then the usual single/multi set-match
protocol. Checkpoints store both encoders + the head (the frozen-probe
run lived under ``checkpoints_supervised_progression/``).

Usage
-----
    # 4-GPU DDP (see train_supervised_progression.sh)
    torchrun --nproc_per_node=4 train_supervised_progression.py --train

    python train_supervised_progression.py --train \\
        --epochs 5 --limit-train 20000

    # CPU smoke (tiny silver + tiny gold set-match):
    python train_supervised_progression.py --train \\
        --device cpu --epochs 1 --limit-train 32 --limit-gold 40 \\
        --batch-size 4 --num-workers 0

    python eval_progression_gold_setmatch.py --backend supervised --eval \\
        --ckpt checkpoints_supervised_progression_unfrozen/epoch_3.pt

    # Continue after epoch 5 (writes epoch_6.pt …; fresh LR schedule
    # over the remaining epochs — the original 5-epoch run already
    # decayed LR to 0, so this is not a continuation of that schedule):
    python train_supervised_progression.py --train \\
        --resume checkpoints_supervised_progression_unfrozen/epoch_5.pt \\
        --epochs 10
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset_combined_jepa import DEFAULT_FINDINGS, DEFAULT_SPLITS_FILE
from eval_progression_biovilt import BioViLTPairModel
from eval_progression_jepa_silver import load_silver_progression_df
from gold_progression_setmatch import (
    format_running_setmatch,
    group_gold_by_pair_finding,
    print_setmatch_report,
    topk_set_match,
)
from infer_jepa import IMAGE_ROOTS
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER

DEFAULT_CKPT_DIR = "checkpoints_supervised_progression_unfrozen"
CLS_TO_IDX = {c: i for i, c in enumerate(CLS_ORDER)}
# Same as resume_train_jepa.py
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.03


class SupervisedHead(nn.Module):
    def __init__(self, in_dim: int = 256, n_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)


class SupervisedProgressionModel(nn.Module):
    """Unfrozen BioViL-T pair + text encoders and a 5-way head."""

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        head: Optional[SupervisedHead] = None,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.head = head if head is not None else SupervisedHead()

    def encode(
        self,
        priors: torch.Tensor,
        currents: torch.Tensor,
        findings,
    ) -> torch.Tensor:
        img_global, _ = self.image_encoder(currents, priors)
        img_emb = F.normalize(img_global.float(), dim=-1)
        txt, _, _ = self.text_encoder.forward_contrastive(list(findings))
        txt = F.normalize(txt.float(), dim=-1)
        return torch.cat([img_emb, txt], dim=-1)

    def forward(self, priors, currents, findings) -> torch.Tensor:
        return self.head(self.encode(priors, currents, findings))


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def encode_pair_and_finding(
    model: BioViLTPairModel,
    prior: torch.Tensor,
    current: torch.Tensor,
    finding: str,
    device: torch.device,
    text_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Return L2-normalized concat ``[img_pair; finding_text]`` (1, 256)."""
    prior_b = prior.unsqueeze(0).to(device)
    current_b = current.unsqueeze(0).to(device)
    img_global, _ = model.image_encoder(current_b, prior_b)
    img_emb = F.normalize(img_global.float(), dim=-1)

    key = finding.strip().lower()
    if text_cache is not None and key in text_cache:
        txt = text_cache[key].to(device)
    else:
        txt, _, _ = model.text_encoder.forward_contrastive([key])
        txt = F.normalize(txt.float(), dim=-1)
        if text_cache is not None:
            text_cache[key] = txt.detach().cpu()
            txt = txt.to(device)
    return torch.cat([img_emb, txt], dim=-1)


class SilverProgressionDataset(Dataset):
    def __init__(self, df, image_roots: Dict[str, str]):
        self.df = df.reset_index(drop=True)
        self.image_roots = image_roots

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            prior = load_image_tensor(
                row["dataset"], row["parent_image_prev"], self.image_roots,
            )
            current = load_image_tensor(
                row["dataset"], row["parent_image_curr"], self.image_roots,
            )
        except (FileNotFoundError, OSError):
            return None
        y = CLS_TO_IDX[str(row["progression"])]
        return prior, current, str(row["finding"]), y


def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    priors, currents, findings, labels = zip(*batch)
    return (
        torch.stack(priors, dim=0),
        torch.stack(currents, dim=0),
        list(findings),
        torch.tensor(labels, dtype=torch.long),
    )


def _log(msg: str, rank: int = 0) -> None:
    if rank == 0:
        print(msg, flush=True)


def _setup_distributed(device_arg: str) -> Tuple[int, int, torch.device]:
    """DDP when launched with torchrun; otherwise a single process."""
    if device_arg == "cpu" or "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device(device_arg)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    return local_rank, dist.get_world_size(), torch.device(f"cuda:{local_rank}")


@torch.no_grad()
def eval_gold_setmatch(
    model: nn.Module,
    groups,
    image_roots: Dict[str, str],
    device: torch.device,
    epoch: int,
    backend_name: str = "supervised",
):
    """Full gold single/multi/combined set-match using the current model."""
    raw = _unwrap(model)
    raw.eval()
    results = []
    skipped = 0
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
        logits = raw(
            prior.unsqueeze(0).to(device),
            current.unsqueeze(0).to(device),
            [str(row["finding"])],
        ).squeeze(0).float().tolist()
        results.append(
            topk_set_match(
                logits,
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
        _log(f"[supervised] gold skipped missing images: {skipped}")
    if not results:
        _log("[supervised] gold set-match: no groups evaluated")
        return
    print_setmatch_report(
        results, backend_name, f", epoch={epoch} (in-training)",
    )
    sys.stdout.flush()


def _class_weights(labels) -> torch.Tensor:
    counts = Counter(labels)
    n = sum(counts.values())
    w = []
    for cls in CLS_ORDER:
        c = counts.get(cls, 0)
        w.append(n / (len(CLS_ORDER) * max(1, c)))
    return torch.tensor(w, dtype=torch.float32)


def _save_ckpt(model: nn.Module, path: str, meta: dict) -> None:
    raw = _unwrap(model)
    torch.save(
        {
            "head": raw.head.state_dict(),
            "image_encoder": raw.image_encoder.state_dict(),
            "text_encoder": raw.text_encoder.state_dict(),
            "in_dim": 256,
            "classes": list(CLS_ORDER),
            "unfrozen": True,
            **meta,
        },
        path,
    )


def train(args):
    rank, world_size, device = _setup_distributed(args.device)
    parquet_dir = os.path.dirname(os.path.abspath(args.findings_parquet))
    gold_parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    auto_roots = {
        **discover_gold_image_roots(parquet_dir),
        **discover_gold_image_roots(gold_parquet_dir),
    }
    overrides: Dict[str, str] = {}
    for spec in args.image_root:
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(f"unknown dataset {d!r}")
        overrides[d] = p
    # Silver lives under all_data; gold prefers final_gold_* if discovered.
    silver_roots = {**IMAGE_ROOTS, **auto_roots, **overrides}
    gold_roots = {**IMAGE_ROOTS, **overrides, **auto_roots}
    _log("[supervised] unfrozen BioViL-T image + text + linear head", rank)
    _log(f"[supervised] DDP world_size={world_size} rank={rank} device={device}", rank)
    _log("[supervised] silver image roots:", rank)
    for d in DATASETS:
        _log(f"  {d}: {silver_roots.get(d, '<missing>')}", rank)
    _log("[supervised] gold image roots:", rank)
    for d in DATASETS:
        _log(f"  {d}: {gold_roots.get(d, '<missing>')}", rank)

    df = load_silver_progression_df(
        args.findings_parquet, args.splits_file, split="train",
    )
    if args.limit_train is not None:
        df = df.sample(
            n=min(args.limit_train, len(df)),
            random_state=args.seed,
        ).reset_index(drop=True)
        _log(f"[supervised] --limit-train → {len(df)} silver rows", rank)

    gold_groups = None
    if not args.skip_gold:
        gold_df = load_gold_pairs(
            args.gold_parquet,
            args.findings_parquet,
        )
        gold_groups = group_gold_by_pair_finding(gold_df)
        if args.limit_gold is not None:
            gold_groups = gold_groups.head(args.limit_gold).reset_index(drop=True)
            _log(f"[supervised] --limit-gold → {len(gold_groups)} groups", rank)
        _log(
            f"[supervised] gold set-match after every epoch "
            f"({len(gold_groups)} groups, rank-0 only)",
            rank,
        )

    weights = _class_weights(df["progression"].tolist()).to(device)
    _log("[supervised] class-balanced CE weights:", rank)
    for cls, w in zip(CLS_ORDER, weights.tolist()):
        _log(f"  {cls:<10} {w:.3f}", rank)

    ds = SilverProgressionDataset(df, silver_roots)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=args.seed,
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=_collate,
        drop_last=(world_size > 1),
        pin_memory=(device.type == "cuda"),
    )

    holder = BioViLTPairModel(device)
    model = SupervisedProgressionModel(
        holder.image_encoder, holder.text_encoder,
    ).to(device)

    start_epoch = 1
    resumed_train_loss = None
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(
                f"--resume checkpoint not found: {args.resume}"
            )
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "finding_vocab" in ckpt:
            raise ValueError(
                f"{args.resume} is a finding-ID (embed/onehot) checkpoint."
            )
        model.head.load_state_dict(ckpt["head"])
        if "image_encoder" in ckpt:
            model.image_encoder.load_state_dict(ckpt["image_encoder"])
        if "text_encoder" in ckpt:
            model.text_encoder.load_state_dict(ckpt["text_encoder"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "train_loss" in ckpt:
            resumed_train_loss = float(ckpt["train_loss"])
        _log(
            f"[supervised] resumed {args.resume} "
            f"(completed epoch {start_epoch - 1}) → "
            f"will train epochs {start_epoch}..{args.epochs}",
            rank,
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"--resume finished epoch {start_epoch - 1} but "
                f"--epochs {args.epochs}. Pass --epochs N with "
                f"N >= {start_epoch} (e.g. --epochs 10)."
            )

    for p in model.parameters():
        p.requires_grad = True
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            find_unused_parameters=True,
        )

    remaining_epochs = args.epochs - start_epoch + 1
    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    num_steps = max(1, len(loader) * remaining_epochs)
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(args.warmup_ratio * num_steps),
        num_training_steps=num_steps,
    )
    _log(
        f"[supervised] AdamW lr={args.lr} wd={args.weight_decay} "
        f"warmup_ratio={args.warmup_ratio} steps={num_steps} "
        f"over {remaining_epochs} remaining epoch(s) "
        f"(fresh warmup/decay; original 5-epoch run already hit lr=0)",
        rank,
    )

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    best_loss = (
        float(resumed_train_loss)
        if resumed_train_loss is not None
        else float("inf")
    )

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        n_ok = 0
        n_seen = 0
        skipped = 0
        pbar = tqdm(
            loader,
            desc=f"train ep{epoch}/{args.epochs}",
            dynamic_ncols=True,
            file=sys.stdout,
            disable=(rank != 0),
        )
        for batch in pbar:
            if batch is None:
                skipped += args.batch_size
                if rank == 0:
                    pbar.set_postfix(acc="-", skipped=skipped)
                continue
            priors, currents, findings, labels = batch
            priors = priors.to(device, non_blocking=True)
            currents = currents.to(device, non_blocking=True)
            y = labels.to(device, non_blocking=True)
            logits = model(priors, currents, findings)
            loss = F.cross_entropy(logits, y, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            running += float(loss.item()) * y.size(0)
            n_ok += int((logits.argmax(dim=-1) == y).sum().item())
            n_seen += int(y.size(0))
            if rank == 0:
                pbar.set_postfix(
                    loss=f"{running / max(1, n_seen):.3f}",
                    acc=f"{n_ok / max(1, n_seen):.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    skipped=skipped,
                )
        pbar.close()

        stats = torch.tensor(
            [running, float(n_ok), float(n_seen), float(skipped)],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        running, n_ok, n_seen, skipped = stats.tolist()
        mean_loss = running / max(1.0, n_seen)
        acc = n_ok / max(1.0, n_seen)
        _log(
            f"[supervised] epoch {epoch}/{args.epochs}  "
            f"train_loss={mean_loss:.4f} train_acc={acc:.4f}  "
            f"seen={int(n_seen)} skipped={int(skipped)}   "
            f"(train_acc = silver row argmax, not gold set-match)",
            rank,
        )
        if rank == 0:
            meta = {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_acc": acc,
            }
            _save_ckpt(
                model, os.path.join(args.out_dir, f"epoch_{epoch}.pt"), meta,
            )
            if mean_loss < best_loss:
                best_loss = mean_loss
                _save_ckpt(model, os.path.join(args.out_dir, "best.pt"), meta)
                _log(f"[supervised]   saved {args.out_dir}/best.pt", rank)

            if gold_groups is not None:
                eval_gold_setmatch(
                    model, gold_groups, gold_roots, device, epoch,
                )
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


def load_supervised_scorer(
    ckpt_path: str,
    device: torch.device,
    image_roots: Dict[str, str],
):
    """Return ``score_fn(row, text_cache)`` using logits as class scores."""
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"supervised checkpoint not found: {ckpt_path}. "
            "Train first with python train_supervised_progression.py --train"
        )
    print(f"[supervised] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "finding_vocab" in ckpt:
        raise ValueError(
            f"{ckpt_path} is a finding-ID (embed/onehot) checkpoint. "
            "Use --backend supervised_embed."
        )
    holder = BioViLTPairModel(device)
    head = SupervisedHead(in_dim=int(ckpt.get("in_dim", 256)))
    head.load_state_dict(ckpt["head"])
    model = SupervisedProgressionModel(
        holder.image_encoder, holder.text_encoder, head,
    ).to(device)
    if "image_encoder" in ckpt:
        model.image_encoder.load_state_dict(ckpt["image_encoder"])
        print("[supervised] loaded trained image encoder")
    if "text_encoder" in ckpt:
        model.text_encoder.load_state_dict(ckpt["text_encoder"])
        print("[supervised] loaded trained text encoder")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def score_fn(row, text_cache):
        del text_cache  # finding text is encoded by the (possibly trained) text encoder
        prior = load_image_tensor(
            row["dataset"], row["parent_image_prev"], image_roots,
        )
        current = load_image_tensor(
            row["dataset"], row["parent_image_curr"], image_roots,
        )
        logits = model(
            prior.unsqueeze(0).to(device),
            current.unsqueeze(0).to(device),
            [str(row["finding"])],
        ).squeeze(0)
        scores = logits.float().tolist()
        pred_class = int(logits.argmax().item())
        return {
            "cos_class_scores": scores,
            "pred_class": pred_class,
        }

    return score_fn


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", action="store_true", required=True)
    parser.add_argument("--findings-parquet", default=DEFAULT_FINDINGS)
    parser.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE)
    parser.add_argument("--out-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Load a previous epoch_N.pt (weights only; no optimizer) "
            "and continue from epoch N+1 through --epochs. "
            "Uses a fresh warmup/decay over the remaining epochs."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Last epoch index to train (inclusive). With --resume "
        "epoch_5.pt use --epochs 10 to run 6..10.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO,
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument(
        "--gold-parquet",
        default=DEFAULT_GOLD_PARQUET,
        help="Gold parquet for the in-training set-match report.",
    )
    parser.add_argument(
        "--limit-gold",
        type=int,
        default=None,
        help="Only first N gold (pair, finding) groups (smoke tests).",
    )
    parser.add_argument(
        "--skip-gold",
        action="store_true",
        help="Skip the gold set-match report after each epoch.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
