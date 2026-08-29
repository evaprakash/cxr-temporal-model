#!/usr/bin/env python3
"""Supervised pair-head CE + quiet JEPA dynamic-sentence aux.

Gold is **unchanged**: ``Linear([pair_global; finding]) → 5``. Dynamic
text and the predictor are train-only.

    L = CE(head, silver) + W_JEPA * mean_p (1 - cos(ẑ, z_cur))

with ``ẑ = predictor(z_prior, dynamic_sentences)`` and ``z_cur`` from an
EMA target encoder. Default ``W_JEPA=0.1``. Rows without a dynamic
sentence still get CE; the JEPA term is skipped for those rows.

Writes to ``checkpoints_supervised_progression_jepaaux/``.

Usage
-----
    torchrun --nproc_per_node=4 train_supervised_jepa_aux.py --train

    python train_supervised_jepa_aux.py --train --device cpu \\
        --epochs 1 --limit-train 32 --limit-gold 40 --batch-size 4
"""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from dataset_combined_jepa import (
    DEFAULT_FINDINGS,
    DEFAULT_SENTENCES,
    DEFAULT_SPLITS_FILE,
    _nonempty,
    _normalize_ids,
)
from eval_progression_biovilt import BioViLTPairModel
from eval_progression_jepa_silver import load_silver_progression_df
from infer_jepa import IMAGE_ROOTS
from losses_jepa import jepa_cosine_loss
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from gold_progression_setmatch import group_gold_by_pair_finding
from progression_phrases import CLS_ORDER
from tempcxr.modules.jepa import (
    EMA_END,
    EMA_START,
    IJEPATemporalPredictor,
    _build_target_encoder,
    _update_ema,
    make_momentum_scheduler,
)
from train_supervised_progression import (
    DEFAULT_LR,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    SilverProgressionDataset,
    SupervisedProgressionModel,
    _class_weights,
    _log,
    _setup_distributed,
    _unwrap,
    eval_gold_setmatch,
)

DEFAULT_OUT_DIR = "checkpoints_supervised_progression_jepaaux"
DEFAULT_W_JEPA = 0.1
CLS_TO_IDX = {c: i for i, c in enumerate(CLS_ORDER)}


def attach_dynamic_reports(
    df: pd.DataFrame,
    sentences_parquet: str,
) -> pd.DataFrame:
    """Left-join joined ``label==dynamic`` sentences onto silver finding rows."""
    if not os.path.isfile(sentences_parquet):
        raise FileNotFoundError(f"silver sentences not found: {sentences_parquet}")
    sent = _normalize_ids(pd.read_parquet(sentences_parquet))
    if "label" not in sent.columns or "sentence" not in sent.columns:
        raise ValueError(
            f"{sentences_parquet} needs label + sentence; got {list(sent.columns)}"
        )
    dyn = sent[sent["label"].astype(str).str.strip() == "dynamic"].copy()
    dyn = dyn[dyn["sentence"].apply(_nonempty)]
    grouped = (
        dyn.groupby(["dataset", "patient_id", "study_id"], sort=False)["sentence"]
        .apply(lambda s: " ".join(str(x).strip() for x in s if _nonempty(x)))
        .reset_index()
        .rename(columns={"sentence": "dynamic_report", "study_id": "study_id_curr"})
    )
    out = df.merge(
        grouped,
        on=["dataset", "patient_id", "study_id_curr"],
        how="left",
    )
    out["dynamic_report"] = (
        out["dynamic_report"].fillna("").astype(str).map(lambda x: x.strip())
    )
    return out


class SilverJEPAAuxDataset(SilverProgressionDataset):
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
        dyn = str(row.get("dynamic_report", "") or "")
        return prior, current, str(row["finding"]), y, dyn


def _collate_aux(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    priors, currents, findings, labels, dynamics = zip(*batch)
    return (
        torch.stack(priors, dim=0),
        torch.stack(currents, dim=0),
        list(findings),
        torch.tensor(labels, dtype=torch.long),
        list(dynamics),
    )


class SupervisedJEPAAuxModel(SupervisedProgressionModel):
    """Pair-head CE model plus a JEPA predictor / EMA target."""

    def __init__(self, image_encoder, text_encoder, head=None):
        super().__init__(image_encoder, text_encoder, head)
        self.predictor = IJEPATemporalPredictor()
        self.target_image_encoder = _build_target_encoder(image_encoder)

    def forward(self, priors, currents, findings, dynamic_texts=None):
        logits = self.head(self.encode(priors, currents, findings))
        if dynamic_texts is None:
            return logits
        return logits, self._jepa_aux(priors, currents, dynamic_texts)

    def _jepa_aux(self, priors, currents, dynamic_texts: List[str]):
        active = [i for i, t in enumerate(dynamic_texts) if str(t).strip()]
        if not active:
            return priors.new_zeros(())
        idx = torch.tensor(active, device=priors.device, dtype=torch.long)
        texts = [dynamic_texts[i] for i in active]
        _, z_prior = self.image_encoder(priors.index_select(0, idx))
        with torch.no_grad():
            _, z_cur = self.target_image_encoder(currents.index_select(0, idx))
        z_cur = z_cur.detach()
        _, txt_local, token_mask = self.text_encoder.forward_contrastive(texts)
        zhat = self.predictor(z_prior, txt_local, token_mask)
        return jepa_cosine_loss(zhat.float(), z_cur.float())

    def update_ema(self, momentum: float) -> None:
        _update_ema(self.image_encoder, self.target_image_encoder, momentum)


def _save_ckpt(model, path: str, meta: dict) -> None:
    raw = _unwrap(model)
    torch.save(
        {
            "head": raw.head.state_dict(),
            "image_encoder": raw.image_encoder.state_dict(),
            "text_encoder": raw.text_encoder.state_dict(),
            "predictor": raw.predictor.state_dict(),
            "target_image_encoder": raw.target_image_encoder.state_dict(),
            "in_dim": 256,
            "classes": list(CLS_ORDER),
            "unfrozen": True,
            "jepa_aux": True,
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
    overrides = {}
    for spec in args.image_root:
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(f"unknown dataset {d!r}")
        overrides[d] = p
    silver_roots = {**IMAGE_ROOTS, **auto_roots, **overrides}
    gold_roots = {**IMAGE_ROOTS, **overrides, **auto_roots}

    _log(
        f"[jepa-aux] pair-head CE + W_JEPA={args.jepa_weight} "
        f"dynamic-sentence JEPA (gold = finding head only)",
        rank,
    )
    _log(f"[jepa-aux] DDP world_size={world_size} rank={rank} device={device}", rank)

    df = load_silver_progression_df(
        args.findings_parquet, args.splits_file, split="train",
    )
    df = attach_dynamic_reports(df, args.sentences_parquet)
    n_dyn = int((df["dynamic_report"].str.len() > 0).sum())
    _log(
        f"[jepa-aux] silver train rows={len(df)}  "
        f"with dynamic text={n_dyn} ({n_dyn / max(1, len(df)):.3f})",
        rank,
    )
    if args.limit_train is not None:
        df = df.sample(
            n=min(args.limit_train, len(df)),
            random_state=args.seed,
        ).reset_index(drop=True)
        _log(f"[jepa-aux] --limit-train → {len(df)} rows", rank)

    gold_groups = None
    if not args.skip_gold:
        gold_df = load_gold_pairs(args.gold_parquet, args.findings_parquet)
        gold_groups = group_gold_by_pair_finding(gold_df)
        if args.limit_gold is not None:
            gold_groups = gold_groups.head(args.limit_gold).reset_index(drop=True)
        _log(
            f"[jepa-aux] gold set-match after every epoch "
            f"({len(gold_groups)} groups, rank-0 only, supervised head)",
            rank,
        )

    weights = _class_weights(df["progression"].tolist()).to(device)
    _log("[jepa-aux] class-balanced CE weights:", rank)
    for cls, w in zip(CLS_ORDER, weights.tolist()):
        _log(f"  {cls:<10} {w:.3f}", rank)

    ds = SilverJEPAAuxDataset(df, silver_roots)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True,
            drop_last=True, seed=args.seed,
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=_collate_aux,
        drop_last=(world_size > 1),
        pin_memory=(device.type == "cuda"),
    )

    holder = BioViLTPairModel(device)
    model = SupervisedJEPAAuxModel(
        holder.image_encoder, holder.text_encoder,
    ).to(device)
    for p in model.parameters():
        if p is not None:
            p.requires_grad = True
    for p in model.target_image_encoder.parameters():
        p.requires_grad = False
    model.target_image_encoder.eval()

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index],
            find_unused_parameters=True,
        )

    opt = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    num_steps = max(1, len(loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        opt,
        num_warmup_steps=int(args.warmup_ratio * num_steps),
        num_training_steps=num_steps,
    )
    momentum_sched = make_momentum_scheduler(
        EMA_START, EMA_END, total_iters=num_steps,
    )
    _log(
        f"[jepa-aux] AdamW lr={args.lr} W_JEPA={args.jepa_weight} "
        f"steps={num_steps} epochs={args.epochs}",
        rank,
    )

    if rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        _unwrap(model).target_image_encoder.eval()
        running = 0.0
        running_ce = 0.0
        running_jepa = 0.0
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
                continue
            priors, currents, findings, labels, dynamics = batch
            priors = priors.to(device, non_blocking=True)
            currents = currents.to(device, non_blocking=True)
            y = labels.to(device, non_blocking=True)
            logits, jepa = model(priors, currents, findings, dynamics)
            ce = F.cross_entropy(logits, y, weight=weights)
            loss = ce + args.jepa_weight * jepa
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()
            try:
                m = next(momentum_sched)
            except StopIteration:
                m = EMA_END
            _unwrap(model).update_ema(m)

            running += float(loss.item()) * y.size(0)
            running_ce += float(ce.item()) * y.size(0)
            running_jepa += float(jepa.detach().item()) * y.size(0)
            n_ok += int((logits.argmax(dim=-1) == y).sum().item())
            n_seen += int(y.size(0))
            if rank == 0:
                pbar.set_postfix(
                    loss=f"{running / max(1, n_seen):.3f}",
                    ce=f"{running_ce / max(1, n_seen):.3f}",
                    jepa=f"{running_jepa / max(1, n_seen):.3f}",
                    acc=f"{n_ok / max(1, n_seen):.3f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
        pbar.close()

        stats = torch.tensor(
            [
                running, running_ce, running_jepa,
                float(n_ok), float(n_seen), float(skipped),
            ],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        running, running_ce, running_jepa, n_ok, n_seen, skipped = stats.tolist()
        mean_loss = running / max(1.0, n_seen)
        _log(
            f"[jepa-aux] epoch {epoch}/{args.epochs}  "
            f"total={mean_loss:.4f} ce={running_ce / max(1.0, n_seen):.4f} "
            f"jepa={running_jepa / max(1.0, n_seen):.4f} "
            f"train_acc={n_ok / max(1.0, n_seen):.4f}  "
            f"seen={int(n_seen)} skipped={int(skipped)}",
            rank,
        )
        if rank == 0:
            meta = {
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_acc": n_ok / max(1.0, n_seen),
                "jepa_weight": args.jepa_weight,
            }
            _save_ckpt(
                model, os.path.join(args.out_dir, f"epoch_{epoch}.pt"), meta,
            )
            if mean_loss < best_loss:
                best_loss = mean_loss
                _save_ckpt(model, os.path.join(args.out_dir, "best.pt"), meta)
                _log(f"[jepa-aux]   saved {args.out_dir}/best.pt", rank)
            if gold_groups is not None:
                eval_gold_setmatch(
                    model, gold_groups, gold_roots, device, epoch,
                    backend_name="supervised_jepaaux",
                )
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", action="store_true", required=True)
    parser.add_argument("--findings-parquet", default=DEFAULT_FINDINGS)
    parser.add_argument("--sentences-parquet", default=DEFAULT_SENTENCES)
    parser.add_argument("--splits-file", default=DEFAULT_SPLITS_FILE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--jepa-weight", type=float, default=DEFAULT_W_JEPA)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--gold-parquet", default=DEFAULT_GOLD_PARQUET)
    parser.add_argument("--limit-gold", type=int, default=None)
    parser.add_argument("--skip-gold", action="store_true")
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
