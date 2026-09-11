#!/usr/bin/env python3
"""5-way JEPA progression classification on CheXTemporal **silver**.

Same image–image scoring rule as gold JEPA eval, but over
``silver_findings.parquet`` (the train/val distribution). Use this for
the train≈test overfitting check: if silver-train accuracy is close to
silver-val (and gold) accuracy, the model is not memorizing.

Default scoring is **per-patch** mean cosine (the pre-global-pool /
``eval_progression_jepa_perpatch`` rule). Pass ``--pooling global`` for
checkpoints trained with global-pool JEPA + progression CE.

Each silver row is one ``(pair, finding, progression)`` — same granularity
as gold. Split filtering uses ``splits_jepa.csv`` (the same file training
writes / reads).

Usage
-----
    # Overfitting check (recommended): stratified subsample
    python eval_progression_jepa_silver.py --eval --split train --limit 5000 \\
        --ckpt checkpoints_jepa_dynamic_cbw99999/best.pt
    python eval_progression_jepa_silver.py --eval --split val --limit 5000 \\
        --ckpt checkpoints_jepa_dynamic_cbw99999/best.pt

    # Full silver val (slow; ~tens of thousands of rows)
    python eval_progression_jepa_silver.py --eval --split val \\
        --ckpt checkpoints_jepa_dynamic_cbw99999/best.pt

    # SLURM
    sbatch eval_progression_jepa_silver.sh
    sbatch eval_progression_jepa_silver.sh --split val --limit 5000
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from dataset_combined_jepa import (
    DEFAULT_FINDINGS,
    DEFAULT_SPLITS_FILE,
    SPLIT_KEY_COLS,
    _ensure_split_assignments,
    _normalize_ids,
)
from eval_progression_jepa import (
    PROMPT_TEMPLATE,
    _compute_balanced_metrics,
    _encode_prompts,
)
from infer_jepa import load_jepa_model
from losses_jepa import global_pool_normalize
from progression_classify import load_image_tensor
from progression_phrases import CLS_ORDER, SILVER_TO_CLS
from tempcxr.modules.jepa import TempCXRJEPA

DATASETS = ["mimic", "chexpert", "rexgradient"]


def _default_image_roots() -> Dict[str, str]:
    """Match ``resume_train_jepa.IMAGE_ROOTS`` (JEPA_IMAGE_ROOTS_DIR)."""
    base = os.environ.get(
        "JEPA_IMAGE_ROOTS_DIR",
        os.path.expanduser("~/all_data"),
    )
    return {
        "mimic": os.path.join(base, "mimic"),
        "chexpert": os.path.join(base, "chexpert", "train"),
        "rexgradient": os.path.join(base, "rexgradient", "deid_png"),
    }


@torch.no_grad()
def score_one_pair(
    model: TempCXRJEPA,
    prior_img: torch.Tensor,
    current_img: torch.Tensor,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, Tuple]] = None,
    pooling: str = "perpatch",
) -> Dict:
    """N-way image–image scoring (per-patch or global-pool)."""
    prompts, txt_local, token_mask = _encode_prompts(
        model, finding, template, device, text_cache,
    )
    n_prompts = len(prompts)

    prior = prior_img.unsqueeze(0).to(device)
    current = current_img.unsqueeze(0).to(device)

    _, z_prior = model.image_encoder(prior)
    _, z_cur = model.target_image_encoder(current)
    z_cur = z_cur.detach()

    z_prior_b = z_prior.expand(n_prompts, -1, -1).contiguous()
    preds = model.predictor(z_prior_b, txt_local, token_mask)

    pred_f = preds.float()
    target_f = z_cur.float()

    if pooling == "global":
        pred_g = global_pool_normalize(pred_f)
        target_g = global_pool_normalize(target_f)
        cos_class_scores = F.cosine_similarity(
            pred_g, target_g.expand_as(pred_g), dim=-1,
        ).tolist()
        cos_naive = F.cosine_similarity(
            global_pool_normalize(z_prior.float()),
            target_g,
            dim=-1,
        ).item()
    else:
        target_exp = target_f.expand_as(pred_f)
        cos_per_patch = F.cosine_similarity(pred_f, target_exp, dim=-1)
        cos_class_scores = cos_per_patch.mean(dim=1).tolist()
        cos_naive = F.cosine_similarity(
            z_prior.float(), z_cur.float(), dim=-1,
        ).mean().item()

    pred_class = max(range(n_prompts), key=lambda k: cos_class_scores[k])
    return {
        "prompts": prompts,
        "cos_class_scores": cos_class_scores,
        "pred_class": pred_class,
        "cos_naive": cos_naive,
    }


def load_silver_progression_df(
    findings_parquet: str,
    splits_file: str,
    split: Optional[str],
) -> pd.DataFrame:
    """Load finding-level silver rows; optionally filter to train/val."""
    print(f"[silver] loading {findings_parquet}")
    df = _normalize_ids(pd.read_parquet(findings_parquet))
    print(f"[silver]   {len(df)} raw rows, columns: {list(df.columns)}")

    need = [
        "dataset",
        "patient_id",
        "study_id_curr",
        "study_id_prev",
        "finding",
        "progression",
        "parent_image_curr",
        "parent_image_prev",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f"silver_findings missing columns {missing}; got {list(df.columns)}"
        )

    df = df.copy()
    df["progression"] = (
        df["progression"].astype(str).str.strip().map(SILVER_TO_CLS)
    )
    df = df[df["progression"].isin(CLS_ORDER)].copy()
    df["finding"] = df["finding"].astype(str).str.strip().str.lower()
    df = df[df["finding"].str.len() > 0].copy()
    df = df[
        df["parent_image_curr"].astype("string").str.strip().ne("")
        & df["parent_image_prev"].astype("string").str.strip().ne("")
    ].copy()
    print(f"[silver]   {len(df)} rows after label / path filters")

    if split is None or split == "all":
        return df.reset_index(drop=True)

    if split not in ("train", "val"):
        raise ValueError(f"--split must be train|val|all; got {split!r}")

    # Pair-level splits — prefer the cached training file; only generate
    # if missing (never rewrite an existing splits_jepa.csv from a
    # findings-only subset, which would shrink training's split table).
    if os.path.exists(splits_file):
        cached = pd.read_csv(
            splits_file,
            dtype={
                "dataset": "string",
                "patient_id": "string",
                "study_id_curr": "string",
                "study_id_prev": "string",
                "split": "string",
            },
        )
        print(f"[silver] loaded splits from {splits_file} ({len(cached)} pairs)")
        df = df.merge(
            cached[SPLIT_KEY_COLS + ["split"]],
            on=SPLIT_KEY_COLS,
            how="inner",
        )
    else:
        print(
            f"[silver] WARNING: {splits_file} missing; generating pair "
            f"splits (same helper as training)"
        )
        pairs = df[SPLIT_KEY_COLS].drop_duplicates().copy()
        pairs = _ensure_split_assignments(
            pairs,
            splits_file=splits_file,
            val_fraction=0.1,
            seed=42,
        )
        df = df.merge(
            pairs[SPLIT_KEY_COLS + ["split"]],
            on=SPLIT_KEY_COLS,
            how="inner",
        )

    before = len(df)
    df = df[df["split"] == split].reset_index(drop=True)
    print(
        f"[silver]   split={split}: {len(df)}/{before} finding-rows "
        f"(splits_file={splits_file})"
    )
    return df


def _stratified_limit(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Cap to ``limit`` rows, roughly preserving class proportions."""
    if limit is None or limit >= len(df):
        return df.reset_index(drop=True)
    rng = random.Random(seed)
    counts = df["progression"].value_counts()
    # Proportional allocation, at least 1 per present class when possible.
    alloc = {}
    remaining = limit
    classes = list(counts.index)
    for i, cls in enumerate(classes):
        if i == len(classes) - 1:
            alloc[cls] = remaining
        else:
            n = max(1, int(round(limit * (counts[cls] / len(df)))))
            n = min(n, int(counts[cls]), remaining - (len(classes) - i - 1))
            alloc[cls] = n
            remaining -= n
    parts = []
    for cls, n in alloc.items():
        idxs = df.index[df["progression"] == cls].tolist()
        rng.shuffle(idxs)
        parts.append(df.loc[idxs[:n]])
    out = pd.concat(parts, ignore_index=True)
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(
        f"[silver] stratified --limit {limit}: kept {len(out)} rows; "
        f"per-class { {c: int((out['progression']==c).sum()) for c in CLS_ORDER} }"
    )
    return out


def _print_eval_summary(
    n_correct: int,
    n_seen: int,
    confusion: Dict,
    per_finding: Dict,
    cos_class_sums: List[float],
    naive_sum: float,
    n_above_naive: int,
    pooling: str,
):
    chance = 1.0 / len(CLS_ORDER)
    print(f"\n{'=' * 60}")
    print(
        f"=== Results: 5-way JEPA silver ({pooling}) image–image matching"
    )
    print(f"{'=' * 60}")
    acc = n_correct / max(1, n_seen)
    print(
        f"Overall accuracy: {n_correct}/{n_seen} = {acc:.4f}    "
        f"(chance = {chance:.3f})"
    )

    print("\nPer-class accuracy (= per-class recall):")
    print(f"  {'gt class':<10} {'n':>6} {'acc':>8}")
    for cls in CLS_ORDER:
        n = sum(confusion[cls].values())
        c = confusion[cls].get(cls, 0)
        a = c / n if n else float("nan")
        print(f"  {cls:<10} {n:>6} {a:>8.4f}")

    print("\nConfusion matrix (rows=gt, cols=pred):")
    header = " ".join(f"{c[:9]:>9}" for c in CLS_ORDER)
    print(f"  {'':<10} {header}")
    for gt in CLS_ORDER:
        cells = " ".join(f"{confusion[gt].get(p, 0):>9}" for p in CLS_ORDER)
        print(f"  {gt:<10} {cells}")

    print("\nMean class cosine (avg over eval samples):")
    print(f"  {'class':<10} {'mean_cos':>10}")
    for k, cls in enumerate(CLS_ORDER):
        print(f"  {cls:<10} {cos_class_sums[k] / max(1, n_seen):>10.4f}")
    print(
        f"  {'naive':<10} {naive_sum / max(1, n_seen):>10.4f}  "
        f"(cos(z_prior, z_cur))"
    )
    print(
        f"\n  Pred beats naive: {n_above_naive}/{n_seen} "
        f"({100.0 * n_above_naive / max(1, n_seen):.1f}%)"
    )

    print("\nPer-finding accuracy:")
    print(f"  {'finding':<26} {'n':>6} {'acc':>8}")
    for finding in sorted(per_finding):
        c, n = per_finding[finding]
        print(f"  {finding:<26} {n:>6} {c / n if n else float('nan'):>8.4f}")

    m = _compute_balanced_metrics(confusion, CLS_ORDER, n_correct)
    print("\nBalanced (imbalance-corrected) metrics:")
    print(f"  macro recall       {m['macro_recall']:>8.4f}")
    print(f"  macro precision    {m['macro_precision']:>8.4f}")
    print(f"  macro F1           {m['macro_f1']:>8.4f}")
    print(f"  Cohen's kappa      {m['cohen_kappa']:>8.4f}")
    print(
        f"  majority baseline  {m['majority_acc']:>8.4f}   "
        f"(always predict {m['majority_class']!r})"
    )


def run_eval(args, model, silver_df, device):
    n_correct = 0
    n_seen = 0
    n_above_naive = 0
    confusion: Dict[str, Counter] = defaultdict(Counter)
    per_finding: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    cos_class_sums = [0.0] * len(CLS_ORDER)
    naive_sum = 0.0
    skipped = 0
    text_cache: Dict[str, Tuple] = {}

    print(
        f"\n[eval] silver 5-way JEPA ({args.pooling}) on {len(silver_df)} rows "
        f"(split={args.split})"
    )
    print(f"[eval] template: {args.prompt_template!r}")
    print(f"[eval] ckpt: {args.ckpt}")

    for i in range(len(silver_df)):
        row = silver_df.iloc[i]
        finding = str(row["finding"])
        gt_label = str(row["progression"])
        try:
            prior = load_image_tensor(
                row["dataset"], row["parent_image_prev"], args.image_roots,
            )
            current = load_image_tensor(
                row["dataset"], row["parent_image_curr"], args.image_roots,
            )
        except (FileNotFoundError, OSError) as e:
            skipped += 1
            if skipped <= 5:
                print(f"[eval] skipping row {i} (missing image: {e})")
            continue

        out = score_one_pair(
            model, prior, current, finding, args.prompt_template, device,
            text_cache=text_cache,
            pooling=args.pooling,
        )
        pred_label = CLS_ORDER[out["pred_class"]]

        n_seen += 1
        n_correct += int(pred_label == gt_label)
        confusion[gt_label][pred_label] += 1
        per_finding[finding][1] += 1
        per_finding[finding][0] += int(pred_label == gt_label)
        if out["cos_class_scores"][out["pred_class"]] > out["cos_naive"]:
            n_above_naive += 1
        for k in range(len(CLS_ORDER)):
            cos_class_sums[k] += out["cos_class_scores"][k]
        naive_sum += out["cos_naive"]

        if (i + 1) % max(1, len(silver_df) // 20) == 0:
            print(
                f"[eval]   {i + 1}/{len(silver_df)}  "
                f"acc={n_correct / max(1, n_seen):.4f}  skipped={skipped}"
            )

    if n_seen == 0:
        print("No samples evaluated.")
        return
    if skipped:
        print(f"\nSkipped {skipped} rows due to missing images")

    _print_eval_summary(
        n_correct=n_correct,
        n_seen=n_seen,
        confusion=confusion,
        per_finding=per_finding,
        cos_class_sums=cos_class_sums,
        naive_sum=naive_sum,
        n_above_naive=n_above_naive,
        pooling=args.pooling,
    )


def run_demo(args, model, silver_df, device):
    if args.idx is None:
        rng = random.Random(args.seed)
        args.idx = rng.randrange(len(silver_df))
    row = silver_df.iloc[args.idx]
    finding = str(row["finding"])
    gt_label = str(row["progression"])

    print(f"\n=== Silver sample {args.idx} of {len(silver_df)} ===")
    print(f"  dataset:       {row['dataset']}")
    print(f"  patient:       {row['patient_id']}")
    print(f"  studies:       {row['study_id_prev']} → {row['study_id_curr']}")
    print(f"  finding:       {finding}")
    print(f"  gt progression:{gt_label}")
    print(f"  split:         {row.get('split', args.split)}")

    prior = load_image_tensor(
        row["dataset"], row["parent_image_prev"], args.image_roots,
    )
    current = load_image_tensor(
        row["dataset"], row["parent_image_curr"], args.image_roots,
    )
    out = score_one_pair(
        model, prior, current, finding, args.prompt_template, device,
        pooling=args.pooling,
    )
    pred_label = CLS_ORDER[out["pred_class"]]
    print(f"\n  predicted: {pred_label}  correct={pred_label == gt_label}")
    print("\nPer-class cosines:")
    for k, cls in enumerate(CLS_ORDER):
        mark = " <-- pred" if k == out["pred_class"] else ""
        gt = " <-- gt" if cls == gt_label else ""
        print(f"  {cls:<10}  {out['cos_class_scores'][k]:>+.4f}{mark}{gt}")
    print(f"  naive       {out['cos_naive']:>+.4f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        default=os.environ.get(
            "JEPA_CKPT",
            "checkpoints_jepa_dynamic_cbw99999/best.pt",
        ),
        help="JEPA checkpoint (default: env JEPA_CKPT or "
             "checkpoints_jepa_dynamic_cbw99999/best.pt).",
    )
    parser.add_argument(
        "--findings-parquet",
        default=os.environ.get("FINDINGS_PARQUET", DEFAULT_FINDINGS),
        help="Path to silver_findings.parquet.",
    )
    parser.add_argument(
        "--splits-file",
        default=os.environ.get("JEPA_SPLITS_FILE", DEFAULT_SPLITS_FILE),
        help="Pair-level train/val split CSV (same as training).",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "all"],
        help="Which silver split to score (default: train).",
    )
    parser.add_argument(
        "--pooling",
        default="perpatch",
        choices=["perpatch", "global"],
        help="Similarity rule. perpatch = mean_p cos (default); "
             "global = cos(pool(ẑ), pool(z_cur)).",
    )
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Two-slot template; default '{} is {}.' (train format).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="Override image root. Can repeat.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stratified subsample size (recommended for train check).",
    )
    parser.add_argument("--seed", type=int, default=0)

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--idx", type=int, default=None, help="(demo) row index.",
    )

    args = parser.parse_args()
    try:
        _ = args.prompt_template.format("test_disease", "test_class")
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"--prompt-template must accept two positional slots: {e}"
        )

    image_roots = _default_image_roots()
    print("[silver] image roots:")
    for d, p in image_roots.items():
        print(f"  {d}: {p}  {'OK' if os.path.isdir(p) else 'MISSING'}")
    for spec in args.image_root:
        if "=" not in spec:
            raise ValueError(f"--image-root expects DATASET=PATH, got {spec!r}")
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {d!r}")
        image_roots[d] = p
        print(f"[silver] override: {d} -> {p}")
    args.image_roots = image_roots

    silver_df = load_silver_progression_df(
        args.findings_parquet,
        args.splits_file,
        None if args.split == "all" else args.split,
    )
    if args.limit is not None:
        silver_df = _stratified_limit(silver_df, args.limit, args.seed)
    if len(silver_df) == 0:
        raise RuntimeError("No silver rows after filtering.")

    device = torch.device(args.device)
    model = load_jepa_model(args.ckpt, device)

    if args.demo:
        run_demo(args, model, silver_df, device)
    else:
        run_eval(args, model, silver_df, device)


if __name__ == "__main__":
    main()
