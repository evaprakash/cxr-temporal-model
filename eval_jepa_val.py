"""Average JEPA inference metrics over the entire val split.

Same metrics as ``infer_jepa.py``, but computed once per pair and
averaged across the val set rather than reported for a single sample.

What gets averaged
------------------
JEPA Smooth L1 (predictor)
    ``F.smooth_l1_loss(ẑ_cur, z_cur)`` averaged across val pairs. Should
    closely match the ``val_jepa`` column in ``logs/val_metrics_jepa.csv``
    at the resumed epoch (modulo augmentation differences — eval runs
    without augmentation).

JEPA Smooth L1 (do-nothing)
    Same loss but with ``ẑ_cur := LN(z_prior)`` (i.e., Δz = 0). The
    "trivial" baseline. If the predictor's loss is below this, the
    model genuinely beats "assume nothing changed" on average.

Slide-deck cos(Δẑ, Δz_true)
    Per-sample cosine between predicted change and actual change,
    averaged across val. ``cos(ẑ_cur - z_prior, z_cur - z_prior)``.
    Positive numbers mean the predictor is, on average, picking the
    right direction of change.

Per-patch cosine (predictor / do-nothing)
    Per-sample mean over the 196 patches of the cosine between
    predicted-vs-target and prior-vs-target patch features. Sanity
    check that the absolute prediction quality is reasonable.

Each metric is reduced **per sample first** and then averaged across
samples, so each pair contributes equally regardless of batch.

Usage
-----
    python eval_jepa_val.py
    python eval_jepa_val.py --limit 500            # quick smoke test
    python eval_jepa_val.py --batch-size 32        # if VRAM allows
    python eval_jepa_val.py --ckpt checkpoints_jepa/epoch_25.pt
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_combined_jepa import JEPACombinedDataset, jepa_collate_fn
from infer_jepa import IMAGE_ROOTS, load_jepa_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.environ.get("JEPA_CKPT", "checkpoints_jepa/best.pt"),
        help="Path to a JEPA checkpoint (default: checkpoints_jepa/best.pt).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N val samples (debug / smoke test).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Must match training (default 0.1) so the val split agrees.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Must match training (default 42) so the val split agrees.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_jepa_model(args.ckpt, device)

    val_ds = JEPACombinedDataset(
        image_roots=IMAGE_ROOTS,
        split="val",
        train=False,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    if args.limit is not None:
        val_ds.df = val_ds.df.head(args.limit).reset_index(drop=True)
        print(f"[eval] limiting to first {len(val_ds)} val samples")

    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=jepa_collate_fn,
        pin_memory=True,
    )

    # Per-sample sums; divide by n at the end so each pair contributes equally
    # regardless of batch composition.
    sums = {
        "smooth_l1_pred": 0.0,
        "smooth_l1_naive": 0.0,
        "cos_delta": 0.0,
        "cos_patch_pred": 0.0,
        "cos_patch_naive": 0.0,
    }
    n = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="val"):
            prior = batch["prior_image"].to(device, non_blocking=True)
            current = batch["current_image"].to(device, non_blocking=True)
            out = model(
                prior, current,
                batch["prior_report"], batch["current_report"], batch["dynamic_report"],
            )
            pred    = out["pred_current_patches"].float()      # (B, N, D)
            target  = out["current_patches_target"].float()    # (B, N, D)
            z_prior = out["prior_patches"].float()             # (B, N, D)
            B = pred.size(0)

            # Per-sample Smooth L1 (mean over the N*D feature axes so each
            # sample contributes one number).
            l1_pred  = F.smooth_l1_loss(pred,    target, reduction="none").mean(dim=(1, 2))
            l1_naive = F.smooth_l1_loss(z_prior, target, reduction="none").mean(dim=(1, 2))

            # Slide-deck cosine: treat the entire (N*D) Δz as one vector
            # per sample, take cos between predicted Δẑ and true Δz_true.
            dpred = (pred   - z_prior).flatten(start_dim=1)
            dtrue = (target - z_prior).flatten(start_dim=1)
            cos_delta = F.cosine_similarity(dpred, dtrue, dim=1)  # (B,)

            # Per-patch cosine, averaged over the 196 patches per sample.
            cos_p = F.cosine_similarity(pred,    target, dim=-1).mean(dim=1)
            cos_n = F.cosine_similarity(z_prior, target, dim=-1).mean(dim=1)

            sums["smooth_l1_pred"]  += l1_pred.sum().item()
            sums["smooth_l1_naive"] += l1_naive.sum().item()
            sums["cos_delta"]       += cos_delta.sum().item()
            sums["cos_patch_pred"]  += cos_p.sum().item()
            sums["cos_patch_naive"] += cos_n.sum().item()
            n += B

    if n == 0:
        raise RuntimeError("No val samples evaluated — check IMAGE_ROOTS and CHEXTEMPORAL_DIR.")

    means = {k: v / n for k, v in sums.items()}

    print(f"\n=== Averaged over {n} val pairs ===")
    print(f"  Smooth L1 (predictor):       {means['smooth_l1_pred']:.4f}")
    print(f"  Smooth L1 (do-nothing):      {means['smooth_l1_naive']:.4f}")
    print(f"  Slide-deck cos(Δẑ, Δz_true): {means['cos_delta']:.4f}")
    print(f"  Per-patch cos (predictor):   {means['cos_patch_pred']:.4f}")
    print(f"  Per-patch cos (do-nothing):  {means['cos_patch_naive']:.4f}")

    if means["smooth_l1_naive"] > 0:
        delta = (
            (means["smooth_l1_naive"] - means["smooth_l1_pred"])
            / means["smooth_l1_naive"]
            * 100.0
        )
        print(f"\n  Smooth L1 improvement of predictor over do-nothing: {delta:+.1f}%")

    print(
        "\nInterpretation:\n"
        "  - Predictor Smooth L1 < do-nothing  => model genuinely beats "
        "the trivial baseline on average.\n"
        "  - Slide-deck cosine clearly > 0     => predicted change "
        "directions match actual change directions.\n"
        "  - Per-patch cos sims should both be high (0.9+) if the "
        "encoders are healthy.\n"
        "Compare predictor Smooth L1 to the val_jepa column in your "
        "training log at the resumed epoch — they should match modulo "
        "augmentation (eval here runs without augmentation)."
    )


if __name__ == "__main__":
    main()
