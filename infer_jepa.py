"""Single-example inference demo for the JEPA model.

Loads ``best.pt``, picks one paired sample from the val split, and runs
the full prior + condition-text -> predicted-current pipeline. Reports
how close the prediction is to the actual current image's encoding.

The condition text comes from the dataset's ``condition_mode`` (see
``dataset_combined_jepa.JEPACombinedDataset``); pick the same mode your
checkpoint was trained with via the ``CONDITION_MODE`` env var (default
``dynamic``).

Reported metrics
----------------
JEPA cosine distance
    ``1 - cos(ẑ_cur, z_cur)`` averaged over patches. This is the
    training-time loss on the unit-sphere model. Lower is better.

JEPA Smooth L1 (diagnostic)
    ``F.smooth_l1_loss(ẑ_cur, z_cur)``. On unit-norm vectors this is
    monotonically related to the cosine distance; reported for
    historical comparison with pre-unit-sphere runs.

Slide-deck inference score
    ``cos(ẑ_cur - z_prior, z_cur - z_prior)``: cosine similarity between
    the predicted change vector and the actual change vector. Range
    [-1, 1]; higher is better. This is the inference-time score
    written on the slide deck.

Per-patch cosine similarity
    Mean / min / max cosine between predicted and target patch features
    across the 196 patches.

Do-nothing baseline
    Same metrics but using ``z_prior`` as the prediction (i.e., a
    delta of zero). Confirms that the predictor is doing better than
    "assume nothing changed".

Usage
-----
    # Random val sample on the default best.pt
    python infer_jepa.py

    # Specific val sample
    python infer_jepa.py --idx 42

    # Different checkpoint
    python infer_jepa.py --ckpt checkpoints_jepa_dynamic/epoch_25.pt --idx 42
"""

import argparse
import os
import random

import torch
import torch.nn.functional as F

from dataset_combined_jepa import JEPACombinedDataset
from tempcxr.modules.jepa import TempCXRJEPA


# ============================================================
# IMAGE ROOTS (must match training)
# ============================================================
IMAGE_ROOTS = {
    "mimic": "/home/evaprakash/all_data/mimic",
    "chexpert": "/home/evaprakash/all_data/chexpert/train",
    "rexgradient": "/home/evaprakash/all_data/rexgradient/deid_png",
}


# ============================================================
# SHARED HELPERS (also imported by progression_classify.py)
# ============================================================
def load_jepa_model(ckpt_path: str, device: torch.device) -> TempCXRJEPA:
    """Build a TempCXRJEPA and load weights from ``ckpt_path``.

    The checkpoint dict is the one produced by ``resume_train_jepa.py``:
    ``ckpt["model"]`` is the full state dict (online + EMA target image
    encoders, text encoder, predictor).
    """
    print(f"[infer] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise ValueError(
            f"{ckpt_path} does not look like a JEPA checkpoint "
            f"(missing 'model' key). Top-level keys: {list(ckpt.keys())}"
        )
    print(f"[infer]   epoch={ckpt.get('epoch', '?')}  "
          f"best_val_loss={ckpt.get('best_val_loss', '?')}")
    model = TempCXRJEPA()
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[infer] WARNING: {len(missing)} missing keys "
              f"(first 5: {missing[:5]})")
    if unexpected:
        print(f"[infer] WARNING: {len(unexpected)} unexpected keys "
              f"(first 5: {unexpected[:5]})")
    model.to(device).eval()
    return model


@torch.no_grad()
def encode_pair_with_text(
    model: TempCXRJEPA,
    prior: torch.Tensor,
    current: torch.Tensor,
    condition_text: list,
    device: torch.device,
):
    """Run the full forward pass for ONE (prior, current, condition) sample.

    Uses placeholder text for prior_report / current_report since this
    helper is for inference where we only care about the predictor side.
    The text encoder is still run on those placeholders because the
    forward pass batches all three text inputs together; their outputs
    are returned but ignored by callers that only need the predictor's
    output.

    Returns a dict with at least:
        prior_patches            (1, 196, D)  unit-norm, online encoder
        current_patches_target   (1, 196, D)  unit-norm, EMA encoder, detached
        pred_current_patches     (1, 196, D)  unit-norm predictor output ẑ_cur
    """
    prior = prior.to(device)
    current = current.to(device)
    placeholder = [""] * len(condition_text)
    return model(prior, current, placeholder, placeholder, condition_text)


def jepa_metrics(pred: torch.Tensor, target: torch.Tensor, prior: torch.Tensor):
    """Standard JEPA-side metrics for a single (B=1) sample.

    Returns a dict of python floats.
    """
    pred_f = pred.float()
    target_f = target.float()
    prior_f = prior.float()

    # Training-time loss on the unit sphere: 1 - cos(pred, target).
    cos_patches = F.cosine_similarity(pred_f, target_f, dim=-1)  # (B, N)
    cosine_dist = (1.0 - cos_patches).mean().item()

    # Smooth L1 kept as a diagnostic — monotone in cosine distance on
    # unit-norm vectors, but lets you compare to pre-unit-sphere runs.
    smooth_l1 = F.smooth_l1_loss(pred_f, target_f).item()

    delta_pred = (pred_f - prior_f).flatten()
    delta_true = (target_f - prior_f).flatten()
    cos_delta = F.cosine_similarity(delta_pred, delta_true, dim=0).item()

    return {
        "cosine_dist": cosine_dist,
        "smooth_l1": smooth_l1,
        "cos_delta": cos_delta,
        "cos_patch_mean": cos_patches.mean().item(),
        "cos_patch_min": cos_patches.min().item(),
        "cos_patch_max": cos_patches.max().item(),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.environ.get("JEPA_CKPT", "checkpoints_jepa_dynamic/best.pt"),
        help="Path to a JEPA checkpoint (default: checkpoints_jepa_dynamic/best.pt).",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Val sample index. If omitted, picks one at random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used when --idx is not given.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
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
    parser.add_argument(
        "--condition-mode",
        default=os.environ.get("CONDITION_MODE", "templated"),
        choices=("dynamic", "templated"),
        help=(
            "Which text condition to feed the predictor. Should match "
            "what the checkpoint was trained with. Defaults to the "
            "CONDITION_MODE env var, falling back to 'templated' (the "
            "current training default)."
        ),
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Model ----
    model = load_jepa_model(args.ckpt, device)

    # ---- Val dataset (no augmentation) ----
    val_ds = JEPACombinedDataset(
        image_roots=IMAGE_ROOTS,
        split="val",
        train=False,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        condition_mode=args.condition_mode,
    )
    if len(val_ds) == 0:
        raise RuntimeError("Val split is empty — check IMAGE_ROOTS and CHEXTEMPORAL_DIR.")

    if args.idx is None:
        rng = random.Random(args.seed)
        args.idx = rng.randrange(len(val_ds))
    if not (0 <= args.idx < len(val_ds)):
        raise IndexError(f"--idx {args.idx} out of range [0, {len(val_ds)})")

    sample = val_ds[args.idx]
    row = val_ds.df.iloc[args.idx]

    print(f"\n=== Val sample {args.idx} of {len(val_ds)} ===")
    print(f"  dataset:       {row['dataset']}")
    print(f"  patient_id:    {row['patient_id']}")
    print(f"  study_id_prev: {row['study_id_prev']}")
    print(f"  study_id_curr: {row['study_id_curr']}")
    print(f"  condition_mode: {val_ds.condition_mode}")
    cond_preview = sample["condition_text"][:300].replace("\n", " ")
    print(f"  condition_text (first 300 chars):\n    {cond_preview}")

    # ---- Forward ----
    out = encode_pair_with_text(
        model,
        prior=sample["prior_image"].unsqueeze(0),
        current=sample["current_image"].unsqueeze(0),
        condition_text=[sample["condition_text"]],
        device=device,
    )
    pred = out["pred_current_patches"]
    target = out["current_patches_target"]
    z_prior = out["prior_patches"]

    # ---- Metrics: predictor ----
    m_pred = jepa_metrics(pred, target, z_prior)

    # ---- Metrics: do-nothing baseline (ẑ_cur = z_prior, i.e. Δz = 0) ----
    cos_patches_naive = F.cosine_similarity(z_prior.float(), target.float(), dim=-1)
    naive_patch_mean = cos_patches_naive.mean().item()
    cosine_dist_naive = (1.0 - cos_patches_naive).mean().item()
    smooth_l1_naive = F.smooth_l1_loss(z_prior.float(), target.float()).item()

    print("\n=== Predictor (z_prior + Δz, L2-norm'd) vs target z_cur ===")
    print(f"  JEPA cosine distance:           {m_pred['cosine_dist']:.4f}")
    print(f"  JEPA Smooth L1 (diagnostic):    {m_pred['smooth_l1']:.4f}")
    print(f"  Slide-deck cos(Δẑ, Δz_true):    {m_pred['cos_delta']:.4f}")
    print(f"  Per-patch cos sim mean/min/max: "
          f"{m_pred['cos_patch_mean']:.4f} / "
          f"{m_pred['cos_patch_min']:.4f} / "
          f"{m_pred['cos_patch_max']:.4f}")

    print("\n=== Do-nothing baseline (ẑ_cur := z_prior) ===")
    print(f"  JEPA cosine distance:           {cosine_dist_naive:.4f}")
    print(f"  JEPA Smooth L1 (diagnostic):    {smooth_l1_naive:.4f}")
    print(f"  Per-patch cos sim mean:         {naive_patch_mean:.4f}")

    # ---- Improvement ----
    if cosine_dist_naive > 0:
        improvement = (cosine_dist_naive - m_pred["cosine_dist"]) / cosine_dist_naive * 100.0
        print(f"\nCosine distance improvement of predictor over do-nothing: {improvement:+.1f}%")

    print("\nInterpretation:")
    print("  - Lower cosine distance than do-nothing => predictor moves in the right direction.")
    print("  - Slide-deck cosine close to +1         => predicted change aligns with actual change.")
    print("  - Slide-deck cosine close to  0         => predicted change is uninformative.")
    print("  - Slide-deck cosine close to -1         => predicted change opposes the actual change.")


if __name__ == "__main__":
    main()
