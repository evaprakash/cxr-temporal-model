"""Smoke test for sometimes-masked progression loss.

Verifies:
  1. Uniform / all-ones ``patch_weights`` match the legacy unmasked
     ``progression_classification_loss`` (bit-identical logits path).
  2. A concentrated mask changes the CE vs global mean, and planting
     ``ẑ_cur^label ≈ z_cur`` *inside the masked patches* drives loss ~0
     while planting only outside the mask does not.
  3. ``TempCXRJEPA.forward`` + masked progression CE + ``backward()``
     succeed; gradients reach the online encoder / predictor / text
     encoder, not the EMA target.
  4. (Optional) If ``CheXTemporal/filtered_masks`` exists, load one real
     JSON, warp it, and check ``(196,)`` weights with positive mass.

Usage
-----
    # Loss + synthetic-mask checks (CPU ok; GPU preferred for #3):
    python smoke_test_masked_prog.py

    # Also try loading a real filtered mask (cluster):
    python smoke_test_masked_prog.py --try-real-mask

    # Point at a non-default CheXTemporal tree:
    python smoke_test_masked_prog.py --try-real-mask \\
        --masks-root /path/to/CheXTemporal/filtered_masks
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from losses_jepa import progression_classification_loss
from progression_phrases import CLS_ORDER
from silver_masks import (
    N_PATCHES,
    PATCH_GRID,
    default_masks_root,
    load_union_mask_hw,
    mask_hw_to_patch_weights,
    resolve_mask_json_path,
)
from tempcxr.modules.jepa import TempCXRJEPA


N_CLS = len(CLS_ORDER)


def _section(title: str):
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _pass(msg: str):
    print(f"  [PASS] {msg}")


def _fail(msg: str):
    print(f"  [FAIL] {msg}")


def _info(msg: str):
    print(f"         {msg}")


def _grad_norm(module: torch.nn.Module) -> float:
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += p.grad.detach().float().norm(2).item() ** 2
    return sq ** 0.5


def check_uniform_matches_legacy(device: str, failures: list) -> None:
    _section("1. Uniform patch_weights == legacy global mean")
    B, N, D = 4, N_PATCHES, 128
    torch.manual_seed(0)
    pred = F.normalize(torch.randn(B, N_CLS, N, D, device=device), dim=-1)
    target = F.normalize(torch.randn(B, N, D, device=device), dim=-1)
    labels = torch.randint(0, N_CLS, (B,), device=device)

    loss_none = progression_classification_loss(pred, target, labels)
    loss_ones = progression_classification_loss(
        pred, target, labels, patch_weights=torch.ones(B, N, device=device)
    )
    diff = abs(loss_none.item() - loss_ones.item())
    _info(f"loss(None)={loss_none.item():.6f}  loss(ones)={loss_ones.item():.6f}  |Δ|={diff:.2e}")
    if diff < 1e-5:
        _pass("all-ones weights match patch_weights=None")
    else:
        _fail("all-ones weights diverge from legacy mean")
        failures.append("uniform_vs_legacy")


def check_mask_focuses_loss(device: str, failures: list) -> None:
    _section("2. Masked mean focuses CE on selected patches")
    B, N, D = 2, N_PATCHES, 32
    torch.manual_seed(1)

    # Build a hard mask on the first 16 patches only.
    w = torch.zeros(B, N, device=device)
    w[:, :16] = 1.0

    target = F.normalize(torch.randn(B, N, D, device=device), dim=-1)
    # Five candidates: class 0 matches target ONLY on masked patches;
    # class 1 matches ONLY on unmasked patches; others are noise.
    pred = F.normalize(torch.randn(B, N_CLS, N, D, device=device), dim=-1)
    pred = pred.clone()
    pred[:, 0, :16, :] = target[:, :16, :]
    pred[:, 1, 16:, :] = target[:, 16:, :]
    pred = F.normalize(pred, dim=-1)

    labels = torch.zeros(B, dtype=torch.long, device=device)  # class 0

    loss_masked = progression_classification_loss(
        pred, target, labels, temperature=0.1, patch_weights=w
    )
    loss_global = progression_classification_loss(
        pred, target, labels, temperature=0.1, patch_weights=None
    )

    # With mask, class 0 should be a near-perfect match → low CE.
    # Globally, class 0 is only good on 16/196 patches so CE is higher.
    _info(f"loss_masked(label=0)={loss_masked.item():.4f}")
    _info(f"loss_global(label=0)={loss_global.item():.4f}")

    if loss_masked.item() < 0.25:
        _pass("masked loss near 0 when label matches inside the mask")
    else:
        _fail(f"masked loss unexpectedly high: {loss_masked.item():.4f}")
        failures.append("masked_low_loss")

    if loss_masked.item() + 0.05 < loss_global.item():
        _pass("masked loss < global loss (mask focuses the match)")
    else:
        _fail("masked loss did not improve vs global as expected")
        failures.append("masked_vs_global")

    # Swap label to class 1 (good only outside mask) → masked CE should jump.
    labels_bad = torch.ones(B, dtype=torch.long, device=device)
    loss_bad = progression_classification_loss(
        pred, target, labels_bad, temperature=0.1, patch_weights=w
    )
    _info(f"loss_masked(label=1, match outside mask)={loss_bad.item():.4f}")
    if loss_bad.item() > loss_masked.item() + 0.5:
        _pass("wrong-class-outside-mask yields higher masked CE")
    else:
        _fail("masked CE did not rise for outside-mask class")
        failures.append("outside_mask_class")


def check_forward_backward(device: str, failures: list) -> None:
    _section("3. TempCXRJEPA forward + masked prog CE + backward")
    B = 2
    model = TempCXRJEPA().to(device)
    model.train()

    prior = torch.randn(B, 3, 448, 448, device=device)
    curr = torch.randn(B, 3, 448, 448, device=device)
    reports = ["Prior report findings."] * B
    curr_reports = ["Current report findings."] * B
    conditions = ["opacity is improving."] * B

    # Pair-major class-minor prompts for finding "edema"
    prompts = []
    for _ in range(B):
        for cls in CLS_ORDER:
            prompts.append(f"Edema is {cls}.")

    # Soft mask: center 6×6 of the 14×14 grid
    w = torch.zeros(B, N_PATCHES, device=device)
    grid = w.view(B, PATCH_GRID, PATCH_GRID)
    grid[:, 4:10, 4:10] = 1.0
    w = grid.view(B, N_PATCHES)
    labels = torch.tensor([0, 2], device=device)

    out = model(
        prior,
        curr,
        reports,
        curr_reports,
        conditions,
        progression_prompts_flat=prompts,
    )
    prog = out["pred_progression_patches"]
    target = out["current_patches_target"]

    if tuple(prog.shape) != (B, N_CLS, N_PATCHES, prog.shape[-1]):
        _fail(f"unexpected pred_progression_patches shape {tuple(prog.shape)}")
        failures.append("prog_shape")
        return
    _pass(f"pred_progression_patches shape={tuple(prog.shape)}")

    if target.requires_grad:
        _fail("current_patches_target unexpectedly requires grad")
        failures.append("target_grad")
    else:
        _pass("current_patches_target is stop-grad")

    loss = progression_classification_loss(
        prog.float(),
        target.float(),
        labels,
        temperature=0.1,
        patch_weights=w,
    )
    _info(f"masked prog CE={loss.item():.4f}")
    if not torch.isfinite(loss):
        _fail("non-finite masked prog loss")
        failures.append("nonfinite")
        return
    _pass("masked prog CE is finite")

    model.zero_grad(set_to_none=True)
    loss.backward()

    g_img = _grad_norm(model.image_encoder)
    g_txt = _grad_norm(model.text_encoder)
    g_pred = _grad_norm(model.predictor)
    g_tgt = _grad_norm(model.target_image_encoder)
    _info(f"grad norms: image={g_img:.3e} text={g_txt:.3e} "
          f"predictor={g_pred:.3e} target_ema={g_tgt:.3e}")

    if g_img > 0 and g_pred > 0 and g_txt > 0:
        _pass("grads flow to online image / text / predictor")
    else:
        _fail("missing grads on online modules")
        failures.append("online_grads")

    if g_tgt == 0.0:
        _pass("EMA target encoder has zero grad")
    else:
        _fail(f"EMA target got grad norm {g_tgt:.3e}")
        failures.append("ema_grad")


def check_real_mask(masks_root: str, failures: list) -> None:
    _section("4. Optional real filtered_masks JSON load")
    root = Path(masks_root)
    if not root.is_dir():
        _info(f"masks_root missing ({root}); skipping real-mask check")
        _pass("skipped (no filtered_masks tree)")
        return

    # Walk a few JSONs; stop at the first successful decode.
    json_paths = list(root.rglob("*.json"))
    _info(f"found {len(json_paths)} json files under {root}")
    if not json_paths:
        _fail("filtered_masks exists but contains no .json files")
        failures.append("no_json")
        return

    # Prefer a path we can reverse into resolve_mask_json_path.
    loaded = False
    for jp in json_paths[:50]:
        try:
            mask_hw = load_union_mask_hw(jp)
            weights = mask_hw_to_patch_weights(mask_hw, aug_params=None)
            if float(weights.sum()) <= 0:
                continue
            _info(f"loaded {jp.relative_to(root)}")
            _info(f"  mask HxW={mask_hw.shape}  "
                  f"patch weight sum={float(weights.sum()):.3f}  "
                  f"nnz={(weights > 0).sum().item()}/{N_PATCHES}")
            # Path join round-trip: dirname layout under filtered_masks
            # is chexpert/... or mimic/...; stem is finding__image
            rel = jp.relative_to(root)
            name = jp.stem  # e.g. atelectasis__view1_frontal
            if "__" not in name:
                continue
            finding_key, img_stem = name.split("__", 1)
            parent_image = str(rel.parent / f"{img_stem}.jpg")
            dataset = rel.parts[0]
            finding = finding_key.replace("_", " ")
            resolved = resolve_mask_json_path(
                root, dataset, parent_image, finding
            )
            if resolved is None or resolved.resolve() != jp.resolve():
                _fail(
                    f"resolve_mask_json_path missed {jp} "
                    f"(got {resolved})"
                )
                failures.append("resolve_path")
            else:
                _pass("resolve_mask_json_path round-trips for a real file")
            if weights.numel() != N_PATCHES:
                _fail(f"expected {N_PATCHES} weights, got {weights.numel()}")
                failures.append("weight_len")
            else:
                _pass("warped mask → (196,) patch weights with mass > 0")
            loaded = True
            break
        except Exception as e:
            _info(f"skip {jp.name}: {e}")
            continue

    if not loaded:
        _fail("could not decode any of the first 50 mask JSONs "
              "(is pycocotools installed?)")
        failures.append("decode_failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--try-real-mask", action="store_true")
    parser.add_argument(
        "--masks-root",
        default=None,
        help="Default: $CHEXTEMPORAL_DIR/filtered_masks or "
             "./CheXTemporal/filtered_masks",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / cpu (default: cuda if available)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning smoke_test_masked_prog on device={device}")
    failures: list = []

    try:
        check_uniform_matches_legacy(device, failures)
        check_mask_focuses_loss(device, failures)
        check_forward_backward(device, failures)
        if args.try_real_mask:
            masks_root = args.masks_root or default_masks_root()
            check_real_mask(masks_root, failures)
        else:
            _section("4. Optional real filtered_masks JSON load")
            _info("skipped (pass --try-real-mask to exercise on cluster)")
            _pass("skipped by default")
    except Exception:
        traceback.print_exc()
        failures.append("exception")

    _section("SUMMARY")
    if failures:
        _fail(f"{len(failures)} failure(s): {failures}")
        return 1
    _pass("all smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
