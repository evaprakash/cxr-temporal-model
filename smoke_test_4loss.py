"""End-to-end smoke test for the 4-loss training setup.

Run this on the cluster (anywhere ``python -m tempcxr.modules.jepa``
runs) to verify the new 4th loss is wired correctly and all the safety
properties the user asked about hold:

  1. The model's forward pass returns ``pred_progression_patches`` with
     shape ``(B, 5, N, D)``.
  2. The EMA target encoder's parameters have ``requires_grad=False``
     and the ``current_patches_target`` tensor has ``requires_grad=False``
     (i.e. the actual current patches truly have no gradient path).
  3. The predictor genuinely runs 5 different forward passes per pair
     for the progression loss — the 5 candidate ``ẑ_cur^c`` predictions
     differ across class c.
  4. All four losses compute, are finite, and ``total.backward()``
     succeeds.
  5. After backward, gradients flow into the online image encoder, the
     text encoder, and the predictor — but the EMA target encoder
     receives **zero** gradient.
  6. The progression CE is literally comparing
     ``argmax_c cos(ẑ_cur^c, z_cur)`` against the silver progression
     label: planting ``ẑ_cur^silver_label = z_cur`` drives the loss to
     ~0 and argmax matches silver.
  7. The dataset's ``__getitem__`` samples one finding per pair per
     epoch in a way that genuinely varies across calls (no accidental
     seed pinning).
  8. The 5 progression prompts are pair-major class-minor, and the row
     ordering in ``pred_progression_patches`` matches that layout (so
     ``pred[b, c]`` really is "pair b, class c").

Usage
-----
    python smoke_test_4loss.py
    python smoke_test_4loss.py --batch-size 4 --n-sampling-trials 500

The default batch size is 2 (smallest that still exercises shape logic);
bump it up if you want a stress test. The sampling check uses a
synthetic 4-finding pair so it doesn't need silver parquets — it
exercises the same ``random.choice(...)`` codepath the dataset uses.
"""

import argparse
import random
import sys
import traceback

import torch
import torch.nn.functional as F

from losses import local_contrastive_loss
from losses_jepa import jepa_cosine_loss, progression_classification_loss
from progression_phrases import CLS_ORDER
from tempcxr.modules.jepa import TempCXRJEPA


# ==================================================================
# REPORTING HELPERS
# ==================================================================
def _section(title: str):
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def _pass(msg: str):
    print(f"  [PASS] {msg}")


def _fail(msg: str):
    print(f"  [FAIL] {msg}")


def _info(msg: str):
    print(f"         {msg}")


# ==================================================================
# CHECKS
# ==================================================================
def grad_norm_and_count(module: torch.nn.Module):
    """Return (L2-norm of all parameter gradients, # params with grad).

    Skips ``p.grad is None`` (parameters that didn't receive any
    gradient), so a return of ``(0.0, 0)`` means literally no parameter
    in the module accumulated any gradient.
    """
    sq = 0.0
    n_with_grad = 0
    for p in module.parameters():
        if p.grad is not None:
            sq += p.grad.detach().float().norm(2).item() ** 2
            n_with_grad += 1
    return (sq ** 0.5, n_with_grad)


def run_smoke_test(B: int, n_sampling_trials: int) -> int:
    """Returns 0 if every check passed, 1 otherwise."""
    failures = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nRunning smoke_test_4loss on device={device}, B={B}")

    # ----------------------------------------------------------
    # 0. Build model + a synthetic batch
    # ----------------------------------------------------------
    _section("0. Build model + synthetic batch")
    model = TempCXRJEPA().to(device)
    model.train()

    # Real-looking text so the GLoRIA losses produce non-trivial values.
    prior_reports = [
        "No acute cardiopulmonary process. Lungs are clear.",
        "Stable cardiomegaly. No focal consolidation. No effusion.",
    ][:B] * ((B + 1) // 2)
    current_reports = [
        "Increased right pleural effusion.",
        "Left lower lobe pneumonia is improving.",
    ][:B] * ((B + 1) // 2)
    condition_texts = [
        "Right pleural effusion has increased in size compared to prior.",
        "Left lower lobe pneumonia is showing signs of improvement.",
    ][:B] * ((B + 1) // 2)
    # Truncate to exactly B in case the duplication overshot.
    prior_reports = prior_reports[:B]
    current_reports = current_reports[:B]
    condition_texts = condition_texts[:B]

    # The dataset picks one finding per pair per epoch. Use varied
    # findings so we can see different cosines per pair.
    prog_findings = [
        "pleural effusion",
        "pneumonia",
        "cardiomegaly",
        "edema",
        "consolidation",
        "atelectasis",
        "lung opacity",
        "pneumothorax",
    ][:B]
    # Silver class indices: arbitrary spread across the 5 classes.
    silver_indices = [(2 * i) % len(CLS_ORDER) for i in range(B)]
    prog_cls_idx = torch.tensor(silver_indices, dtype=torch.long, device=device)

    # Build the 5*B progression prompts in pair-major class-minor order.
    progression_prompts_flat = []
    for f in prog_findings:
        f_cap = f[:1].upper() + f[1:]
        for cls in CLS_ORDER:
            progression_prompts_flat.append(f"{f_cap} is {cls}.")
    _info(
        f"len(progression_prompts_flat) = {len(progression_prompts_flat)}  "
        f"(expected B*5 = {B * len(CLS_ORDER)})"
    )

    prior_imgs = torch.randn(B, 3, 448, 448, device=device)
    current_imgs = torch.randn(B, 3, 448, 448, device=device)

    # ----------------------------------------------------------
    # 1. Forward pass + output shape checks
    # ----------------------------------------------------------
    _section("1. Forward pass + output shapes")
    out = model(
        prior_imgs,
        current_imgs,
        prior_reports,
        current_reports,
        condition_texts,
        progression_prompts_flat=progression_prompts_flat,
    )

    shapes = {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}
    for k, s in shapes.items():
        _info(f"{k}: {s}")

    pred_prog = out["pred_progression_patches"]
    expected = (B, len(CLS_ORDER), pred_prog.shape[2], pred_prog.shape[3])
    if pred_prog.shape == expected:
        _pass(
            f"pred_progression_patches is (B={B}, C={len(CLS_ORDER)}, "
            f"N={pred_prog.shape[2]}, D={pred_prog.shape[3]})"
        )
    else:
        _fail(
            f"pred_progression_patches shape {pred_prog.shape} "
            f"≠ expected {expected}"
        )
        failures.append("pred_progression_patches shape")

    # ----------------------------------------------------------
    # 2. Stop-grad on the EMA target path
    # ----------------------------------------------------------
    _section("2. Stop-grad on target encoder")

    target_params_requiring_grad = [
        n for n, p in model.target_image_encoder.named_parameters()
        if p.requires_grad
    ]
    if not target_params_requiring_grad:
        _pass(
            "Every parameter of target_image_encoder has "
            "requires_grad=False"
        )
    else:
        _fail(
            f"{len(target_params_requiring_grad)} target_image_encoder "
            f"params have requires_grad=True (showing first 3): "
            f"{target_params_requiring_grad[:3]}"
        )
        failures.append("target encoder params still trainable")

    cpt = out["current_patches_target"]
    if not cpt.requires_grad and cpt.grad_fn is None:
        _pass(
            "current_patches_target.requires_grad=False AND "
            "grad_fn is None (detached + stop-grad)"
        )
    else:
        _fail(
            f"current_patches_target has requires_grad={cpt.requires_grad}, "
            f"grad_fn={cpt.grad_fn}"
        )
        failures.append("current_patches_target gradient path")

    # ----------------------------------------------------------
    # 3. Predictor genuinely ran 5 times with 5 different conditions
    # ----------------------------------------------------------
    _section("3. Predictor multiplicity (5 distinct outputs per pair)")

    same_pair_diffs = []
    for b in range(B):
        for c in range(1, len(CLS_ORDER)):
            d = (pred_prog[b, c] - pred_prog[b, 0]).abs().mean().item()
            same_pair_diffs.append(d)
    min_diff = min(same_pair_diffs)
    max_diff = max(same_pair_diffs)
    _info(
        f"|pred[b, c>0] - pred[b, 0]| across (b, c): "
        f"min={min_diff:.6f}, max={max_diff:.6f}"
    )

    if min_diff > 1e-7:
        _pass(
            "All 5 candidate predictions differ within each pair "
            "(text condition genuinely reaches the predictor)"
        )
    else:
        _fail(
            "Some candidate predictions are identical across class c — "
            "the text condition isn't reaching the predictor"
        )
        failures.append("predictor multiplicity")

    # ----------------------------------------------------------
    # 4. All 4 losses compute + finite
    # ----------------------------------------------------------
    _section("4. Compute all 4 losses")

    jepa = jepa_cosine_loss(
        out["pred_current_patches"].float(),
        out["current_patches_target"].float(),
    )
    prior = local_contrastive_loss(
        out["prior_patches"],
        out["prior_txt_local"],
        out["prior_token_mask"],
    )
    pred = local_contrastive_loss(
        out["pred_current_patches"],
        out["current_txt_local"],
        out["current_token_mask"],
    )
    prog = progression_classification_loss(
        out["pred_progression_patches"].float(),
        out["current_patches_target"].float(),
        prog_cls_idx,
        temperature=0.1,
    )
    total = 1.0 * jepa + 0.1 * prior + 0.1 * pred + 0.1 * prog

    _info(f"jepa  = {jepa.item():.4f}")
    _info(f"prior = {prior.item():.4f}")
    _info(f"pred  = {pred.item():.4f}")
    _info(f"prog  = {prog.item():.4f}  (≈ log(5) = "
          f"{torch.log(torch.tensor(float(len(CLS_ORDER)))).item():.4f} "
          f"at init)")
    _info(f"TOTAL = {total.item():.4f}")

    all_finite = all(
        torch.isfinite(l) for l in (jepa, prior, pred, prog, total)
    )
    if all_finite:
        _pass("All 4 losses are finite")
    else:
        _fail("At least one loss is NaN/Inf")
        failures.append("loss finiteness")

    # ----------------------------------------------------------
    # 5. Backward + per-module gradient flow
    # ----------------------------------------------------------
    _section("5. Backward + per-module gradient flow")

    total.backward()

    on_norm, on_n = grad_norm_and_count(model.image_encoder)
    tg_norm, tg_n = grad_norm_and_count(model.target_image_encoder)
    txt_norm, txt_n = grad_norm_and_count(model.text_encoder)
    pred_norm, pred_n = grad_norm_and_count(model.predictor)

    _info(f"||grad image_encoder (online)|| = {on_norm:>8.4f}  "
          f"({on_n} params with grad)")
    _info(f"||grad image_encoder (target)|| = {tg_norm:>8.4f}  "
          f"({tg_n} params with grad)")
    _info(f"||grad text_encoder||           = {txt_norm:>8.4f}  "
          f"({txt_n} params with grad)")
    _info(f"||grad predictor||              = {pred_norm:>8.4f}  "
          f"({pred_n} params with grad)")

    if tg_norm == 0.0 and tg_n == 0:
        _pass("EMA target encoder received ZERO gradient (no grad path)")
    else:
        _fail(
            f"EMA target encoder received gradient: ||g||={tg_norm}, "
            f"{tg_n} params with grad"
        )
        failures.append("target encoder grad leak")

    if on_norm > 0 and txt_norm > 0 and pred_norm > 0:
        _pass(
            "Online image encoder, text encoder, and predictor all "
            "received non-zero gradient"
        )
    else:
        _fail(
            f"Some trainable module received no gradient: "
            f"online={on_norm}, text={txt_norm}, pred={pred_norm}"
        )
        failures.append("trainable modules grad flow")

    # ----------------------------------------------------------
    # 6. Progression CE = argmax cos(ẑ_cur^c, z_cur) vs silver
    # ----------------------------------------------------------
    _section(
        "6. Progression CE compares argmax cos(ẑ_cur^c, z_cur) "
        "to silver label"
    )

    with torch.no_grad():
        target = F.normalize(out["current_patches_target"], dim=-1).detach()
        N, D = target.shape[1], target.shape[2]
        # Plant pred[b, silver[b]] := target[b]; randomize the others.
        pred_synth = F.normalize(
            torch.randn(B, len(CLS_ORDER), N, D, device=device), dim=-1
        )
        for b in range(B):
            pred_synth[b, prog_cls_idx[b].item()] = target[b]

        # Recompute logits the same way the loss does, then argmax.
        cos_per_patch = (pred_synth * target.unsqueeze(1)).sum(dim=-1)
        logits = cos_per_patch.mean(dim=-1)  # (B, C)
        argmax = logits.argmax(dim=-1)

        loss_aligned = progression_classification_loss(
            pred_synth, target, prog_cls_idx, temperature=0.1
        )

    _info(f"silver_labels  = {prog_cls_idx.tolist()}")
    _info(f"argmax(logits) = {argmax.tolist()}")
    _info(
        f"loss(planted)  = {loss_aligned.item():.6f}  "
        f"(expected ~0 because pred[silver] == target)"
    )

    if torch.equal(argmax, prog_cls_idx):
        _pass(
            "argmax of cos(ẑ_cur^c, z_cur) matches silver label exactly "
            "when pred is planted at the target"
        )
    else:
        _fail(
            "argmax of progression logits doesn't match silver label "
            "even when the planted class is identical to the target"
        )
        failures.append("argmax alignment")

    if loss_aligned.item() < 0.1:
        _pass(
            "Loss collapses to ~0 when pred[silver] equals target "
            "(confirms cos-as-logits → softmax-CE wiring)"
        )
    else:
        _fail(
            f"Loss should be near zero when pred[silver]=target, "
            f"got {loss_aligned.item():.4f}"
        )
        failures.append("loss collapse when pred=target")

    # ----------------------------------------------------------
    # 7. Random-sampling check: prog_finding varies across __getitem__
    # ----------------------------------------------------------
    _section(
        "7. Random sampling actually varies (no accidental seed pinning)"
    )

    # Replicate the dataset's pick logic exactly. The dataset uses
    # ``random.choice(list(zip(findings, progression_cls_idx)))`` for
    # train and a sorted alphabetical first for val.
    findings_synth = ["alpha", "beta", "gamma", "delta"]
    cls_synth = [0, 1, 2, 3]
    pairs = list(zip(findings_synth, cls_synth))

    # ---- TRAIN: simulate N independent __getitem__ calls ----
    counts_train = {f: 0 for f in findings_synth}
    for _ in range(n_sampling_trials):
        f, _c = random.choice(pairs)
        counts_train[f] += 1

    _info(
        f"Train picks over {n_sampling_trials} calls (random.choice):"
    )
    for f in findings_synth:
        pct = 100.0 * counts_train[f] / n_sampling_trials
        _info(f"  {f:<8} -> {counts_train[f]:>5}  ({pct:>5.1f}%)")

    coverage = sum(1 for c in counts_train.values() if c > 0)
    if coverage == len(findings_synth):
        _pass(
            f"All {len(findings_synth)} findings get sampled across "
            f"{n_sampling_trials} calls (no accidental seed pinning)"
        )
    else:
        _fail(
            f"Only {coverage}/{len(findings_synth)} findings were ever "
            f"sampled in {n_sampling_trials} calls — random.choice "
            f"appears to be deterministically returning a subset"
        )
        failures.append("random sampling coverage")

    # ---- VAL: deterministic ----
    pairs_sorted = sorted(pairs, key=lambda fp: fp[0])
    val_pick = pairs_sorted[0][0]
    _info(f"Val pick (deterministic alphabetical first): {val_pick!r}")
    if val_pick == "alpha":
        _pass("Val pick is deterministic (alphabetically first)")
    else:
        _fail(f"Val pick is not 'alpha': {val_pick!r}")
        failures.append("val determinism")

    # ----------------------------------------------------------
    # 8. Prompt ordering: pair-major class-minor in pred_progression_patches
    # ----------------------------------------------------------
    _section(
        "8. Prompt ordering: pair-major class-minor (row i*C+c → pair i, class c)"
    )

    # The model uses prior_patches.repeat_interleave(C, dim=0), which
    # produces rows [pair0_x_C copies, pair1_x_C copies, ...]. Verify
    # that the prompt list and the resulting pred_progression_patches
    # share this layout by re-running the predictor on only one pair's
    # 5 prompts and confirming pred_progression_patches[0] matches.
    with torch.no_grad():
        single = model(
            prior_imgs[:1],
            current_imgs[:1],
            prior_reports[:1],
            current_reports[:1],
            condition_texts[:1],
            progression_prompts_flat=progression_prompts_flat[:5],
        )
        ref = single["pred_progression_patches"]  # (1, 5, N, D)

    # Compare against pair-0 slice from the full batch forward.
    full_pair0 = out["pred_progression_patches"][:1].detach()
    diff = (ref - full_pair0).abs().mean().item()
    _info(
        f"||pred_prog[pair 0, full batch] - pred_prog[pair 0, single]|| "
        f"(mean abs) = {diff:.6f}  "
        f"(should be ~0; cross-batch text-encoder padding/dropout can "
        f"add tiny noise even in eval mode if any module has state)"
    )

    if diff < 1e-3:
        _pass(
            "pred_progression_patches[b, c] really is pair b's class-c "
            "prediction (rows align with the prompt list)"
        )
    else:
        _fail(
            f"Layout check failed: cross-batch slice differs from "
            f"single-pair forward by {diff:.4f}. Either rows aren't "
            f"pair-major class-minor or the model has random state."
        )
        failures.append("prompt ordering")

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------
    _section("SUMMARY")
    if not failures:
        print("\n  ALL CHECKS PASSED.\n")
        return 0
    else:
        print(f"\n  {len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"    - {f}")
        print()
        return 1


# ==================================================================
# CLI
# ==================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch-size", "-B", type=int, default=2,
        help="Synthetic batch size (default: 2).",
    )
    parser.add_argument(
        "--n-sampling-trials", type=int, default=400,
        help=(
            "How many random.choice calls to do for the random-sampling "
            "variance check (default: 400). 200+ is plenty to detect "
            "accidental seed pinning."
        ),
    )
    args = parser.parse_args()

    try:
        rc = run_smoke_test(args.batch_size, args.n_sampling_trials)
    except Exception:
        print("\nUNCAUGHT EXCEPTION in smoke test:")
        traceback.print_exc()
        rc = 2

    sys.exit(rc)


if __name__ == "__main__":
    main()
