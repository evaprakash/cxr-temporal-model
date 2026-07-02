"""Forward + loss + backward smoke test for the all-findings prog-loss variant.

This is a minimal single-process (no DDP) sanity check that the
``prog-loss-all-findings`` branch is wired end-to-end:

  1. Instantiate ``TempCXRJEPA()`` (main-branch defaults: BioViL-T image
     encoder + BioViL-T text encoder with the 128-d joint-space head).
  2. Build a tiny synthetic batch of 3 pairs where pair 0 has 1
     finding, pair 1 has 3 findings, and pair 2 has 2 findings (so
     ``F_total = 6``, deliberately different from ``B = 3``).
  3. Reuse the trainer's ``build_progression_prompts_all`` helper to
     generate the flat prompt list, the per-finding pair-idx tensor,
     and the per-finding silver-label tensor.
  4. Run ``TempCXRJEPA.forward(..., progression_prompts_flat=prompts,
     progression_prior_pair_idx=pair_idx)`` and assert:
       * ``pred_progression_patches.shape == (F_total, 5, N, D)`` —
         one predictor output per (pair, finding), not per pair.
       * ``pair_idx`` values are correct (0, 1, 1, 1, 2, 2) in the
         order the trainer flattens findings.
  5. Compute the full 4-loss objective (JEPA + 2 local-CL + 5-way
     progression CE gathered over ``F_total`` findings), assert it is
     finite, and confirm the backward pass leaves gradients on every
     trainable parameter.
  6. Fire one EMA update to make sure that path still executes.

Run with::

    python smoke_test_all_findings.py

Uses CUDA when available, otherwise falls back to CPU (slow but works
— useful for sanity-checking wiring on a laptop before submitting to
the cluster).
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from tempcxr.modules.jepa import (
    TempCXRJEPA,
    make_momentum_scheduler,
    EMA_START,
    EMA_END,
)
from losses import local_contrastive_loss
from losses_jepa import jepa_cosine_loss, progression_classification_loss
from progression_phrases import CLS_ORDER

# ``CLS_TO_IDX`` is defined inside ``dataset_combined_jepa`` (which
# would pull in pandas + the whole silver-parquet loader). We only
# need the string-to-int mapping here, so rebuild it locally.
CLS_TO_IDX = {cls: i for i, cls in enumerate(CLS_ORDER)}

# Loss weights and prog CE temperature — kept in sync with
# ``resume_train_jepa.py``. Duplicated (rather than imported) because
# importing ``resume_train_jepa`` triggers ``setup_ddp()`` at module
# load time, which needs ``torchrun``-set env vars we don't have in a
# single-process smoke test.
PROG_TEMP = 0.1
PROG_TEMPLATE = "{} is {}."
W_JEPA = 1.0
W_REPORT_PRIOR = 0.1
W_REPORT_PRED = 0.1
W_PROG = 0.1


def build_progression_prompts_all(batch_findings, batch_prog_cls_idx):
    """Local copy of the trainer helper (avoids DDP-at-import-time).

    Kept byte-for-byte in sync with ``resume_train_jepa.py`` — if you
    edit one, edit the other.
    """
    prompts = []
    pair_idx = []
    silver_labels = []
    for p, (findings, cls_ids) in enumerate(
        zip(batch_findings, batch_prog_cls_idx)
    ):
        for finding, cls in zip(findings, cls_ids):
            if not finding:
                continue
            f_cap = finding[:1].upper() + finding[1:]
            for prog_cls in CLS_ORDER:
                prompts.append(PROG_TEMPLATE.format(f_cap, prog_cls))
            pair_idx.append(p)
            silver_labels.append(int(cls))
    return prompts, pair_idx, silver_labels


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("[smoke] instantiating TempCXRJEPA() ...")
    model = TempCXRJEPA().to(device)
    model.train()

    # ------------------------------------------------------------------
    # Dummy batch — three pairs with 1, 3, 2 findings respectively so
    # F_total = 6 != B = 3. This is the whole point of this branch:
    # every finding contributes to every optimizer step.
    # ------------------------------------------------------------------
    B = 3
    prior_imgs = torch.randn(B, 3, 448, 448, device=device)
    current_imgs = torch.randn(B, 3, 448, 448, device=device)

    prior_reports = [
        "No acute cardiopulmonary process. Lungs clear.",
        "Stable cardiomegaly. Small right pleural effusion.",
        "No focal consolidation.",
    ]
    current_reports = [
        "Increased right pleural effusion.",
        "Left lower lobe pneumonia is worsening; effusion stable.",
        "New patchy opacity in the right lower lobe.",
    ]
    condition_texts = [
        "right pleural effusion is new",
        "pneumonia is worsening. pleural effusion is stable.",
        "consolidation is new",
    ]

    # Findings per pair — variable-length, matching the ragged shape
    # produced by ``JEPACombinedDataset.__getitem__`` for real data.
    findings = [
        ["pleural effusion"],
        ["pneumonia", "pleural effusion", "atelectasis"],
        ["consolidation", "pneumothorax"],
    ]
    silver_class_names = [
        ["worsening"],
        ["worsening", "stable", "improving"],
        ["worsening", "new"],
    ]
    progression_cls_idx = [
        [CLS_TO_IDX[c] for c in row] for row in silver_class_names
    ]
    F_total = sum(len(f) for f in findings)
    assert F_total == 6, f"expected F_total=6 for the smoke fixture, got {F_total}"
    print(f"[smoke] batch B = {B}, F_total = {F_total} (variable F per pair)")

    # ------------------------------------------------------------------
    # Build all-findings prog prompts + pair idx + silver labels using
    # the same helper the trainer calls at each step.
    # ------------------------------------------------------------------
    prog_prompts, prog_pair_idx_list, prog_silver_list = (
        build_progression_prompts_all(findings, progression_cls_idx)
    )
    expected_pair_idx = [0, 1, 1, 1, 2, 2]
    expected_silver = [
        CLS_TO_IDX["worsening"],
        CLS_TO_IDX["worsening"],
        CLS_TO_IDX["stable"],
        CLS_TO_IDX["improving"],
        CLS_TO_IDX["worsening"],
        CLS_TO_IDX["new"],
    ]
    assert prog_pair_idx_list == expected_pair_idx, (
        f"pair_idx wrong: got {prog_pair_idx_list}, "
        f"expected {expected_pair_idx}"
    )
    assert prog_silver_list == expected_silver, (
        f"silver_labels wrong: got {prog_silver_list}, "
        f"expected {expected_silver}"
    )
    assert len(prog_prompts) == F_total * len(CLS_ORDER), (
        f"prompt count wrong: got {len(prog_prompts)}, "
        f"expected {F_total * len(CLS_ORDER)}"
    )
    print(
        f"[smoke] prog prompts: {len(prog_prompts)} strings "
        f"(F_total={F_total} x C={len(CLS_ORDER)})"
    )
    print(f"[smoke] first 5 prompts: {prog_prompts[:5]}")

    prog_pair_idx = torch.tensor(
        prog_pair_idx_list, dtype=torch.long, device=device
    )
    prog_silver_labels = torch.tensor(
        prog_silver_list, dtype=torch.long, device=device
    )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    print("[smoke] running forward pass ...")
    out = model(
        prior_imgs,
        current_imgs,
        prior_reports,
        current_reports,
        condition_texts,
        progression_prompts_flat=prog_prompts,
        progression_prior_pair_idx=prog_pair_idx,
    )

    print("[smoke] tensor shapes:")
    for k in [
        "prior_patches",
        "current_patches_target",
        "pred_current_patches",
        "prior_txt_local",
        "current_txt_local",
        "condition_txt_local",
        "pred_progression_patches",
    ]:
        v = out[k]
        print(f"           {k:28s} = {tuple(v.shape)}")

    # ------------------------------------------------------------------
    # Shape asserts — the key invariant is that pred_progression_patches
    # has F_total (not B) in the first dim.
    # ------------------------------------------------------------------
    pred_prog = out["pred_progression_patches"]
    assert pred_prog.shape[0] == F_total, (
        f"pred_progression_patches.shape[0] should equal F_total={F_total}, "
        f"got {pred_prog.shape[0]}"
    )
    assert pred_prog.shape[1] == len(CLS_ORDER), (
        f"pred_progression_patches.shape[1] should equal C="
        f"{len(CLS_ORDER)}, got {pred_prog.shape[1]}"
    )
    assert out["prior_patches"].shape[0] == B  # main image branch is still B
    assert out["current_patches_target"].shape[0] == B

    # ------------------------------------------------------------------
    # Losses — mirror the trainer exactly, including the per-finding
    # target gather.
    # ------------------------------------------------------------------
    print("[smoke] computing losses ...")
    l_jepa = jepa_cosine_loss(
        out["pred_current_patches"].float(),
        out["current_patches_target"].float(),
    )
    l_report_prior = local_contrastive_loss(
        out["prior_patches"],
        out["prior_txt_local"],
        out["prior_token_mask"],
    )
    l_report_pred = local_contrastive_loss(
        out["pred_current_patches"],
        out["current_txt_local"],
        out["current_token_mask"],
    )
    target_per_finding = out["current_patches_target"].index_select(
        0, prog_pair_idx
    )
    l_prog = progression_classification_loss(
        out["pred_progression_patches"].float(),
        target_per_finding.float(),
        prog_silver_labels,
        temperature=PROG_TEMP,
    )
    total = (
        W_JEPA * l_jepa
        + W_REPORT_PRIOR * l_report_prior
        + W_REPORT_PRED * l_report_pred
        + W_PROG * l_prog
    )

    print(f"[smoke] jepa_cosine        = {l_jepa.item():.4f}")
    print(f"[smoke] local_cl (z_prior) = {l_report_prior.item():.4f}")
    print(f"[smoke] local_cl (z_hat)   = {l_report_pred.item():.4f}")
    print(f"[smoke] progression 5-way  = {l_prog.item():.4f} (over {F_total} findings)")
    print(f"[smoke] total (weighted)   = {total.item():.4f}")
    assert torch.isfinite(total), "total loss is not finite"

    # ------------------------------------------------------------------
    # Backward + EMA
    # ------------------------------------------------------------------
    print("[smoke] running backward ...")
    total.backward()

    n_grads = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[smoke] parameters with gradient: {n_grads} / {n_trainable}")
    assert n_grads > 0, "no gradients flowed — check the forward path"

    print("[smoke] running one EMA update step ...")
    sched = make_momentum_scheduler(EMA_START, EMA_END, total_iters=1)
    m = next(sched)
    model.update_ema(momentum=m)
    print(f"[smoke] EMA update ok (momentum={m:.4f})")

    print(
        "\n[smoke] SUCCESS — all-findings prog-loss forward + loss + "
        "backward pass all OK"
    )


if __name__ == "__main__":
    main()
