"""Forward + loss + backward smoke test for the RAD-DINO variant of TempCXRJEPA.

This is a minimal, single-process (no DDP) sanity check that the
``raddino-image-encoder`` branch is wired end-to-end:

  1. Instantiate ``TempCXRJEPA(mode="raddino")`` — pulls
     ``microsoft/rad-dino-maira-2`` for the image encoder and pairs it
     with the BioViL-T CXR-BERT text encoder configured with
     ``use_projection=False``.
  2. Build a tiny synthetic batch (2 pairs of 3x448x448 random images in
     ``[0, 1]``, matching the shape emitted by the training dataset's
     ``TF.to_tensor``) with placeholder reports and 5 templated
     progression prompts per pair.
  3. Run the full forward pass (JEPA branch + 5-way progression branch).
  4. Compute all 4 losses (JEPA cosine, GLoRIA local CL on z_prior,
     GLoRIA local CL on ẑ_cur, and the 5-way image-image CE) and their
     weighted sum.
  5. Assert:
       * image / text feature dims agree (768 both);
       * ``pred_current_patches.shape == (B, N, 768)`` with
         ``N == 1369`` (RAD-DINO's 37x37 patch grid at 518x518);
       * ``pred_progression_patches.shape == (B, 5, 1369, 768)``.
  6. Run ``total.backward()`` and confirm the EMA update path executes
     one step without raising.

Run with::

    python smoke_test_raddino.py

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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("[smoke] instantiating TempCXRJEPA(mode='raddino') ...")
    model = TempCXRJEPA(mode="raddino").to(device)
    model.train()

    img_dim = model.image_encoder.embed_dim
    txt_dim = model.text_encoder.output_dim
    print(f"[smoke] image feature dim = {img_dim}")
    print(f"[smoke] text  feature dim = {txt_dim}")
    assert img_dim == txt_dim == 768, (
        f"expected image & text dims to both be 768; got img={img_dim}, "
        f"txt={txt_dim} — the JEPA / contrastive loss requires matching D"
    )
    assert model.num_patches == 1369, (
        f"expected 1369 patches (37x37 at 518x518), got {model.num_patches}"
    )
    assert model.d_model == 768, (
        f"expected predictor d_model == 768, got {model.d_model}"
    )

    # ------------------------------------------------------------------
    # Dummy batch — mimics what the dataset produces post-TF.to_tensor.
    # Values in [0, 1], shape (B, 3, 448, 448). The image encoder
    # resizes to 518x518 and normalizes internally.
    # ------------------------------------------------------------------
    B = 2
    prior_imgs = torch.rand(B, 3, 448, 448, device=device)
    current_imgs = torch.rand(B, 3, 448, 448, device=device)

    prior_reports = [
        "No acute cardiopulmonary process. Lungs clear.",
        "Stable cardiomegaly. No effusion.",
    ]
    current_reports = [
        "Increased right pleural effusion.",
        "Left lower lobe pneumonia is improving.",
    ]
    condition_texts = [
        "right pleural effusion is new and increased",
        "pneumonia is getting better",
    ]

    prog_findings = ["pleural effusion", "pneumonia"]
    prog_cls_idx = torch.tensor([2, 0], device=device, dtype=torch.long)
    progression_prompts_flat = []
    for f in prog_findings:
        f_cap = f[:1].upper() + f[1:]
        for cls in CLS_ORDER:
            progression_prompts_flat.append(f"{f_cap} is {cls}.")

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    print("[smoke] running forward pass ...")
    out = model(
        prior_imgs,
        current_imgs,
        prior_reports,
        current_reports,
        condition_texts,
        progression_prompts_flat=progression_prompts_flat,
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

    assert out["prior_patches"].shape == (B, 1369, 768)
    assert out["current_patches_target"].shape == (B, 1369, 768)
    assert out["pred_current_patches"].shape == (B, 1369, 768)
    assert out["pred_progression_patches"].shape == (B, len(CLS_ORDER), 1369, 768)

    # Local text tokens: (B, T, 768). T depends on tokenizer max_length
    # (currently 112 in the encoder), minus the CLS drop; we just assert
    # the feature dim.
    assert out["prior_txt_local"].shape[-1] == 768
    assert out["current_txt_local"].shape[-1] == 768
    assert out["condition_txt_local"].shape[-1] == 768

    # ------------------------------------------------------------------
    # Losses (same combination the trainer uses)
    # ------------------------------------------------------------------
    print("[smoke] computing losses ...")
    l_jepa = jepa_cosine_loss(
        out["pred_current_patches"],
        out["current_patches_target"],
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
    l_prog = progression_classification_loss(
        out["pred_progression_patches"],
        out["current_patches_target"],
        prog_cls_idx,
    )
    total = l_jepa + 0.1 * l_report_prior + 0.1 * l_report_pred + 0.1 * l_prog

    print(f"[smoke] jepa_cosine        = {l_jepa.item():.4f}")
    print(f"[smoke] local_cl (z_prior) = {l_report_prior.item():.4f}")
    print(f"[smoke] local_cl (z_hat)   = {l_report_pred.item():.4f}")
    print(f"[smoke] progression 5-way  = {l_prog.item():.4f}")
    print(f"[smoke] total (weighted)   = {total.item():.4f}")

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

    print("\n[smoke] SUCCESS — RAD-DINO JEPA forward + loss + backward pass all OK")


if __name__ == "__main__":
    main()
