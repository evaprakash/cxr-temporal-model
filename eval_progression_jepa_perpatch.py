"""5-way progression eval with the **original per-patch** scoring rule.

Same pipeline as ``eval_progression_jepa.py`` (predictor image–image
matching), but scores each class by::

    mean over patches of cos(ẑ_cur^c[p], z_cur[p])

instead of global-pool ``cos(pool(ẑ), pool(z_cur))``.

Use this to evaluate checkpoints trained with per-patch JEPA /
progression CE (pre-global-pool main), so train rule = test rule.

Usage
-----
    python eval_progression_jepa_perpatch.py --eval \\
        --ckpt /path/to/old_perpatch_checkpoint/best.pt

    # Optional: drop (pair, finding) rows with conflicting lesion-level
    # progression labels (see audit_gold_multi_progression.py).
    python eval_progression_jepa_perpatch.py --eval --drop-multi-progression \\
        --ckpt /path/to/old_perpatch_checkpoint/best.pt

    python eval_progression_jepa_perpatch.py --demo --idx 17 \\
        --ckpt /path/to/old_perpatch_checkpoint/best.pt
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

import eval_progression_jepa as base
from tempcxr.modules.jepa import TempCXRJEPA


@torch.no_grad()
def score_one_pair(
    model: TempCXRJEPA,
    prior_img: torch.Tensor,
    current_img: torch.Tensor,
    finding: str,
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, Tuple]] = None,
    classes: Optional[List[str]] = None,
) -> Dict:
    """N-way image-image scoring with mean per-patch cosine (original)."""
    prompts, txt_local, token_mask = base._encode_prompts(
        model, finding, template, device, text_cache, classes=classes,
    )
    n_prompts = len(prompts)

    prior = prior_img.unsqueeze(0).to(device)
    current = current_img.unsqueeze(0).to(device)

    _, z_prior = model.image_encoder(prior)               # (1, N, D)
    _, z_cur = model.target_image_encoder(current)        # (1, N, D)
    z_cur = z_cur.detach()

    z_prior_b = z_prior.expand(n_prompts, -1, -1).contiguous()
    preds = model.predictor(z_prior_b, txt_local, token_mask)

    pred_f = preds.float()
    target_f = z_cur.float().expand_as(pred_f)
    cos_per_patch = F.cosine_similarity(pred_f, target_f, dim=-1)  # (C, N)
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


# Reuse CLI / gold loading / metrics from the global-pool script, but
# force the original per-patch scorer.
base.score_one_pair = score_one_pair


if __name__ == "__main__":
    print(
        "[eval_progression_jepa_perpatch] scoring = "
        "mean_p cos(ẑ[p], z_cur[p])  (original per-patch rule)"
    )
    base.main()
