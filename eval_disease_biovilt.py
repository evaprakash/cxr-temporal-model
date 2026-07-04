"""N-way disease classification with zero-shot BioViL-T (no checkpoint).

Direct BioViL-T counterpart to ``eval_disease_jepa.py`` — same gold
parquet, same vocabulary, same top-1 / top-3 / top-5 + per-finding +
per-progression + confusion + balanced-metrics output layout, so the
two ``cnr.csv`` / stdout blocks are byte-for-byte comparable.

Design
------
* Model:  ``TempCXR(mode="biovilt")`` (same as
  ``biovilt_progression_pairs.py`` — no checkpoint fine-tuning; whatever
  weights BioViL-T's image + text encoders ship with).
* Image:  ``model.image_encoder(curr, prev)`` gives a pair-conditioned
  128-D global embedding. L2-normalize.
* Text:   the prompt for each candidate finding ``d`` is
  ``"{finding} is {gt_progression}."`` — same template as the JEPA
  disease eval, with the finding swept and the progression class held
  fixed to the row's ground-truth label. Encoded through
  ``model.text_encoder.forward_contrastive``. L2-normalize.
* Score:  ``cos(img_global, text_global[d])`` per candidate finding.
* Predict: ``argmax_d``.

Why the prompts include the GT progression
------------------------------------------
Apples-to-apples with ``eval_disease_jepa.py``. The JEPA disease eval
feeds its predictor ``"{finding} is {gt_progression}."`` and asks
"which finding, given the GT progression, makes the predictor land
closest to the true current image?" — i.e. the predictor gets *both*
the finding token and the progression token as hints. If the BioViL-T
counterpart only got the finding name, BioViL-T would be answering a
harder question (no progression hint) and any accuracy gap would
conflate model capability with the extra information JEPA received.
Conditioning both on the GT progression is also well within
BioViL-T's operating regime — its own zero-shot progression eval
(``biovilt_progression_pairs.py``) uses the same ``{finding} is
{phrase}`` template style.

If you want the plain "no progression hint" baseline instead, pass
``--prompt-template "{}"`` — the script auto-detects 1-slot vs.
2-slot templates and drops the progression when the template has
only one positional slot.

Usage
-----
    # Sanity-check one gold row (top-5 predictions shown)
    python eval_disease_biovilt.py --demo

    # Full N-way eval over the gold parquet (progression-conditioned)
    python eval_disease_biovilt.py --eval

    # Restrict vocab to findings with >=20 gold rows, run full eval
    python eval_disease_biovilt.py --eval --min-per-finding 20

    # Only score rows whose GT progression is "worsening"
    python eval_disease_biovilt.py --eval --progression worsening

    # Diagnostic: bare finding name (no progression conditioning)
    python eval_disease_biovilt.py --eval --prompt-template "{}"

    # Natural-language wrapper (2-slot: finding + progression)
    python eval_disease_biovilt.py --eval \
        --prompt-template "Findings suggest {} is {}."
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ``tempcxr_biovilt.modules.tempcxr_model`` imports ``losses`` from the
# project root. We don't need the training losses here (we're only
# running inference), so stub the module out if it isn't present before
# importing TempCXR — same shim biovilt_progression_pairs.py uses.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
try:
    import losses  # noqa: F401
except ModuleNotFoundError:
    import types as _types
    _stub = _types.ModuleType("losses")
    _stub.global_contrastive_loss = None
    _stub.local_contrastive_loss = None
    _stub.mlm_loss = None
    sys.modules["losses"] = _stub

from dataset_combined_jepa import DEFAULT_FINDINGS  # noqa: E402
from eval_disease_jepa import build_findings_vocab  # noqa: E402
from eval_progression_jepa import _compute_balanced_metrics  # noqa: E402
from infer_jepa import IMAGE_ROOTS  # noqa: E402
from progression_classify import (  # noqa: E402
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    _normalize_label,
    _resolve_with_fallbacks,
    discover_gold_image_roots,
    load_gold_pairs,
)
from progression_phrases import CLS_ORDER  # noqa: E402
from tempcxr_biovilt.modules.tempcxr_model import TempCXR  # noqa: E402


# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_SIZE = 448           # matches BASE_TRANSFORM / biovilt_progression_pairs
RESIZE_SHORT = 512

# Default prompt template — matches eval_disease_jepa exactly: the
# (capitalized) finding is inserted in the first slot and the row's GT
# progression class in the second slot ({improving, stable, worsening,
# new, resolved}). Users can drop back to a 1-slot template like
# ``"{}"`` for the bare-finding baseline; the script auto-detects the
# slot count.
PROMPT_TEMPLATE = "{} is {}."


# ============================================================
# IMAGE TRANSFORM (must match biovilt_progression_pairs.py)
# ============================================================
transform = T.Compose([
    T.Resize(RESIZE_SHORT),
    T.CenterCrop(INPUT_SIZE),
    T.ToTensor(),
])


def load_image(path) -> torch.Tensor:
    img = Image.open(str(path)).convert("RGB")
    img_t = transform(img)
    if img_t.shape[0] == 1:
        img_t = img_t.repeat(3, 1, 1)
    return img_t.to(DEVICE)


# ============================================================
# MODEL LOAD
# ============================================================
def load_model() -> TempCXR:
    print("🔧 Initializing TempCXR with mode='biovilt' (no checkpoint).")
    model = TempCXR(mode="biovilt").to(DEVICE)
    model.eval()
    print("✅ Model loaded.")
    return model


# ============================================================
# PROMPT BUILDING
# ============================================================
def _count_slots(template: str) -> int:
    """Count positional ``{}`` slots in the template. Only positional
    slots are supported here — keyword templates like ``"{finding}"``
    are rejected upstream by ``main``'s validation. The count decides
    whether progression is inserted (2-slot) or ignored (1-slot)."""
    n = template.count("{}")
    if n not in (1, 2):
        raise ValueError(
            f"--prompt-template must contain 1 or 2 positional {{}} "
            f"slots, got {n} in {template!r}"
        )
    return n


def build_disease_prompts(
    findings_vocab: List[str],
    progression: Optional[str],
    template: str = PROMPT_TEMPLATE,
) -> List[str]:
    """One prompt per candidate finding. Capitalizes the first letter of
    the finding to match how reports typically render pathology names.

    * 2-slot templates (default ``"{} is {}."``): ``progression`` is
      required and inserted as the second slot. Mirrors
      ``eval_disease_jepa.build_disease_prompts``.
    * 1-slot templates (e.g. ``"{}"``): ``progression`` is ignored —
      this is the bare-finding baseline reachable via
      ``--prompt-template "{}"``.
    """
    n_slots = _count_slots(template)
    if n_slots == 2 and progression is None:
        raise ValueError(
            "2-slot template requires a progression class; got None. "
            "Pass a 1-slot template (--prompt-template \"{}\") if you "
            "really want the bare-finding baseline."
        )

    out: List[str] = []
    for f in findings_vocab:
        f_cap = f[:1].upper() + f[1:] if f else f
        if n_slots == 1:
            out.append(template.format(f_cap))
        else:
            out.append(template.format(f_cap, progression))
    return out


@torch.no_grad()
def _encode_text_bank(
    model: TempCXR,
    findings_vocab: List[str],
    progression: Optional[str],
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Encode the full finding bank once per (progression, template,
    vocab) triple. In progression-conditioned mode the eval only has 5
    distinct progression classes, so the cache fires
    ``len(gold_df) − 5`` times over a full run; in 1-slot mode it fires
    ``len(gold_df) − 1`` times."""
    cache_key = (
        f"{template}||{progression or ''}||{','.join(findings_vocab)}"
    )
    if text_cache is not None and cache_key in text_cache:
        return text_cache[cache_key].to(device)

    prompts = build_disease_prompts(findings_vocab, progression, template)
    txt_global, _, _ = model.text_encoder.forward_contrastive(prompts)
    txt_global = F.normalize(txt_global, dim=-1)  # (N, D)
    if text_cache is not None:
        text_cache[cache_key] = txt_global.detach().cpu()
    return txt_global


# ============================================================
# CORE SCORING
# ============================================================
@torch.no_grad()
def score_one_pair_biovilt(
    model: TempCXR,
    prev_path,
    curr_path,
    findings_vocab: List[str],
    progression: Optional[str],
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict:
    """N-way BioViL-T image-text disease classification for ONE (prev,
    curr, progression) row.

    Sweeps the finding axis with the progression held fixed (to the GT
    progression class, in 2-slot mode). Mirrors the signature of
    ``eval_disease_jepa.score_one_pair_disease``.

    Returns
    -------
    cos_finding_scores : list[float] — per-candidate cos similarity
    pred_idx           : int         — argmax over cos_finding_scores
    """
    prev_img = load_image(prev_path).unsqueeze(0)
    curr_img = load_image(curr_path).unsqueeze(0)

    # Pair-conditioned image encoding — BioViL-T's temporal transformer
    # cross-attends prior <-> current, so the global embedding captures
    # both images and their relationship.
    img_global, _ = model.image_encoder(curr_img, prev_img)
    img_emb = F.normalize(img_global, dim=-1)  # (1, D)

    phrase_embs = _encode_text_bank(
        model, findings_vocab, progression, template, device, text_cache,
    )
    sims = (img_emb @ phrase_embs.T).squeeze(0)  # (N,)
    cos_finding_scores = sims.detach().cpu().float().tolist()
    pred_idx = int(sims.argmax().item())

    return {
        "cos_finding_scores": cos_finding_scores,
        "pred_idx": pred_idx,
    }


# ============================================================
# DEMO MODE
# ============================================================
def run_demo(args, model, gold_df, device, findings_vocab):
    if args.idx is None:
        rng = random.Random(args.seed)
        args.idx = rng.randrange(len(gold_df))
    if not (0 <= args.idx < len(gold_df)):
        raise IndexError(f"--idx {args.idx} out of range [0, {len(gold_df)})")

    row = gold_df.iloc[args.idx]
    gt_finding = str(row["finding"]).lower()
    gt_progression = _normalize_label(row["progression"])

    print(f"\n=== Gold sample {args.idx} of {len(gold_df)} ===")
    print(f"  dataset:        {row['dataset']}")
    print(f"  patient_id:     {row['patient_id']}")
    print(f"  study_id_prev:  {row['study_id_prev']}")
    print(f"  study_id_curr:  {row['study_id_curr']}")
    print(f"  gt progression: {gt_progression}")
    print(f"  gt finding:     {gt_finding}")
    print(f"  vocab size:     {len(findings_vocab)}")

    prev_path = _resolve_with_fallbacks(
        row["dataset"], row["parent_image_prev"], args.image_roots,
    )
    curr_path = _resolve_with_fallbacks(
        row["dataset"], row["parent_image_curr"], args.image_roots,
    )

    prog_arg = gt_progression if _count_slots(args.prompt_template) == 2 else None
    out = score_one_pair_biovilt(
        model, prev_path, curr_path, findings_vocab,
        prog_arg, args.prompt_template, device,
    )

    scored = sorted(
        [(f, s) for f, s in zip(findings_vocab, out["cos_finding_scores"])],
        key=lambda p: p[1],
        reverse=True,
    )
    k = min(args.top_k, len(scored))

    print(f"\nTop-{k} predictions (of {len(findings_vocab)} candidates):")
    print(f"  {'rank':<4} {'finding':<30} {'cos':>10}")
    gt_rank = next(
        (r for r, (f, _) in enumerate(scored) if f == gt_finding),
        None,
    )
    for r in range(k):
        finding, score = scored[r]
        marker = "  <-- GT" if finding == gt_finding else ""
        print(f"  {r + 1:<4} {finding:<30} {score:>+10.4f}{marker}")
    if gt_rank is not None and gt_rank >= k:
        gt_score = scored[gt_rank][1]
        print(
            f"\n  GT finding {gt_finding!r} ranked #{gt_rank + 1} "
            f"(cos={gt_score:+.4f}, below top-{k})"
        )

    pred_finding = scored[0][0]
    print(
        f"\n  Top-1 prediction: {pred_finding:<30} "
        f"vs GT {gt_finding:<30}  "
        f"=> {'CORRECT' if pred_finding == gt_finding else 'WRONG'}"
    )


# ============================================================
# EVAL MODE
# ============================================================
def _print_eval_summary_disease(
    n_correct: int,
    n_top3: int,
    n_top5: int,
    n_seen: int,
    confusion: Dict,
    per_finding_ct: Dict,
    per_progression_ct: Dict,
    findings_vocab: List[str],
    top_k_full: int,
):
    """Same layout as eval_disease_jepa._print_eval_summary_disease so
    the two outputs are directly comparable side-by-side. Naive-baseline
    section is dropped because BioViL-T zero-shot has no predictor to
    contrast against."""
    n_vocab = len(findings_vocab)
    chance = 1.0 / max(1, n_vocab)

    print(f"\n{'=' * 70}")
    print(f"=== Results: {n_vocab}-way BioViL-T zero-shot disease "
          f"classification")
    print(f"{'=' * 70}")
    print(f"Vocab size: {n_vocab}    Chance (top-1) = {chance:.4f}")

    acc1 = n_correct / max(1, n_seen)
    acc3 = n_top3 / max(1, n_seen)
    acc5 = n_top5 / max(1, n_seen)
    print(f"\nOverall accuracy:")
    print(f"  top-1:  {n_correct}/{n_seen} = {acc1:.4f}    "
          f"(chance = {chance:.4f})")
    print(f"  top-3:  {n_top3}/{n_seen} = {acc3:.4f}    "
          f"(chance = {min(1.0, 3 * chance):.4f})")
    print(f"  top-5:  {n_top5}/{n_seen} = {acc5:.4f}    "
          f"(chance = {min(1.0, 5 * chance):.4f})")

    print("\nPer-finding accuracy (= per-finding recall):")
    print(f"  {'gt finding':<30} {'n':>6} {'acc':>8}")
    for finding in sorted(per_finding_ct.keys()):
        c, total = per_finding_ct[finding]
        a = c / total if total else float("nan")
        print(f"  {finding:<30} {total:>6} {a:>8.4f}")

    print("\nPer-progression breakdown "
          "(same disease classification metric, sliced by GT progression):")
    print(f"  {'progression':<12} {'n':>6} {'acc':>8}")
    for prog in CLS_ORDER:
        c, total = per_progression_ct.get(prog, (0, 0))
        if total == 0:
            continue
        print(f"  {prog:<12} {total:>6} {c / total:>8.4f}")

    m = _compute_balanced_metrics(confusion, findings_vocab, n_correct)
    print("\nBalanced (imbalance-corrected) metrics:")
    print(
        f"  macro recall       {m['macro_recall']:>8.4f}   "
        "(mean per-finding accuracy)"
    )
    print(f"  macro precision    {m['macro_precision']:>8.4f}")
    print(f"  macro F1           {m['macro_f1']:>8.4f}")
    print(
        f"  Cohen's kappa      {m['cohen_kappa']:>8.4f}   "
        "(0 = chance given marginals, 1 = perfect)"
    )
    if m["total"] > 0:
        print(
            f"  majority baseline  {m['majority_acc']:>8.4f}   "
            f"(always predict {m['majority_class']!r})"
        )

    print("\nPredicted vs true finding distribution "
          "(top rows only, sorted by n_true):")
    print(
        f"  {'finding':<30} {'n_pred':>7} {'pred%':>7} "
        f"{'n_true':>7} {'true%':>7}"
    )
    total = m["total"]
    ranked = sorted(
        findings_vocab, key=lambda f: m["n_true"][f], reverse=True,
    )
    for finding in ranked[:top_k_full]:
        npred = m["n_pred"][finding]
        ntrue = m["n_true"][finding]
        pred_pct = 100.0 * npred / total if total else float("nan")
        true_pct = 100.0 * ntrue / total if total else float("nan")
        print(
            f"  {finding:<30} {npred:>7} {pred_pct:>6.1f}% "
            f"{ntrue:>7} {true_pct:>6.1f}%"
        )
    if len(ranked) > top_k_full:
        n_hidden = len(ranked) - top_k_full
        print(f"  ... ({n_hidden} more findings hidden; "
              f"use --show-full-distribution to see all)")


def run_eval(args, model, gold_df, device, findings_vocab):
    if args.limit is not None:
        gold_df = gold_df.head(args.limit).reset_index(drop=True)

    finding_to_idx = {f: i for i, f in enumerate(findings_vocab)}

    n_seen = 0
    n_correct = 0
    n_top3 = 0
    n_top5 = 0
    skipped_oov = 0
    skipped_io = 0

    confusion: Dict[str, Counter] = defaultdict(Counter)
    per_finding_ct: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    per_progression_ct: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    text_cache: Dict[str, torch.Tensor] = {}
    n_slots = _count_slots(args.prompt_template)

    print(
        f"\n[eval] running {len(findings_vocab)}-way BioViL-T zero-shot "
        f"disease classification on {len(gold_df)} rows"
    )
    cond_str = (
        "progression-conditioned (GT progression in prompt)"
        if n_slots == 2
        else "bare finding name (no progression conditioning)"
    )
    print(
        f"[eval] one prompt per finding "
        f"(template: {args.prompt_template!r}, {cond_str})"
    )

    for i in range(len(gold_df)):
        row = gold_df.iloc[i]
        gt_finding = str(row["finding"]).lower()
        gt_progression = _normalize_label(row["progression"])

        if gt_finding not in finding_to_idx:
            skipped_oov += 1
            if skipped_oov <= 3:
                print(
                    f"[eval] skipping row {i} — GT finding {gt_finding!r} "
                    f"not in vocab"
                )
            continue

        try:
            prev_path = _resolve_with_fallbacks(
                row["dataset"], row["parent_image_prev"], args.image_roots,
            )
            curr_path = _resolve_with_fallbacks(
                row["dataset"], row["parent_image_curr"], args.image_roots,
            )
        except (FileNotFoundError, OSError) as e:
            skipped_io += 1
            if skipped_io <= 5:
                print(f"[eval] skipping row {i} (missing image: {e})")
            continue

        try:
            prog_arg = gt_progression if n_slots == 2 else None
            out = score_one_pair_biovilt(
                model, prev_path, curr_path, findings_vocab,
                prog_arg, args.prompt_template, device,
                text_cache=text_cache,
            )
        except Exception as e:
            skipped_io += 1
            if skipped_io <= 5:
                print(f"[eval] skipping row {i} (encoder error: {e})")
            continue

        ranked_indices = sorted(
            range(len(out["cos_finding_scores"])),
            key=lambda k: out["cos_finding_scores"][k],
            reverse=True,
        )
        gt_idx = finding_to_idx[gt_finding]
        gt_rank = ranked_indices.index(gt_idx)

        pred_idx = ranked_indices[0]
        pred_finding = findings_vocab[pred_idx]

        n_seen += 1
        n_correct += int(gt_rank == 0)
        n_top3 += int(gt_rank < 3)
        n_top5 += int(gt_rank < 5)

        confusion[gt_finding][pred_finding] += 1
        per_finding_ct[gt_finding][1] += 1
        per_finding_ct[gt_finding][0] += int(pred_finding == gt_finding)
        per_progression_ct[gt_progression][1] += 1
        per_progression_ct[gt_progression][0] += int(
            pred_finding == gt_finding
        )

        if (i + 1) % max(1, len(gold_df) // 20) == 0:
            acc = n_correct / max(1, n_seen)
            print(
                f"[eval]   {i + 1}/{len(gold_df)}  "
                f"top-1 acc={acc:.4f}  "
                f"oov={skipped_oov}  io_missing={skipped_io}"
            )

    if n_seen == 0:
        print("No samples evaluated.")
        return

    if skipped_oov:
        print(
            f"\nSkipped {skipped_oov} rows with GT findings outside the vocab "
            f"(check --min-per-finding / --top-n-findings)"
        )
    if skipped_io:
        print(f"Skipped {skipped_io} rows due to missing images / encoder errors")

    _print_eval_summary_disease(
        n_correct=n_correct,
        n_top3=n_top3,
        n_top5=n_top5,
        n_seen=n_seen,
        confusion=confusion,
        per_finding_ct=per_finding_ct,
        per_progression_ct=per_progression_ct,
        findings_vocab=findings_vocab,
        top_k_full=(
            len(findings_vocab) if args.show_full_distribution else 25
        ),
    )


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # No --ckpt: this is the zero-shot BioViL-T baseline. The image + text
    # encoders come from BioViL-T's official pretrained weights (whatever
    # ``TempCXR(mode="biovilt")`` loads by default).
    parser.add_argument(
        "--gold-parquet",
        default=DEFAULT_GOLD_PARQUET,
        help=f"Path to gold_progression_pairs.parquet "
             f"(default: {DEFAULT_GOLD_PARQUET}).",
    )
    parser.add_argument(
        "--findings-parquet",
        default=DEFAULT_FINDINGS,
        help="Path to silver_findings.parquet (used to join image paths "
             "if the gold parquet does not have them).",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Override the gold-parquet column name for the progression "
             "label. Auto-detected by default.",
    )
    parser.add_argument(
        "--finding-col",
        default=None,
        help="Override the gold-parquet column name for the "
             "finding/disease. Auto-detected by default.",
    )
    parser.add_argument(
        "--prompt-template",
        default=PROMPT_TEMPLATE,
        help="Positional-slot template applied to each candidate "
             "finding. Default (\"{} is {}.\") is 2-slot: finding + GT "
             "progression, matching eval_disease_jepa. Pass a 1-slot "
             "template like \"{}\" for the bare-finding baseline.",
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
        help="Override an image root for one dataset. Can repeat. "
             "Example: --image-root mimic=/data/final_gold_mimic_images.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="(demo only) How many top-K predictions to display. "
             "Eval mode always reports top-1 / top-3 / top-5.",
    )
    parser.add_argument(
        "--min-per-finding",
        type=int,
        default=1,
        help="Drop findings with fewer than N rows in the gold parquet.",
    )
    parser.add_argument(
        "--top-n-findings",
        type=int,
        default=None,
        help="Keep only the N most-frequent findings in the vocab.",
    )
    parser.add_argument(
        "--progression",
        default=None,
        choices=list(CLS_ORDER),
        help="Optionally restrict eval to rows whose GT progression is "
             "one of {improving, stable, worsening, new, resolved}. "
             "Filters the eval slice; the per-row GT progression is "
             "still what's inserted into the 2-slot prompt.",
    )
    parser.add_argument(
        "--show-full-distribution",
        action="store_true",
        help="Print the pred-vs-true distribution for every finding "
             "(default: top 25 only).",
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--demo", action="store_true",
        help="Print top-K disease predictions for one gold row.",
    )
    mode.add_argument(
        "--eval", action="store_true",
        help="Compute overall + per-finding + per-progression accuracy.",
    )

    parser.add_argument(
        "--idx", type=int, default=None,
        help="(demo only) gold-row index. Default: random.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="(demo only) RNG seed for picking a random row.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="(eval only) Only evaluate the first N rows.",
    )

    args = parser.parse_args()

    # Prompt-template sanity check — must accept 1 or 2 positional {}
    # slots. 2-slot = finding + progression (default, matches JEPA
    # disease eval). 1-slot = finding only (bare-finding baseline).
    n_slots = _count_slots(args.prompt_template)
    try:
        if n_slots == 2:
            _ = args.prompt_template.format("test_disease", "worsening")
        else:
            _ = args.prompt_template.format("test_disease")
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"--prompt-template failed to format with {n_slots} "
            f"positional arg(s). Got {args.prompt_template!r}: {e}"
        )

    # Image roots: same resolution rule as eval_progression_jepa.py.
    parquet_dir = os.path.dirname(os.path.abspath(args.gold_parquet))
    auto_gold_roots = discover_gold_image_roots(parquet_dir)
    image_roots: Dict[str, str] = {**IMAGE_ROOTS, **auto_gold_roots}
    if auto_gold_roots:
        print("[gold] auto-detected gold image roots:")
        for d, p in auto_gold_roots.items():
            print(f"  {d}: {p}")
    for spec in args.image_root:
        if "=" not in spec:
            raise ValueError(
                f"--image-root expects DATASET=PATH, got: {spec!r}"
            )
        d, p = spec.split("=", 1)
        if d not in DATASETS:
            raise ValueError(
                f"--image-root dataset must be one of {DATASETS}, got {d!r}"
            )
        image_roots[d] = p
        print(f"[gold] override: {d} -> {p}")
    args.image_roots = image_roots

    device = torch.device(args.device)
    model = load_model()
    gold_df = load_gold_pairs(
        args.gold_parquet,
        args.findings_parquet,
        label_col=args.label_col,
        finding_col=args.finding_col,
    )
    if len(gold_df) == 0:
        raise RuntimeError("No usable gold rows after filtering.")

    if args.progression is not None:
        before = len(gold_df)
        gold_df = gold_df[gold_df["progression"] == args.progression]
        gold_df = gold_df.reset_index(drop=True)
        print(
            f"[gold] restricted to progression={args.progression!r}: "
            f"{len(gold_df)}/{before} rows retained"
        )
        if len(gold_df) == 0:
            raise RuntimeError(
                f"No gold rows with progression={args.progression!r}"
            )

    findings_vocab, full_counts = build_findings_vocab(
        gold_df,
        min_per_finding=args.min_per_finding,
        top_n=args.top_n_findings,
    )
    if len(findings_vocab) < 2:
        raise RuntimeError(
            f"Vocabulary size = {len(findings_vocab)} is too small for "
            "N-way classification. Relax --min-per-finding or "
            "--top-n-findings."
        )

    print(
        f"\n[vocab] {len(findings_vocab)} unique findings in vocab "
        f"(from {len(full_counts)} total in the (filtered) gold slice)"
    )
    if args.min_per_finding > 1 or args.top_n_findings is not None:
        n_dropped = len(full_counts) - len(findings_vocab)
        print(
            f"[vocab]   {n_dropped} finding(s) dropped by "
            f"--min-per-finding={args.min_per_finding} / "
            f"--top-n-findings={args.top_n_findings}"
        )
    print(f"[vocab] top-10 findings by count:")
    for f, c in full_counts.most_common(10):
        in_vocab = "" if f in findings_vocab else "  (dropped)"
        print(f"    {f:<30} n={c:<6}{in_vocab}")

    vocab_set = set(findings_vocab)
    before = len(gold_df)
    gold_df = gold_df[
        gold_df["finding"].astype(str).str.lower().isin(vocab_set)
    ].reset_index(drop=True)
    if len(gold_df) < before:
        print(
            f"[gold] dropped {before - len(gold_df)} rows whose GT "
            f"finding was outside the vocab"
        )

    if args.demo:
        run_demo(args, model, gold_df, device, findings_vocab)
    else:
        run_eval(args, model, gold_df, device, findings_vocab)


if __name__ == "__main__":
    main()
