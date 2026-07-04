"""N-way disease classification via image-image cosine matching (JEPA).

Mirror image of ``eval_progression_jepa.py``: instead of sweeping the
5-way progression axis with a fixed finding, this eval sweeps the
**finding axis** with the fixed ground-truth progression class. For
each gold row ``(prior_image, current_image, gt_finding, gt_progression)``:

  1. Encode the prior with the online image encoder → ``z_prior``.
  2. Encode the current with the EMA target encoder → ``z_cur`` (detached).
  3. Build the findings vocabulary ``D`` from the unique lower-cased
     ``finding`` values in the gold parquet (optionally filtered by
     ``--min-per-finding`` / ``--top-n-findings``).
  4. For each candidate finding ``d ∈ D``, build one templated prompt::

         "{d.capitalize()} is {gt_progression}."

     Encode all ``|D|`` prompts through the text encoder.
  5. Batch the predictor across the ``|D|`` prompts (same ``z_prior``
     expanded to a batch of ``|D|``) to obtain ``|D|`` candidate
     predictions ``ẑ_cur^d``.
  6. Score each candidate by ``mean over patches of cos(ẑ_cur^d, z_cur)``.
  7. **Predicted finding = argmax_d**.

Why do this?
------------
The 5-way progression eval measures whether the *progression* token in
the prompt drives the predictor's output. If it does, classification
wins. But an alternate explanation is that the *finding* token is
being ignored — that the model has learned a shortcut of the form
"any 'worsening' prompt produces a slightly larger predicted change,
regardless of what disease it names." This eval falsifies (or
confirms) that shortcut: if disease classification is well above
chance, the finding token is doing meaningful work. If it's near
chance, the model is class-blind to the specific pathology being
described.

This is the natural symmetry to progression classification: the same
image-image cosine rule, the same predictor forward, the same target,
just with the finding axis being swept instead of the progression axis.

Do-nothing baseline
-------------------
For each pair we also report ``cos(z_prior, z_cur)``. If the best
``ẑ_cur^d`` scores below this baseline, the predictor's finding-
conditioned delta was worse than predicting "no change" on this pair.

Usage
-----
    # Sanity-check one gold row with the full vocab (top-5 predictions)
    python eval_disease_jepa.py --demo

    # Full N-way eval over the gold parquet
    python eval_disease_jepa.py --eval

    # Restrict vocab to findings with ≥20 gold rows and top-1/3/5 report
    python eval_disease_jepa.py --eval --min-per-finding 20

    # Only score rows whose GT progression is "worsening"
    python eval_disease_jepa.py --eval --progression worsening
"""

import argparse
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from dataset_combined_jepa import DEFAULT_FINDINGS
from eval_progression_jepa import _compute_balanced_metrics
from infer_jepa import IMAGE_ROOTS, load_jepa_model
from progression_classify import (
    DATASETS,
    DEFAULT_GOLD_PARQUET,
    _normalize_label,
    discover_gold_image_roots,
    load_gold_pairs,
    load_image_tensor,
)
from progression_phrases import CLS_ORDER
from tempcxr.modules.jepa import TempCXRJEPA


# ============================================================
# CONSTANTS
# ============================================================
# Same canonical template as the 5-way progression eval — capitalized
# finding, lowercase progression, trailing period. This matches how
# ``JEPACombinedDataset._build_templated_condition`` formats the per-
# finding training clauses.
PROMPT_TEMPLATE = "{} is {}."


# ============================================================
# PROMPT BUILDING
# ============================================================
def build_disease_prompts(
    progression: str,
    findings_vocab: List[str],
    template: str = PROMPT_TEMPLATE,
) -> List[str]:
    """One templated prompt per candidate finding, holding the GT
    progression class fixed. Mirror of ``build_class_prompts`` in
    ``eval_progression_jepa.py`` — same template, swept axis inverted.

    Capitalizes the first letter of each finding to match the templated
    training condition format.
    """
    out: List[str] = []
    for f in findings_vocab:
        if not f:
            out.append(template.format(f, progression))
            continue
        f_cap = f[:1].upper() + f[1:]
        out.append(template.format(f_cap, progression))
    return out


# ============================================================
# CORE SCORING (image-image cosine, findings axis)
# ============================================================
@torch.no_grad()
def _encode_disease_prompts(
    model: TempCXRJEPA,
    progression: str,
    findings_vocab: List[str],
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, Tuple]] = None,
):
    """Cache-aware text encoding for the finding-sweep bank.

    The cache is keyed on ``(progression, template, vocab)`` — since
    there are only 5 progression classes and a single fixed vocab, the
    cache fires ``len(gold_df) − 5`` times over a full eval.
    """
    cache_key = (
        f"disease||{progression}||{template}||"
        f"{','.join(findings_vocab)}"
    )
    if text_cache is not None and cache_key in text_cache:
        prompts, txt_local, token_mask = text_cache[cache_key]
        return prompts, txt_local.to(device), token_mask.to(device)

    prompts = build_disease_prompts(progression, findings_vocab, template)
    _, txt_local, token_mask = model.text_encoder.forward_contrastive(prompts)
    if text_cache is not None:
        text_cache[cache_key] = (
            prompts,
            txt_local.detach().cpu(),
            token_mask.detach().cpu(),
        )
    return prompts, txt_local, token_mask


@torch.no_grad()
def score_one_pair_disease(
    model: TempCXRJEPA,
    prior_img: torch.Tensor,
    current_img: torch.Tensor,
    progression: str,
    findings_vocab: List[str],
    template: str,
    device: torch.device,
    text_cache: Optional[Dict[str, Tuple]] = None,
) -> Dict:
    """N-way image-image scoring for ONE (prior, current, progression) row.

    ``N = len(findings_vocab)``. The correct progression class is fixed
    (the row's GT), and we sweep the finding axis.

    Returns
    -------
    prompts             : list[str]   — per-finding templated prompts
    cos_finding_scores  : list[float] — mean per-patch cos per candidate
    pred_idx            : int          — argmax over cos_finding_scores
    cos_naive           : float        — cos(z_prior, z_cur) baseline
    """
    prompts, txt_local, token_mask = _encode_disease_prompts(
        model, progression, findings_vocab, template, device, text_cache,
    )
    n_prompts = len(prompts)

    prior = prior_img.unsqueeze(0).to(device)
    current = current_img.unsqueeze(0).to(device)

    _, z_prior = model.image_encoder(prior)               # (1, N, D)
    _, z_cur = model.target_image_encoder(current)        # (1, N, D)
    z_cur = z_cur.detach()

    # Batch the predictor across all N candidate findings by broadcasting
    # the same z_prior. delta_z differs per candidate because txt_local
    # differs per candidate.
    z_prior_b = z_prior.expand(n_prompts, -1, -1).contiguous()
    preds = model.predictor(z_prior_b, txt_local, token_mask)

    pred_f = preds.float()
    target_f = z_cur.float().expand_as(pred_f)
    cos_per_patch = F.cosine_similarity(pred_f, target_f, dim=-1)  # (N, P)
    cos_finding_scores = cos_per_patch.mean(dim=1).tolist()

    cos_naive = F.cosine_similarity(
        z_prior.float(), z_cur.float(), dim=-1,
    ).mean().item()

    pred_idx = max(range(n_prompts), key=lambda k: cos_finding_scores[k])

    return {
        "prompts": prompts,
        "cos_finding_scores": cos_finding_scores,
        "pred_idx": pred_idx,
        "cos_naive": cos_naive,
    }


# ============================================================
# FINDINGS VOCABULARY
# ============================================================
def build_findings_vocab(
    gold_df,
    min_per_finding: int = 1,
    top_n: Optional[int] = None,
) -> Tuple[List[str], Counter]:
    """Extract sorted, lowercased unique finding names from the gold set.

    ``min_per_finding`` drops any finding with fewer than that many rows
    (useful because rare findings contribute noisy per-class metrics and
    aren't really testable). ``top_n`` keeps only the N most frequent
    findings — useful when the vocab is huge and you want a bounded
    N-way problem.

    Returns
    -------
    findings_vocab : list[str]  — sorted alphabetically, lower-cased
    counts         : Counter    — full per-finding counts (unfiltered
                                   so the caller can print them)
    """
    counts = Counter(str(f).lower() for f in gold_df["finding"])
    filtered = counts
    if min_per_finding > 1:
        filtered = Counter(
            {f: c for f, c in filtered.items() if c >= min_per_finding}
        )
    if top_n is not None and top_n > 0:
        filtered = Counter(dict(filtered.most_common(top_n)))
    return sorted(filtered.keys()), counts


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

    prior = load_image_tensor(
        row["dataset"], row["parent_image_prev"], args.image_roots,
    )
    current = load_image_tensor(
        row["dataset"], row["parent_image_curr"], args.image_roots,
    )

    out = score_one_pair_disease(
        model, prior, current, gt_progression, findings_vocab,
        args.prompt_template, device,
    )

    # Sort by cos score descending -> top-K predictions
    scored = sorted(
        [(f, s) for f, s in zip(findings_vocab, out["cos_finding_scores"])],
        key=lambda p: p[1],
        reverse=True,
    )
    k = min(args.top_k, len(scored))

    print(f"\nTop-{k} predictions (of {len(findings_vocab)} candidates):")
    print(f"  {'rank':<4} {'finding':<30} {'cos':>10} {'Δ vs naive':>12}")
    gt_rank = next(
        (r for r, (f, _) in enumerate(scored) if f == gt_finding),
        None,
    )
    for r in range(k):
        finding, score = scored[r]
        delta = score - out["cos_naive"]
        marker = "  <-- GT" if finding == gt_finding else ""
        print(
            f"  {r + 1:<4} {finding:<30} {score:>+10.4f} "
            f"{delta:>+12.4f}{marker}"
        )
    print(
        f"  (naive baseline cos(z_prior, z_cur) = {out['cos_naive']:+.4f})"
    )
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
    naive_sum: float,
    n_above_naive: int,
    top_k_full: int,
):
    n_vocab = len(findings_vocab)
    chance = 1.0 / max(1, n_vocab)

    print(f"\n{'=' * 70}")
    print(f"=== Results: {n_vocab}-way disease classification "
          f"(fixed GT progression)")
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

    naive_mean = naive_sum / max(1, n_seen)
    above_pct = 100.0 * n_above_naive / max(1, n_seen)
    print(
        f"\n  naive baseline cos(z_prior, z_cur) = {naive_mean:+.4f}"
    )
    print(
        f"  Pairs where top-1 cos > baseline: "
        f"{n_above_naive}/{n_seen} ({above_pct:.1f}%)"
    )
    print(
        "  (Higher = predictor's finding-conditioned best guess beats "
        "the do-nothing baseline.)"
    )

    # -----------------------------------------------------------------
    # Per-finding (per-class recall on the finding axis)
    # -----------------------------------------------------------------
    print("\nPer-finding accuracy (= per-finding recall):")
    print(f"  {'gt finding':<30} {'n':>6} {'acc':>8}")
    for finding in sorted(per_finding_ct.keys()):
        c, total = per_finding_ct[finding]
        a = c / total if total else float("nan")
        print(f"  {finding:<30} {total:>6} {a:>8.4f}")

    # -----------------------------------------------------------------
    # Per-progression breakdown — is the model better at distinguishing
    # findings when the progression is worsening vs stable vs improving?
    # -----------------------------------------------------------------
    print("\nPer-progression breakdown "
          "(same disease classification metric, sliced by GT progression):")
    print(f"  {'progression':<12} {'n':>6} {'acc':>8}")
    for prog in CLS_ORDER:
        c, total = per_progression_ct.get(prog, (0, 0))
        if total == 0:
            continue
        print(f"  {prog:<12} {total:>6} {c / total:>8.4f}")

    # -----------------------------------------------------------------
    # Balanced (imbalance-corrected) metrics.
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Predicted vs true finding distribution (only most-common shown for
    # readability; the full table would be N_vocab lines long).
    # -----------------------------------------------------------------
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
    n_above_naive = 0
    naive_sum = 0.0
    skipped_oov = 0
    skipped_io = 0

    confusion: Dict[str, Counter] = defaultdict(Counter)
    # each entry: [correct, total]
    per_finding_ct: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    per_progression_ct: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    text_cache: Dict[str, Tuple] = {}

    print(
        f"\n[eval] running {len(findings_vocab)}-way image-image cosine "
        f"disease classification on {len(gold_df)} rows"
    )
    print(
        f"[eval] one prompt per finding, GT progression held fixed "
        f"(template: {args.prompt_template!r})"
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
                    f"is not in the vocab"
                )
            continue

        try:
            prior = load_image_tensor(
                row["dataset"], row["parent_image_prev"], args.image_roots,
            )
            current = load_image_tensor(
                row["dataset"], row["parent_image_curr"], args.image_roots,
            )
        except (FileNotFoundError, OSError) as e:
            skipped_io += 1
            if skipped_io <= 5:
                print(f"[eval] skipping row {i} (missing image: {e})")
            continue

        out = score_one_pair_disease(
            model, prior, current, gt_progression, findings_vocab,
            args.prompt_template, device, text_cache=text_cache,
        )

        # Rank of GT finding in the sorted score list (0 = top-1).
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

        best_cos = out["cos_finding_scores"][pred_idx]
        if best_cos > out["cos_naive"]:
            n_above_naive += 1
        naive_sum += out["cos_naive"]

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
        print(f"Skipped {skipped_io} rows due to missing images")

    _print_eval_summary_disease(
        n_correct=n_correct,
        n_top3=n_top3,
        n_top5=n_top5,
        n_seen=n_seen,
        confusion=confusion,
        per_finding_ct=per_finding_ct,
        per_progression_ct=per_progression_ct,
        findings_vocab=findings_vocab,
        naive_sum=naive_sum,
        n_above_naive=n_above_naive,
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
    parser.add_argument(
        "--ckpt",
        default=os.environ.get(
            "JEPA_CKPT", "checkpoints_jepa_dynamic/best.pt"
        ),
        help="Path to a JEPA checkpoint "
             "(default: checkpoints_jepa_dynamic/best.pt).",
    )
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
        help="Two-slot positional template: {finding} is filled in first, "
             "{progression} second. Default: '{} is {}.' (matches the "
             "templated training condition format).",
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
             "The eval mode always reports top-1 / top-3 / top-5.",
    )
    parser.add_argument(
        "--min-per-finding",
        type=int,
        default=1,
        help="Drop findings with fewer than N rows in the gold parquet. "
             "Rare findings contribute noisy per-class metrics; setting "
             "this to 10 or 20 gives a cleaner N-way problem. "
             "Default: 1 (keep all).",
    )
    parser.add_argument(
        "--top-n-findings",
        type=int,
        default=None,
        help="Keep only the N most-frequent findings in the vocab. "
             "Overrides --min-per-finding when both would apply.",
    )
    parser.add_argument(
        "--progression",
        default=None,
        choices=list(CLS_ORDER),
        help="Optionally restrict eval to rows whose GT progression is "
             "one of {improving, stable, worsening, new, resolved}. "
             "Useful to answer 'within worsening pairs, does the model "
             "distinguish diseases?' — tests the finding token in "
             "isolation.",
    )
    parser.add_argument(
        "--show-full-distribution",
        action="store_true",
        help="Print the predicted-vs-true distribution for every finding "
             "in the vocab (default: top 25 only).",
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

    # Prompt-template sanity check.
    try:
        _ = args.prompt_template.format("test_disease", "test_class")
    except (IndexError, KeyError) as e:
        raise ValueError(
            f"--prompt-template must accept two positional {{}} slots "
            f"({{finding}}, {{progression}}). "
            f"Got {args.prompt_template!r}: {e}"
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
    model = load_jepa_model(args.ckpt, device)
    gold_df = load_gold_pairs(
        args.gold_parquet,
        args.findings_parquet,
        label_col=args.label_col,
        finding_col=args.finding_col,
    )
    if len(gold_df) == 0:
        raise RuntimeError("No usable gold rows after filtering.")

    # Optional progression-class filter (before vocab build so per-slice
    # runs use the vocab that actually appears in that slice).
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

    # Build the disease vocabulary from the (possibly filtered) gold set.
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

    # Drop gold rows whose GT finding isn't in the vocab so we're not
    # forcing the model to pick between only-wrong candidates. Rows are
    # counted as skipped in run_eval / run_demo, but pre-filtering here
    # keeps random --demo picks from all landing on OOV rows.
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
