#!/usr/bin/env python3
"""Evaluate a dynamic-sentence VL-JEPA checkpoint.

  gold       CheXTemporal gold; saves a score file that
             ``baap_eval/eval_baap.py compare-gold`` reads
  csv        a clean MS-CXR-T or Chest ImaGenome CSV; saves a score file in
             the ``vljepa.eval_mscxrt compare`` format
  retrieval  silver val: does Ŝ rank its own dynamic sentence first?

``--mode perfinding`` scores each finding with the 37-phrase bank (the
zero-shot baselines' protocol). ``--mode combined`` scores every class
combination over the pair's findings ("Edema is worsening. Pleural effusion
is stable.") and takes each finding's soft maximum per class.

    python -m vljepa.eval_dynamic gold --ckpt checkpoints_vljepa_dynamic/epoch_5.pt
    python -m vljepa.eval_dynamic gold --ckpt ... --mode combined
    python -m vljepa.eval_dynamic csv --ckpt ... --clean mscxrt_clean_patient.csv \\
        --out-dir mscxrt_scores_patient
    python -m vljepa.eval_dynamic retrieval --ckpt ...

Score files hold per-pair labels; keep them out of git.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from progression_phrases import CLS_ORDER


def build_scorer(args, device):
    from vljepa.dynamic import DynamicScorer, load_dynamic_from_ckpt

    model = load_dynamic_from_ckpt(args.ckpt, device)
    return DynamicScorer(model, mode=args.mode, max_k=args.max_k,
                         combo_temperature=args.combo_temperature)


def default_name(args) -> str:
    return args.name or f"vljepa_dyn_{args.mode}"


def gold(args) -> int:
    import torch

    from baap_eval.eval_baap import print_gold_metrics
    from progression_classify import DEFAULT_GOLD_PARQUET, discover_gold_image_roots
    from vljepa.cluster_paths import gold_image_dirs, image_roots
    from vljepa.dynamic import score_gold_groups
    from vljepa.eval_gold import load_gold_groups

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = build_scorer(args, device)
    groups = load_gold_groups()
    if args.limit:
        groups = groups.iloc[:args.limit].reset_index(drop=True)
    roots = {
        **image_roots(),
        **discover_gold_image_roots(os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))),
        **gold_image_dirs(),
    }
    scores, gts, findings, keys, datasets = score_gold_groups(scorer, groups, roots, device)
    name = default_name(args)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   f"gold_{name}_{os.path.splitext(os.path.basename(args.ckpt))[0]}.npz")
    np.savez(out, gold_scores=scores, gold_gts=gts, gold_findings=findings, gold_keys=keys,
             gold_datasets=datasets, score_kind="cosine", model_name=name, ckpt=args.ckpt,
             mode=args.mode)
    print(f"[dynamic] saved {out}")
    print_gold_metrics(name, scores, gts, findings, datasets)
    return 0


def csv(args) -> int:
    import pandas as pd
    import torch
    from tqdm import tqdm

    from progression_classify import load_image_tensor
    from vljepa.eval_mscxrt import print_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.clean, dtype=str)
    if args.limit:
        df = df.iloc[:args.limit]
    scorer = build_scorer(args, device)
    pair = ["patient_id", "study_id_prev", "study_id_curr"]
    siblings = df.groupby(pair)["finding"].apply(lambda s: sorted(set(s))).to_dict()
    per_pair = {}
    scores = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=default_name(args), file=sys.stdout):
            pk = tuple(row[c] for c in pair)
            if pk not in per_pair:
                prior = load_image_tensor("mimic", row["prev_path"], {"mimic": "/"}).to(device)
                current = load_image_tensor("mimic", row["curr_path"], {"mimic": "/"}).to(device)
                per_pair[pk] = scorer.score_pair(prior, current, siblings[pk])
            scores.append(per_pair[pk][row["finding"]])
    if scorer.n_fallback:
        print(f"[dynamic] combined: {scorer.n_fallback} pairs had more than {scorer.max_k} "
              "findings and were scored per finding")
    scores = np.stack(scores).astype(np.float64)
    labels = np.array([CLS_ORDER.index(c) for c in df["label"]], dtype=np.int64)
    name = default_name(args)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{name}.npz")
    np.savez(out, scores=scores, labels=labels, findings=df["finding"].to_numpy(dtype=object),
             keys=df["key"].to_numpy(dtype=object), name=name, model="vljepa_dynamic",
             ckpt=str(args.ckpt), score_kind="cosine")
    print(f"[dynamic] saved {out}")
    print_metrics({name: (scores, labels, df["finding"].to_numpy(dtype=object))})
    return 0


def retrieval(args) -> int:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    from vljepa.cluster_paths import image_roots
    from vljepa.dynamic import (
        VLJEPADynamicDataset,
        dynamic_collate_fn,
        load_dynamic_from_ckpt,
        retrieval_recall,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dynamic_from_ckpt(args.ckpt, device)
    ds = VLJEPADynamicDataset(image_roots(), split="val", train=False)
    if args.limit:
        ds = Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=dynamic_collate_fn)
    preds, tgts, texts = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="retrieval", file=sys.stdout):
            out = model(batch["prior_image"].to(device), batch["current_image"].to(device),
                        batch["query_text"], target_texts=batch["target_text"])
            preds.append(F.normalize(out["pred"].float(), dim=-1))
            tgts.append(F.normalize(out["target_global"].float(), dim=-1))
            texts.extend(batch["target_text"])
    r = retrieval_recall(torch.cat(preds), torch.cat(tgts), texts)
    print(f"[dynamic] silver val retrieval over {r['gallery']} distinct sentences "
          f"({len(texts)} pairs): R@1 {r['recall@1']:.3f}  R@5 {r['recall@5']:.3f}  "
          f"R@10 {r['recall@10']:.3f}  median rank {r['median_rank']:.0f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(a, scoring=True):
        a.add_argument("--ckpt", required=True)
        a.add_argument("--limit", type=int)
        if scoring:
            a.add_argument("--mode", default="perfinding", choices=["perfinding", "combined"])
            a.add_argument("--max-k", type=int, default=4,
                           help="combined: pairs with more findings are scored per finding")
            a.add_argument("--combo-temperature", type=float, default=0.07)
            a.add_argument("--name", help="score name (default vljepa_dyn_<mode>)")

    a = sub.add_parser("gold")
    common(a)
    a.add_argument("--out")
    a = sub.add_parser("csv")
    common(a)
    a.add_argument("--clean", required=True, help="mscxrt_clean_patient.csv or cig_clean_patient_ours.csv")
    a.add_argument("--out-dir", required=True, help="the folder with the other models' score files")
    a = sub.add_parser("retrieval")
    common(a, scoring=False)

    args = p.parse_args()
    return {"gold": gold, "csv": csv, "retrieval": retrieval}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
