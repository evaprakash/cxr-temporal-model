#!/usr/bin/env python3
"""MS-CXR-T 3-way progression (improving / stable / worsening) for BioViL-T,
the supervised baseline, and VL-JEPA, on pairs that share no image and no
study with silver.

  prepare  dedupe, rewrite image paths onto the local MIMIC-CXR-JPG copy,
           drop pairs overlapping silver, write the clean CSV
  score    score one model on the clean CSV; saves all five class scores
  compare  metrics for several score files, plus a paired bootstrap of
           each model minus a reference model

The prediction is the argmax over the three MS-CXR-T classes only. VL-JEPA
adds the class shift saved in its checkpoint (only the improving / stable /
worsening entries matter here). BioViL-T is scored zero-shot with the same
phrase bank as its gold eval (eval_progression_biovilt.py).

The CSV and score files hold PhysioNet labels; keep them out of git.

    python -m vljepa.eval_mscxrt prepare
    python -m vljepa.eval_mscxrt score --model biovilt
    python -m vljepa.eval_mscxrt score --model supervised \\
        --ckpt checkpoints_supervised_progression_unfrozen/epoch_5.pt
    python -m vljepa.eval_mscxrt score --model vljepa \\
        --ckpt checkpoints_vljepa_lora_5way/epoch_5_shifted.pt
    python -m vljepa.eval_mscxrt compare --ref supervised
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

from progression_phrases import CLS_ORDER

VALID = ["improving", "stable", "worsening"]
VALID_IDX = [CLS_ORDER.index(c) for c in VALID]
STABLE = CLS_ORDER.index("stable")
LABEL_MAP = {
    "improving": "improving", "improved": "improving",
    "stable": "stable", "no change": "stable", "no-change": "stable", "unchanged": "stable",
    "worsening": "worsening", "worsened": "worsening",
}
# A prior study with several frames is listed once per frame.
DEDUP = ["patient_id", "study_id_prev", "study_id_curr", "disease_name"]
DEFAULT_CSV = os.environ.get("MSCXRT_CSV", os.path.join(REPO_ROOT, "mscxrt_labels_new.csv"))
DEFAULT_CLEAN = os.path.join(REPO_ROOT, "mscxrt_clean.csv")
DEFAULT_MIMIC = os.environ.get("MIMIC_CXR_JPG", "/nvme/data/cxr/mimic-cxr-jpg/2.0.0")
DEFAULT_SCORES = os.path.join(REPO_ROOT, "mscxrt_scores")
PHYSIONET = "https://physionet.org/files/mimic-cxr-jpg/2.0.0/"


def study_digits(x) -> str:
    return re.sub(r"\D", "", str(x)).lstrip("0")


def local_path(path: str, roots) -> str:
    """First existing copy under any root (as root/files/pXX/... or root/pXX/...);
    the first candidate if none exists."""
    path = str(path).strip()
    i = path.find("/files/")
    if i < 0:
        raise ValueError(f"no /files/ in MS-CXR-T image path: {path}")
    rel = path[i + 1:]
    cands = [os.path.join(r, rel) for r in roots] + [os.path.join(r, rel[len("files/"):]) for r in roots]
    return next((c for c in cands if os.path.isfile(c)), cands[0])


def explain_missing(paths, root):
    """Where the copy under ``root`` stops: no patient folder, no study folder, or no file."""
    from collections import Counter

    why, by_group = Counter(), Counter()
    for p in paths:
        study_dir = os.path.dirname(p)
        patient_dir = os.path.dirname(study_dir)
        if not os.path.isdir(patient_dir):
            why["no patient folder"] += 1
        elif not os.path.isdir(study_dir):
            why["no study folder"] += 1
        else:
            why["study folder without this file"] += 1
        by_group[os.path.basename(os.path.dirname(patient_dir))] += 1
    print("  why: " + ", ".join(f"{k} {v}" for k, v in why.most_common()))
    print("  by folder: " + ", ".join(f"{k} {v}" for k, v in sorted(by_group.items())))


# ----------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------
def prepare(args) -> int:
    from vljepa.cluster_paths import silver_findings_parquet

    df = pd.read_csv(args.csv, dtype=str)
    n_raw = len(df)
    df["label"] = df["comparison"].str.strip().str.lower().map(LABEL_MAP)
    df = df[df["label"].notna()]
    n_labeled = len(df)
    df = df.drop_duplicates(subset=DEDUP).reset_index(drop=True)
    print(f"[mscxrt] {args.csv}: {n_raw} rows, {n_labeled} with a 3-way label, "
          f"{len(df)} after removing repeated prior frames")

    silver_path = args.silver or silver_findings_parquet()
    sil = pd.read_parquet(silver_path)
    print(f"[mscxrt] silver: {silver_path} ({len(sil)} rows)")
    names = set()
    for col in ("parent_image_curr", "parent_image_prev"):
        names.update(os.path.basename(str(p).strip()) for p in sil[col].dropna())
    names.discard("")
    mimic = sil[sil["dataset"].astype(str).str.lower() == "mimic"] if "dataset" in sil else sil
    studies = set()
    for col in ("study_id_curr", "study_id_prev"):
        if col in mimic:
            studies.update(study_digits(s) for s in mimic[col].dropna())
    studies.discard("")
    if not studies:
        raise SystemExit("[mscxrt] silver has no study_id_curr / study_id_prev; cannot check study overlap")

    img_hit = (df["img_path_prev"].map(os.path.basename).isin(names)
               | df["img_path_curr"].map(os.path.basename).isin(names))
    study_hit = (df["study_id_prev"].map(study_digits).isin(studies)
                 | df["study_id_curr"].map(study_digits).isin(studies))
    print(f"[mscxrt] overlap with silver: image {int(img_hit.sum())}, study {int(study_hit.sum())}, "
          f"either {int((img_hit | study_hit).sum())} of {len(df)}")
    df = df[~(img_hit | study_hit)].reset_index(drop=True)

    roots = [args.mimic_root] + [r for r in args.alt_root if os.path.isdir(r)]
    print(f"[mscxrt] image roots, in order: {roots}")
    df["prev_path"] = df["img_path_prev"].map(lambda p: local_path(p, roots))
    df["curr_path"] = df["img_path_curr"].map(lambda p: local_path(p, roots))
    present = df["prev_path"].map(os.path.isfile) & df["curr_path"].map(os.path.isfile)
    if not present.all():
        missing = [p for p in pd.concat([df.loc[~present, "prev_path"], df.loc[~present, "curr_path"]])
                   if not os.path.isfile(p)]
        print(f"[mscxrt] {int((~present).sum())} clean pairs have a missing image "
              f"({len(set(missing))} distinct files), e.g. {missing[0]}")
        explain_missing(sorted(set(missing)), args.mimic_root)
        list_path = os.path.splitext(args.out)[0] + "_missing_urls.txt"
        with open(list_path, "w") as f:
            for p in sorted(set(missing)):
                f.write(PHYSIONET + p[p.find("files/"):] + "\n")
        print(f"[mscxrt] wrote the missing files' PhysioNet URLs to {list_path}")
        if not args.allow_missing:
            print("[mscxrt] stopping without writing the clean CSV. Fix the image copy, or pass "
                  "--allow-missing to score only the pairs that have both images.")
            return 1
    df = df[present].reset_index(drop=True)
    df["finding"] = df["disease_name"].str.strip().str.lower()
    df["key"] = (df["patient_id"] + "_" + df["study_id_prev"] + "_"
                 + df["study_id_curr"] + "_" + df["finding"])

    print(f"[mscxrt] clean pairs: {len(df)}")
    print("  " + ", ".join(f"{c} {int((df['label'] == c).sum())}" for c in VALID))
    print("  " + ", ".join(f"{f} {n}" for f, n in df["finding"].value_counts().items()))
    df.to_csv(args.out, index=False)
    print(f"[mscxrt] wrote {args.out}")
    return 0


# ----------------------------------------------------------------------
# score
# ----------------------------------------------------------------------
def make_scorer(args, device):
    """Returns (score kind, fn(prior, current, finding) -> np.ndarray of 5 scores)."""
    if args.model == "vljepa":
        from vljepa.eval_gold import load_vljepa_from_ckpt

        model = load_vljepa_from_ckpt(args.ckpt, device)
        shift = model.class_shift
        cache = {}

        def fn(prior, current, finding):
            s = model.score_classes(prior.unsqueeze(0), current.unsqueeze(0), finding, text_cache=cache)
            s = s.float().cpu()
            return (s + shift if shift is not None else s).numpy()

        return "cosine", fn
    if args.model == "supervised":
        from vljepa.check_class_signal import SupervisedScorer

        scorer = SupervisedScorer(args.ckpt, device)
        return "logit", lambda p, c, f: scorer.one(
            p.unsqueeze(0), c.unsqueeze(0), f).float().cpu().numpy()

    from eval_progression_biovilt import PROMPT_TEMPLATE, BioViLTPairModel, score_one_pair

    model = BioViLTPairModel(device)
    cache = {}
    return "cosine", lambda p, c, f: np.array(score_one_pair(
        model, p, c, f, PROMPT_TEMPLATE, device, cache)["cos_class_scores"])


def score(args) -> int:
    import torch
    from tqdm import tqdm

    from progression_classify import load_image_tensor

    if args.model != "biovilt" and not args.ckpt:
        raise SystemExit(f"[mscxrt] --model {args.model} needs --ckpt")
    name = args.name or args.model
    df = pd.read_csv(args.clean, dtype=str)
    if args.limit:
        df = df.iloc[:args.limit]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mscxrt] {name}: {args.model} {args.ckpt or '(pretrained)'} on {len(df)} pairs, {device}")
    kind, fn = make_scorer(args, device)

    scores = []
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=name, file=sys.stdout):
            prior = load_image_tensor("mimic", row["prev_path"], {"mimic": "/"}).to(device)
            current = load_image_tensor("mimic", row["curr_path"], {"mimic": "/"}).to(device)
            scores.append(fn(prior, current, row["finding"]))
    scores = np.stack(scores).astype(np.float64)
    labels = np.array([CLS_ORDER.index(c) for c in df["label"]], dtype=np.int64)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{name}.npz")
    np.savez(out, scores=scores, labels=labels, findings=df["finding"].to_numpy(dtype=object),
             keys=df["key"].to_numpy(dtype=object), name=name, model=args.model,
             ckpt=str(args.ckpt), score_kind=kind)
    print(f"[mscxrt] saved {out}")
    print_metrics({name: (scores, labels, df["finding"].to_numpy(dtype=object))})
    return 0


# ----------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------
def three_way(scores):
    return scores[:, VALID_IDX]


def metrics(scores, labels):
    """Accuracy, macro accuracy (mean recall), macro F1, per-class F1, AUROC vs stable."""
    from vljepa.compare_shift_rules import fast_auroc

    pred = np.array(VALID_IDX)[three_way(scores).argmax(1)]
    rec, f1 = [], []
    for c in VALID_IDX:
        tp = int(((pred == c) & (labels == c)).sum())
        n_pred, n_true = int((pred == c).sum()), int((labels == c).sum())
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_true if n_true else 0.0
        rec.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    auc = {}
    for cls in ("improving", "worsening"):
        c = CLS_ORDER.index(cls)
        keep = (labels == c) | (labels == STABLE)
        auc[cls] = fast_auroc(scores[keep, c] - scores[keep, STABLE], labels[keep] == c)
    return {
        "acc": float((pred == labels).mean()),
        "macro_acc": float(np.mean(rec)),
        "macro_f1": float(np.mean(f1)),
        "f1": dict(zip(VALID, f1)),
        "calls": {c: int((pred == CLS_ORDER.index(c)).sum()) for c in VALID},
        "auc": auc,
        "pred": pred,
    }


def print_metrics(runs):
    labels = next(iter(runs.values()))[1]
    print(f"\n=== MS-CXR-T clean pairs (n={len(labels)}; "
          + ", ".join(f"{c} {int((labels == CLS_ORDER.index(c)).sum())}" for c in VALID) + ")")
    print(f"  {'model':<16} {'acc':>6} {'macroAcc':>8} {'macroF1':>8}"
          + "".join(f"{c[:4] + ' F1':>9}" for c in VALID)
          + "".join(f"{c[:4] + ' n':>7}" for c in VALID)
          + f"{'AUC impr':>9}{'AUC wors':>9}")
    out = {}
    for name, (scores, labs, _) in runs.items():
        m = metrics(scores, labs)
        out[name] = m
        print(f"  {name:<16} {m['acc']:>6.3f} {m['macro_acc']:>8.3f} {m['macro_f1']:>8.3f}"
              + "".join(f"{m['f1'][c]:>9.3f}" for c in VALID)
              + "".join(f"{m['calls'][c]:>7d}" for c in VALID)
              + f"{m['auc']['improving']:>9.3f}{m['auc']['worsening']:>9.3f}")
    findings = next(iter(runs.values()))[2]
    print("\n  accuracy per finding")
    print(f"  {'finding':<18} {'n':>4}" + "".join(f"{n[:12]:>13}" for n in runs))
    for f in sorted(set(findings)):
        mask = findings == f
        print(f"  {f:<18} {int(mask.sum()):>4d}" + "".join(
            f"{(out[n]['pred'][mask] == labels[mask]).mean():>13.3f}" for n in runs))
    return out


def compare(args) -> int:
    from vljepa.compare_shift_rules import ci

    paths = args.npz or sorted(glob.glob(os.path.join(args.out_dir, "*.npz")))
    if not paths:
        raise SystemExit(f"[mscxrt] no score files in {args.out_dir}")
    loaded = {}
    for path in paths:
        d = np.load(path, allow_pickle=True)
        loaded[str(d["name"])] = d
    if args.ref not in loaded:
        raise SystemExit(f"[mscxrt] no score file named {args.ref!r}; have {sorted(loaded)}")
    common = set.intersection(*(set(d["keys"].tolist()) for d in loaded.values()))
    if any(len(d["keys"]) != len(common) for d in loaded.values()):
        print(f"[mscxrt] score files cover different pairs; using the {len(common)} they share")
    ref_keys = [k for k in loaded[args.ref]["keys"].tolist() if k in common]
    runs = {}
    for name, d in loaded.items():
        pos = {k: i for i, k in enumerate(d["keys"].tolist())}
        idx = np.array([pos[k] for k in ref_keys])
        runs[name] = (d["scores"][idx], d["labels"][idx], d["findings"][idx])
    m = print_metrics(runs)

    labels = runs[args.ref][1]
    rng = np.random.default_rng(args.seed)
    boots = rng.integers(0, len(labels), size=(args.boot, len(labels)))
    keys = ["acc", "macro_acc", "macro_f1"]

    def boot_stats(scores):
        rows = []
        for b in boots:
            r = metrics(scores[b], labels[b])
            rows.append([r[k] for k in keys] + [r["f1"][c] for c in VALID]
                        + [r["auc"]["improving"], r["auc"]["worsening"]])
        return np.array(rows)

    print(f"\n=== Paired bootstrap ({args.boot} resamples of the same pairs), model minus {args.ref}")
    print("  95% range = where the true difference most likely lies; if it spans 0, it is a tie.")
    print("  P = share of resamples where the model is ahead.")
    ref_boot = boot_stats(runs[args.ref][0])
    cols = keys + [f"f1_{c}" for c in VALID] + ["auc_improving", "auc_worsening"]
    for name in runs:
        if name == args.ref:
            continue
        diff = boot_stats(runs[name][0]) - ref_boot
        point = ([m[name][k] - m[args.ref][k] for k in keys]
                 + [m[name]["f1"][c] - m[args.ref]["f1"][c] for c in VALID]
                 + [m[name]["auc"][c] - m[args.ref]["auc"][c] for c in ("improving", "worsening")])
        print(f"\n  {name} minus {args.ref}")
        for j, col in enumerate(cols):
            lo, hi, pr = ci(diff[:, j])
            print(f"  {col:<14} {point[j]:>+8.4f}   [{lo:>+7.4f}, {hi:>+7.4f}]   P {pr:.2f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("prepare")
    a.add_argument("--csv", default=DEFAULT_CSV)
    a.add_argument("--silver", help="silver_findings.parquet (default: CheXTemporal on this machine)")
    a.add_argument("--mimic-root", default=DEFAULT_MIMIC)
    a.add_argument("--alt-root", nargs="*",
                   default=[os.path.join(os.environ.get("JEPA_IMAGE_ROOTS_DIR", "/nvme/data/temporal/all_data"), "mimic")],
                   help="other folders to look in when an image is missing under --mimic-root")
    a.add_argument("--allow-missing", action="store_true")
    a.add_argument("--out", default=DEFAULT_CLEAN)

    a = sub.add_parser("score")
    a.add_argument("--model", required=True, choices=["vljepa", "supervised", "biovilt"])
    a.add_argument("--ckpt")
    a.add_argument("--name", help="score file name (default: the model type)")
    a.add_argument("--clean", default=DEFAULT_CLEAN)
    a.add_argument("--out-dir", default=DEFAULT_SCORES)
    a.add_argument("--limit", type=int)

    a = sub.add_parser("compare")
    a.add_argument("--npz", nargs="*", help="score files (default: all in --out-dir)")
    a.add_argument("--out-dir", default=DEFAULT_SCORES)
    a.add_argument("--ref", default="supervised")
    a.add_argument("--boot", type=int, default=2000)
    a.add_argument("--seed", type=int, default=0)

    args = p.parse_args()
    return {"prepare": prepare, "score": score, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
