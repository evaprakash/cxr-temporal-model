#!/usr/bin/env python3
"""Did gold recall move because the answer sentences moved, or because Ŝ did?

Uses the first VL-JEPA run (Y unfrozen, no stop-grad).

Part A — answer drift, no images, no Llama
    Encode ``{Finding} is {class}.`` with pretrained BioViL-T and with the
    text encoder inside each checkpoint. Cosine near 1 means that sentence
    did not move. Also prints the pretrained 5×5, including resolved–stable.

Part B — predictor vs fixed answers
    Run epoch-1 and epoch-5 Ŝ on stratified single-label gold. Score the same
    Ŝ against that epoch's sentences and against the pretrained sentences.

Part C — zero-shot, no predictor
    Frozen BioViL-T image encoder vs the five original sentences. If resolved
    films already lose this argmax, the pretrained sentence is a weak target.

    sbatch vljepa/probe_answer_drift.sh
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "vljepa"

from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from . import _root  # noqa: F401
from .cluster_paths import gold_image_dirs, image_roots, llama_dir
from .model import resolve_llama_local
from .prompts import class_target_texts, query_text

from progression_classify import discover_gold_image_roots, load_gold_pairs, load_image_tensor
from progression_phrases import CLS_ORDER
from dataset_combined_jepa import DEFAULT_FINDINGS
from gold_progression_setmatch import group_gold_by_pair_finding
from tempcxr.modules.text_encoder import BioViLTTextEncoder

FINDINGS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged cardiomediastinum",
    "lung lesion",
    "lung opacity",
    "pleural effusion",
    "pleural other",
    "pneumonia",
    "pneumothorax",
]

DEFAULT_CKPT_DIR = "/scratch/m000081/eprakash/ckpts/checkpoints_vljepa"


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _epoch_ckpts(ckpt_dir: str) -> List[str]:
    paths = sorted(
        glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")),
        key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
    )
    if not paths:
        raise FileNotFoundError(f"no epoch_*.pt under {ckpt_dir}")
    return paths


def _load_text_sd(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_sd = ckpt["model"]
    prefix = "text_encoder."
    sd = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
    if not sd:
        raise RuntimeError(f"no text_encoder weights in {ckpt_path}")
    del ckpt, model_sd
    return sd


def _fresh_text() -> BioViLTTextEncoder:
    enc = BioViLTTextEncoder(mode="biovilt")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return enc


@torch.no_grad()
def _embed_bank(enc: BioViLTTextEncoder, findings: List[str]) -> Dict[str, torch.Tensor]:
    """finding -> (C, D) unit-norm global embeddings in CLS_ORDER."""
    bank = {}
    for finding in findings:
        texts = class_target_texts(finding)
        g, _, _ = enc.forward_contrastive(texts)
        bank[finding] = F.normalize(g.float(), dim=-1).cpu()
    return bank


def _offdiag(vecs: torch.Tensor) -> float:
    sim = vecs @ vecs.T
    n = sim.shape[0]
    return float((sim.sum() - sim.trace()) / (n * (n - 1)))


def part_a(ckpt_dir: str) -> None:
    _print("=" * 72)
    _print("PART A  answer sentences: pretrained BioViL-T vs each checkpoint")
    _print("cosine(init, epoch) per class, mean over findings. 1.000 = did not move.")
    _print("=" * 72)
    init = _fresh_text()
    init_bank = _embed_bank(init, FINDINGS)
    init_off = sum(_offdiag(v) for v in init_bank.values()) / len(init_bank)
    _print(f"init mean off-diag among the 5 sentences: {init_off:.4f}")
    _print("")
    header = f"{'ckpt':12}" + "".join(f"{c[:10]:>12}" for c in CLS_ORDER) + f"{'offdiag':>10}"
    _print(header)
    for path in _epoch_ckpts(ckpt_dir):
        enc = _fresh_text()
        missing, unexpected = enc.load_state_dict(_load_text_sd(path), strict=False)
        if missing:
            _print(f"  warn {os.path.basename(path)} missing {len(missing)} text keys")
        if unexpected:
            _print(f"  warn {os.path.basename(path)} unexpected {len(unexpected)} text keys")
        bank = _embed_bank(enc, FINDINGS)
        cos_by_cls = []
        for c_i, _cls in enumerate(CLS_ORDER):
            vals = [
                float((init_bank[f][c_i] * bank[f][c_i]).sum())
                for f in FINDINGS
            ]
            cos_by_cls.append(sum(vals) / len(vals))
        off = sum(_offdiag(v) for v in bank.values()) / len(bank)
        name = os.path.basename(path).replace(".pt", "")
        row = f"{name:12}" + "".join(f"{v:12.4f}" for v in cos_by_cls) + f"{off:10.4f}"
        _print(row)
        del enc, bank
    _print("")
    _print_pair_matrix(init_bank)
    _print("")
    _print("Read part A: a class column falling well below 1 means that")
    _print("sentence's embedding moved. offdiag is how similar the five")
    _print("sentences are to each other (the first run stayed near 0.5).")
    _print("The 5×5 is the pretrained geometry. resolved–stable near 0.9")
    _print("means argmax cannot separate those two sentences.")
    del init


def _print_pair_matrix(bank: Dict[str, torch.Tensor]) -> None:
    acc = None
    for vecs in bank.values():
        sim = vecs @ vecs.T
        acc = sim if acc is None else acc + sim
    acc = acc / max(len(bank), 1)
    _print("PART A2  pretrained 5×5 cosine, mean over findings")
    header = f"{'':12}" + "".join(f"{c[:10]:>12}" for c in CLS_ORDER)
    _print(header)
    for i, cls in enumerate(CLS_ORDER):
        _print(f"{cls[:12]:12}" + "".join(f"{float(acc[i, j]):12.4f}" for j in range(len(CLS_ORDER))))
    i_res = CLS_ORDER.index("resolved")
    i_sta = CLS_ORDER.index("stable")
    _print(
        f"resolved–stable cosine = {float(acc[i_res, i_sta]):.4f}  "
        f"(1 = same sentence)"
    )


def _gold_roots() -> dict:
    from progression_classify import DEFAULT_GOLD_PARQUET

    pq_dir = os.path.dirname(os.path.abspath(DEFAULT_GOLD_PARQUET))
    return {
        **image_roots(),
        **discover_gold_image_roots(pq_dir),
        **gold_image_dirs(),
    }


def _stratified_single(groups, limit: int):
    single = groups[~groups["is_multi"]].copy()
    buckets = {cls: [] for cls in CLS_ORDER}
    for idx, row in single.iterrows():
        gt = list(row["gt_labels"])
        if len(gt) == 1 and gt[0] in buckets:
            buckets[gt[0]].append(idx)
    picked = []
    while len(picked) < limit:
        grew = False
        for cls in CLS_ORDER:
            if buckets[cls] and len(picked) < limit:
                picked.append(buckets[cls].pop(0))
                grew = True
        if not grew:
            break
    return single.loc[picked]


def _llama_local(cfg: dict) -> Optional[str]:
    saved = cfg.get("llama_local")
    if saved and os.path.isfile(os.path.join(saved, "config.json")):
        return saved
    resolved = resolve_llama_local()
    if resolved:
        return resolved
    fallback = llama_dir()
    if os.path.isfile(os.path.join(fallback, "config.json")):
        return fallback
    return saved


@torch.no_grad()
def _score_epoch(ckpt_path: str, rows, roots, init_bank: Dict[str, torch.Tensor], device: torch.device) -> None:
    from .model import VLJEPA

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("vljepa_cfg", {})
    local = _llama_local(cfg)
    if local:
        os.environ["VLJEPA_LLAMA_LOCAL"] = local
    model = VLJEPA(
        image_mode="biovilt_no_pretrained",
        smoke=False,
        n_llama_layers=int(cfg.get("n_llama_layers", 8)),
        llama_name=cfg.get("llama_name", "meta-llama/Llama-3.2-1B"),
        llama_local=local,
        freeze_image_encoder=True,
        freeze_text_encoder=True,
        gradient_checkpointing=False,
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    del ckpt
    if missing:
        _print(f"  warn missing keys: {missing[:6]}")
    if unexpected:
        _print(f"  warn unexpected keys: {unexpected[:6]}")
    model = model.to(device).eval()

    # Cache this epoch's own phrase bank (same findings the gold rows use).
    findings = sorted({str(r["finding"]).strip().lower() for _, r in rows.iterrows()})
    epoch_bank = {}
    for finding in findings:
        texts = class_target_texts(finding)
        g, _, _ = model.encode_targets(texts)
        epoch_bank[finding] = F.normalize(g.float(), dim=-1).cpu()

    names = ("epoch_Y", "init_Y")
    correct = {n: 0 for n in names}
    n_by = {n: defaultdict(int) for n in names}
    hit_by = {n: defaultdict(int) for n in names}
    cos_sum = {n: defaultdict(float) for n in names}
    cos_n = {n: defaultdict(int) for n in names}
    n_ok = 0
    n_skip = 0
    for _, row in rows.iterrows():
        finding = str(row["finding"]).strip().lower()
        gt = list(row["gt_labels"])
        if len(gt) != 1 or finding not in epoch_bank:
            continue
        try:
            prior = load_image_tensor(
                row["dataset"], row["parent_image_prev"], roots,
            )
            current = load_image_tensor(
                row["dataset"], row["parent_image_curr"], roots,
            )
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        img_g, img_p = model.encode_images(
            prior.unsqueeze(0).to(device), current.unsqueeze(0).to(device),
        )
        pred = model.predict(img_g, img_p, [query_text(finding)])
        pred_n = F.normalize(pred.float(), dim=-1).cpu().squeeze(0)
        if finding not in epoch_bank or finding not in init_bank:
            continue
        banks = {"epoch_Y": epoch_bank, "init_Y": init_bank}
        n_ok += 1
        gold_i = CLS_ORDER.index(gt[0])
        for name, bank in banks.items():
            tgt = bank[finding]
            cos = tgt @ pred_n
            pred_i = int(cos.argmax())
            n_by[name][gt[0]] += 1
            if pred_i == gold_i:
                correct[name] += 1
                hit_by[name][gt[0]] += 1
            for c_i, cls in enumerate(CLS_ORDER):
                cos_sum[name][cls] += float(cos[c_i])
                cos_n[name][cls] += 1
        if n_ok % 25 == 0:
            _print(f"  scored {n_ok}  skipped_images={n_skip}")

    _print(f"  usable single-label groups: {n_ok}  missing images: {n_skip}")
    _print(f"  {'bank':10} {'acc':>8}" + "".join(f"{c[:8]:>10}" for c in CLS_ORDER))
    for name in names:
        acc = correct[name] / max(n_ok, 1)
        recalls = []
        for cls in CLS_ORDER:
            recalls.append(hit_by[name][cls] / max(n_by[name][cls], 1))
        _print(
            f"  {name:10} {acc:8.3f}"
            + "".join(f"{r:10.3f}" for r in recalls)
        )
    _print("  mean cos(Ŝ, sentence) over the same groups")
    for name in names:
        vals = [
            cos_sum[name][cls] / max(cos_n[name][cls], 1) for cls in CLS_ORDER
        ]
        _print(f"  {name:10} {'':8}" + "".join(f"{v:10.3f}" for v in vals))
    del model


def part_b(ckpt_dir: str, epochs: List[int], limit: int, device: torch.device) -> None:
    _print("")
    _print("=" * 72)
    _print("PART B  same Ŝ scored against epoch sentences and pretrained sentences")
    _print(f"stratified single-label gold, up to {limit} groups, device={device}")
    _print("=" * 72)
    from progression_classify import DEFAULT_GOLD_PARQUET

    gold = load_gold_pairs(DEFAULT_GOLD_PARQUET, DEFAULT_FINDINGS)
    groups = group_gold_by_pair_finding(gold)
    rows = _stratified_single(groups, limit)
    _print(f"sample n={len(rows)}")
    counts = defaultdict(int)
    for _, row in rows.iterrows():
        counts[list(row["gt_labels"])[0]] += 1
    _print("  " + " ".join(f"{c[:3]}={counts[c]}" for c in CLS_ORDER))
    roots = _gold_roots()
    init = _fresh_text()
    # Cover every finding that shows up in the sample, not only FINDINGS.
    findings = sorted(set(FINDINGS) | {str(r["finding"]).strip().lower() for _, r in rows.iterrows()})
    init_bank = _embed_bank(init, findings)
    del init
    by_epoch = {
        int(os.path.basename(p).split("_")[1].split(".")[0]): p
        for p in _epoch_ckpts(ckpt_dir)
    }
    for ep in epochs:
        path = by_epoch.get(ep)
        if path is None:
            _print(f"epoch {ep}: checkpoint missing, skip")
            continue
        _print("")
        _print(f"--- epoch {ep}  {path} ---")
        _score_epoch(path, rows, roots, init_bank, device)
    _print("")
    _print("Read part B: recall columns are per true class.")
    _print("epoch_Y = sentences from that checkpoint. init_Y = pretrained BioViL-T.")
    _print("Resolved recall only on epoch_Y means the answer sentence moved.")
    _print("Resolved recall on init_Y as well means Ŝ learned the original sentence.")


@torch.no_grad()
def part_c(limit: int, device: torch.device) -> None:
    """Frozen image encoder vs the five original sentences. No Llama."""
    from tempcxr.modules.image_encoder_jepa import BioViLTImageEncoderJEPA
    from progression_classify import DEFAULT_GOLD_PARQUET

    _print("")
    _print("=" * 72)
    _print("PART C  zero-shot: frozen BioViL-T image vs original sentences")
    _print("No predictor. If resolved films already lose argmax, the sentence is a weak target.")
    _print(f"device={device}")
    _print("=" * 72)
    gold = load_gold_pairs(DEFAULT_GOLD_PARQUET, DEFAULT_FINDINGS)
    groups = group_gold_by_pair_finding(gold)
    rows = _stratified_single(groups, limit)
    roots = _gold_roots()
    enc = _fresh_text().to(device)
    findings = sorted({str(r["finding"]).strip().lower() for _, r in rows.iterrows()})
    bank = _embed_bank(enc, findings)
    del enc
    image = BioViLTImageEncoderJEPA(mode="biovilt").to(device).eval()
    for p in image.parameters():
        p.requires_grad = False

    correct = 0
    n_by = defaultdict(int)
    hit_by = defaultdict(int)
    cos_on_true = defaultdict(lambda: defaultdict(float))
    n_ok = 0
    n_skip = 0
    for _, row in rows.iterrows():
        finding = str(row["finding"]).strip().lower()
        gt = list(row["gt_labels"])
        if len(gt) != 1 or finding not in bank:
            continue
        try:
            prior = load_image_tensor(row["dataset"], row["parent_image_prev"], roots)
            current = load_image_tensor(row["dataset"], row["parent_image_curr"], roots)
        except (FileNotFoundError, OSError):
            n_skip += 1
            continue
        img_g, _patches = image(
            current.unsqueeze(0).to(device), prior.unsqueeze(0).to(device),
        )
        img_n = F.normalize(img_g.float(), dim=-1).cpu().squeeze(0)
        cos = bank[finding] @ img_n
        pred_i = int(cos.argmax())
        gold_i = CLS_ORDER.index(gt[0])
        n_ok += 1
        n_by[gt[0]] += 1
        if pred_i == gold_i:
            correct += 1
            hit_by[gt[0]] += 1
        for c_i, cls in enumerate(CLS_ORDER):
            cos_on_true[gt[0]][cls] += float(cos[c_i])
        if n_ok % 50 == 0:
            _print(f"  scored {n_ok}  skipped_images={n_skip}")

    _print(f"  usable single-label groups: {n_ok}  missing images: {n_skip}")
    acc = correct / max(n_ok, 1)
    _print(f"  overall acc {acc:.3f}")
    _print(f"  {'':12}" + "".join(f"{c[:8]:>10}" for c in CLS_ORDER))
    recalls = [hit_by[cls] / max(n_by[cls], 1) for cls in CLS_ORDER]
    _print("  recall      " + "".join(f"{r:10.3f}" for r in recalls))
    _print("  n           " + "".join(f"{n_by[cls]:10d}" for cls in CLS_ORDER))
    _print("  mean cos(image, sentence) on films of each true class (rows=true class)")
    for gt_cls in CLS_ORDER:
        vals = [
            cos_on_true[gt_cls][cls] / max(n_by[gt_cls], 1) for cls in CLS_ORDER
        ]
        _print(f"  {gt_cls[:12]:12}" + "".join(f"{v:10.3f}" for v in vals))
    _print("")
    _print("Read part C: the resolved row is resolved films only.")
    _print("If that row's resolved column is not the largest, the original")
    _print("sentence already loses to another class before any training.")
    del image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--skip-pred", action="store_true")
    parser.add_argument("--skip-zeroshot", action="store_true")
    parser.add_argument("--pred-epochs", default="1,5")
    parser.add_argument("--pred-limit", type=int, default=0,
                        help="single-label groups. 0 = all")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    limit = args.pred_limit if args.pred_limit > 0 else 10 ** 9
    _print(f"device={device}  pred_limit={'all' if args.pred_limit <= 0 else args.pred_limit}")
    if not os.path.isdir(args.ckpt_dir):
        _print(f"checkpoint dir not found: {args.ckpt_dir}")
        return 1
    part_a(args.ckpt_dir)
    if not args.skip_zeroshot:
        part_c(limit, device)
    if not args.skip_pred:
        epochs = [int(x) for x in args.pred_epochs.split(",") if x.strip()]
        part_b(args.ckpt_dir, epochs, limit, device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
