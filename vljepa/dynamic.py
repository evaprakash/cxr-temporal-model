"""Dynamic-sentence VL-JEPA (option 1).

One example per silver study pair:

    query  : "Describe what has changed between these two chest X-rays."
    target : one of the current report's dynamic sentences (a new one each
             epoch), or all of them joined (``target_mode="report"``)
    Y      : original BioViL-T text encoder, frozen, no LoRA
    loss   : batch InfoNCE. Row i's positive is its own sentence; the other
             rows' sentences (from every GPU) are negatives, except those
             with the same text as row i's sentence.

Benchmarks have no reports, so they are scored zero-shot from the same Ŝ:

    perfinding  per finding, best cosine over the 37-phrase bank
                "{finding} is {phrase}", the zero-shot baselines' protocol
    combined    every class combination over the pair's findings,
                "Edema is worsening. Pleural effusion is stable.", then each
                finding's class score is a soft maximum (temperature τ) over
                the combinations giving it that class. Pairs with more than
                ``max_k`` findings use "{Finding} is {class}." per finding.

Silver val retrieval (does Ŝ find its own sentence?) is a training check.
"""

from __future__ import annotations

import itertools
import random
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import _root  # noqa: F401

from progression_phrases import CLS_ORDER, PROGRESSION_PHRASES

from .prompts import target_text

DYNAMIC_QUERY = "Describe what has changed between these two chest X-rays."
PHRASE_TEMPLATE = "{} is {}"
TARGET_MODES = ("sentence", "report")
SCORE_MODES = ("perfinding", "combined")
PAIR_COLS = ["dataset", "patient_id", "study_id_curr", "study_id_prev"]


def normalize_sentence(text: str) -> str:
    """Key for "same sentence": case, spacing and final punctuation ignored."""
    s = re.sub(r"\s+", " ", str(text).strip().lower())
    return s.rstrip(" .;:")


def same_text_mask(row_texts: Sequence[str], all_texts: Sequence[str]) -> torch.Tensor:
    """(B, N) bool, True where all_texts[n] matches row_texts[b]."""
    rows = [normalize_sentence(t) for t in row_texts]
    cols = [normalize_sentence(t) for t in all_texts]
    return torch.tensor([[r == c for c in cols] for r in rows], dtype=torch.bool)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
def dynamic_sentence_lists(sentences_path: str, pair_df) -> List[List[str]]:
    """Each pair's dynamic sentences from its current report, in report
    order, repeats removed."""
    import pandas as pd

    from dataset_combined_jepa import _nonempty, _normalize_ids

    cols = ["dataset", "patient_id", "study_id", "label", "sentence"]
    sent = _normalize_ids(pd.read_parquet(sentences_path, columns=cols))
    sent = sent[(sent["label"] == "dynamic") & sent["sentence"].apply(_nonempty)]
    keys = (
        pair_df[["dataset", "patient_id", "study_id_curr"]]
        .rename(columns={"study_id_curr": "study_id"})
        .drop_duplicates()
    )
    sent = sent.merge(keys, on=["dataset", "patient_id", "study_id"], how="inner")
    lookup: Dict[Tuple[str, str, str], List[str]] = {}
    for key, group in sent.groupby(["dataset", "patient_id", "study_id"], sort=False)["sentence"]:
        seen, out = set(), []
        for s in group:
            s = str(s).strip()
            k = normalize_sentence(s)
            if k and k not in seen:
                seen.add(k)
                out.append(s)
        lookup[tuple(str(x) for x in key)] = out
    return [
        lookup.get((str(r.dataset), str(r.patient_id), str(r.study_id_curr)), [])
        for r in pair_df.itertuples()
    ]


class VLJEPADynamicDataset(Dataset):
    """One silver study pair per example; target is a dynamic sentence.

    Uses the same pair split as the 5-way run (``splits_jepa.csv``)."""

    def __init__(
        self,
        image_roots: Dict[str, str],
        split: Optional[str] = None,
        train: bool = True,
        val_fraction: float = 0.1,
        split_seed: int = 42,
        target_mode: str = "sentence",
        splits_file: Optional[str] = None,
        findings_path: Optional[str] = None,
        studies_path: Optional[str] = None,
        sentences_path: Optional[str] = None,
    ):
        from dataset_combined_jepa import DEFAULT_SENTENCES, JEPACombinedDataset

        if target_mode not in TARGET_MODES:
            raise ValueError(f"target_mode must be one of {TARGET_MODES}, got {target_mode!r}")
        kwargs = dict(
            image_roots=image_roots,
            split=split,
            train=train,
            val_fraction=val_fraction,
            split_seed=split_seed,
            condition_mode="dynamic",
            load_anatomy_masks=False,
            load_finding_masks=False,
        )
        for name, value in (("splits_file", splits_file), ("findings_path", findings_path),
                            ("studies_path", studies_path), ("sentences_path", sentences_path)):
            if value is not None:
                kwargs[name] = value
        self.pairs = JEPACombinedDataset(**kwargs)
        self.train = train
        self.target_mode = target_mode
        self.sentences = dynamic_sentence_lists(sentences_path or DEFAULT_SENTENCES, self.pairs.df)
        n_sent = [len(s) for s in self.sentences]
        empty = sum(n == 0 for n in n_sent)
        if empty:
            raise RuntimeError(f"[vljepa dynamic] {empty} pairs have no dynamic sentence")
        counts = Counter(normalize_sentence(s) for ss in self.sentences for s in ss)
        top = ", ".join(f"{s!r} x{n}" for s, n in counts.most_common(3))
        print(
            f"[vljepa dynamic] split={split or 'all'}: {len(self.sentences)} pairs, "
            f"{sum(n_sent)} dynamic sentences ({np.mean(n_sent):.2f} per pair, "
            f"{len(counts)} distinct), target={target_mode}; most common: {top}"
        )

    def __len__(self) -> int:
        return len(self.sentences)

    def target_for(self, idx: int) -> str:
        sents = self.sentences[idx]
        if self.target_mode == "report":
            return " ".join(sents)
        if self.train:
            return random.choice(sents)
        return sents[random.Random(idx).randrange(len(sents))]

    def __getitem__(self, idx: int) -> dict:
        sample = self.pairs._getitem_one(idx)
        return {
            "prior_image": sample["prior_image"],
            "current_image": sample["current_image"],
            "query_text": DYNAMIC_QUERY,
            "target_text": self.target_for(idx),
            "dataset": sample["dataset"],
        }


def dynamic_collate_fn(batch: List[dict]) -> dict:
    return {
        "prior_image": torch.stack([b["prior_image"] for b in batch]),
        "current_image": torch.stack([b["current_image"] for b in batch]),
        "query_text": [b["query_text"] for b in batch],
        "target_text": [b["target_text"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }


# ----------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------
def load_dynamic_from_ckpt(ckpt_path: str, device: torch.device):
    from .model import VLJEPA, resolve_llama_local

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("vljepa_cfg", {})
    if cfg.get("objective") != "dynamic":
        raise SystemExit(f"[vljepa dynamic] {ckpt_path} is not a dynamic-sentence checkpoint")
    model = VLJEPA(
        image_mode="biovilt_no_pretrained",
        smoke=False,
        n_llama_layers=int(cfg.get("n_llama_layers", 8)),
        llama_name=cfg.get("llama_name", "meta-llama/Llama-3.2-1B"),
        llama_local=resolve_llama_local() or cfg.get("llama_local"),
        freeze_image_encoder=True,
        freeze_text_encoder=True,
        gradient_checkpointing=False,
        text_lora=False,
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[vljepa dynamic] missing={missing[:8]} unexpected={unexpected[:8]}")
    print(f"[vljepa dynamic] loaded {ckpt_path} (epoch {ckpt.get('epoch')})")
    return model.to(device).eval()


# ----------------------------------------------------------------------
# Zero-shot class scores from Ŝ
# ----------------------------------------------------------------------
class DynamicScorer:
    """Five class scores per (pair, finding) from one Ŝ per pair."""

    def __init__(self, model, mode: str = "perfinding", max_k: int = 4,
                 combo_temperature: float = 0.07, chunk: int = 256):
        if mode not in SCORE_MODES:
            raise ValueError(f"mode must be one of {SCORE_MODES}, got {mode!r}")
        self.model = model
        self.mode = mode
        self.max_k = int(max_k)
        self.tau = float(combo_temperature)
        self.chunk = int(chunk)
        self.cache: Dict[object, object] = {}
        self.n_fallback = 0

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), self.chunk):
            g, _, _ = self.model.encode_targets(texts[i:i + self.chunk])
            out.append(F.normalize(g.float(), dim=-1))
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def predict(self, prior: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        """(3, H, W) prior and current -> unit-norm Ŝ of shape (D,)."""
        img_g, img_p = self.model.encode_images(prior.unsqueeze(0), current.unsqueeze(0))
        pred = self.model.predict(img_g, img_p, [DYNAMIC_QUERY])
        return F.normalize(pred.float(), dim=-1).squeeze(0)

    def phrase_bank(self, finding: str):
        key = ("bank", finding)
        if key not in self.cache:
            texts, cls_idx = [], []
            for c, cls in enumerate(CLS_ORDER):
                for phrase in PROGRESSION_PHRASES[cls]:
                    texts.append(PHRASE_TEMPLATE.format(finding, phrase))
                    cls_idx.append(c)
            self.cache[key] = (self.encode(texts), torch.tensor(cls_idx))
        return self.cache[key]

    def class_sentences(self, finding: str) -> torch.Tensor:
        key = ("cls", finding)
        if key not in self.cache:
            self.cache[key] = self.encode([target_text(finding, c) for c in CLS_ORDER])
        return self.cache[key]

    def combos(self, findings: Tuple[str, ...]):
        """All class combinations over sorted findings: (texts, (M, K) class ids, embeddings)."""
        key = ("combo", findings)
        if key not in self.cache:
            ids = list(itertools.product(range(len(CLS_ORDER)), repeat=len(findings)))
            texts = [" ".join(target_text(f, CLS_ORDER[c]) for f, c in zip(findings, combo))
                     for combo in ids]
            self.cache[key] = (texts, torch.tensor(ids), self.encode(texts))
        return self.cache[key]

    def score_perfinding(self, pred: torch.Tensor, finding: str) -> np.ndarray:
        emb, cls_idx = self.phrase_bank(finding)
        sims = (emb.to(pred.device) @ pred).cpu()
        return np.array([float(sims[cls_idx == c].max()) for c in range(len(CLS_ORDER))])

    def score_combined(self, pred: torch.Tensor, findings: Sequence[str]) -> Dict[str, np.ndarray]:
        findings = tuple(sorted(set(findings)))
        if len(findings) > self.max_k:
            self.n_fallback += 1
            return {f: (self.class_sentences(f).to(pred.device) @ pred).cpu().numpy()
                    for f in findings}
        _, ids, emb = self.combos(findings)
        sims = (emb.to(pred.device) @ pred).cpu() / self.tau
        out = {}
        for j, f in enumerate(findings):
            out[f] = np.array([
                float(self.tau * torch.logsumexp(sims[ids[:, j] == c], dim=0))
                for c in range(len(CLS_ORDER))
            ])
        return out

    def score_pair(self, prior, current, findings: Sequence[str]) -> Dict[str, np.ndarray]:
        pred = self.predict(prior, current)
        if self.mode == "combined":
            return self.score_combined(pred, findings)
        return {f: self.score_perfinding(pred, f) for f in findings}


def score_gold_groups(scorer: DynamicScorer, groups, roots, device, desc: str = "gold"):
    """Score gold (pair, finding) groups in ``load_gold_groups`` order, skipping
    groups with a missing image exactly like ``check_class_signal.score_gold``,
    so the arrays line up with the other gold score files."""
    import sys

    from tqdm import tqdm

    from progression_classify import load_image_tensor

    siblings: Dict[tuple, List[str]] = {}
    for _, row in groups.iterrows():
        siblings.setdefault(tuple(str(row[c]) for c in PAIR_COLS), []).append(str(row["finding"]))
    per_pair: Dict[tuple, Optional[Dict[str, np.ndarray]]] = {}
    scores, gts, findings, keys, datasets = [], [], [], [], []
    skipped = 0
    for i in tqdm(range(len(groups)), desc=desc, file=sys.stdout):
        row = groups.iloc[i]
        pk = tuple(str(row[c]) for c in PAIR_COLS)
        if pk not in per_pair:
            try:
                prior = load_image_tensor(row["dataset"], row["parent_image_prev"], roots)
                current = load_image_tensor(row["dataset"], row["parent_image_curr"], roots)
                per_pair[pk] = scorer.score_pair(prior.to(device), current.to(device), siblings[pk])
            except (FileNotFoundError, OSError):
                per_pair[pk] = None
        if per_pair[pk] is None:
            skipped += 1
            continue
        finding = str(row["finding"]).strip().lower()
        scores.append(per_pair[pk][str(row["finding"])])
        gts.append("|".join(sorted(str(x).strip().lower() for x in row["gt_labels"])))
        findings.append(finding)
        datasets.append(str(row["dataset"]))
        keys.append("|".join(str(row[c]) for c in
                             ("dataset", "patient_id", "study_id_prev", "study_id_curr", "finding")))
    if skipped:
        print(f"[vljepa dynamic] gold skipped missing images: {skipped}")
    if scorer.n_fallback:
        print(f"[vljepa dynamic] combined: {scorer.n_fallback} pairs had more than "
              f"{scorer.max_k} findings and were scored per finding")
    obj = lambda x: np.array(x, dtype=object)  # noqa: E731
    return (np.stack(scores).astype(np.float64), obj(gts), obj(findings), obj(keys), obj(datasets))


# ----------------------------------------------------------------------
# Silver val retrieval
# ----------------------------------------------------------------------
def retrieval_recall(pred: torch.Tensor, target: torch.Tensor, texts: Sequence[str],
                     ks: Sequence[int] = (1, 5, 10), chunk: int = 1024) -> Dict[str, float]:
    """Rank every distinct val sentence for each Ŝ. A row is a hit at k when
    fewer than k other sentences score above its own (same-text duplicates
    count once)."""
    keys = [normalize_sentence(t) for t in texts]
    first: Dict[str, int] = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    gallery = F.normalize(target[list(first.values())].float(), dim=-1)
    col = {k: j for j, k in enumerate(first)}
    own = torch.tensor([col[k] for k in keys], device=gallery.device)
    pred = F.normalize(pred.float(), dim=-1).to(gallery.device)
    ranks = []
    for i in range(0, len(keys), chunk):
        sims = pred[i:i + chunk] @ gallery.T
        mine = sims.gather(1, own[i:i + chunk, None])
        ranks.append((sims > mine).sum(dim=1))
    ranks = torch.cat(ranks)
    out = {f"recall@{k}": float((ranks < k).float().mean()) for k in ks}
    out["gallery"] = len(first)
    out["median_rank"] = float(ranks.float().median()) + 1
    return out
