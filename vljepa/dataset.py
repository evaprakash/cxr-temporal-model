"""Finding-level CheXTemporal silver dataset for option-2 VL-JEPA.

Each ``(pair, finding)`` is one example. A pair with three annotated
findings becomes three independent samples that share the same prior
and current films but have their own query / target class.

Built on top of ``JEPACombinedDataset`` so train/val pair splits stay
identical to the existing JEPA run (``splits_jepa.csv``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

from . import _root  # noqa: F401

from dataset_combined_jepa import CLS_TO_IDX, JEPACombinedDataset
from progression_phrases import CLS_ORDER

from .prompts import class_target_texts, query_text


class VLJEPAFindingDataset(Dataset):
    """Explode JEPA study-pairs into one row per finding."""

    def __init__(
        self,
        image_roots: Dict[str, str],
        split: Optional[str] = None,
        train: bool = True,
        val_fraction: float = 0.1,
        split_seed: int = 42,
        splits_file: Optional[str] = None,
        findings_path: Optional[str] = None,
        studies_path: Optional[str] = None,
        sentences_path: Optional[str] = None,
    ):
        kwargs = dict(
            image_roots=image_roots,
            split=split,
            train=train,
            val_fraction=val_fraction,
            split_seed=split_seed,
            condition_mode="templated",
            load_anatomy_masks=False,
            load_finding_masks=False,
        )
        if splits_file is not None:
            kwargs["splits_file"] = splits_file
        if findings_path is not None:
            kwargs["findings_path"] = findings_path
        if studies_path is not None:
            kwargs["studies_path"] = studies_path
        if sentences_path is not None:
            kwargs["sentences_path"] = sentences_path
        self.pairs = JEPACombinedDataset(**kwargs)
        self.train = train
        self.index: List[tuple] = []
        for pair_i in range(len(self.pairs.df)):
            row = self.pairs.df.iloc[pair_i]
            findings = list(row["finding"])
            classes = list(row["progression_cls"])
            for finding, cls_name in zip(findings, classes):
                finding = str(finding).strip().lower()
                cls_name = str(cls_name).strip().lower()
                if finding not in ("", "nan") and cls_name in CLS_TO_IDX:
                    self.index.append(
                        (pair_i, finding, int(CLS_TO_IDX[cls_name]))
                    )
        print(
            f"[vljepa dataset] split={split or 'all'}: "
            f"{len(self.pairs)} pairs → {len(self.index)} finding-examples"
        )

    def __len__(self) -> int:
        return len(self.index)

    def class_counts(self) -> torch.Tensor:
        counts = torch.zeros(len(CLS_ORDER), dtype=torch.long)
        for _, _, cls_idx in self.index:
            counts[cls_idx] += 1
        return counts

    def __getitem__(self, idx: int) -> dict:
        pair_i, finding, cls_idx = self.index[idx]
        sample = self.pairs._getitem_one(pair_i)
        return {
            "prior_image": sample["prior_image"],
            "current_image": sample["current_image"],
            "finding": finding,
            "cls_idx": int(cls_idx),
            "query_text": query_text(finding),
            "target_texts": class_target_texts(finding),
            "dataset": sample["dataset"],
        }


def vljepa_collate_fn(batch: List[dict]) -> dict:
    return {
        "prior_image": torch.stack([b["prior_image"] for b in batch]),
        "current_image": torch.stack([b["current_image"] for b in batch]),
        "finding": [b["finding"] for b in batch],
        "cls_idx": torch.tensor([b["cls_idx"] for b in batch], dtype=torch.long),
        "query_text": [b["query_text"] for b in batch],
        "target_texts": [b["target_texts"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }


def flatten_target_texts(target_texts: List[List[str]]) -> List[str]:
    """``(B, C)`` nested phrase lists → flat ``B*C`` list, class-major per row."""
    flat: List[str] = []
    for row in target_texts:
        flat.extend(row)
    return flat
