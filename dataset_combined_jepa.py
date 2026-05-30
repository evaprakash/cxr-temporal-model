"""JEPA-style dataset for the silver CheXTemporal corpus.

Differences from ``dataset_combined.BioViLTCombinedDataset``:

  * Reads the silver parquet trio (findings, studies, sentences) directly,
    rather than a precomputed CSV. This is the "full silver dataset
    currently being used and subsampled in jepa.py" per the user's note.
  * Every sample is a PAIR (prior + current image), filtered to rows
    that also have:
      - both impression+findings reports (prior and current),
      - at least one ``label == "dynamic"`` sentence (the change
        description used as the predictor's textual condition).
    There are no single-image samples; the JEPA objective requires both.
  * Augmentation is reused unchanged from ``dataset_combined`` so the
    image preprocessing matches the existing biovilt pipeline. The same
    augmentation parameters are sampled once per pair and applied
    identically to both images, which keeps prior/current geometrically
    consistent.

The class follows the same constructor + ``__getitem__`` pattern as
``BioViLTCombinedDataset`` so it slots into a torch DataLoader the same
way.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset_combined import (
    BASE_TRANSFORM,
    apply_augmentation,
    sample_augmentation,
)


# ============================================================
# DEFAULT SILVER DATASET PATHS (HF)
# ============================================================
DEFAULT_FINDINGS = "hf://datasets/anonaccount107240/CheXTemporal/silver_findings.parquet"
DEFAULT_STUDIES = "hf://datasets/anonaccount107240/CheXTemporal/silver_studies.parquet"
DEFAULT_SENTENCES = "hf://datasets/anonaccount107240/CheXTemporal/silver_sentences.parquet"


# ============================================================
# HELPERS
# ============================================================
def _nonempty(x):
    return pd.notna(x) and str(x).strip() != ""


def _build_report(impression, findings):
    parts = []
    if _nonempty(impression):
        parts.append(str(impression).strip())
    if _nonempty(findings):
        parts.append(str(findings).strip())
    return "\n".join(parts)


def _normalize_ids(df):
    for c in ["patient_id", "study_id", "study_id_curr", "study_id_prev"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df


def _resolve_image_path(dataset: str, rel_path: str, roots: Dict[str, str]) -> Path:
    """Strip dataset-prefix junk from the silver paths and resolve onto roots."""
    rel_path = str(rel_path).strip()

    for prefix in ["mimic/", "chexpert/", "rexgradient/"]:
        if rel_path.startswith(prefix):
            rel_path = rel_path[len(prefix):]

    if dataset == "chexpert" and rel_path.startswith("train/"):
        rel_path = rel_path[len("train/"):]

    if dataset == "rexgradient" and rel_path.startswith("deid_png/"):
        rel_path = rel_path[len("deid_png/"):]

    return Path(roots[dataset]) / rel_path


# ============================================================
# DATASET
# ============================================================
class JEPACombinedDataset(Dataset):
    """JEPA dataset over the silver CheXTemporal corpus.

    Each sample yields:
        prior_image    (3,H,W) tensor (synced augmentation with current)
        current_image  (3,H,W) tensor
        prior_report   (str)  prior impression + findings
        current_report (str)  current impression + findings
        dynamic_report (str)  joined "dynamic" sentences from current study
                              (the change description used to condition
                              the JEPA predictor)
        dataset        (str)  one of "mimic", "chexpert", "rexgradient"

    Parameters
    ----------
    image_roots
        Mapping ``{dataset_name: filesystem_root}``. Same shape as the
        biovilt pipeline's ``IMAGE_ROOTS``.
    findings_path / studies_path / sentences_path
        Paths to the silver parquet files. Defaults to the HF locations
        used by ``jepa.py`` smoke tests.
    split
        Optional. If provided AND the studies parquet has a ``split``
        column, rows are filtered to that split. Otherwise the parameter
        is ignored (with a printed warning) and all rows are used.
    train
        Whether to apply random augmentation. Eval/val should pass False.
    """

    def __init__(
        self,
        image_roots: Dict[str, str],
        findings_path: str = DEFAULT_FINDINGS,
        studies_path: str = DEFAULT_STUDIES,
        sentences_path: str = DEFAULT_SENTENCES,
        split: Optional[str] = None,
        train: bool = True,
    ):
        self.image_roots = {k: str(v) for k, v in image_roots.items()}
        self.train = train
        self.split = split

        # ------------------------------------------------------------
        # Load + filter
        # ------------------------------------------------------------
        findings = _normalize_ids(pd.read_parquet(findings_path))
        studies = _normalize_ids(pd.read_parquet(studies_path))
        sentences = _normalize_ids(pd.read_parquet(sentences_path))

        if split is not None:
            if "split" in studies.columns:
                studies = studies[studies["split"] == split].copy()
            else:
                print(
                    f"[JEPA dataset] WARNING: split={split!r} requested but no "
                    f"'split' column in studies parquet; using all rows."
                )

        # ------------------------------------------------------------
        # Build current/prior report strings (impression + findings)
        # ------------------------------------------------------------
        studies["current_report"] = studies.apply(
            lambda r: _build_report(r["current_impression"], r["current_findings"]),
            axis=1,
        )
        studies["prior_report"] = studies.apply(
            lambda r: _build_report(r["prior_impression"], r["prior_findings"]),
            axis=1,
        )

        # ------------------------------------------------------------
        # Merge findings ↔ studies on the curr↔study_id key
        # ------------------------------------------------------------
        rows = findings.merge(
            studies[
                ["dataset", "patient_id", "study_id", "current_report", "prior_report"]
            ],
            left_on=["dataset", "patient_id", "study_id_curr"],
            right_on=["dataset", "patient_id", "study_id"],
            how="inner",
        )

        # Keep only rows with both image paths and both reports
        rows = rows[
            rows["parent_image_curr"].apply(_nonempty)
            & rows["parent_image_prev"].apply(_nonempty)
            & rows["current_report"].apply(_nonempty)
            & rows["prior_report"].apply(_nonempty)
        ].copy()

        rows = rows.drop_duplicates(
            ["dataset", "patient_id", "study_id_curr", "study_id_prev"]
        )

        # ------------------------------------------------------------
        # Pre-aggregate dynamic sentences into one string per study.
        # This replaces the O(N · |sentences|) per-row loop the smoke
        # test used in `jepa.py::load_smoke_batch`.
        # ------------------------------------------------------------
        dyn = sentences[sentences["label"] == "dynamic"].copy()
        dyn = dyn[dyn["sentence"].apply(_nonempty)]

        dyn_grouped = (
            dyn.groupby(["dataset", "patient_id", "study_id"])["sentence"]
            .apply(lambda s: " ".join(str(x).strip() for x in s if _nonempty(x)))
            .reset_index()
            .rename(columns={"sentence": "dynamic_report", "study_id": "study_id_curr"})
        )

        rows = rows.merge(
            dyn_grouped,
            on=["dataset", "patient_id", "study_id_curr"],
            how="inner",
        )
        rows = rows[rows["dynamic_report"].apply(_nonempty)].copy()

        rows = rows.reset_index(drop=True)
        self.df = rows

        print(
            f"[JEPA dataset] split={split or 'all'}: "
            f"{len(self.df)} usable paired samples"
        )

    def __len__(self):
        return len(self.df)

    # --------------------------------------------------------
    # I/O
    # --------------------------------------------------------
    def _load_image(self, dataset: str, rel_path: str) -> Image.Image:
        path = _resolve_image_path(dataset, rel_path, self.image_roots)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dataset = row["dataset"]

        prior_raw = self._load_image(dataset, row["parent_image_prev"])
        curr_raw = self._load_image(dataset, row["parent_image_curr"])

        prior_raw = BASE_TRANSFORM(prior_raw)
        curr_raw = BASE_TRANSFORM(curr_raw)

        params = sample_augmentation(self.train)

        prior_img = apply_augmentation(prior_raw, params)
        curr_img = apply_augmentation(curr_raw, params)

        return {
            "prior_image": prior_img,
            "current_image": curr_img,
            "prior_report": row["prior_report"],
            "current_report": row["current_report"],
            "dynamic_report": row["dynamic_report"],
            "dataset": dataset,
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================
def jepa_collate_fn(batch):
    """
    Stack paired images; keep texts as lists of strings (the BioViL-T
    text encoder tokenizes inside `forward_contrastive`).
    """
    return {
        "prior_image": torch.stack([b["prior_image"] for b in batch]),
        "current_image": torch.stack([b["current_image"] for b in batch]),
        "prior_report": [b["prior_report"] for b in batch],
        "current_report": [b["current_report"] for b in batch],
        "dynamic_report": [b["dynamic_report"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }


# ============================================================
# SANITY CHECK
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-root", default="/home/evaprakash/all_data/mimic")
    parser.add_argument("--chexpert-root", default="/home/evaprakash/all_data/chexpert/train")
    parser.add_argument("--rexgradient-root", default="/home/evaprakash/all_data/rexgradient/deid_png")
    parser.add_argument("--split", default=None)
    parser.add_argument("--load-images", action="store_true")
    args = parser.parse_args()

    IMAGE_ROOTS = {
        "mimic": args.mimic_root,
        "chexpert": args.chexpert_root,
        "rexgradient": args.rexgradient_root,
    }

    ds = JEPACombinedDataset(
        image_roots=IMAGE_ROOTS,
        split=args.split,
        train=True,
    )

    print(f"\nTotal samples: {len(ds)}")
    print("\nPer-dataset breakdown:")
    for name in ("mimic", "chexpert", "rexgradient"):
        sub = ds.df[ds.df["dataset"] == name]
        print(f"  {name:12s}: {len(sub):>7d} paired samples")

    if args.load_images and len(ds) > 0:
        print("\n--- Image loading test ---")
        for name in ("mimic", "chexpert", "rexgradient"):
            indices = ds.df.index[ds.df["dataset"] == name].tolist()
            if not indices:
                continue
            idx = indices[0]
            try:
                sample = ds[idx]
                print(
                    f"  [{name}] idx={idx}  "
                    f"prior={tuple(sample['prior_image'].shape)}  "
                    f"current={tuple(sample['current_image'].shape)}  "
                    f"dyn={sample['dynamic_report'][:60]}..."
                )
            except FileNotFoundError as e:
                print(f"  [{name}] idx={idx}  FileNotFoundError: {e}")

    print("\nJEPA combined dataset sanity check passed.")
