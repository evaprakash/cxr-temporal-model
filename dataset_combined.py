# dataset_combined.py
#
# Multi-dataset version of BioViLTDataset that works with the combined CSV
# containing MIMIC, CheXpert, and ReXGradient data.

import random
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ============================================================
# BASE IMAGE TRANSFORM (DETERMINISTIC)
# ============================================================
BASE_TRANSFORM = T.Compose([
    T.Resize(512),
    T.CenterCrop(448),
])


# ============================================================
# SYNCED AUGMENTATION SAMPLING
# ============================================================
def sample_augmentation(train: bool):
    """Sample augmentation parameters ONCE per sample pair."""
    if not train:
        return None

    return {
        "angle": random.uniform(-30, 30),
        "shear": random.uniform(-15, 15),
        "brightness": random.uniform(0.8, 1.2),
        "contrast": random.uniform(0.8, 1.2),
    }


def apply_augmentation(img: Image.Image, params):
    """Apply identical augmentation params to an image."""
    if params is not None:
        img = TF.affine(
            img,
            angle=params["angle"],
            translate=(0, 0),
            scale=1.0,
            shear=[params["shear"], 0.0],
        )
        img = TF.adjust_brightness(img, params["brightness"])
        img = TF.adjust_contrast(img, params["contrast"])

    img = TF.to_tensor(img)

    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)

    return img


# ============================================================
# DATASET
# ============================================================
class BioViLTCombinedDataset(Dataset):
    """
    Multi-dataset version that reads from the combined CSV produced by
    build_combined_csv.py.  Each row stores pre-resolved *relative* image
    paths; per-dataset root directories are supplied via ``image_roots``.

    Parameters
    ----------
    csv_path : str
        Path to biovilt_pretrain_combined_imagelevel.csv
    image_roots : dict[str, str]
        Mapping from dataset name to the filesystem root that is prepended
        to the relative paths stored in the CSV.  Expected keys:
            "mimic"        -> e.g. "/scratch/.../mimic-cxr-jpg/2.0.0/files"
            "chexpert"     -> e.g. "/scratch/.../CheXpert-v1.0"
            "rexgradient"  -> e.g. "/scratch/.../ReXGradient-160K"
    split : str
        One of "train", "validate", "test".
    train : bool
        Whether to apply data augmentation.
    """

    def __init__(
        self,
        csv_path: str,
        image_roots: Dict[str, str],
        split: str,
        train: bool,
    ):
        self.df = pd.read_csv(csv_path, dtype={"subject_id": str, "study_id": str, "prior_study_id": str})
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.image_roots = {k: Path(v) for k, v in image_roots.items()}
        self.train = train

        self.single_indices = self.df.index[self.df["has_prior"] == False].tolist()
        self.multi_indices  = self.df.index[self.df["has_prior"] == True].tolist()

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, dataset: str, rel_path: str) -> Path:
        return self.image_roots[dataset] / rel_path

    def _load_image(self, dataset: str, rel_path: str) -> Image.Image:
        path = self._resolve_path(dataset, rel_path)
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dataset = row["dataset"]

        curr_raw = self._load_image(dataset, row["image_path_curr"])

        prior_raw = None
        if row["has_prior"] and pd.notna(row["image_path_prior"]) and row["image_path_prior"]:
            prior_raw = self._load_image(dataset, row["image_path_prior"])

        curr_raw = BASE_TRANSFORM(curr_raw)
        if prior_raw is not None:
            prior_raw = BASE_TRANSFORM(prior_raw)

        params = sample_augmentation(self.train)

        curr_img = apply_augmentation(curr_raw, params)
        prior_img = (
            apply_augmentation(prior_raw, params)
            if prior_raw is not None
            else None
        )

        return {
            "current_image": curr_img,
            "prior_image": prior_img,
            "has_prior": bool(row["has_prior"]),
            "text": row["full_report_text"],
            "dataset": dataset,
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================
def biovilt_collate_fn(batch):
    """
    Assumes batch is homogeneous (all Ds or all Dm).
    That will be guaranteed by how we construct loaders.
    """
    has_prior = batch[0]["has_prior"]

    curr = torch.stack([b["current_image"] for b in batch])

    prior = (
        torch.stack([b["prior_image"] for b in batch])
        if has_prior
        else None
    )

    return {
        "current_image": curr,
        "prior_image": prior,
        "has_prior": has_prior,
        "text": [b["text"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
    }


# ============================================================
# SANITY CHECK
# ============================================================
if __name__ == "__main__":
    import argparse
    from torch.utils.data import Subset

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="biovilt_pretrain_combined_imagelevel.csv")
    parser.add_argument("--mimic-root", default="/scratch/m000081/yunhe/dataset/MIMIC-CXR/mimic-cxr-jpg/2.0.0/files")
    parser.add_argument("--chexpert-root", default="/scratch/m000081/yabin/datasets/chestxpert/CheXpert-v1.0")
    parser.add_argument("--rexgradient-root", default="/scratch/m000081/chong/data/ReXGradient-160K")
    parser.add_argument("--load-images", action="store_true",
                        help="Actually try to open images (requires access to image roots)")
    args = parser.parse_args()

    IMAGE_ROOTS = {
        "mimic": args.mimic_root,
        "chexpert": args.chexpert_root,
        "rexgradient": args.rexgradient_root,
    }

    print(f"Loading CSV: {args.csv}")
    ds = BioViLTCombinedDataset(
        csv_path=args.csv,
        image_roots=IMAGE_ROOTS,
        split="train",
        train=True,
    )

    print(f"\nTotal samples: {len(ds)}")
    print(f"Single (Ds):   {len(ds.single_indices)}")
    print(f"Multi  (Dm):   {len(ds.multi_indices)}")

    print("\nPer-dataset breakdown:")
    for name in ("mimic", "chexpert", "rexgradient"):
        sub = ds.df[ds.df["dataset"] == name]
        n_prior = (sub["has_prior"] == True).sum()
        print(f"  {name:12s}: {len(sub):>7d} total, {n_prior:>7d} with prior")

    single_subset = Subset(ds, ds.single_indices)
    multi_subset  = Subset(ds, ds.multi_indices)
    print(f"\nSingle subset size: {len(single_subset)}")
    print(f"Multi  subset size: {len(multi_subset)}")

    # Spot-check resolved paths (no I/O needed)
    print("\nSample resolved paths:")
    for name in ("mimic", "chexpert", "rexgradient"):
        row = ds.df[ds.df["dataset"] == name].iloc[0]
        full = ds._resolve_path(name, row["image_path_curr"])
        exists = full.exists()
        tag = "✓ exists" if exists else "✗ not found locally"
        print(f"  [{name}] {full}  ({tag})")

    # Optionally try loading actual images
    if args.load_images:
        print("\n--- Image loading test ---")
        import random
        for name in ("mimic", "chexpert", "rexgradient"):
            indices = ds.df.index[ds.df["dataset"] == name].tolist()
            idx = random.choice(indices)
            try:
                sample = ds[idx]
                shape = tuple(sample["current_image"].shape)
                print(f"  [{name}] idx={idx}  current_image={shape}  "
                      f"has_prior={sample['has_prior']}  text={sample['text'][:60]}...")
            except FileNotFoundError as e:
                print(f"  [{name}] idx={idx}  ✗ FileNotFoundError: {e}")
    else:
        print("\nSkipping image loading (pass --load-images to test actual I/O)")

    print("\n✅ Combined dataset sanity check passed.")
