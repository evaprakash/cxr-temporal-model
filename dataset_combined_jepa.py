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

import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset_combined import (
    BASE_TRANSFORM,
    apply_augmentation,
    sample_augmentation,
)
from progression_phrases import CLS_ORDER, SILVER_TO_CLS
from silver_masks import (
    default_masks_root,
    load_prog_patch_weights,
    uniform_patch_weights,
)

CLS_TO_IDX = {cls: i for i, cls in enumerate(CLS_ORDER)}


# ============================================================
# CONDITION MODES
# ============================================================
# ``dynamic``   : use the study-level concatenation of all
#                 ``label == "dynamic"`` sentences from
#                 ``silver_sentences.parquet`` (the original behavior).
# ``templated`` : build the condition from the per-finding
#                 ``(finding, progression)`` rows of
#                 ``silver_findings.parquet`` joined to the same study
#                 pair, formatted as
#                 ``"{finding.lower()} is {progression_class}"`` and
#                 joined with ". ". Order is shuffled per
#                 ``__getitem__`` call when ``train=True`` and
#                 deterministically sorted otherwise, so the val
#                 condition is stable epoch-to-epoch.
CONDITION_MODES = ("dynamic", "templated")


# ============================================================
# DEFAULT SILVER DATASET PATHS
# ============================================================
# Local copy of the CheXTemporal silver annotation parquets. Default
# is ``CheXTemporal/`` next to this file (same convention as the
# ``final_gold_<dataset>_images/`` lookup in ``progression_classify``).
# Override via the ``CHEXTEMPORAL_DIR`` env var or pass explicit paths
# to ``JEPACombinedDataset(...)``.
DEFAULT_DATASET_DIR = os.environ.get(
    "CHEXTEMPORAL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "CheXTemporal"),
)
DEFAULT_FINDINGS = os.path.join(DEFAULT_DATASET_DIR, "silver_findings.parquet")
DEFAULT_STUDIES = os.path.join(DEFAULT_DATASET_DIR, "silver_studies.parquet")
DEFAULT_SENTENCES = os.path.join(DEFAULT_DATASET_DIR, "silver_sentences.parquet")

# Where the auto-generated train/val split assignments are persisted
# when the studies parquet does not already have a ``split`` column.
DEFAULT_SPLITS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "splits_jepa.csv",
)
SPLIT_KEY_COLS = ["dataset", "patient_id", "study_id_curr", "study_id_prev"]


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


def _ensure_split_assignments(
    rows: pd.DataFrame,
    splits_file: str,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> pd.DataFrame:
    """Add a ``split`` column to ``rows`` (values: ``'train'`` / ``'val'``).

    If ``splits_file`` already exists, load it and merge by
    ``SPLIT_KEY_COLS``. Otherwise compute a stratified split (per
    ``dataset``) using a seeded RNG, save to ``splits_file``, and merge.

    The stratified procedure: for each dataset, sort the row indices
    deterministically, shuffle them with ``np.random.RandomState(seed)``,
    take the first ``round(n * val_fraction)`` as ``val``, the rest as
    ``train``. This guarantees that:

      - each dataset is represented in both splits with the requested
        proportion,
      - the assignment is reproducible from ``seed`` alone, and
      - the cached file lets you pin a split across re-runs even if the
        underlying data changes (delete the file to regenerate).
    """
    rows = rows.reset_index(drop=True)

    # ------------------------------------------------------------
    # Load cached splits if present and consistent
    # ------------------------------------------------------------
    if os.path.exists(splits_file):
        cached = pd.read_csv(
            splits_file,
            dtype={
                "dataset": "string",
                "patient_id": "string",
                "study_id_curr": "string",
                "study_id_prev": "string",
                "split": "string",
            },
        )
        merged = rows.merge(cached[SPLIT_KEY_COLS + ["split"]], on=SPLIT_KEY_COLS, how="left")
        n_missing = int(merged["split"].isna().sum())
        if n_missing == 0:
            print(f"[JEPA dataset] Loaded cached splits from {splits_file}")
            return merged
        print(
            f"[JEPA dataset] WARNING: {n_missing} rows not in cached splits "
            f"({splits_file}); regenerating."
        )
        os.remove(splits_file)

    # ------------------------------------------------------------
    # Generate stratified split deterministically
    # ------------------------------------------------------------
    rng = np.random.RandomState(seed)
    rows = rows.copy()
    rows["split"] = "train"

    for dataset_name, group in rows.groupby("dataset", sort=True):
        n = len(group)
        n_val = max(1, int(round(n * val_fraction)))
        sorted_idx = sorted(group.index.tolist())
        shuffled = rng.permutation(sorted_idx)
        val_idx = shuffled[:n_val].tolist()
        rows.loc[val_idx, "split"] = "val"

    # ------------------------------------------------------------
    # Persist for transparency / pinning
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(splits_file) or ".", exist_ok=True)
    rows[SPLIT_KEY_COLS + ["split"]].to_csv(splits_file, index=False)

    val_counts = rows[rows["split"] == "val"].groupby("dataset").size().to_dict()
    train_counts = rows[rows["split"] == "train"].groupby("dataset").size().to_dict()
    print(
        f"[JEPA dataset] Generated stratified split (val_fraction={val_fraction}, "
        f"seed={seed}); saved to {splits_file}"
    )
    print(f"[JEPA dataset]   train per dataset: {train_counts}")
    print(f"[JEPA dataset]   val   per dataset: {val_counts}")

    return rows


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
        condition_text (str)  the predictor's textual condition; content
                              depends on ``condition_mode``:
                                * ``"dynamic"``: joined ``label=="dynamic"``
                                  sentences from the current study.
                                * ``"templated"``: per-finding clauses
                                  ``"{finding} is {progression}"`` joined
                                  with ". " (order shuffled per call when
                                  ``train=True``, sorted otherwise).
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
        Optional. If ``"train"`` or ``"val"``:
          - if the studies parquet has a ``split`` column, filter on it
          - else, fall back to a deterministic stratified split (per
            ``dataset``) generated from ``split_seed`` and cached to
            ``splits_file``.
        ``None`` returns all rows.
    train
        Whether to apply random augmentation. Eval/val should pass False.
    val_fraction
        Fraction of rows assigned to ``"val"`` when generating the
        fallback stratified split. Default 0.1 (10%).
    split_seed
        Seed for the fallback stratified split. Same seed → same split.
    splits_file
        Where to read/write the cached split assignments. Defaults to
        ``./splits_jepa.csv`` next to this file.
    condition_mode
        Which text condition to feed the predictor. ``"dynamic"`` (the
        default) returns the joined ``label=="dynamic"`` sentences from
        ``silver_sentences.parquet``. ``"templated"`` returns the
        capitalized per-finding clauses
        ``"{Finding} is {progression_class}."`` built from
        ``silver_findings.parquet``. Both modes carry the same set of
        study-pair rows under the hood — only the value of
        ``condition_text`` differs.

        Regardless of ``condition_mode``, every sample also exposes the
        per-pair ``findings`` and ``progression_cls_idx`` lists (kept for
        backward compatibility / inspection), plus two new fields
        consumed by the 4th progression-classification loss:

          ``prog_finding``     : str
              One finding sampled from the pair's findings list. Randomly
              chosen each ``__getitem__`` call when ``train=True`` (so
              across epochs every finding gets sampled), deterministically
              chosen (alphabetically first) when ``train=False`` so val
              metrics are comparable epoch-to-epoch.
          ``prog_cls_idx``     : int
              Silver progression-class index (into ``CLS_ORDER``) for the
              ``prog_finding`` above.
          ``prog_patch_weights`` : Tensor ``(N,)`` float
              Soft 14×14 patch weights for the progression loss. When a
              ``filtered_masks`` JSON exists for
              ``(parent_image_curr, prog_finding)``, weights concentrate
              on that finding region (same geometry as the current
              image's resize/crop/affine). Otherwise all-ones (= legacy
              global mean).
          ``prog_mask_used``   : bool
              True iff a non-empty mask was loaded for this sample.

        Training time uses these fields to build a per-pair 5-prompt
        bank (one ``"{prog_finding} is {class}."`` per class) and runs
        the predictor 5 times to score image-image cosine for a 5-way CE
        — see ``progression_classification_loss`` in ``losses_jepa.py``.
    """

    def __init__(
        self,
        image_roots: Dict[str, str],
        findings_path: str = DEFAULT_FINDINGS,
        studies_path: str = DEFAULT_STUDIES,
        sentences_path: str = DEFAULT_SENTENCES,
        split: Optional[str] = None,
        train: bool = True,
        val_fraction: float = 0.1,
        split_seed: int = 42,
        splits_file: Optional[str] = None,
        condition_mode: str = "dynamic",
        masks_root: Optional[str] = None,
    ):
        if condition_mode not in CONDITION_MODES:
            raise ValueError(
                f"condition_mode must be one of {CONDITION_MODES!r}; "
                f"got {condition_mode!r}"
            )
        self.image_roots = {k: str(v) for k, v in image_roots.items()}
        self.train = train
        self.split = split
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.splits_file = splits_file or DEFAULT_SPLITS_FILE
        self.condition_mode = condition_mode
        self.masks_root = masks_root or default_masks_root()
        if not os.path.isdir(self.masks_root):
            print(
                f"[JEPA dataset] WARNING: masks_root not found "
                f"({self.masks_root}); progression loss will use "
                f"uniform patch weights for every sample."
            )
        else:
            print(f"[JEPA dataset] masks_root={self.masks_root}")

        # ------------------------------------------------------------
        # Load + filter
        # ------------------------------------------------------------
        findings = _normalize_ids(pd.read_parquet(findings_path))
        studies = _normalize_ids(pd.read_parquet(studies_path))
        sentences = _normalize_ids(pd.read_parquet(sentences_path))

        # Track whether the studies parquet provides its own split column;
        # if not, we'll generate a stratified split AFTER all the joining
        # is done so the cached file describes what's actually used.
        studies_has_split_col = "split" in studies.columns
        if split is not None and studies_has_split_col:
            studies = studies[studies["split"] == split].copy()

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

        # ------------------------------------------------------------
        # Normalize the per-finding progression labels onto CLS_ORDER.
        # Anything outside SILVER_TO_CLS (empty / corrupt) becomes
        # NaN and is dropped before grouping so the templated condition
        # never contains an unrecognized progression class.
        # ------------------------------------------------------------
        rows["progression_cls"] = (
            rows["progression"].astype(str).str.strip().map(SILVER_TO_CLS)
        )
        rows = rows[rows["progression_cls"].notna()].copy()
        rows = rows[rows["finding"].apply(_nonempty)].copy()
        rows["finding"] = rows["finding"].astype(str).str.strip().str.lower()

        # ------------------------------------------------------------
        # Collapse per-finding rows into one row per study pair, keeping
        # the *list* of (finding, progression_cls) pairs so the dataset
        # can build a multi-clause condition at __getitem__ time. The
        # image paths and reports are study-pair-level so we take the
        # first occurrence under each group.
        # ------------------------------------------------------------
        pair_keys = ["dataset", "patient_id", "study_id_curr", "study_id_prev"]
        rows = (
            rows.groupby(pair_keys, sort=False, as_index=False)
            .agg(
                finding=("finding", list),
                progression_cls=("progression_cls", list),
                parent_image_curr=("parent_image_curr", "first"),
                parent_image_prev=("parent_image_prev", "first"),
                current_report=("current_report", "first"),
                prior_report=("prior_report", "first"),
            )
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

        # ------------------------------------------------------------
        # Fallback split: studies parquet didn't have a split column,
        # so generate a deterministic stratified one (per dataset) over
        # the fully-joined rows and cache to disk.
        # ------------------------------------------------------------
        if split is not None and not studies_has_split_col:
            rows = _ensure_split_assignments(
                rows,
                splits_file=self.splits_file,
                val_fraction=self.val_fraction,
                seed=self.split_seed,
            )
            rows = rows[rows["split"] == split].reset_index(drop=True)

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

    def _build_templated_condition(self, row) -> str:
        """Build the per-finding templated condition for one study pair.

        Order is shuffled per call when ``self.train`` is True so the
        predictor doesn't learn position-dependent shortcuts; deterministic
        (alphabetical by finding name) otherwise so val Smooth L1 is
        comparable across epochs.

        Each clause is ``"{Finding} is {progression_class}."`` (first
        letter capitalized, trailing period) with the canonical class
        name from ``progression_phrases.CLS_ORDER``. Synonym sampling is
        intentionally not done here — the eval-time prompt ensembling
        already covers the synonym bank.
        """
        findings = list(row["finding"])
        prog_cls = list(row["progression_cls"])
        pairs = list(zip(findings, prog_cls))
        if self.train:
            random.shuffle(pairs)
        else:
            pairs.sort(key=lambda fp: fp[0])
        clauses = [
            f"{finding[:1].upper()}{finding[1:]} is {cls}."
            for finding, cls in pairs
        ]
        return " ".join(clauses)

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

        if self.condition_mode == "dynamic":
            condition_text = row["dynamic_report"]
        else:  # "templated" — validated in __init__
            condition_text = self._build_templated_condition(row)

        # The per-pair finding list and matching silver progression class
        # are surfaced unconditionally for backward compat / inspection.
        # ``progression_cls_idx`` is the integer index into CLS_ORDER (0–4)
        # so downstream code never has to re-map class names to ints.
        findings = [str(f) for f in row["finding"]]
        progression_cls_idx = [
            CLS_TO_IDX[str(c)] for c in row["progression_cls"]
        ]

        # For the 4th (progression-classification) loss we score one
        # finding per pair per epoch. Sampling one here keeps the 5-prompt
        # bank flat (5*B prompts per batch instead of 5*Σfindings) and lets
        # the dataloader return a fixed shape; across epochs the random
        # pick exposes every finding for the pair anyway. Val uses the
        # alphabetically first finding so val loss is reproducible.
        if findings:
            pairs = list(zip(findings, progression_cls_idx))
            if self.train:
                prog_finding, prog_cls_idx = random.choice(pairs)
            else:
                pairs.sort(key=lambda fp: fp[0])
                prog_finding, prog_cls_idx = pairs[0]
        else:
            # Defensive: rows without findings should have been dropped in
            # __init__, but if any slip through, surface an empty finding
            # with class 0 so the collate doesn't crash; the trainer will
            # have to mask these out (currently it doesn't, because the
            # filter in __init__ is strict).
            prog_finding, prog_cls_idx = "", 0

        # Sometimes-masked progression: if filtered_masks has a JSON for
        # (current image, prog_finding), warp it with the same geometry
        # as curr_img and pool onto the 14×14 patch grid. Otherwise
        # uniform weights (= legacy global mean over all patches).
        if prog_finding:
            prog_patch_weights, prog_mask_used = load_prog_patch_weights(
                self.masks_root,
                dataset,
                str(row["parent_image_curr"]),
                prog_finding,
                aug_params=params,
            )
        else:
            prog_patch_weights = uniform_patch_weights()
            prog_mask_used = False

        return {
            "prior_image": prior_img,
            "current_image": curr_img,
            "prior_report": row["prior_report"],
            "current_report": row["current_report"],
            "condition_text": condition_text,
            "dataset": dataset,
            "findings": findings,
            "progression_cls_idx": progression_cls_idx,
            "prog_finding": prog_finding,
            "prog_cls_idx": int(prog_cls_idx),
            "prog_patch_weights": prog_patch_weights,
            "prog_mask_used": bool(prog_mask_used),
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================
def jepa_collate_fn(batch):
    """
    Stack paired images; keep texts as lists of strings (the BioViL-T
    text encoder tokenizes inside ``forward_contrastive``). The ragged
    per-pair ``findings`` and ``progression_cls_idx`` lists are kept as
    nested Python lists rather than padded into a tensor — the trainer
    flattens them when building the prompt batch for the progression
    loss, so a tensor of variable shape per pair would just add a
    pad/unpad step.

    ``prog_finding`` and ``prog_cls_idx`` carry the per-epoch sampled
    finding + silver-progression-class for the 4th loss. They are
    fixed-shape (B,) — one finding string per pair, one integer label per
    pair — so the trainer can build a flat (B*5,) prompt list of
    ``"{Finding} is {class}."`` strings without any padding.

    ``prog_patch_weights`` is ``(B, N)`` float; ``prog_mask_used`` is
    ``(B,)`` bool marking which rows used a real filtered mask.
    """
    return {
        "prior_image": torch.stack([b["prior_image"] for b in batch]),
        "current_image": torch.stack([b["current_image"] for b in batch]),
        "prior_report": [b["prior_report"] for b in batch],
        "current_report": [b["current_report"] for b in batch],
        "condition_text": [b["condition_text"] for b in batch],
        "dataset": [b["dataset"] for b in batch],
        "findings": [b["findings"] for b in batch],
        "progression_cls_idx": [b["progression_cls_idx"] for b in batch],
        "prog_finding": [b["prog_finding"] for b in batch],
        "prog_cls_idx": torch.tensor(
            [b["prog_cls_idx"] for b in batch], dtype=torch.long
        ),
        "prog_patch_weights": torch.stack(
            [b["prog_patch_weights"] for b in batch]
        ),
        "prog_mask_used": torch.tensor(
            [b["prog_mask_used"] for b in batch], dtype=torch.bool
        ),
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
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--splits-file", default=None,
                        help="Override the cached splits CSV path.")
    parser.add_argument("--load-images", action="store_true")
    parser.add_argument(
        "--condition-mode",
        choices=list(CONDITION_MODES),
        default="dynamic",
        help="Which text condition to surface in the smoke test.",
    )
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
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        splits_file=args.splits_file,
        condition_mode=args.condition_mode,
    )

    print(f"\nTotal samples: {len(ds)}")
    print("\nPer-dataset breakdown:")
    for name in ("mimic", "chexpert", "rexgradient"):
        sub = ds.df[ds.df["dataset"] == name]
        print(f"  {name:12s}: {len(sub):>7d} paired samples")

    n_findings = ds.df["finding"].apply(len)
    print(
        f"\nFindings per study pair: min={int(n_findings.min())}, "
        f"max={int(n_findings.max())}, mean={n_findings.mean():.2f}"
    )
    print(f"Condition mode: {ds.condition_mode}")

    if args.load_images and len(ds) > 0:
        print("\n--- Image loading test ---")
        for name in ("mimic", "chexpert", "rexgradient"):
            indices = ds.df.index[ds.df["dataset"] == name].tolist()
            if not indices:
                continue
            idx = indices[0]
            try:
                sample = ds[idx]
                preview = sample["condition_text"][:80].replace("\n", " ")
                print(
                    f"  [{name}] idx={idx}  "
                    f"prior={tuple(sample['prior_image'].shape)}  "
                    f"current={tuple(sample['current_image'].shape)}  "
                    f"condition={preview}..."
                )
            except FileNotFoundError as e:
                print(f"  [{name}] idx={idx}  FileNotFoundError: {e}")

    print("\nJEPA combined dataset sanity check passed.")
