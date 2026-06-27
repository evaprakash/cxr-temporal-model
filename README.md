# cxr-temporal-model

I-JEPA-style temporal chest X-ray model. Predicts current-image patch
features from prior-image patch features conditioned on a textual
description of the change. The text condition can come from one of two
sources (see "Condition modes" below): per-finding templated clauses
of the form `"{finding} is {progression}"` built from
`silver_findings.parquet` (the default), or the joined "dynamic"
sentences of the current study.

## Architecture

Everything lives on the unit sphere — both encoders L2-normalize their
outputs along the feature dim, and the predictor renormalizes its
``z_prior + Δz`` sum so ``ẑ_cur`` is unit-norm too. There are no
LayerNorms anywhere in the JEPA path (and no extra projection heads on
top of each encoder's built-in 128-d projection).

```
Prior CXR  ──►  E (online, trained, L2-norm)  ──►  z_prior ──┐
                                                              ├──►  Predictor  ──►  ẑ_cur = normalize(z_prior + Δz)
Condition  ──►  text_encoder (L2-norm)        ──►  τ_cond  ──┘
Current CXR──►  E_target (EMA, stop-grad, L2) ──►  z_cur
```

Losses:

- **JEPA cosine** between `ẑ_cur` and stop-gradient `z_cur`:
  ``1 - cos(ẑ_cur, z_cur)`` averaged over patches. Both inputs are
  already unit-norm, so this is a directional loss by construction.
  Computed in fp32 inside the autocast region.
- **GLoRIA local contrastive** on `(z_prior, prior_report)`.
- **GLoRIA local contrastive** on `(ẑ_cur, current_report)`.

The image encoder is a shared BioViL-T (ResNet50 + multi-image
transformer) whose built-in `joint_feature_size=128` head projects to
the joint 128-d space; the prior path receives gradients while the
current path is encoded by an EMA copy under stop-gradient. The text
encoder is BioViL-T's CXR-BERT specialized model with the 768→128
`BertProjectionHead` followed by L2-norm. Both encoders return
unit-norm features directly — no extra `proj_clip` / `proj_jepa` /
`target_proj_jepa` heads sit on top.

The predictor is a small transformer that outputs a delta `Δz` at the
transformer's natural residual scale (no final LayerNorm). The
prediction is reconstructed as `ẑ_cur = normalize(z_prior + Δz)` — the
"do nothing" baseline (`Δz ≈ 0`) yields `ẑ_cur ≈ z_prior` on the
sphere for unchanged anatomy, leaving the predictor to spend capacity
only on the actual change described by `τ_cond`.

## Repository layout

```
cxr-temporal-model/
├── dataset_combined.py         # base image transforms + augmentation helpers
├── dataset_combined_jepa.py    # JEPA dataset (silver corpus, paired)
├── progression_phrases.py      # shared CLS_ORDER / phrase bank / label maps
├── losses.py                   # local_contrastive_loss (GLoRIA)
├── losses_jepa.py              # JEPA cosine loss
├── resume_train_jepa.py        # JEPA DDP training entry (invoked via torchrun)
├── run_jepa.py                 # direct launcher (auto-detects GPUs, no SLURM needed)
├── infer_jepa.py               # single-example inference demo
├── eval_jepa_val.py            # average inference metrics over the full val split
├── progression_classify.py     # 5-way progression on gold pairs (image-text scoring)
├── eval_progression_jepa.py    # 5-way progression on gold pairs (image-image cosine; matches train rule)
├── csv_progression_eval.py     # shared helpers for CSV-based 3-way benchmarks
├── eval_mscxrt.py              # 3-way progression classification on MS-CXR-T
├── eval_cig.py                 # 3-way progression classification on Chest ImaGenome
└── tempcxr/
    └── modules/
        ├── image_encoder_jepa.py   # JEPA image encoder (unit-sphere outputs)
        ├── text_encoder.py         # CXR-BERT text encoder
        └── jepa.py                 # JEPA forward orchestration + EMA + predictor
```

`dataset_combined.py` is included for the shared image transforms
(`BASE_TRANSFORM`, `sample_augmentation`, `apply_augmentation`) that
`dataset_combined_jepa.py` imports. `losses.py` is included for the
shared `local_contrastive_loss` that the JEPA model and training script
import.

## Setup

### 1. Python environment

```bash
conda create -n jepa python=3.10
conda activate jepa
pip install torch torchvision transformers pandas pillow tqdm
# plus whatever the hi-ml submodule needs
```

### 2. hi-ml-multimodal (BioViL-T)

The image and text encoders are built on top of Microsoft's
[`hi-ml`](https://github.com/microsoft/hi-ml) repo. Clone it as a
sibling under `tempcxr/modules/`:

```bash
cd tempcxr/modules
git clone https://github.com/microsoft/hi-ml.git
```

Both `image_encoder_jepa.py` and `text_encoder.py` add
`tempcxr/modules/hi-ml/hi-ml-multimodal/src` to `sys.path` and import
`health_multimodal.image.model.MultiImageModel` and
`health_multimodal.text.model.modelling_cxrbert.CXRBertModel` from
there. If you put hi-ml elsewhere, edit the `HI_ML_SRC` constant at the
top of those two files.

### 3. Pretrained text-encoder weights

`text_encoder.py` resolves both pretrained text-model paths relative
to itself:

```
tempcxr/modules/pretrained/BiomedVLP-CXR-BERT-specialized/
tempcxr/modules/pretrained/BiomedVLP-BioViL-T/
```

Drop the local copies of the BiomedVLP and BioViL-T text models into
those two directories. Alternatively, edit the `BIOVIL_TEXT_MODEL` and
`BIOVILT_TEXT_MODEL` constants at the top of `text_encoder.py` to
point elsewhere (or to the HF identifiers
`microsoft/BiomedVLP-CXR-BERT-specialized` /
`microsoft/BiomedVLP-BioViL-T` for `transformers` to download).

`tempcxr/modules/pretrained/` is gitignored.

### 4. Image roots

`resume_train_jepa.py` hardcodes an `IMAGE_ROOTS` dict that maps each
dataset name to the filesystem location of its images. Edit it at the
top of the file to match your filesystem.

### 5. CheXTemporal annotation parquets

`dataset_combined_jepa.py` loads three silver parquets
(`silver_findings.parquet`, `silver_studies.parquet`,
`silver_sentences.parquet`). The default location is
`/home/evaprakash/CheXTemporal/`; override with the
`CHEXTEMPORAL_DIR` env var:

```bash
export CHEXTEMPORAL_DIR=/path/to/CheXTemporal
```

### 5a. Train / val split

If the studies parquet has a `split` column, rows are filtered by it.
Otherwise `JEPACombinedDataset` falls back to a **deterministic
stratified split** (per dataset) seeded by `split_seed=42` and
defaulting to a 10% val fraction. The train/val DataLoaders in
`resume_train_jepa.py` share the same `(val_fraction, split_seed)`,
so they always agree on which rows are val.

The assignments are cached to `splits_jepa.csv` next to
`dataset_combined_jepa.py` so you can audit which study pairs are in
val. The CSV is gitignored. To regenerate (e.g., after updating the
silver parquets), delete it and re-run any script that constructs a
`JEPACombinedDataset` with `split="train"` or `split="val"`.

To pick a different fraction or seed, edit the constants at the top
of `resume_train_jepa.py`:

```python
VAL_FRACTION = 0.1
SPLIT_SEED = 42
```

### 6. Checkpoint and log directories

`resume_train_jepa.py` writes checkpoints to
`./checkpoints_jepa_<CONDITION_MODE>/` and validation metrics to
`./logs_<CONDITION_MODE>/val_metrics_jepa.csv` (both relative to the
script). With the default `CONDITION_MODE=templated` that's
`./checkpoints_jepa_templated/` and `./logs_templated/`. Override with
environment variables:

```bash
export JEPA_CHECKPOINT_DIR=/data/ckpts
export JEPA_LOG_DIR=/data/logs
```

**Schedule:** epoch 1 is always saved, then every `SAVE_EVERY_N_EPOCHS`
(default 5), plus `best.pt` whenever the validation total improves.
So a 50-epoch run produces `epoch_{1,5,10,...,50}.pt` (11 snapshots)
plus `best.pt`.

**Contents** of each `.pt` (one self-contained dict): full model state
(online encoder, **EMA target encoder**, text encoder, predictor),
AdamW state, LR scheduler state, GradScaler state, current epoch,
and best-val-so-far. The EMA momentum schedule is deterministic from
the step count and is reconstructed on resume.

**Resume:** automatic from the latest `epoch_*.pt` if you rerun
`python run_jepa.py` with no flags, or explicit via
`python run_jepa.py --resume checkpoints_jepa_templated/epoch_25.pt`.

Both directories are excluded from version control via `.gitignore`.

## Smoke tests (run before training)

From the repo root:

```bash
# Model-only forward + losses + backward + EMA update on random tensors
python -m tempcxr.modules.jepa

# Dataset construction + filtering + (optional) image loading
python dataset_combined_jepa.py --load-images

# Side-by-side check that both condition modes parse the silver
# parquets correctly and produce the expected condition_text strings
# (prior/current image paths, per-finding metadata, condition preview).
python smoke_test_dataset.py
python smoke_test_dataset.py --num-examples 5 --split val
python smoke_test_dataset.py --load-images   # also call __getitem__
```

`python -m tempcxr.modules.jepa` validates:

- The shared image encoder + EMA target encoder are wired correctly.
- The text encoder forwards three reports (prior / current / condition)
- The predictor produces patch-shape outputs.
- All three losses (JEPA cosine, prior contrastive, current
  contrastive) are finite and `total.backward()` succeeds.
- `update_ema(momentum)` runs without error.

## Training

```bash
python run_jepa.py
```

Auto-detects the number of visible GPUs and spawns torchrun with the
right `--nproc_per_node`. Equivalent to:

```bash
torchrun --nproc_per_node=$(python -c "import torch; print(torch.cuda.device_count())") \
         resume_train_jepa.py
```

Resume from a checkpoint:

```bash
python run_jepa.py --resume checkpoints_jepa_templated/epoch_5.pt
```

Without `--resume`, `resume_train_jepa.py` auto-resumes from the
latest `epoch_*.pt` in `CHECKPOINT_DIR` if any exist.

## Key design choices

### Condition modes

The predictor's text condition is selected at dataset-construction time
via `JEPACombinedDataset(condition_mode=...)`. Two modes are supported:

| Mode        | Source                                                          | Per-sample text                                                              |
|-------------|-----------------------------------------------------------------|------------------------------------------------------------------------------|
| `templated` (default) | Per-finding rows of `silver_findings.parquet` for the study pair | `"{Finding1} is {prog1}. {Finding2} is {prog2}."` (capitalized, period-terminated, shuffled per-call at train time) |
| `dynamic`             | All `label == "dynamic"` sentences for the current study, joined | Free-text change description ("Right pleural effusion has increased…") |

Regardless of `condition_mode`, every dataset row also exposes
`findings: list[str]` and `progression_cls_idx: list[int]` (indices
into `CLS_ORDER`) — these are not consumed by the training loop on
`main` but are kept on the dataset so the eval-time progression
classifiers (`progression_classify.py`, `eval_mscxrt.py`, `eval_cig.py`)
and the templated condition mode have access to them. Per-finding
progression supervision as a training loss lives on the
`progression-loss` branch.

The mode is selected via the `CONDITION_MODE` env var on the training
side and via `--condition-mode` (or `CONDITION_MODE`) on the eval side.
Both modes write to independent default checkpoint / log dirs
(`checkpoints_jepa_<mode>/` + `logs_<mode>/`), so switching modes never
clobbers the other mode's checkpoints. Legacy `checkpoints_jepa/` and
`logs/` dirs from earlier (pre-3-loss, pre-softmax-fix) runs are left
untouched as an archive.

```bash
# current default: templated finding + progression sentences as the
# predictor's text condition
python run_jepa.py
# → writes to checkpoints_jepa_templated/ and logs_templated/

# dynamic report sentences (opt in via the env var)
CONDITION_MODE=dynamic python run_jepa.py
# → writes to checkpoints_jepa_dynamic/ and logs_dynamic/

# evaluate the templated-mode checkpoint (matches the default)
JEPA_CKPT=checkpoints_jepa_templated/best.pt python eval_jepa_val.py

# evaluate a dynamic-mode checkpoint
CONDITION_MODE=dynamic \
    JEPA_CKPT=checkpoints_jepa_dynamic/best.pt \
    python eval_jepa_val.py
```

Important: the `condition_mode` you train with should match the
`condition_mode` you evaluate with — otherwise the val cosine distance
and the per-pair JEPA metrics aren't directly comparable to the training
loss curve. The 5-way / 3-way progression-classification scripts
(`progression_classify.py`, `eval_mscxrt.py`, `eval_cig.py`) are
independent of the condition mode because they build their own
single-finding prompts at score time.

### What gets trained vs. frozen

| Component                    | Gradient? |
|------------------------------|-----------|
| online image encoder         | yes (via prior path + report losses) |
| target image encoder (EMA)   | no — updated via EMA, used under stop-gradient |
| text encoder                 | yes (via report losses + predictor's condition text) |
| predictor                    | yes (via JEPA cosine + report-on-pred losses) |

### EMA target encoder

The current image is encoded by an EMA copy of the online encoder
(I-JEPA / BYOL recipe). After every `optimizer.step()`, the target's
weights are nudged toward the online weights:

```
target ← m * target + (1 - m) * online
```

`m` follows a linear ramp from `EMA_START = 0.996` to `EMA_END = 1.0`
over the full course of training. BatchNorm running stats are copied
verbatim from the online encoder rather than EMA-averaged (since BN
running stats are themselves already running averages).

### Normalization (unit sphere)

Everything in the JEPA path lives on the unit sphere:

- The image encoder L2-normalizes both patch and global outputs along
  the feature dim inside `image_encoder_jepa.py` (one `F.normalize` at
  the end of `forward`).
- The text encoder L2-normalizes its per-token (`txt_local`) and global
  (`txt_global`) outputs inside `text_encoder.py:forward_contrastive`.
- The predictor takes unit-norm prior patches and unit-norm text
  tokens, computes `Δz` at the transformer's natural residual scale
  (no final LayerNorm), and renormalizes the sum: `ẑ_cur =
  normalize(prior + Δz)`.
- The JEPA loss is `1 - cos(ẑ_cur, z_cur)`, a directional loss
  consistent with the geometry.
- The downstream contrastive losses (`local_contrastive_loss`)
  re-L2-normalize their inputs defensively — that's a no-op on
  already-unit-norm inputs but keeps the loss correct if used
  standalone.

There are no extra `proj_clip` / `proj_jepa` / `target_proj_jepa`
heads on top of either encoder. The image encoder's built-in
`joint_feature_size=128` projection and the text encoder's
`BertProjectionHead` are the only projections that turn raw backbone
features into the joint 128-d space.

### Loss reuse

The JEPA pipeline does not duplicate `local_contrastive_loss`. It is
defined once in `losses.py` and imported by the JEPA model and
training script. Only the JEPA cosine loss lives in `losses_jepa.py`.

### Branches

- `main` (this branch): three-loss model — JEPA cosine + two GLoRIA
  local contrastive losses — on the dynamic sentence condition by
  default.
- `progression-loss`: adds a 4th per-finding 5-way progression
  classification loss on top of `main` (see that branch's README for
  details). Kept around as a separate experiment; not the current
  canonical training setup.

## Inference

Two scripts are provided for evaluating a trained checkpoint
(`checkpoints_jepa_templated/best.pt` by default; override with
`--ckpt` or `JEPA_CKPT`).

### `infer_jepa.py` — single-example demo

Picks one paired sample from the val split, runs the predictor, and
prints how close the prediction is to the actual current image's
encoding:

```bash
python infer_jepa.py            # random val sample
python infer_jepa.py --idx 42   # specific val sample
```

Reports the JEPA cosine distance (`1 - cos(ẑ_cur, z_cur)`, the
training-time loss on the unit sphere), the slide-deck inference score
`cos(ẑ_cur - z_prior, z_cur - z_prior)`, per-patch cosine similarity,
and a do-nothing baseline (`ẑ_cur := z_prior`) for comparison. The
Smooth L1 is also printed as a diagnostic for comparison with
pre-unit-sphere runs.

Single-example numbers can be misleading: most paired CXR studies are
"stable" so on any given pair the do-nothing baseline is hard to beat.
Use `eval_jepa_val.py` for the averaged comparison.

### `eval_jepa_val.py` — averaged metrics over the full val split

Same metrics as `infer_jepa.py` but computed once per pair and
averaged across all val samples. This is the meaningful comparison
against the do-nothing baseline (cherry-picked stable / change samples
will systematically favor one or the other).

```bash
python eval_jepa_val.py                    # full val set
python eval_jepa_val.py --limit 500        # quick smoke test
python eval_jepa_val.py --batch-size 32    # if VRAM allows
```

The reported `Cosine distance (predictor)` should closely match the
`val_jepa` column in `logs/val_metrics_jepa.csv` at the loaded epoch
(modulo augmentation — eval runs without).

### `progression_classify.py` — 5-way progression classification (gold)

Loads `gold_progression_pairs.parquet` and, for each (study pair,
finding) example, builds a multi-phrase prompt bank for five candidate
progression classes (`improving`, `stable`, `worsening`, `new`,
`resolved`). Every phrase is fed through the predictor; per-phrase
scores are averaged within each class to produce one cosine score and
one Smooth L1 score per class. The script reports two parallel sets of
results — one for `argmax cos(Δẑ, Δz_true)` (direction-only, the
slide-deck rule) and one for `argmin SmoothL1(ẑ_cur, z_cur)`
(direction + magnitude, the JEPA training loss).

```bash
# Sanity-check one gold row with the full 5-way breakdown
python progression_classify.py --demo

# Full evaluation (both cosine and Smooth L1)
python progression_classify.py --eval

# First 200 rows only (quick smoke test)
python progression_classify.py --eval --limit 200

# Override the disease/phrase template
python progression_classify.py --eval --prompt-template "{} appears {}"

# Override an image root for a single dataset
python progression_classify.py --eval \
  --image-root mimic=/data/final_gold_mimic_images
```

Both image encoders follow the training-time convention: online
encoder for the prior (matches what the predictor was trained on) and
EMA target encoder for the current (matches the JEPA loss target). All
phrase prompts for one finding are batched through both the text
encoder and the predictor, and the image forward pass runs once per
(prior, current) pair, so the cost scales with `O(N_pairs)` rather
than with the size of the prompt bank.

### `eval_progression_jepa.py` — 5-way progression on gold pairs (image-image)

Mirrors the training-time JEPA invariant exactly: ``cos(ẑ_cur, z_cur)``
on the unit sphere. Where `progression_classify.py` scores predicted
patches against text prompts (image-text cosine), this script scores
predicted patches against the *actual* current latent (image-image
cosine), so the test-time question matches the training-time question.

Per gold row `(prior_image, current_image, finding, gt_progression)`:

  1. Encode prior with the online image encoder → unit-norm `z_prior`.
  2. Encode current with the EMA target image encoder → unit-norm
     detached `z_cur`.
  3. Build 5 templated prompts — one per class in `CLS_ORDER` — using
     the **same canonical format as the templated training condition**:
     `"{Finding} is {class}."`.
  4. Batch the predictor with the same `z_prior` and the 5 different
     text prompts → 5 candidate `ẑ_cur^c` (one per class).
  5. Score each by `mean over patches of cos(ẑ_cur^c, z_cur)`.
  6. **Predicted class = argmax_c** over the 5 cosines.

Also reports the do-nothing baseline `cos(z_prior, z_cur)` and the
fraction of pairs where the argmax predicted class actually beats it,
as a sanity check on whether the text-conditioned predictor is doing
useful work vs. just predicting "no change".

```bash
# Sanity-check one gold row (random or specific --idx)
python eval_progression_jepa.py --demo
python eval_progression_jepa.py --demo --idx 17

# Full 5-way eval over the gold parquet
python eval_progression_jepa.py --eval
python eval_progression_jepa.py --eval --limit 200

# Custom checkpoint and per-dataset image roots
python eval_progression_jepa.py --eval \
    --ckpt checkpoints_jepa_templated/epoch_30.pt \
    --image-root mimic=/data/final_gold_mimic_images
```

The output is overall accuracy, per-class recall, 5×5 confusion
matrix, mean cosine score per candidate class, the do-nothing baseline
cosine, and per-finding accuracy — directly comparable to the cosine
column of `progression_classify.py --eval` but using the image-image
scoring rule instead of image-text.

### `eval_mscxrt.py` and `eval_cig.py` — 3-way progression classification

Same multi-phrase scoring machinery as `progression_classify.py`, but
applied to two CSV-format external benchmarks where every label is one
of `{improving, stable, worsening}`:

| Script              | Default CSV                  | Label mapping                                                            |
|---------------------|------------------------------|--------------------------------------------------------------------------|
| `eval_mscxrt.py`    | `mscxrt_labels_new.csv`      | `improving / stable / worsening` (also accepts `no change`, `unchanged`) |
| `eval_cig.py`       | `cig_gold_labels_new.csv`    | `improved → improving`, `worsened → worsening`, `no change → stable`     |

`eval_cig.py` drops every CIG row whose `comparison` is anything other
than the three accepted labels (so e.g. `new`, `resolved`, and `n/a`
are excluded). Both scripts restrict the predictor's argmax / argmin
to the 3-class subset of `CLS_ORDER`, so the model is never penalized
for predicting `new` or `resolved` on benchmarks that don't have those
classes.

Both CSVs are expected to contain absolute MIMIC-CXR-JPG image paths
in `img_path_prev` / `img_path_curr` (we open them directly with no
prefix-stripping or root remapping). The reported overall accuracy,
per-class recall, 3×3 confusion matrix, and per-finding accuracy are
directly comparable to the MS-CXR-T and CIG numbers in the
CheXTemporal paper.

```bash
# MS-CXR-T
python eval_mscxrt.py --demo
python eval_mscxrt.py --eval
python eval_mscxrt.py --eval --csv /path/to/mscxrt_labels_new.csv

# Chest ImaGenome
python eval_cig.py --demo
python eval_cig.py --eval
python eval_cig.py --eval --csv /path/to/cig_gold_labels_new.csv
```

The shared 3-way evaluation logic lives in `csv_progression_eval.py`;
the two scripts above are thin wrappers that just supply a default
CSV path and a dataset-specific `comparison`-to-canonical-class map.

## Reference

The JEPA recipe here closely follows the official I-JEPA training
script:
[`facebookresearch/ijepa/src/train.py`](https://github.com/facebookresearch/ijepa/blob/main/src/train.py),
adapted for the temporal CXR setup (cross-image prediction conditioned
on text instead of masked-block prediction within a single image).
