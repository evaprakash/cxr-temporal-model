# cxr-temporal-model

I-JEPA-style temporal chest X-ray model. Predicts current-image patch
features from prior-image patch features conditioned on a textual
description of the change ("dynamic" sentences).

## Architecture

```
Prior CXR  ──►  E (online, trained)         ──►  LN(z_prior) ──┐
                                                                 ├──►  Predictor  ──►  ẑ_cur = LN(z_prior) + Δz
Condition  ──►  text_encoder                ──►  τ_dyn       ──┘
Current CXR──►  E_target (EMA, stop-grad)   ──►  LN(z_cur)
```

Losses:

- **JEPA Smooth L1** between `ẑ_cur` and stop-gradient
  LayerNorm-normalized `z_cur` (mirrors the I-JEPA reference loss).
  Computed in fp32 inside the autocast region.
- **GLoRIA local contrastive** on `(LN(z_prior), prior_report)`.
- **GLoRIA local contrastive** on `(ẑ_cur, current_report)`.

The image encoder is a shared BioViL-T (ResNet50 + multi-image
transformer); the prior path receives gradients while the current path
is encoded by an EMA copy under stop-gradient. The text encoder is
BioViL-T's CXR-BERT specialized model. The predictor is a small
transformer that outputs a delta `Δz`, so `ẑ_cur` is reconstructed as
`LN(z_prior) + Δz` — the "do nothing" baseline (`Δz = 0`) yields
`ẑ_cur ≈ LN(z_cur)` for unchanged anatomy, leaving the predictor to
spend capacity only on the actual change described by `τ_dyn`.

## Repository layout

```
cxr-temporal-model/
├── dataset_combined.py         # base image transforms + augmentation helpers
├── dataset_combined_jepa.py    # JEPA dataset (silver corpus, paired)
├── losses.py                   # local_contrastive_loss (GLoRIA)
├── losses_jepa.py              # JEPA Smooth L1
├── resume_train_jepa.py        # JEPA DDP training entry (invoked via torchrun)
├── run_jepa.py                 # direct launcher (auto-detects GPUs, no SLURM needed)
└── tempcxr/
    └── modules/
        ├── image_encoder_jepa.py   # JEPA image encoder (raw outputs)
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
`/home/eprakash/jepa/CheXTemporal/`; override with the
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

`resume_train_jepa.py` writes checkpoints to `./checkpoints_jepa/` and
validation metrics to `./logs/val_metrics_jepa.csv` (both relative to
the script). Override with environment variables:

```bash
export JEPA_CHECKPOINT_DIR=/data/ckpts
export JEPA_LOG_DIR=/data/logs
```

Both directories are excluded from version control via `.gitignore`.

## Smoke tests (run before training)

From the repo root:

```bash
# Model-only forward + losses + backward + EMA update on random tensors
python -m tempcxr.modules.jepa

# Dataset construction + filtering + (optional) image loading
python dataset_combined_jepa.py --load-images
```

`python -m tempcxr.modules.jepa` validates:

- The shared image encoder + EMA target encoder are wired correctly.
- The text encoder forwards three reports (prior / current / dynamic).
- The predictor produces patch-shape outputs.
- All three losses are finite and `total.backward()` succeeds.
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
python run_jepa.py --resume checkpoints_jepa/epoch_5.pt
```

Without `--resume`, `resume_train_jepa.py` auto-resumes from the
latest `epoch_*.pt` in `CHECKPOINT_DIR` if any exist.

## Key design choices

### What gets trained vs. frozen

| Component                    | Gradient? |
|------------------------------|-----------|
| online image encoder         | yes (via prior path + report loss) |
| target image encoder (EMA)   | no — updated via EMA, used under stop-gradient |
| text encoder                 | yes (via report losses + dynamic-text conditioning) |
| predictor                    | yes (via JEPA + report-on-pred losses) |

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

### Normalization

Following I-JEPA's recipe rather than BioViL-T's CLIP-style recipe:

- Image encoder patch + global outputs are returned **raw** (no
  `F.normalize` to the unit sphere) by `image_encoder_jepa.py`.
- Both the **prior patches** (going into the predictor) and the
  **target patches** (used as the JEPA loss target) get a feature-dim
  **LayerNorm with no learnable parameters** applied inside
  `TempCXRJEPA.forward`. This puts both sides of the JEPA loss on the
  same scale (≈ √D) so the delta predictor has a meaningful
  "do-nothing" baseline rather than fighting a 10× scale gap.
- Predictor output is `LN(prior) + Δz` (raw, no extra normalization).
- The downstream contrastive losses (`local_contrastive_loss`)
  re-normalize their inputs internally with `F.normalize`, so they are
  invariant to whether they receive raw or LayerNorm-normed patches.

### Loss reuse

The JEPA pipeline does not duplicate `local_contrastive_loss`. It is
defined once in `losses.py` and imported by the JEPA model and training
script. Only the new JEPA loss lives in `losses_jepa.py`.

## Reference

The JEPA recipe here closely follows the official I-JEPA training
script:
[`facebookresearch/ijepa/src/train.py`](https://github.com/facebookresearch/ijepa/blob/main/src/train.py),
adapted for the temporal CXR setup (cross-image prediction conditioned
on text instead of masked-block prediction within a single image).
