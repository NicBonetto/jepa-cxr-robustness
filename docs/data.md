# Data

Datasets are never committed. Raw data and derived files live in the private GCS bucket.
Copy them under `data/` (gitignored) to work locally.

## CheXpert

### Download

1. Register and accept the research use agreement on Stanford AIMI.
2. Download CheXpert v1.0 (the downsampled `CheXpert-v1.0-small` release is enough at 224×224).
3. Upload to the project bucket, or copy from it:
   ```bash
   gcloud storage rsync -r gs://$GCS_BUCKET/<chexpert-prefix> data/raw
   ```

### Preprocess

```bash
uv run python scripts/preprocess_chexpert.py \
    --root data/raw --csv data/raw/CheXpert-v1.0-small/train.csv --out data/chexpert
```

Outputs `data/chexpert/images/<split>/<image_id>.png`, `labels.csv`, `splits.csv` and the
`config.yaml` used. Settings are in `configs/data.yaml`.

### Decisions

| Setting | Choice | Why |
|---|---|---|
| Views | Frontal only | Both encoders are trained on frontal views, and VinDr-CXR is frontal only. |
| Uncertain (`-1`) | **U-Ignore**: treated as missing (NaN) | No guess at the true label. The classifiers already skip NaN per finding. `zeros`/`ones` are available in the config. |
| Blank (not mentioned) | Negative (0) | Standard practice: the report did not mention the finding. |
| Split | 80/10/10 **by patient**, seed 0 | A patient's images never appear in more than one split (prevents leakage). |
| Image ids | `patient00001_study1_view1_frontal` | The file stem alone (`view1_frontal`) repeats across patients. |
| Size | 224×224, bilinear, aspect ratio not preserved | Corruptions are applied at this size (`CORRUPT_SIZE` in `src/jepa_cxr/extract.py`). See Input resolution. |

Under U-Ignore the test split has missing labels. The metrics skip them (see Shared labels).

## VinDr-CXR

### Download

1. Get PhysioNet credentialed access and sign the VinDr-CXR data use agreement.
2. Download `vindr-cxr/1.0.0` (folders `train/`, `test/`, `annotations/`) to the bucket, or copy
   it from there to `data/raw/vindr-cxr`.

### Preprocess

```bash
uv run python scripts/preprocess_vindr.py --root data/raw/vindr-cxr --out data/vindr
```

Same outputs as CheXpert, and it prints shared-label positives per split. The official
train/test split is kept. VinDr is never used for training (Aim 3).

### Decisions

| Setting | Choice | Why |
|---|---|---|
| Labels | All image-level labels in `image_labels_<split>.csv` | Mapped to the shared set below. |
| Multiple readers | **Majority** (at least 2 of 3 radiologists on train) | Single-reader positives are noisy. `any` is available in the config. Test labels are already a consensus. |
| Pixels | VOI LUT / window from the DICOM, MONOCHROME1 inverted, min-max to uint8 | The corruptions assume MONOCHROME2 in [0, 1]. |
| Size | 224×224, bilinear, aspect ratio not preserved | Same as CheXpert. |

## Shared labels

Both preprocessing scripts also write `labels_shared.csv` with the ten labels below
(`src/jepa_cxr/labels.py`). Use these for anything that compares or crosses datasets.

| Shared | CheXpert | VinDr-CXR |
|---|---|---|
| Atelectasis | Atelectasis | Atelectasis |
| Cardiomegaly | Cardiomegaly | Cardiomegaly |
| Consolidation | Consolidation | Consolidation |
| Edema | Edema | Edema |
| Pleural Effusion | Pleural Effusion | Pleural effusion |
| Pneumothorax | Pneumothorax | Pneumothorax |
| Pneumonia | Pneumonia | Pneumonia |
| Lung Lesion | Lung Lesion | Nodule/Mass |
| Fracture | Fracture | Rib fracture OR Clavicle fracture |
| No Finding | No Finding | No finding |

When several source labels map to one, the result is positive if any source is positive and
missing only if all are missing.

### Dropped

| Label | Why |
|---|---|
| Support Devices (CheXpert) | No VinDr equivalent. |
| Lung Opacity | CheXpert uses it as an umbrella over consolidation, edema and others. In VinDr it is a specific local finding. |
| Enlarged Cardiomediastinum / Aortic enlargement | Only partial overlap. |
| Pleural Other / Pleural thickening | Only partial overlap. |
| Other VinDr labels (COPD, Tuberculosis, ILD, ...) | No CheXpert equivalent. |

### Rare labels

`compare_methods.py` drops any label with fewer than `labels.min_positives` (default 20)
positives in any test set, since AUROC on a handful of positives is noise. It prints which
labels were dropped. Override with `--min-positives`.

### Missing labels in evaluation

Metrics skip missing (NaN) labels: AUROC/AUPRC per label, ECE over observed entries, and
per-image error over that image's observed labels. Images with no observed labels are left
out of the referral metrics.

## Embeddings

After preprocessing, embed every condition in `configs/embeddings.yaml`:

```bash
uv run python scripts/run_embedding_grid.py --dry-run     # list what would run
uv run python scripts/run_embedding_grid.py               # all encoders and datasets
uv run python scripts/run_embedding_grid.py --encoder radjepa --dataset vindr
```

Outputs go to `embeddings/<encoder>/<dataset>/<split>/<condition>.parquet`, where the
condition is `clean` or `<corruption>_s<severity>`. Each encoder is loaded once per run.

| Setting | Choice | Why |
|---|---|---|
| Clean splits | CheXpert train/val/test, VinDr test | Classifiers are fit and tuned on clean CheXpert only. |
| Corrupted splits | CheXpert test, VinDr test | Corruptions are a test-time shift (Aim 2). VinDr adds shift plus corruption (Aim 3). |
| Grid | 5 corruptions × severities 1–5 | Severity values are in `jepa_cxr.corruptions.PARAMS`. |
| Corruption seed | `seed + i` for the i-th image (sorted by id) in a split | The same image gets the same noise for both encoders. |

### Input resolution

Both encoders get the same 224×224 image, clean or corrupted. It is then resized (bicubic)
to each encoder's native size: 224 for RadJEPA, 518 for RAD-DINO (`image_size` in
`configs/encoders.yaml`, matching each model's pinned `config.json`).

| Option | Why not |
|---|---|
| Run RAD-DINO at 224 | It was trained at 518 with 14-px patches (37×37 grid). At 224 the position embeddings are interpolated to 16×16 and each patch covers ~2.3× more anatomy than in training, which would understate the baseline. |
| Preprocess at 518 for RAD-DINO | RAD-DINO would see more detail than RadJEPA, confounding encoder with resolution. Noise applied at 518 is also a different corruption from noise at 224, so severities would not match across encoders. |

Caveats:

- Upsampling smooths high-frequency corruptions (Gaussian and Poisson noise, JPEG
  artifacts) before RAD-DINO sees them. RAD-DINO's robustness may be partly a filtering effect.
- RAD-DINO costs roughly 5× more per image (1369 vs 256 tokens). Run the full grid on CUDA.

### Manifest

Every file written is recorded in `embeddings/manifest.csv`: encoder and pinned revision,
dataset, split, corruption, severity, seed, image count, dimension, SHA-256, git commit
(suffixed `-dirty` if the tree had uncommitted changes) and time. The grid runner skips any
condition that already has a file and a manifest row, so an interrupted run resumes.
`--force` redoes them.

`--limit N` embeds only the first N images per split and requires `--out-dir`, so a partial
run never counts as done in the real manifest.

Sync with the bucket, manifest included:

```bash
gcloud storage rsync -r embeddings gs://$GCS_BUCKET/<embeddings-prefix>
```
