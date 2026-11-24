# Data

Datasets are never committed. Raw data and derived files live in the private GCS bucket;
copy them under `data/` (gitignored) to work locally.

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
| Size | 224×224, bilinear, aspect ratio not preserved | Matches the corruption pipeline in `scripts/extract_embeddings.py`. |

**Open:** under U-Ignore the test split has missing labels, which the evaluation metrics do
not yet handle. To be resolved in label harmonization (#4).
