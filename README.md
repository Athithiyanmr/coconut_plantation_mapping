# Coconut Plantation Mapping

> **A deep learning pipeline for coconut plantation mapping from Sentinel-2 imagery using UNet semantic segmentation — applied to Tamil Nadu districts.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-003DA5?style=flat-square)](https://sentinel.esa.int)
[![Deep Learning](https://img.shields.io/badge/UNet--Transformer-Deep%20Learning-FF4500?style=flat-square)](https://arxiv.org/abs/1505.04597)
[![Planetary Computer](https://img.shields.io/badge/Microsoft%20Planetary%20Computer-0078D4?style=flat-square&logo=microsoft&logoColor=white)](https://planetarycomputer.microsoft.com)
[![Canopy Height](https://img.shields.io/badge/WRI%20%2F%20Meta-Canopy%20Height%202024-2E7D32?style=flat-square)](https://ai.meta.com/ai-for-good/datasets/canopy-height-maps/)
[![MIT License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## What Is This?

Coconut palms are one of the most important tropical plantation crops, yet accurate large-scale mapping remains challenging due to spectral similarity with other tree crops. Precise mapping of coconut plantations supports agricultural planning, carbon stock estimation, and biodiversity monitoring.

This project builds a **reproducible deep learning pipeline** that maps coconut plantations from Sentinel-2 satellite imagery using a **multi-spectral UNet-Transformer segmentation model** — trained on Tamil Nadu districts using labels from the Descals et al. (2023) global coconut palm layer.

It applies convolutional neural networks with a transformer bottleneck to learn spatial patterns directly from satellite image patches, achieving pixel-level segmentation at 10 m resolution.

---

## Scientific Objective

To learn pixel-level representations of coconut plantation canopy from multi-spectral Sentinel-2 imagery, enriched with vegetation-discriminative spectral indices and canopy structural height, using deep convolutional semantic segmentation with class-imbalance-aware training.

---

## Area of Interest

**Tamil Nadu & Puducherry, India**

- Primary AOI: Coimbatore district (76.5°–77.3°E, 10.8°–11.5°N) — major coconut-growing region
- Extended to: Puducherry, Dindigul, and other TN districts via `--aoi` flag
- CRS: EPSG:32643 (UTM Zone 43N)

---

## Label Data Source

**Descals et al. (2023) Global Coconut Palm Layer**

- Paper: [https://essd.copernicus.org/articles/15/3991/2023/](https://essd.copernicus.org/articles/15/3991/2023/)
- Dataset: [https://zenodo.org/records/8128183](https://zenodo.org/records/8128183)
- Binary labels: `[0]` Not coconut, `[1]` Coconut palm (closed-canopy)
- Original resolution: 20 m (resampled to 10 m using nearest-neighbor interpolation)
- Produced using U-Net on Sentinel-1 + Sentinel-2 annual composites for 2020

> **Note on class imbalance:** Coconut pixels represent approximately **1% of total AOI area** in most Tamil Nadu districts. The pipeline includes specific fixes to handle this extreme imbalance — see [Class Imbalance Strategy](#class-imbalance-strategy) below.

---

## Full Pipeline Workflow

```
1. Download lowest-cloud Sentinel-2 scenes (Planetary Computer STAC)
       |
2. Mosaic & clip scenes to AOI boundary
       |
3. Build 12-band feature stack
   ├── 8 raw Sentinel-2 bands
   ├── 3 vegetation indices (NDVI, EVI, NDMI)
   └── 1 canopy height band (WRI/Meta, optional)
       |
4. Prepare coconut labels from Descals et al. (clip + resample to 10m)
       |
5. Generate class-balanced image patches
   ├── Hard-positive mining  (--min_pos_px)
   └── Negative subsampling  (--neg_sample)
       |
6. Train UNet-Transformer segmentation model
   ├── Dynamic class-aware Focal + Dice loss
   ├── Stratified train/val split
   ├── WeightedRandomSampler (5x positive oversampling)
   └── Albumentations data augmentation
       |
7. Sliding-window inference over full AOI
       |
8. Evaluate segmentation performance (IoU, F1, Precision, Recall)
```

---

## Input Feature Stack

### Sentinel-2 Level-2A Bands

| Band | Name | Resolution |
|---|---|---|
| B02 | Blue | 10 m |
| B03 | Green | 10 m |
| B04 | Red | 10 m |
| B05 | Red Edge 1 | 20 m → resampled to 10 m |
| B06 | Red Edge 2 | 20 m → resampled to 10 m |
| B08 | Near Infrared | 10 m |
| B11 | Shortwave Infrared 1 | 20 m → resampled to 10 m |
| B12 | Shortwave Infrared 2 | 20 m → resampled to 10 m |

### Computed Spectral Indices

| Index | Formula | Purpose |
|---|---|---|
| NDVI | `(B08 - B04) / (B08 + B04)` | Vegetation greenness |
| EVI | `2.5 * (B08 - B04) / (B08 + 6·B04 - 7.5·B02 + 1)` | Enhanced vegetation — reduces soil/atmosphere noise |
| NDMI | `(B08 - B11) / (B08 + B11)` | Moisture — separates coconut from dry crops |

### Canopy Height Band (Optional — Recommended)

| Source | Resolution | Units | Temporal Coverage |
|---|---|---|---|
| Meta & WRI High Resolution Canopy Height Maps (2024) | 1 m native → 10 m resampled | metres above ground | 2018–2020 |

**Why it matters:** Coconut palms (15–30 m tall) are spectrally similar to banana, scrub, and paddy, but structurally much taller. Adding the height band gives the model a clean separator that spectral indices cannot provide.

**Citation:** Tolan et al. (2024). *Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer and convolutional decoder trained on aerial lidar.* Remote Sensing of Environment, 300, 113888. CC BY 4.0.

**How to download for your AOI:**

```bash
# Option A — one-time TN-wide mosaic download (recommended)
python scripts/00_download_canopy_height_tn.py

# Then add to stack build with auto-clip to AOI boundary:
python scripts/02_build_stack.py --aoi puducherry --year 2022 --canopy_tn

# Option B — pre-clipped GeoTIFF for specific AOI:
python scripts/02_build_stack.py --aoi puducherry --year 2022 \
    --canopy_height data/raw/canopy_height_puducherry.tif
```

**Final model input without canopy:** 11-channel stack `[B02 B03 B04 B05 B06 B08 B11 B12 NDVI EVI NDMI]`  
**Final model input with canopy:** 12-channel stack `[B02 B03 B04 B05 B06 B08 B11 B12 NDVI EVI NDMI CanopyHeight_m]`

> The training script auto-detects the channel count from patches — no flag needed.

---

## Model Architecture

**UNet-Transformer Semantic Segmentation**

- Encoder: 3-block CNN (64 → 128 → 256 channels) with MaxPool downsampling
- Bottleneck: Transformer block with multi-head self-attention (4 heads, MLP dim 512) for long-range spatial context
- Decoder: Bilinear upsampling + skip connections
- Output: Pixel-level binary sigmoid (coconut / non-coconut)
- Input channels: auto-detected at runtime from patch shape

```
Input (B × C × 128 × 128)
    ↓
[Encoder]
  Conv-BN-ReLU × 2  →  64ch
  MaxPool 2×2
  Conv-BN-ReLU × 2  →  128ch
  MaxPool 2×2
  Conv-BN-ReLU × 2  →  256ch
    ↓
[Transformer Bottleneck]
  Flatten → MultiHeadAttention (4 heads) + MLP → Reshape
    ↓
[Decoder]
  Upsample + Cat(skip) → Conv 384→128
  Upsample + Cat(skip) → Conv 192→64
  Conv 64→1 + Sigmoid
    ↓
Output (B × 1 × 128 × 128)
```

---

## Loss Function

```
Loss = FocalLoss(dynamic pos_weight, gamma=3.0)  +  DiceLoss(smooth=1e-6)
```

| Component | Role |
|---|---|
| **Focal Loss** | Down-weights easy background pixels; focuses training on hard coconut boundaries. `gamma=3.0` chosen for sparse targets. |
| **Dynamic `pos_weight`** | Computed per-batch as `clamp(neg_count / pos_count, 5, 30)` — automatically scales with the actual imbalance ratio in each batch instead of a hardcoded constant. |
| **Dice Loss** | Optimises spatial overlap directly; prevents the model from learning "predict nothing" as a shortcut. |

---

## Class Imbalance Strategy

Coconut pixels make up **~1% of the AOI** in most Tamil Nadu districts. Without correction, the model learns to predict background everywhere and achieves 99% accuracy while predicting zero coconut. The pipeline addresses this at three stages:

### Stage 1 — Patch Generation (`make_patches.py`)

**Hard-positive mining via `--min_pos_px`**

A patch is kept as a positive training sample if either condition is true:
- `coconut_ratio >= --pos_ratio` (original ratio rule), **OR**
- `coconut_pixels >= --min_pos_px` (new hard-positive rule)

This rescues sparse coconut patches at plantation edges that were previously discarded because they missed the ratio threshold, even though they contain real coconut signal.

```bash
# Recommended settings for Tamil Nadu 1%-class districts:
python -m scripts.dl.make_patches --year 2022 --aoi puducherry \
    --patch 256 --stride 128 \
    --min_pos_px 10 --neg_sample 0.15 --clean
```

The summary output now reports `borderline_positive_patches` — the count of patches rescued by the new rule.

### Stage 2 — Training Loader (`train_unet.py`)

**Stratified train/val split**

Replaces `random_split` with `sklearn.model_selection.train_test_split(stratify=...)`. This guarantees that both the training set and validation set contain a representative proportion of coconut patches, so early stopping and threshold search are based on meaningful signal rather than nearly-empty validation batches.

**WeightedRandomSampler**

Positive patches (those containing any coconut pixel) are sampled `--pos_oversample` × (default **5×**) more often during training. Every batch is therefore guaranteed to contain coconut pixels, giving the model a consistent learning signal throughout training.

```bash
python -m scripts.dl.train_unet --year 2022 --aoi puducherry \
    --pos_oversample 5.0 --epochs 40 --batch 8
```

### Stage 3 — Loss Function (`train_unet.py`)

**Dynamic `pos_weight` in FocalLoss**

The BCE term is weighted per-batch based on the actual positive/negative pixel ratio:

```python
pos_weight = clamp(neg_count / pos_count, min=5.0, max=30.0)
```

This is much more robust than a fixed constant — it adapts to whatever batch composition the sampler produces and prevents numerical instability on very imbalanced batches.

---

## Data Augmentation

`dataset.py` now supports **Albumentations augmentation** for training only. Validation and test sets are always instantiated with `augment=False` to keep metrics unbiased.

| Augmentation | Probability | Purpose |
|---|---|---|
| Horizontal flip | 0.5 | Rotational invariance |
| Vertical flip | 0.5 | Rotational invariance |
| Random 90° rotation | 0.5 | Rotational invariance |
| Random brightness/contrast | 0.3 | Illumination robustness |
| Gaussian blur | 0.2 | Atmospheric haze simulation |
| Gaussian noise | 0.2 | Sensor noise robustness |

```bash
pip install albumentations
```

> If `albumentations` is not installed, the dataset falls back gracefully with a clear warning.

---

## Performance

**Primary metric: Intersection over Union (IoU)**

```
IoU = TP / (TP + FP + FN)
```

| IoU Range | Interpretation |
|---|---|
| < 0.40 | Weak |
| 0.40 – 0.59 | Moderate |
| 0.60 – 0.70 | Strong |
| > 0.70 | Research-grade |

The training script also reports F1, Precision, Recall, and performs an **automatic two-stage threshold search** (coarse + fine) to find the optimal decision boundary per epoch.

---

## Project Structure

```
coconut_plantation_mapping/
|
+-- scripts/
|   +-- 00_download_sentinel2_best_per_year.py   # Sentinel-2 acquisition
|   +-- 00_download_canopy_height_tn.py          # WRI/Meta canopy height download
|   +-- 01_prepare_aoi_raw.py                    # AOI preprocessing
|   +-- 02_build_stack.py                        # 11/12-band feature stack builder
|   +-- 03_download_coconut_labels.py            # Descals coconut label preparation
|   +-- evaluate_iou.py                          # IoU / F1 evaluation
|   +-- dl/
|       +-- dataset.py         # CoconutDataset with Albumentations augmentation
|       +-- unet_transformer.py  # UNet-Transformer model (primary)
|       +-- unet_model.py        # Alternate UNet (scratch / smp)
|       +-- make_patches.py      # Patch generation with hard-positive mining
|       +-- train_unet.py        # Training with stratified split + sampler
|       +-- predict_unet.py      # Sliding-window inference
|
+-- data/           # Satellite imagery, AOI boundaries, labels
+-- models/         # Saved model checkpoints + training history JSON
+-- outputs/        # Prediction maps and evaluation results
+-- run.py          # End-to-end runner
+-- environment.yml
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate coconut_mapping
export PYTHONPATH=$(pwd)
export KMP_DUPLICATE_LIB_OK=TRUE

# Additional dependencies for improvements:
pip install albumentations scikit-learn segmentation-models-pytorch
```

---

## Run the Pipeline

**End-to-end:**
```bash
python run.py --year 2022 --aoi puducherry --label_dir /path/to/descals_tiles
```

**Step by step:**
```bash
# Step 1 -- Download Sentinel-2 imagery
python scripts/00_download_sentinel2_best_per_year.py --year 2022 --aoi puducherry

# Step 2 -- Prepare AOI
python scripts/01_prepare_aoi_raw.py --year 2022 --aoi puducherry

# Step 3 -- Build feature stack (with canopy height)
python scripts/02_build_stack.py --year 2022 --aoi puducherry --canopy_tn
# Without canopy height:
# python scripts/02_build_stack.py --year 2022 --aoi puducherry

# Step 4 -- Prepare coconut labels
python scripts/03_download_coconut_labels.py --year 2022 --aoi puducherry \
    --label_dir /path/to/descals_tiles

# Step 5 -- Generate patches (with hard-positive mining)
python -m scripts.dl.make_patches --year 2022 --aoi puducherry \
    --patch 256 --stride 128 --min_pos_px 10 --neg_sample 0.15 --clean

# Step 6 -- Train
python -m scripts.dl.train_unet --year 2022 --aoi puducherry \
    --epochs 40 --batch 8 --workers 0 --pos_oversample 5.0

# Step 7 -- Inference
python -m scripts.dl.predict_unet --year 2022 --aoi puducherry

# Step 8 -- Evaluate
python scripts/evaluate_iou.py --year 2022 --aoi puducherry
```

---

## Key Arguments Reference

### `02_build_stack.py`
| Argument | Default | Description |
|---|---|---|
| `--aoi` | required | AOI name matching `data/raw/boundaries/{aoi}.shp` |
| `--year` | required | Year to process |
| `--canopy_tn` | off | Auto-clip canopy height from TN-wide mosaic (recommended) |
| `--canopy_height` | None | Path to pre-clipped canopy height GeoTIFF |

### `make_patches.py`
| Argument | Default | Description |
|---|---|---|
| `--patch` | 128 | Patch size in pixels (256 recommended for better context) |
| `--stride` | 64 | Stride between patches |
| `--pos_ratio` | 0.02 | Min coconut ratio to force-keep as positive |
| `--min_pos_px` | 10 | **NEW** — Min coconut pixels to force-keep (hard-positive mining) |
| `--neg_sample` | 0.25 | Keep probability for background patches (0.15 recommended) |

### `train_unet.py`
| Argument | Default | Description |
|---|---|---|
| `--epochs` | 40 | Training epochs |
| `--batch` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate (AdamW + CosineAnnealing) |
| `--val_split` | 0.2 | Validation fraction (stratified) |
| `--patience` | 6 | Early stopping patience |
| `--pos_oversample` | 5.0 | **NEW** — Positive patch oversampling multiplier |
| `--metric` | f1 | Threshold optimisation metric (`f1` or `iou`) |

---

## Applications

- Coconut plantation area estimation and district-level monitoring
- Agricultural planning and crop distribution mapping
- Carbon stock and biomass estimation for tropical plantations
- Biodiversity assessment in plantation landscapes
- Land use / land cover change detection
- Input layer for district-scale agricultural yield models

---

## Roadmap

- [x] Multi-spectral Sentinel-2 stack (8 bands + 3 indices)
- [x] UNet-Transformer with transformer bottleneck
- [x] WRI/Meta canopy height as Band 12
- [x] Hard-positive mining for 1%-class imbalance
- [x] Stratified split + WeightedRandomSampler
- [x] Dynamic pos_weight in Focal Loss
- [x] Albumentations augmentation
- [ ] Multi-district generalization (Thanjavur, Kanyakumari, Kerala)
- [ ] Temporal change detection (plantation expansion/loss over years)
- [ ] Sentinel-1 SAR (VV+VH) integration
- [ ] SMP ResNet34-UNet comparison study
- [ ] SegFormer-B2 comparison study
- [ ] Web map visualization of predictions

---

## Author

**Athithiyan M R** — Geospatial Data Scientist | Remote Sensing | Climate Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/athithiyan-m-r-/)
[![GitHub](https://img.shields.io/badge/GitHub-Athithiyanmr-181717?style=flat-square&logo=github)](https://github.com/Athithiyanmr)

---

## Acknowledgements

- ESA Sentinel-2 Mission
- Microsoft Planetary Computer & STAC API
- Descals, A. et al. (2023). High-resolution global map of closed-canopy coconut palm. *Earth System Science Data*, 15, 3991–4010.
- Tolan, J. et al. (2024). Very high resolution canopy height maps from RGB imagery using self-supervised vision transformer. *Remote Sensing of Environment*, 300, 113888.
- Meta AI for Good & World Resources Institute — Canopy Height Maps (CC BY 4.0)
- Zenodo dataset: [10.5281/zenodo.8128183](https://zenodo.org/records/8128183)

---

## License

MIT License (c) 2026 Athithiyan M R
