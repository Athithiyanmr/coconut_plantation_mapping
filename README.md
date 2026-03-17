# Coconut Plantation Mapping

> **A deep learning pipeline for coconut plantation mapping from Sentinel-2 imagery using UNet semantic segmentation — applied to Coimbatore district, Tamil Nadu.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-003DA5?style=flat-square)](https://sentinel.esa.int)
[![Deep Learning](https://img.shields.io/badge/UNet-Deep%20Learning-FF4500?style=flat-square)](https://arxiv.org/abs/1505.04597)
[![Planetary Computer](https://img.shields.io/badge/Microsoft%20Planetary%20Computer-0078D4?style=flat-square&logo=microsoft&logoColor=white)](https://planetarycomputer.microsoft.com)
[![MIT License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## What Is This?

Coconut palms are one of the most important tropical plantation crops, yet accurate large-scale mapping remains challenging due to spectral similarity with other tree crops. Precise mapping of coconut plantations supports agricultural planning, carbon stock estimation, and biodiversity monitoring.

This project builds a **reproducible deep learning pipeline** that maps coconut plantations from Sentinel-2 satellite imagery using a **multi-spectral UNet segmentation model** — trained on Coimbatore district (a major coconut-growing region in western Tamil Nadu) using labels from the Descals et al. (2023) global coconut palm layer.

It applies convolutional neural networks to learn spatial patterns directly from satellite image patches, achieving pixel-level segmentation at 10m resolution.

---

## Scientific Objective

To learn pixel-level representations of coconut plantation canopy from multi-spectral Sentinel-2 imagery, enriched with vegetation-discriminative spectral indices, using deep convolutional semantic segmentation.

---

## Area of Interest

**Coimbatore district, Tamil Nadu, India**

- Approximate bounding box: 76.5°–77.3°E, 10.8°–11.5°N
- Major coconut-growing area in western Tamil Nadu
- CRS: EPSG:32643 (UTM Zone 43N)

---

## Label Data Source

**Descals et al. (2023) Global Coconut Palm Layer**

- Paper: [https://essd.copernicus.org/articles/15/3991/2023/](https://essd.copernicus.org/articles/15/3991/2023/)
- Dataset: [https://zenodo.org/records/8128183](https://zenodo.org/records/8128183)
- Binary labels: [0] Not coconut, [1] Coconut palm (closed-canopy)
- Original resolution: 20m (resampled to 10m using nearest-neighbor interpolation)
- Produced using U-Net on Sentinel-1 + Sentinel-2 annual composites for 2020

---

## Full Pipeline Workflow

```
1. Download lowest-cloud Sentinel-2 scenes (Planetary Computer STAC)
       |
2. Mosaic & clip scenes to Coimbatore AOI
       |
3. Build 11-band spectral feature stack (8 raw + 3 vegetation indices)
       |
4. Prepare coconut labels from Descals et al. (clip + resample to 10m)
       |
5. Generate balanced image patches for training
       |
6. Train UNet-Transformer segmentation model (Focal + Dice loss)
       |
7. Sliding-window inference over full AOI
       |
8. Evaluate segmentation performance (IoU)
```

---

## Input Data

**Sentinel-2 Level-2A bands:**

| Band | Name | Resolution |
|---|---|---|
| B02 | Blue | 10m |
| B03 | Green | 10m |
| B04 | Red | 10m |
| B05 | Red Edge 1 | 20m (resampled to 10m) |
| B06 | Red Edge 2 | 20m (resampled to 10m) |
| B08 | Near Infrared | 10m |
| B11 | Shortwave Infrared 1 | 20m (resampled to 10m) |
| B12 | Shortwave Infrared 2 | 20m (resampled to 10m) |

**Vegetation spectral indices computed:**

| Index | Formula | Purpose |
|---|---|---|
| NDVI | (B08 - B04) / (B08 + B04) | Vegetation greenness |
| EVI | 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1) | Enhanced vegetation sensitivity |
| NDMI | (B08 - B11) / (B08 + B11) | Moisture — separates coconut from dry vegetation |

**Final model input:** 11-channel feature stack `[B02, B03, B04, B05, B06, B08, B11, B12, NDVI, EVI, NDMI]`

---

## Model Architecture

**UNet-Transformer Semantic Segmentation**
- Encoder-decoder structure with skip connections
- Transformer bottleneck with multi-head self-attention for long-range spatial modeling
- 11-channel multi-spectral input
- Pixel-level binary classification output (coconut / non-coconut)

**Loss Function:**
```
Loss = Focal Loss (alpha=0.8, gamma=3.0) + Dice Loss
```
Focal Loss handles class imbalance (coconut is minority class, gamma=3.0 for sparser targets). Dice Loss handles region-level spatial overlap.

---

## Performance

**Primary metric: Intersection over Union (IoU)**

```
IoU = TP / (TP + FP + FN)
```

| IoU Range | Interpretation |
|---|---|
| < 0.40 | Weak |
| 0.40 - 0.59 | Moderate |
| 0.60 - 0.70 | Strong |
| > 0.70 | Research-grade |

---

## Project Structure

```
coconut_plantation_mapping/
|
+-- scripts/
|   +-- 00_download_sentinel2_best_per_year.py   # Satellite acquisition
|   +-- 01_prepare_aoi_raw.py                    # AOI preprocessing
|   +-- 02_build_stack.py                        # Vegetation spectral stack builder
|   +-- 03_download_coconut_labels.py            # Descals coconut label preparation
|   +-- evaluate_iou.py                          # Evaluation metrics
|   +-- dl/
|       +-- dataset.py                           # CoconutDataset class
|       +-- unet_model.py                        # Basic UNet architecture
|       +-- unet_transformer.py                  # UNet + Transformer model
|       +-- make_patches.py                      # Patch generation
|       +-- train_unet.py                        # Training script
|       +-- predict_unet.py                      # Inference script
|
+-- data/           # Satellite imagery, AOI, labels
+-- models/         # Saved model checkpoints
+-- outputs/        # Prediction maps and evaluation results
+-- run.py          # End-to-end runner script
+-- environment.yml
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate coconut_mapping
export PYTHONPATH=$(pwd)
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

## Run the Pipeline

**End-to-end:**
```bash
python run.py --year 2020 --aoi coimbatore --label_dir /path/to/descals_tiles
```

**Step by step:**
```bash
# Step 1 -- Download Sentinel-2 imagery
python scripts/00_download_sentinel2_best_per_year.py --year 2020 --aoi coimbatore

# Step 2 -- Prepare AOI
python scripts/01_prepare_aoi_raw.py --year 2020 --aoi coimbatore

# Step 3 -- Build spectral feature stack
python scripts/02_build_stack.py --year 2020 --aoi coimbatore

# Step 4 -- Prepare coconut labels from Descals et al.
python scripts/03_download_coconut_labels.py --year 2020 --aoi coimbatore \
    --label_dir /path/to/descals_tiles

# Step 5 -- Create image patches
python -m scripts.dl.make_patches --year 2020 --aoi coimbatore

# Step 6 -- Train UNet model
python -m scripts.dl.train_unet --year 2020 --aoi coimbatore

# Step 7 -- Run inference
python -m scripts.dl.predict_unet --year 2020 --aoi coimbatore

# Step 8 -- Evaluate
python scripts/evaluate_iou.py --year 2020 --aoi coimbatore
```

---

## Applications

- Coconut plantation area estimation and monitoring
- Agricultural planning and crop distribution mapping
- Carbon stock and biomass estimation for tropical plantations
- Biodiversity assessment in plantation landscapes
- Land use / land cover change detection
- Input layer for district-scale agricultural models

---

## Roadmap

- [ ] Multi-district generalization (Thanjavur, Kanyakumari, Kerala)
- [ ] Temporal change detection (plantation expansion/loss over years)
- [ ] Integration of Sentinel-1 SAR (VV+VH) as additional input channels
- [ ] DeepLabV3+ comparison study
- [ ] Web map visualization of predictions

---

## Author

**Athithiyan M R** -- Geospatial Data Scientist | Remote Sensing | Climate Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/athithiyan-m-r-/)
[![GitHub](https://img.shields.io/badge/GitHub-Athithiyanmr-181717?style=flat-square&logo=github)](https://github.com/Athithiyanmr)

---

## Acknowledgements

- ESA Sentinel-2 Mission
- Microsoft Planetary Computer & STAC API
- Descals, A. et al. (2023). High-resolution global map of closed-canopy coconut palm. Earth System Science Data, 15, 3991-4010.
- Zenodo dataset: [10.5281/zenodo.8128183](https://zenodo.org/records/8128183)

---

## License

MIT License (c) 2026 Athithiyan M R
