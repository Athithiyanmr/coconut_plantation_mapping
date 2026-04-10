# Coconut Plantation Mapping

> **A deep learning pipeline for coconut plantation mapping from Sentinel-2 imagery using UNet semantic segmentation — applied to Tamil Nadu districts, India.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Sentinel-2](https://img.shields.io/badge/Sentinel--2-003DA5?style=flat-square)](https://sentinel.esa.int)
[![Deep Learning](https://img.shields.io/badge/UNet-Deep%20Learning-FF4500?style=flat-square)](https://arxiv.org/abs/1505.04597)
[![Planetary Computer](https://img.shields.io/badge/Microsoft%20Planetary%20Computer-0078D4?style=flat-square&logo=microsoft&logoColor=white)](https://planetarycomputer.microsoft.com)
[![MIT License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## What Is This?

Coconut palms are one of the most important tropical plantation crops, yet accurate large-scale mapping remains challenging due to spectral similarity with other tree crops. Precise mapping of coconut plantations supports agricultural planning, carbon stock estimation, and biodiversity monitoring.

This project builds a **reproducible deep learning pipeline** that maps coconut plantations from Sentinel-2 satellite imagery using a **multi-spectral UNet segmentation model**. The pipeline supports multiple AOIs across Tamil Nadu and accepts either the Descals et al. (2023) global coconut label tiles or manually digitized shapefiles as training labels. An optional WRI/Meta canopy height layer can be added as a structural band to improve discrimination of coconut palms from other vegetation.

---

## Scientific Objective

To learn pixel-level representations of coconut plantation canopy from multi-spectral Sentinel-2 imagery, enriched with vegetation-discriminative spectral indices and optional canopy height, using deep convolutional semantic segmentation.

---

## Area of Interest

The pipeline is AOI-agnostic — pass any district name with a corresponding boundary file.

**Tested AOIs:**
- Coimbatore district — major coconut-growing region in western Tamil Nadu (EPSG:32643)
- Villupuram district — eastern Tamil Nadu (EPSG:32644)

---

## Label Data Sources

The pipeline supports two label modes, auto-detected from the `--label_dir` argument:

| Mode | Input | When to use |
|---|---|---|
| **Descals tiles** | Path to directory of GeoTIFF tiles | Large-area, automated labelling |
| **Manual shapefile** | Path to a `.shp` polygon file | Custom hand-digitized training areas |

**Descals et al. (2023) Global Coconut Palm Layer**
- Paper: [https://essd.copernicus.org/articles/15/3991/2023/](https://essd.copernicus.org/articles/15/3991/2023/)
- Dataset: [https://zenodo.org/records/8128183](https://zenodo.org/records/8128183)
- Binary labels: [0] Not coconut, [1] Coconut palm (closed-canopy)
- Original resolution: 20m → resampled to 10m (nearest-neighbour)

---

## Full Pipeline Workflow

```
STEP 0 (optional)  Download lowest-cloud Sentinel-2 scene (Planetary Computer STAC)
       |
STEP 1             Mosaic & clip scene bands to AOI boundary
       |
STEP 2             Build spectral feature stack
                   [B02, B03, B04, B05, B06, B08, B11, B12, NDVI, EVI, NDMI]
                   + optional Band 12: WRI/Meta Canopy Height (m)
       |
STEP 3             Prepare coconut labels
                   (Descals tiles OR rasterize manual shapefile)
       |
STEP 4             Generate balanced image patches for training
       |
STEP 5             Train UNet-Transformer segmentation model (Focal + Dice loss)
       |
STEP 6             Sliding-window inference over full AOI → probability map
       |
STEP 7             Evaluate segmentation performance (IoU / F1)
```

---

## Input Data

**Sentinel-2 Level-2A bands:**

| Band | Name | Resolution |
|---|---|---|
| B02 | Blue | 10m |
| B03 | Green | 10m |
| B04 | Red | 10m |
| B05 | Red Edge 1 | 20m → 10m |
| B06 | Red Edge 2 | 20m → 10m |
| B08 | Near Infrared | 10m |
| B11 | Shortwave Infrared 1 | 20m → 10m |
| B12 | Shortwave Infrared 2 | 20m → 10m |

**Vegetation spectral indices computed:**

| Index | Formula | Purpose |
|---|---|---|
| NDVI | (B08 − B04) / (B08 + B04) | Vegetation greenness |
| EVI | 2.5 × (B08 − B04) / (B08 + 6×B04 − 7.5×B02 + 1) | Enhanced vegetation sensitivity |
| NDMI | (B08 − B11) / (B08 + B11) | Moisture — separates coconut from dry vegetation |

**Optional structural band:**

| Band | Source | Purpose |
|---|---|---|
| Canopy Height (m) | WRI / Meta High-Resolution Canopy Height Maps (2024), 1m | Separates coconut palms (15–30m) from low-canopy crops |

Canopy height can be downloaded from:
- **GEE:** `ee.ImageCollection('projects/sat-io/open-datasets/facebook/meta-canopy-height')`
- **AWS:** `aws s3 cp --no-sign-request s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/ .`
- **Meta:** https://ai.meta.com/ai-for-good/datasets/canopy-height-maps/

**Final model input (without canopy height):** 11-channel stack
**Final model input (with canopy height):** 12-channel stack

---

## Model Architecture

**UNet-Transformer Semantic Segmentation**
- Encoder-decoder structure with skip connections
- Transformer bottleneck with multi-head self-attention for long-range spatial modelling
- Auto-detects input channels from the stack (11 or 12)
- Pixel-level binary classification output (coconut / non-coconut)

**Loss Function:**
```
Loss = Focal Loss (alpha=0.8, gamma=3.0) + Dice Loss
```
Focal Loss handles class imbalance (coconut is minority class). Dice Loss handles region-level spatial overlap.

---

## Performance

**Primary metrics: IoU and F1**

```
IoU = TP / (TP + FP + FN)
F1  = 2×TP / (2×TP + FP + FN)
```

| IoU Range | Interpretation |
|---|---|
| < 0.40 | Weak |
| 0.40–0.59 | Moderate |
| 0.60–0.70 | Strong |
| > 0.70 | Research-grade |

---

## Project Structure

```
coconut_plantation_mapping/
|
+-- scripts/
|   +-- 00_download_sentinel2_best_per_year.py   # Download best (lowest-cloud) Sentinel-2 scene
|   +-- 01_prepare_aoi_raw.py                    # Clip bands to AOI boundary
|   +-- 02_build_stack.py                        # Build spectral + canopy height stack
|   +-- 03_download_coconut_labels.py            # Descals et al. label preparation
|   +-- 03_rasterize_manual_labels.py            # Manual shapefile → raster label
|   +-- evaluate_iou.py                          # Threshold sweep + IoU / F1 evaluation
|   +-- dl/
|       +-- dataset.py                           # CoconutDataset class
|       +-- unet_model.py                        # Basic UNet architecture
|       +-- unet_transformer.py                  # UNet + Transformer model
|       +-- make_patches.py                      # Patch generation
|       +-- train_unet.py                        # Training script
|       +-- predict_unet.py                      # Sliding-window inference
|
+-- data/           # Satellite imagery, AOI boundaries, labels, patches
+-- models/         # Saved model checkpoints (.pth)
+-- outputs/        # Prediction probability maps and evaluation results
+-- run.py          # End-to-end one-shot pipeline runner
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

**End-to-end (Descals labels, no canopy height):**
```bash
python run.py --year 2025 --aoi villupuram \
  --label_dir data/raw/training/GlobalCoconutLayer_2020_v1-2
```

**End-to-end (manual shapefile labels + canopy height):**
```bash
python run.py --year 2025 --aoi villupuram \
  --label_dir data/raw/training/my_polygons.shp \
  --canopy_height data/raw/canopy_height/canopy_villupuram.tif
```

**Skip download (imagery already on disk):**
```bash
python run.py --year 2025 --aoi villupuram \
  --label_dir data/raw/training/GlobalCoconutLayer_2020_v1-2 \
  --skip_download
```

**Skip training (re-run prediction + evaluation only):**
```bash
python run.py --year 2025 --aoi villupuram \
  --label_dir data/raw/training/GlobalCoconutLayer_2020_v1-2 \
  --skip_download --skip_train
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--year` | required | Target year for Sentinel-2 imagery |
| `--aoi` | required | AOI name (must match boundary file) |
| `--label_dir` | None | Path to Descals directory **or** `.shp` file |
| `--canopy_height` | None | Path to WRI/Meta canopy height `.tif` |
| `--patch` | 64 | Patch size in pixels |
| `--stride` | 32 | Patch stride in pixels |
| `--threshold` | 0.35 | Probability threshold for binary prediction |
| `--all_touched` | False | (Shapefile mode) burn pixels touching polygon edges |
| `--skip_download` | False | Skip Sentinel-2 download step |
| `--skip_train` | False | Skip model training step |

**Step-by-step:**
```bash
# Step 0 — Download Sentinel-2 (best cloud-free scene)
python scripts/00_download_sentinel2_best_per_year.py --year 2025 --aoi villupuram

# Step 1 — Clip bands to AOI
python scripts/01_prepare_aoi_raw.py --year 2025 --aoi villupuram

# Step 2 — Build spectral stack (+ optional canopy height)
python scripts/02_build_stack.py --year 2025 --aoi villupuram
python scripts/02_build_stack.py --year 2025 --aoi villupuram \
  --canopy_height data/raw/canopy_height/canopy_villupuram.tif

# Step 3 — Prepare labels (Descals tiles)
python scripts/03_download_coconut_labels.py --year 2025 --aoi villupuram \
  --label_dir data/raw/training/GlobalCoconutLayer_2020_v1-2

# Step 3 — Prepare labels (manual shapefile)
python scripts/03_rasterize_manual_labels.py --year 2025 --aoi villupuram \
  --shp data/raw/training/my_polygons.shp

# Step 4 — Generate patches
python -m scripts.dl.make_patches --year 2025 --aoi villupuram

# Step 5 — Train
python -m scripts.dl.train_unet --year 2025 --aoi villupuram

# Step 6 — Predict
python -m scripts.dl.predict_unet --year 2025 --aoi villupuram

# Step 7 — Evaluate
python scripts/evaluate_iou.py --year 2025 --aoi villupuram
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

**Athithiyan M R** — Geospatial Data Scientist | Remote Sensing | Climate Analytics

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/athithiyan-m-r-/)
[![GitHub](https://img.shields.io/badge/GitHub-Athithiyanmr-181717?style=flat-square&logo=github)](https://github.com/Athithiyanmr)

---

## Acknowledgements

- ESA Sentinel-2 Mission
- Microsoft Planetary Computer & STAC API
- Descals, A. et al. (2023). High-resolution global map of closed-canopy coconut palm. *Earth System Science Data*, 15, 3991–4010.
- Meta & WRI High Resolution Canopy Height Maps (2024)
- Zenodo dataset: [10.5281/zenodo.8128183](https://zenodo.org/records/8128183)

---

## License

MIT License © 2026 Athithiyan M R
