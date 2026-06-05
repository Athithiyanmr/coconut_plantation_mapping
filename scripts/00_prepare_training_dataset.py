# scripts/00_prepare_training_dataset.py
#
# Prepares final training dataset GeoJSON from a verified coconut polygon file.
# - Cleans verification_status ('pending' -> 'yes')
# - Maps class: yes -> 1 (coconut), no -> 0 (not-coconut)
# - Clips to district AOI boundary
# - Saves to data/raw/training/<aoi>.geojson and data/raw/boundaries/<aoi>.shp
#
# Usage:
#   python scripts/00_prepare_training_dataset.py \
#       --verified /path/to/<district>_verified.geojson \
#       --aoi      /path/to/<district>_boundary.geojson \
#       --name     dindigul
#
# Output:
#   data/raw/training/<name>_verified_final.geojson   <- labeled training polygons
#   data/raw/boundaries/<name>.shp                    <- AOI boundary for pipeline
#
# Class legend:
#   1  = confirmed coconut plantation
#   0  = confirmed NOT coconut (background / negative sample)
#   (pixels not covered by any polygon = 255 IGNORE in rasterization step)

import argparse
import logging
from pathlib import Path

import geopandas as gpd

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="prepare_dataset.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(
    description="Prepare training dataset GeoJSON with class labels and AOI clip"
)
parser.add_argument("--verified", required=True,
                    help="Path to verified coconut GeoJSON (with verification_status column)")
parser.add_argument("--aoi", required=True,
                    help="Path to district boundary GeoJSON (single dissolved polygon)")
parser.add_argument("--name", required=True,
                    help="District name used for output filenames (e.g. 'villupuram', 'dindigul')")
args = parser.parse_args()

VERIFIED_PATH = Path(args.verified)
AOI_PATH      = Path(args.aoi)
NAME          = args.name.lower().strip()

TRAINING_DIR   = Path("data/raw/training")
BOUNDARIES_DIR = Path("data/raw/boundaries")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
BOUNDARIES_DIR.mkdir(parents=True, exist_ok=True)

OUT_LABELS = TRAINING_DIR   / f"{NAME}_verified_final.geojson"
OUT_AOI    = BOUNDARIES_DIR / f"{NAME}.shp"

print(f"\n{'='*52}")
print(f"Training Dataset Preparation")
print(f"   District  : {NAME}")
print(f"   Verified  : {VERIFIED_PATH}")
print(f"   AOI       : {AOI_PATH}")
print(f"{'='*52}")

# -----------------------------------------
# STEP 1 -- LOAD & CLEAN VERIFIED DATA
# -----------------------------------------
print("\nLoading verified polygon data...")
data = gpd.read_file(VERIFIED_PATH)

required_cols = ["verification_status", "geometry"]
for col in required_cols:
    if col not in data.columns:
        raise ValueError(
            f"Required column '{col}' not found.\n"
            f"Available columns: {data.columns.tolist()}"
        )

# Keep only useful columns (add extras if present)
keep_cols = [c for c in ["area_ha", "district", "verification_status", "geometry"] if c in data.columns]
data = data[keep_cols]

print(f"   Loaded     : {len(data)} polygons")
print(f"   Columns    : {data.columns.tolist()}")
print(f"   CRS        : {data.crs}")

# -----------------------------------------
# STEP 2 -- MAP LABELS -> class column
#
#   verification_status = 'yes'     -> class = 1 (coconut)
#   verification_status = 'no'      -> class = 0 (not-coconut)
#   verification_status = 'pending' -> treated as 'yes' (assumed verified)
# -----------------------------------------
print("\nMapping verification_status to class labels...")

before_counts = data["verification_status"].value_counts().to_dict()
print(f"   Before cleanup : {before_counts}")

# Replace 'pending' with 'yes' — field-verified but not yet updated
data["verification_status"] = data["verification_status"].replace("pending", "yes")

after_counts = data["verification_status"].value_counts().to_dict()
print(f"   After cleanup  : {after_counts}")

# Map to binary class
data["class"] = data["verification_status"].map({"yes": 1, "no": 0})

# Drop any rows with unmapped class (unexpected status values)
unmapped = data["class"].isna().sum()
if unmapped > 0:
    unique_statuses = data[data["class"].isna()]["verification_status"].unique()
    print(f"   WARNING: {unmapped} rows have unrecognized status: {unique_statuses}")
    print(f"   These will be DROPPED from training data.")
    log.warning(f"Dropped {unmapped} unmapped rows with status: {unique_statuses}")
    data = data.dropna(subset=["class"])

data["class"] = data["class"].astype(int)

n_pos = int((data["class"] == 1).sum())
n_neg = int((data["class"] == 0).sum())
print(f"\n   class=1 (coconut)     : {n_pos} polygons")
print(f"   class=0 (not-coconut) : {n_neg} polygons")

if n_pos == 0:
    raise RuntimeError("No coconut (class=1) polygons found. Check your verification_status values.")
if n_neg == 0:
    print("   WARNING: No negative (class=0) polygons found.")
    print("   The model will only see positive samples — consider adding background polygons.")
    log.warning("No negative polygons (class=0) in dataset.")

log.info(f"Class distribution after mapping: class=1: {n_pos}, class=0: {n_neg}")

# -----------------------------------------
# STEP 3 -- LOAD AOI BOUNDARY
# -----------------------------------------
print(f"\nLoading AOI boundary: {AOI_PATH}")
aoi = gpd.read_file(AOI_PATH)

if aoi.crs is None:
    raise ValueError("AOI GeoJSON has no CRS. Set it to EPSG:4326 before using.")

# Dissolve to single polygon in case of multiple features
if len(aoi) > 1:
    print(f"   AOI has {len(aoi)} features — dissolving to single polygon...")
    aoi = aoi.dissolve().reset_index(drop=True)
    log.info(f"AOI dissolved from {len(aoi)} features to 1")

print(f"   AOI features : {len(aoi)}")
print(f"   AOI CRS      : {aoi.crs}")
print(f"   AOI bounds   : {[round(v, 4) for v in aoi.total_bounds]}")

# Reproject AOI to match data CRS for clipping
aoi_reproj = aoi.to_crs(data.crs)

# -----------------------------------------
# STEP 4 -- CLIP LABELS TO AOI
# -----------------------------------------
print("\nClipping labels to AOI boundary...")
before_clip = len(data)
data_clipped = gpd.clip(data, aoi_reproj)
data_clipped = data_clipped[data_clipped.geometry.notnull() & ~data_clipped.geometry.is_empty]
after_clip = len(data_clipped)

dropped_clip = before_clip - after_clip
if dropped_clip > 0:
    print(f"   Dropped {dropped_clip} polygons outside AOI boundary")
    log.info(f"Clip dropped {dropped_clip} polygons outside AOI")

print(f"   Polygons after clip : {after_clip}")
print(f"   class=1 : {int((data_clipped['class'] == 1).sum())}")
print(f"   class=0 : {int((data_clipped['class'] == 0).sum())}")

if data_clipped.empty:
    raise RuntimeError(
        "No polygons remain after clipping to AOI.\n"
        "Check that your verified GeoJSON and AOI are for the same area."
    )

# Tag the district name
data_clipped["district_name"] = NAME
data_clipped.reset_index(drop=True, inplace=True)

# -----------------------------------------
# STEP 5 -- SAVE OUTPUTS
# -----------------------------------------
print(f"\nSaving outputs...")

# Save training labels
data_clipped.to_file(OUT_LABELS, driver="GeoJSON")
size_labels = OUT_LABELS.stat().st_size / 1000
print(f"   Labels saved  -> {OUT_LABELS}  ({size_labels:.1f} KB)")
log.info(f"Labels saved: {OUT_LABELS}")

# Save AOI as shapefile for pipeline compatibility (01_prepare_aoi_raw.py expects .shp)
aoi_save = aoi.to_crs(data.crs)
aoi_save.to_file(OUT_AOI)
print(f"   AOI saved     -> {OUT_AOI}")
log.info(f"AOI saved: {OUT_AOI}")

# -----------------------------------------
# SUMMARY
# -----------------------------------------
print(f"\n{'='*52}")
print(f"Dataset preparation complete for: {NAME}")
print(f"   Total polygons  : {after_clip}")
print(f"   Coconut  (1)    : {int((data_clipped['class']==1).sum())}")
print(f"   Not-coco (0)    : {int((data_clipped['class']==0).sum())}")
print(f"\nNext steps:")
print(f"  1. Download Sentinel-2 imagery:")
print(f"       python scripts/00_download_sentinel2_best_per_year.py --aoi {NAME} --year 2024")
print(f"  2. Preprocess & clip to AOI:")
print(f"       python scripts/01_prepare_aoi_raw.py --aoi {NAME} --year 2024")
print(f"  3. Build raster stack:")
print(f"       python scripts/02_build_stack.py --aoi {NAME} --year 2024")
print(f"  4. Rasterize training labels:")
print(f"       python scripts/03_rasterize_manual_labels.py --aoi {NAME} --year 2024 \\")
print(f"           --shp data/raw/training/{NAME}_verified_final.geojson")
print(f"{'='*52}")
