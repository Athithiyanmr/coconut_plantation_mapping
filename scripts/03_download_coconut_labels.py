
# scripts/03_download_coconut_labels.py
#
# Download Descals et al. (2023) global coconut palm tiles,
# clip to AOI, save RAW labels, and optionally align to Sentinel-2 stack.

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge as rasterio_merge

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="label_coconut.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(
    description="Prepare coconut plantation labels"
)
parser.add_argument("--year", required=True)
parser.add_argument("--aoi", required=True)
parser.add_argument("--label_dir", required=True)
parser.add_argument("--bbox", nargs=4, type=float,
                    default=[76.5, 10.8, 77.3, 11.5])
args = parser.parse_args()

YEAR = args.year
AOI_NAME = args.aoi
LABEL_DIR = Path(args.label_dir)
BBOX = args.bbox

AOI_PATH = f"data/raw/boundaries/{AOI_NAME}.shp"
STACK = f"data/processed/{AOI_NAME}/stack_{YEAR}.tif"

OUT_DIR = Path("data/processed/training")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUT_DIR / f"labels_coconut_raw_{YEAR}_{AOI_NAME}.tif"
ALIGNED_OUT = OUT_DIR / f"labels_coconut_{YEAR}_{AOI_NAME}.tif"

# -----------------------------------------
# STEP 1 -- LOAD AOI
# -----------------------------------------
print("\nLoading AOI...")
if Path(AOI_PATH).exists():
    aoi = gpd.read_file(AOI_PATH).to_crs("EPSG:4326")
    aoi_geom = aoi.geometry.iloc[0]
    bbox = aoi.total_bounds
else:
    from shapely.geometry import box
    aoi_geom = box(*BBOX)
    aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326")
    bbox = BBOX

print(f"AOI loaded. BBox: {bbox}")

# -----------------------------------------
# STEP 2 -- FIND LABEL TILES
# -----------------------------------------
print("\nSearching coconut tiles...")
tif_files = sorted(LABEL_DIR.glob("*.tif")) or sorted(LABEL_DIR.glob("**/*.tif"))

if not tif_files:
    raise FileNotFoundError("No coconut GeoTIFF tiles found.")

print(f"Found {len(tif_files)} tiles")

# -----------------------------------------
# STEP 3 -- FILTER INTERSECTING TILES
# -----------------------------------------
print("\nFiltering intersecting tiles...")
from shapely.geometry import box as shapely_box

intersecting = []
for tif in tif_files:
    with rasterio.open(tif) as src:
        bounds = src.bounds
        tile_box = shapely_box(bounds.left, bounds.bottom,
                               bounds.right, bounds.top)
        if tile_box.intersects(aoi_geom):
            intersecting.append(tif)

if not intersecting:
    raise RuntimeError("No tiles intersect AOI")

print(f"{len(intersecting)} tiles intersect AOI")

# -----------------------------------------
# STEP 4 -- MERGE + CLIP
# -----------------------------------------
print("\nMerging and clipping...")

if len(intersecting) == 1:
    src_file = intersecting[0]
    with rasterio.open(src_file) as src:
        clipped, clipped_transform = mask(
            src, [aoi_geom], crop=True, nodata=0, all_touched=True
        )
        clip_meta = src.meta.copy()
else:
    srcs = [rasterio.open(f) for f in intersecting]
    mosaic, mosaic_transform = rasterio_merge(srcs, nodata=0)

    mosaic_meta = srcs[0].meta.copy()
    mosaic_meta.update(
        transform=mosaic_transform,
        height=mosaic.shape[1],
        width=mosaic.shape[2],
    )

    for s in srcs:
        s.close()

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**mosaic_meta) as tmp:
            tmp.write(mosaic)
            clipped, clipped_transform = mask(
                tmp, [aoi_geom], crop=True, nodata=0, all_touched=True
            )

    clip_meta = mosaic_meta.copy()

clip_meta.update(
    transform=clipped_transform,
    height=clipped.shape[1],
    width=clipped.shape[2],
)

label_data = clipped[0]

print(f"Clipped shape: {label_data.shape}")

# -----------------------------------------
# STEP 5 -- SAVE RAW LABELS
# -----------------------------------------
print("\nSaving RAW labels...")

label_data = np.where(np.isnan(label_data), 0, label_data).astype("uint8")

raw_meta = clip_meta.copy()
raw_meta.update(
    count=1,
    dtype="uint8",
    compress="lzw",
    nodata=0
)

with rasterio.open(RAW_OUT, "w", **raw_meta) as dst:
    dst.write(label_data, 1)

print(f"Raw labels saved → {RAW_OUT}")

# -----------------------------------------
# STEP 6 -- OPTIONAL ALIGNMENT
# -----------------------------------------
if Path(STACK).exists():
    print("\nStack found → aligning labels...")

    with rasterio.open(STACK) as ref:
        ref_meta = ref.meta.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_height = ref.height
        ref_width = ref.width

    label_aligned = np.zeros((ref_height, ref_width), dtype="float32")

    reproject(
        source=label_data,
        destination=label_aligned,
        src_transform=clipped_transform,
        src_crs=clip_meta["crs"],
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,
    )

    # 🔥 FIX: remove NaN + enforce uint8
    label_aligned = np.where(np.isnan(label_aligned), 0, label_aligned).astype("uint8")

    print("Saving aligned labels...")

    out_meta = ref_meta.copy()
    out_meta.update(
        count=1,
        dtype="uint8",
        compress="lzw",
        nodata=0   # ✅ critical fix
    )

    with rasterio.open(ALIGNED_OUT, "w", **out_meta) as dst:
        dst.write(label_aligned, 1)

    print(f"Aligned labels saved → {ALIGNED_OUT}")

    # Debug check
    print("Unique values:", np.unique(label_aligned))

else:
    print("\nNo stack found → skipped alignment")

print("\n✅ Processing complete")
