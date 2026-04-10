#!/usr/bin/env python
"""
00b_download_canopy_height.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Select WRI/Meta canopy height tiles that intersect the AOI from a local
tiles folder, merge them, clip to the AOI boundary, and save:

    data/raw/canopy_height/{aoi}.tif

You must have already downloaded the Meta/WRI canopy height tiles.
The script accepts any flat folder of .tif files — it reads each file's
geographic extent and picks only those that overlap the AOI.

Data source
-----------
Meta & WRI High Resolution Canopy Height Maps (2024)
  - Resolution  : 1 m
  - Temporal    : 2018-2020
  - Units       : metres above ground (vegetation >= 1 m)
  - License     : CC BY 4.0
  - AWS S3:  s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/chm/
  - Tile index:  tiles.geojson (QuadKey named .tif files inside chm/ subfolder)

Usage
-----
    python scripts/00b_download_canopy_height.py \
        --aoi villupuram \
        --tiles_dir /path/to/your/canopy_height_tiles
"""

import argparse
import logging
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_bounds
from shapely.geometry import box

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    filename="canopy_download.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ARGS
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Select & clip WRI/Meta canopy height tiles for an AOI from a local folder"
)
parser.add_argument(
    "--aoi", required=True,
    help="AOI name (matches data/raw/boundaries/{aoi}.shp)"
)
parser.add_argument(
    "--tiles_dir", required=True,
    help="Path to local folder containing Meta/WRI canopy height .tif tiles"
)
args = parser.parse_args()

AOI_NAME  = args.aoi
TILES_DIR = Path(args.tiles_dir)

BOUNDARY_PATH = Path(f"data/raw/boundaries/{AOI_NAME}.shp")
OUT_PATH      = Path(f"data/raw/canopy_height/{AOI_NAME}.tif")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# VALIDATE INPUTS
# ---------------------------------------------------------------------------
if not BOUNDARY_PATH.exists():
    raise FileNotFoundError(
        f"AOI boundary not found: {BOUNDARY_PATH}\n"
        f"Expected: data/raw/boundaries/{AOI_NAME}.shp"
    )

if not TILES_DIR.exists():
    raise FileNotFoundError(
        f"Tiles directory not found: {TILES_DIR}\n"
        "Pass the folder containing your .tif canopy height tiles with --tiles_dir"
    )

# ---------------------------------------------------------------------------
# LOAD AOI
# ---------------------------------------------------------------------------
print(f"\n[canopy height] AOI: {AOI_NAME}")
aoi_gdf  = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")
aoi_bbox = box(*aoi_gdf.total_bounds)
bbox     = aoi_gdf.total_bounds   # minx, miny, maxx, maxy

print(f"   AOI bbox (WGS84): "
      f"{bbox[0]:.4f}E  {bbox[1]:.4f}N  {bbox[2]:.4f}E  {bbox[3]:.4f}N")

# ---------------------------------------------------------------------------
# SCAN TILES FOLDER — find .tif files whose extent intersects AOI
# ---------------------------------------------------------------------------
all_tiles = sorted(TILES_DIR.rglob("*.tif")) + sorted(TILES_DIR.rglob("*.tiff"))

if not all_tiles:
    raise FileNotFoundError(
        f"No .tif files found in: {TILES_DIR}\n"
        "Make sure you're pointing to the folder containing the canopy height tiles."
    )

print(f"   Scanning {len(all_tiles)} tiles in {TILES_DIR} ...")

matching = []
for tile_path in all_tiles:
    try:
        with rasterio.open(tile_path) as src:
            if str(src.crs) != "EPSG:4326":
                tb = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            else:
                tb = src.bounds
            tile_box = box(tb[0], tb[1], tb[2], tb[3])
            if aoi_bbox.intersects(tile_box):
                matching.append(tile_path)
    except Exception as e:
        log.warning(f"Could not read {tile_path.name}: {e}")
        continue

if not matching:
    raise RuntimeError(
        f"No tiles in {TILES_DIR} intersect the AOI bbox.\n"
        f"AOI bbox: {bbox}\n"
        "Double-check that your tiles cover Tamil Nadu / India."
    )

print(f"   Found {len(matching)} intersecting tile(s):")
for m in matching:
    print(f"     {m.name}")
log.info(f"Matching tiles: {[p.name for p in matching]}")

# ---------------------------------------------------------------------------
# MERGE TILES
# ---------------------------------------------------------------------------
print("   Merging tiles...")
open_files = [rasterio.open(p) for p in matching]
merged_arr, merged_transform = merge(open_files)
merged_meta = open_files[0].meta.copy()
merged_meta.update({
    "driver":    "GTiff",
    "height":    merged_arr.shape[1],
    "width":     merged_arr.shape[2],
    "transform": merged_transform,
    "crs":       open_files[0].crs,
    "compress":  "lzw",
})
for f in open_files:
    f.close()

# ---------------------------------------------------------------------------
# REPROJECT AOI GEOMETRIES to tile CRS (for clipping)
# ---------------------------------------------------------------------------
tile_crs = merged_meta["crs"]
aoi_clip = aoi_gdf.to_crs(tile_crs)
geoms    = [g.__geo_interface__ for g in aoi_clip.geometry]

# ---------------------------------------------------------------------------
# CLIP TO AOI
# Write merged to a temp file, then mask.
# Use filled=False to get a masked array back — avoids the
# "Cannot convert fill_value nan to dtype uint8" error when the
# source tiles are integer (uint8) rasters.
# After masking, convert to float32 and fill masked pixels with NaN.
# ---------------------------------------------------------------------------
print("   Clipping to AOI boundary...")
with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_f:
    tmp_path = Path(tmp_f.name)

with rasterio.open(tmp_path, "w", **merged_meta) as tmp_ds:
    tmp_ds.write(merged_arr)

with rasterio.open(tmp_path) as tmp_ds:
    # filled=False returns a numpy masked array — safe for any dtype
    clipped_masked, clipped_transform = rio_mask(
        tmp_ds, geoms, crop=True, filled=False
    )
    # Convert to float32 FIRST, then fill masked pixels with NaN
    clipped_arr = clipped_masked.astype("float32")
    clipped_arr = clipped_arr.filled(np.nan)

    clipped_meta = tmp_ds.meta.copy()
    clipped_meta.update({
        "height":     clipped_arr.shape[1],
        "width":      clipped_arr.shape[2],
        "transform":  clipped_transform,
        "nodata":     np.nan,
        "dtype":      "float32",
        "compress":   "lzw",
        "tiled":      True,
        "blockxsize": 256,
        "blockysize": 256,
    })

tmp_path.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
with rasterio.open(OUT_PATH, "w", **clipped_meta) as dst:
    dst.write(clipped_arr)
    dst.update_tags(
        source="Meta & WRI High Resolution Canopy Height Maps (2024)",
        resolution_native="1m",
        units="metres above ground",
        temporal_coverage="2018-2020",
        license="CC BY 4.0",
        aoi=AOI_NAME,
    )

size_mb = OUT_PATH.stat().st_size / 1_000_000
print(f"\n   Saved -> {OUT_PATH} ({size_mb:.1f} MB)")
log.info(f"Saved: {OUT_PATH} ({size_mb:.1f} MB)")
print("[canopy height] Done")
