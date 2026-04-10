#!/usr/bin/env python
"""
00b_download_canopy_height.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Select WRI/Meta canopy height tiles that intersect the AOI from a local
tiles folder, merge them, clip to the AOI boundary, and save:

    data/raw/canopy_height/{aoi}.tif

Uses GDAL VRT (virtual mosaic) + gdalwarp via Python osgeo API so tiles
are processed block-by-block on disk — no full in-memory load.

Data source
-----------
Meta & WRI High Resolution Canopy Height Maps (2024)
  - Resolution  : 1 m
  - Temporal    : 2018-2020
  - Units       : metres above ground
  - License     : CC BY 4.0

Usage
-----
    python scripts/00b_download_canopy_height.py \
        --aoi villupuram \
        --tiles_dir /path/to/your/canopy_height_tiles
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import rasterio
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
parser = argparse.ArgumentParser()
parser.add_argument("--aoi",       required=True)
parser.add_argument("--tiles_dir", required=True)
args = parser.parse_args()

AOI_NAME  = args.aoi
TILES_DIR = Path(args.tiles_dir)

BOUNDARY_PATH = Path(f"data/raw/boundaries/{AOI_NAME}.shp")
OUT_PATH      = Path(f"data/raw/canopy_height/{AOI_NAME}.tif")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# VALIDATE
# ---------------------------------------------------------------------------
if not BOUNDARY_PATH.exists():
    raise FileNotFoundError(f"AOI boundary not found: {BOUNDARY_PATH}")
if not TILES_DIR.exists():
    raise FileNotFoundError(f"Tiles directory not found: {TILES_DIR}")

# ---------------------------------------------------------------------------
# LOAD AOI
# ---------------------------------------------------------------------------
print(f"\n[canopy height] AOI: {AOI_NAME}")
aoi_gdf  = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")
aoi_bbox = box(*aoi_gdf.total_bounds)
bbox     = aoi_gdf.total_bounds
print(f"   AOI bbox (WGS84): {bbox[0]:.4f}E  {bbox[1]:.4f}N  {bbox[2]:.4f}E  {bbox[3]:.4f}N")

# ---------------------------------------------------------------------------
# FIND INTERSECTING TILES
# ---------------------------------------------------------------------------
all_tiles = sorted(TILES_DIR.rglob("*.tif")) + sorted(TILES_DIR.rglob("*.tiff"))
if not all_tiles:
    raise FileNotFoundError(f"No .tif files found in: {TILES_DIR}")

print(f"   Scanning {len(all_tiles)} tiles in {TILES_DIR} ...")

matching = []
for tile_path in all_tiles:
    try:
        with rasterio.open(tile_path) as src:
            tb = (transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                  if str(src.crs) != "EPSG:4326" else tuple(src.bounds))
            if aoi_bbox.intersects(box(*tb)):
                matching.append(str(tile_path))
    except Exception as e:
        log.warning(f"Could not read {tile_path.name}: {e}")

if not matching:
    raise RuntimeError(f"No tiles intersect the AOI bbox: {bbox}")

print(f"   Found {len(matching)} intersecting tile(s):")
for m in matching:
    print(f"     {Path(m).name}")
log.info(f"Matching tiles: {[Path(p).name for p in matching]}")

# ---------------------------------------------------------------------------
# GET AOI BOUNDS IN TILE CRS
# ---------------------------------------------------------------------------
with rasterio.open(matching[0]) as src:
    tile_crs  = src.crs
    src_nodata = src.nodata  # could be None for these tiles

aoi_tile = aoi_gdf.to_crs(tile_crs)
minx, miny, maxx, maxy = aoi_tile.total_bounds
print(f"   AOI bbox in tile CRS: {minx:.2f} {miny:.2f} {maxx:.2f} {maxy:.2f}")

# ---------------------------------------------------------------------------
# MERGE via GDAL VRT + warp via osgeo.gdal Python API
# This avoids any PATH / shell issues with gdal_merge.py
# ---------------------------------------------------------------------------
try:
    from osgeo import gdal
except ImportError:
    print("ERROR: osgeo.gdal not available. Install with: conda install gdal")
    sys.exit(1)

gdal.UseExceptions()

with tempfile.TemporaryDirectory() as tmpdir:
    vrt_path    = str(Path(tmpdir) / "mosaic.vrt")
    merged_path = str(Path(tmpdir) / "merged.tif")

    # Step 1 — build VRT mosaic (zero memory, just metadata)
    print("   Building VRT mosaic...")
    vrt_options = gdal.BuildVRTOptions(resampleAlg="nearest")
    vrt_ds = gdal.BuildVRT(vrt_path, matching, options=vrt_options)
    if vrt_ds is None:
        raise RuntimeError("gdal.BuildVRT failed — check tile paths")
    vrt_ds.FlushCache()
    vrt_ds = None

    # Step 2 — warp VRT -> clipped Float32 GeoTIFF
    print("   Warping (clip + Float32 convert)...")
    warp_options = gdal.WarpOptions(
        outputBounds=(minx, miny, maxx, maxy),
        outputType=gdal.GDT_Float32,
        dstNodata=float("nan"),
        creationOptions=[
            "COMPRESS=LZW",
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
        ],
        multithread=True,
        warpMemoryLimit=256,   # MB per thread — keeps RAM low
        callback=gdal.TermProgress_nocb,
    )
    out_ds = gdal.Warp(str(OUT_PATH), vrt_path, options=warp_options)
    if out_ds is None:
        raise RuntimeError("gdal.Warp failed")
    out_ds.FlushCache()
    out_ds = None

# ---------------------------------------------------------------------------
# TAG OUTPUT
# ---------------------------------------------------------------------------
with rasterio.open(OUT_PATH, "r+") as dst:
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
