#!/usr/bin/env python
"""
00b_download_canopy_height.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Auto-download WRI / Meta High-Resolution Canopy Height tiles that intersect
the AOI boundary, merge them, clip to the AOI, and save a single GeoTIFF at:

    data/raw/canopy_height/{aoi}.tif

Data source
-----------
Meta & WRI High Resolution Canopy Height Maps (2024)
  - Resolution  : 1 m
  - Temporal    : 2018-2020
  - Units       : metres above ground (vegetation >= 1 m)
  - License     : CC BY 4.0
  - AWS S3 (no auth required):
      s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/
  - Tile naming : {lon}E_{lat}N.tif   e.g. 078E_011N.tif
    where lon / lat are the FLOOR of the 1-degree cell origin.

Usage
-----
    python scripts/00b_download_canopy_height.py --aoi villupuram
"""

import argparse
import logging
from math import floor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import requests
from tqdm import tqdm

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
    description="Download WRI/Meta canopy height tiles for an AOI"
)
parser.add_argument("--aoi", required=True, help="AOI name (matches data/raw/boundaries/{aoi}.shp)")
parser.add_argument(
    "--buffer_deg",
    type=float,
    default=0.05,
    help="Buffer in degrees around AOI bbox before tile selection (default: 0.05)",
)
args = parser.parse_args()

AOI_NAME   = args.aoi
BUFFER_DEG = args.buffer_deg

BOUNDARY_PATH = Path(f"data/raw/boundaries/{AOI_NAME}.shp")
TILE_CACHE    = Path("data/raw/canopy_height/tiles")
OUT_PATH      = Path(f"data/raw/canopy_height/{AOI_NAME}.tif")

TILE_CACHE.mkdir(parents=True, exist_ok=True)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

S3_BASE = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def tile_name(lon_floor: int, lat_floor: int) -> str:
    """Convert integer lon/lat floor values to tile filename.
    e.g. lon=78, lat=11  ->  '078E_011N.tif'
    """
    lon_str = f"{abs(lon_floor):03d}{'E' if lon_floor >= 0 else 'W'}"
    lat_str = f"{abs(lat_floor):03d}{'N' if lat_floor >= 0 else 'S'}"
    return f"{lon_str}_{lat_str}.tif"


def tiles_for_bbox(minx, miny, maxx, maxy):
    """Return list of tile names covering the bbox."""
    tiles = []
    for lat in range(floor(miny), floor(maxy) + 1):
        for lon in range(floor(minx), floor(maxx) + 1):
            tiles.append(tile_name(lon, lat))
    return tiles


def download_tile(name: str, dest: Path) -> bool:
    """Download a single tile from S3. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"   [skip] {name} already cached")
        log.info(f"Cached: {name}")
        return True

    url = f"{S3_BASE}/{name}"
    print(f"   Downloading: {name}")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            if r.status_code == 404:
                print(f"   [skip] {name} not found on S3 (ocean / no data tile)")
                log.warning(f"404: {name}")
                return False
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            tmp = dest.with_suffix(".tmp")
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=name, leave=False
            ) as bar:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            tmp.rename(dest)
            log.info(f"Downloaded: {name} -> {dest}")
            return True
    except Exception as e:
        log.error(f"Failed {name}: {e}")
        print(f"   [error] {name}: {e}")
        return False


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if not BOUNDARY_PATH.exists():
    raise FileNotFoundError(
        f"AOI boundary not found: {BOUNDARY_PATH}\n"
        f"Make sure data/raw/boundaries/{AOI_NAME}.shp exists."
    )

print(f"\n[canopy height] AOI: {AOI_NAME}")
aoi_gdf = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")
bbox    = aoi_gdf.total_bounds   # minx, miny, maxx, maxy

# Add buffer so edge pixels are covered
bminx = bbox[0] - BUFFER_DEG
bminy = bbox[1] - BUFFER_DEG
bmaxx = bbox[2] + BUFFER_DEG
bmaxy = bbox[3] + BUFFER_DEG

print(f"   AOI bbox (WGS84): {bbox[0]:.3f}W {bbox[1]:.3f}S {bbox[2]:.3f}E {bbox[3]:.3f}N")

tile_names = tiles_for_bbox(bminx, bminy, bmaxx, bmaxy)
print(f"   Tiles to download: {tile_names}")
log.info(f"Tiles required: {tile_names}")

# ---------------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------------
downloaded = []
for t in tile_names:
    dest = TILE_CACHE / t
    ok   = download_tile(t, dest)
    if ok:
        downloaded.append(dest)

if not downloaded:
    raise RuntimeError(
        "No canopy height tiles could be downloaded.\n"
        "Check your internet connection or verify the AOI bbox is over land."
    )

print(f"\n   {len(downloaded)} tile(s) ready")

# ---------------------------------------------------------------------------
# MERGE TILES
# ---------------------------------------------------------------------------
print("   Merging tiles...")
open_files = [rasterio.open(p) for p in downloaded]
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
# CLIP TO AOI
# ---------------------------------------------------------------------------
print("   Clipping to AOI boundary...")
geoms = [g.__geo_interface__ for g in aoi_gdf.geometry]

import tempfile, os
with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_f:
    tmp_path = Path(tmp_f.name)

with rasterio.open(tmp_path, "w", **merged_meta) as tmp_ds:
    tmp_ds.write(merged_arr)

with rasterio.open(tmp_path) as tmp_ds:
    clipped_arr, clipped_transform = rio_mask(
        tmp_ds, geoms, crop=True, nodata=np.nan, filled=True
    )
    clipped_meta = tmp_ds.meta.copy()
    clipped_meta.update({
        "height":    clipped_arr.shape[1],
        "width":     clipped_arr.shape[2],
        "transform": clipped_transform,
        "nodata":    np.nan,
        "dtype":     "float32",
        "compress":  "lzw",
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
    })

tmp_path.unlink(missing_ok=True)

# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------
with rasterio.open(OUT_PATH, "w", **clipped_meta) as dst:
    dst.write(clipped_arr.astype("float32"))
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
