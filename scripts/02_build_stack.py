import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="stack.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Build multi-band stack with vegetation spectral indices and canopy height")
parser.add_argument("--aoi",  required=True, help="AOI name")
parser.add_argument("--year", required=True, help="Year to process")
parser.add_argument(
    "--canopy_height",
    default=None,
    help=(
        "Path to WRI/Meta canopy height GeoTIFF (1 m resolution). "
        "Dataset: Meta & WRI High Resolution Canopy Height Maps (2018-2020). "
        "GEE: ee.ImageCollection('projects/sat-io/open-datasets/facebook/meta-canopy-height') "
        "AWS S3: s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/ "
        "If omitted the canopy height band is skipped."
    ),
)
args = parser.parse_args()

AOI  = args.aoi
YEAR = args.year
CANOPY_HEIGHT_PATH = Path(args.canopy_height) if args.canopy_height else None

RAW_DIR = Path("data/processed/sentinel2_clipped") / AOI / YEAR
OUT_DIR = Path("data/processed") / AOI
OUT_DIR.mkdir(parents=True, exist_ok=True)

BANDS = ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12"]

# Band index labels for documentation
# 8 raw Sentinel-2 bands + 3 vegetation indices + (optionally) canopy height
BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12",
              "NDVI", "EVI", "NDMI"]
if CANOPY_HEIGHT_PATH is not None:
    BAND_NAMES.append("CanopyHeight_m")

print(f"\nBuilding stack: {AOI} {YEAR}")
if CANOPY_HEIGHT_PATH:
    print(f"   Canopy height layer : {CANOPY_HEIGHT_PATH}")
log.info(f"Starting stack build: AOI={AOI}, year={YEAR}, canopy_height={CANOPY_HEIGHT_PATH}")

# -----------------------------------------
# LOAD REFERENCE BAND (B02)
# -----------------------------------------
ref_path = RAW_DIR / "B02.tif"
if not ref_path.exists():
    raise FileNotFoundError(f"Reference band not found: {ref_path}")

with rasterio.open(ref_path) as ref:
    ref_arr  = ref.read(1).astype("float32")
    ref_meta = ref.meta.copy()
    nodata   = ref.nodata if ref.nodata is not None else 0

log.info(f"Reference band loaded: shape={ref_arr.shape}, CRS={ref_meta['crs']}")

# -----------------------------------------
# LOAD + ALIGN ALL SENTINEL-2 BANDS
# -----------------------------------------
arrays = [ref_arr]

for band in BANDS[1:]:
    path = RAW_DIR / f"{band}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Band file not found: {path}")

    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")

        if arr.shape != ref_arr.shape or src.transform != ref_meta["transform"]:
            print(f"   Resampling {band} to match reference grid")
            log.info(f"Resampling {band}: {arr.shape} -> {ref_arr.shape}")
            res = np.empty(ref_arr.shape, dtype="float32")
            reproject(
                arr, res,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_meta["transform"],
                dst_crs=ref_meta["crs"],
                resampling=Resampling.bilinear,
            )
            arrays.append(res)
        else:
            arrays.append(arr)

    print(f"   Loaded: {band}")

b2, b3, b4, b5, b6, b8, b11, b12 = arrays

# -----------------------------------------
# NODATA MASK
# -----------------------------------------
nodata_mask = (b2 == nodata)
valid_px    = int(np.sum(~nodata_mask))
total_px    = b2.size
print(f"\nValid pixels : {valid_px:,} / {total_px:,} ({100*valid_px/total_px:.1f}%)")
log.info(f"Valid pixels: {valid_px}/{total_px}")

# -----------------------------------------
# REFLECTANCE SCALING
# Sentinel-2 L2A stores DN as uint16 scaled by 10000
# -----------------------------------------
SCALE = 10_000.0

def scale(arr):
    """Convert DN -> reflectance, clip to [0, 1], preserve nodata."""
    scaled = np.where(arr == nodata, np.nan, arr / SCALE)
    return np.clip(scaled, 0.0, 1.0)

b2s, b3s, b4s, b5s, b6s, b8s, b11s, b12s = [scale(b) for b in [b2, b3, b4, b5, b6, b8, b11, b12]]

# -----------------------------------------
# VEGETATION SPECTRAL INDICES
# -----------------------------------------
eps = 1e-6

ndvi = (b8s  - b4s)  / (b8s  + b4s  + eps)
evi  = 2.5 * (b8s - b4s) / (b8s + 6.0 * b4s - 7.5 * b2s + 1.0 + eps)
ndmi = (b8s  - b11s) / (b8s  + b11s + eps)

for name, arr in [("NDVI", ndvi), ("EVI", evi), ("NDMI", ndmi)]:
    pct1, pct99 = np.nanpercentile(arr[~nodata_mask], [1, 99])
    print(f"   {name:6s} range: [{pct1:.3f}, {pct99:.3f}]")
    log.info(f"{name} P1={pct1:.3f}, P99={pct99:.3f}")

# -----------------------------------------
# CANOPY HEIGHT BAND  (WRI / Meta, 1-m resolution)
#
# The canopy height tile is at 1m resolution — loading it all at once
# into RAM causes OOM / SIGKILL.  Instead we keep the source file open
# and use rasterio.band() to let rasterio stream+reproject block-by-block
# directly onto the Sentinel-2 10m grid.  This avoids any large array
# allocation at the native 1m resolution.
# -----------------------------------------
canopy_arr = None

if CANOPY_HEIGHT_PATH is not None:
    if not CANOPY_HEIGHT_PATH.exists():
        raise FileNotFoundError(
            f"Canopy height file not found: {CANOPY_HEIGHT_PATH}\n"
            "Download from:\n"
            "  GEE  : ee.ImageCollection('projects/sat-io/open-datasets/facebook/meta-canopy-height')\n"
            "  AWS  : s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/\n"
            "  Meta : https://ai.meta.com/ai-for-good/datasets/canopy-height-maps/"
        )

    print("\n   Loading canopy height layer (streaming reproject to 10m grid)...")

    canopy_arr = np.empty(ref_arr.shape, dtype="float32")

    with rasterio.open(CANOPY_HEIGHT_PATH) as src:
        ch_nodata = src.nodata

        # reproject streams block-by-block via rasterio.band() —
        # the full 1m array is NEVER loaded into memory
        reproject(
            source=rasterio.band(src, 1),
            destination=canopy_arr,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            resampling=Resampling.bilinear,
        )

    # Replace source nodata with NaN
    if ch_nodata is not None:
        canopy_arr = np.where(canopy_arr == ch_nodata, np.nan, canopy_arr)

    # Apply same nodata mask as Sentinel-2 bands
    canopy_arr[nodata_mask] = np.nan

    # Clip to physically plausible coconut palm heights: 0 – 45 m
    canopy_arr = np.clip(canopy_arr, 0.0, 45.0)

    ch_valid = canopy_arr[~nodata_mask & ~np.isnan(canopy_arr)]
    p1, p99  = np.nanpercentile(ch_valid, [1, 99])
    print(f"   CanopyHt range: [{p1:.1f}, {p99:.1f}] m  (P1–P99)")
    log.info(f"CanopyHeight P1={p1:.1f}m, P99={p99:.1f}m")

# -----------------------------------------
# STACK
# -----------------------------------------
band_arrays = [b2s, b3s, b4s, b5s, b6s, b8s, b11s, b12s, ndvi, evi, ndmi]
if canopy_arr is not None:
    band_arrays.append(canopy_arr)

stack = np.stack(band_arrays).astype("float32")
stack[:, nodata_mask] = np.nan

# -----------------------------------------
# SAVE
# -----------------------------------------
n_bands = len(BAND_NAMES)
ref_meta.update(
    count=n_bands,
    dtype="float32",
    nodata=np.nan,
    compress="lzw",
    tiled=True,
    blockxsize=256,
    blockysize=256,
)

out_path = OUT_DIR / f"stack_{YEAR}.tif"

with rasterio.open(out_path, "w", **ref_meta) as dst:
    dst.write(stack)
    for i, name in enumerate(BAND_NAMES, start=1):
        dst.update_tags(i, name=name)

    if canopy_arr is not None:
        dst.update_tags(
            canopy_height_source="Meta & WRI High Resolution Canopy Height Maps (2024)",
            canopy_height_resolution_native="1m",
            canopy_height_resampled_to="Sentinel-2 10m grid (bilinear)",
            canopy_height_units="metres above ground",
            canopy_height_temporal_coverage="2018-2020",
            canopy_height_citation=(
                "Tolan et al. (2024) Remote Sensing of Environment 300, 113888. "
                "Meta & WRI High Resolution Canopy Height Maps. CC BY 4.0."
            ),
        )

size_mb = out_path.stat().st_size / 1_000_000
print(f"\nStack saved -> {out_path} ({size_mb:.1f} MB)")
print(f"   Bands ({n_bands}): {', '.join(BAND_NAMES)}")
log.info(f"Stack saved: {out_path}, bands={BAND_NAMES}, size={size_mb:.1f}MB")
