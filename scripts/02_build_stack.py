# scripts/02_build_stack.py
#
# Builds the multi-band Sentinel-2 stack for a given AOI + year.
# Optionally adds a 12th band: CanopyHeight_m (WRI/Meta, 2024).
#
# Canopy height usage modes (mutually exclusive, --canopy_tn takes priority):
#
#   --canopy_tn             Auto-clips from  data/raw/canopy_height_tamilnadu.tif
#                           (the TN-wide mosaic from 00_download_canopy_height_tn.py)
#                           Recommended: download once, reuse for every AOI.
#
#   --canopy_height <path>  Use a pre-clipped AOI-specific GeoTIFF directly.
#
# Output bands (11 without canopy, 12 with):
#   B02 B03 B04 B05 B06 B08 B11 B12  NDVI  EVI  NDMI  [CanopyHeight_m]

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling
import geopandas as gpd

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
parser = argparse.ArgumentParser(
    description="Build multi-band stack with vegetation indices and optional canopy height"
)
parser.add_argument("--aoi",  required=True, help="AOI name (matches data/raw/boundaries/{aoi}.shp)")
parser.add_argument("--year", required=True, help="Year to process")

canopy_group = parser.add_mutually_exclusive_group()
canopy_group.add_argument(
    "--canopy_tn",
    action="store_true",
    default=False,
    help=(
        "Auto-clip canopy height from the TN-wide mosaic "
        "(data/raw/canopy_height_tamilnadu.tif). "
        "Run 00_download_canopy_height_tn.py once to create it."
    ),
)
canopy_group.add_argument(
    "--canopy_height",
    default=None,
    help=(
        "Path to a pre-clipped canopy height GeoTIFF for this AOI. "
        "Use --canopy_tn instead when working with multiple TN districts."
    ),
)
args = parser.parse_args()

AOI  = args.aoi
YEAR = args.year

AOI_SHP    = Path(f"data/raw/boundaries/{AOI}.shp")
RAW_DIR    = Path("data/processed/sentinel2_clipped") / AOI / YEAR
OUT_DIR    = Path("data/processed") / AOI
OUT_DIR.mkdir(parents=True, exist_ok=True)

TN_MOSAIC  = Path("data/raw/canopy_height_tamilnadu.tif")

BANDS = ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12"]
BAND_NAMES = ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12",
              "NDVI", "EVI", "NDMI"]

# Resolve canopy height source
CANOPY_PATH = None
if args.canopy_tn:
    if not TN_MOSAIC.exists():
        raise FileNotFoundError(
            f"TN mosaic not found: {TN_MOSAIC}\n"
            "Run the one-time download first:\n"
            "  python scripts/00_download_canopy_height_tn.py"
        )
    CANOPY_PATH = TN_MOSAIC
    print(f"   Canopy height : TN mosaic (auto-clip to {AOI})")
elif args.canopy_height:
    CANOPY_PATH = Path(args.canopy_height)
    if not CANOPY_PATH.exists():
        raise FileNotFoundError(f"Canopy height file not found: {CANOPY_PATH}")
    print(f"   Canopy height : {CANOPY_PATH}")

if CANOPY_PATH is not None:
    BAND_NAMES.append("CanopyHeight_m")

print(f"\nBuilding stack: {AOI} {YEAR}")
print(f"   Bands : {len(BAND_NAMES)} ({', '.join(BAND_NAMES)})")
log.info(f"Starting stack build: AOI={AOI}, year={YEAR}, canopy={CANOPY_PATH}")

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
# CANOPY HEIGHT BAND
#
# When --canopy_tn is used the TN-wide mosaic is clipped to the AOI
# boundary and reprojected to the Sentinel-2 10 m grid on-the-fly.
# No pre-clipped file is needed — the same mosaic works for every
# Tamil Nadu district (Puducherry, Dindigul, Coimbatore, etc.).
#
# Dataset: Meta & WRI High Resolution Canopy Height Maps (2024)
#   Resolution native : 1 m
#   Resampled to      : Sentinel-2 10 m grid (bilinear)
#   Units             : metres above ground (vegetation >= 1 m)
#   Mean abs. error   : 2.8 m  (Tolan et al., 2024, RSE 300, 113888)
#   Temporal coverage : 2018-2020
#   License           : CC BY 4.0
#
# Why it helps coconut mapping:
#   Coconut palms (15-30 m) are spectrally similar to banana and scrub
#   but structurally taller.  The height band acts as a clean separator.
# -----------------------------------------
canopy_arr = None

if CANOPY_PATH is not None:
    print("\n   Loading canopy height layer...")

    with rasterio.open(CANOPY_PATH) as ch_src:

        # --- If using TN mosaic, clip to AOI boundary first ---
        if args.canopy_tn:
            if not AOI_SHP.exists():
                raise FileNotFoundError(
                    f"AOI shapefile not found: {AOI_SHP}\n"
                    "Needed to clip the TN canopy mosaic to your AOI."
                )
            aoi_gdf  = gpd.read_file(AOI_SHP).to_crs(ch_src.crs)
            geoms    = list(aoi_gdf.geometry)
            raw_ch, _ = rio_mask(ch_src, geoms, crop=True, nodata=np.nan, filled=True)
            raw_ch    = raw_ch[0].astype("float32")
            print(f"   Clipped TN mosaic to {AOI} boundary")
        else:
            raw_ch    = ch_src.read(1).astype("float32")
            ch_nodata = ch_src.nodata
            if ch_nodata is not None:
                raw_ch = np.where(raw_ch == ch_nodata, np.nan, raw_ch)

        # --- Reproject / resample to Sentinel-2 reference grid ---
        canopy_arr = np.empty(ref_arr.shape, dtype="float32")
        reproject(
            raw_ch, canopy_arr,
            src_transform=ch_src.transform,
            src_crs=ch_src.crs,
            dst_transform=ref_meta["transform"],
            dst_crs=ref_meta["crs"],
            resampling=Resampling.bilinear,
        )

    # Apply nodata mask + physical clip
    canopy_arr[nodata_mask] = np.nan
    canopy_arr = np.clip(canopy_arr, 0.0, 45.0)   # coconut palms <= 45 m

    ch_valid = canopy_arr[~nodata_mask & ~np.isnan(canopy_arr)]
    p1, p99  = np.nanpercentile(ch_valid, [1, 99])
    print(f"   CanopyHt range : [{p1:.1f}, {p99:.1f}] m  (P1-P99)")
    log.info(f"CanopyHeight P1={p1:.1f}m P99={p99:.1f}m source={CANOPY_PATH}")

# -----------------------------------------
# ASSEMBLE STACK
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
