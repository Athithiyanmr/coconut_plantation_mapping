import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="preprocess.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--aoi", required=True)
parser.add_argument("--year", required=True)
parser.add_argument("--target_crs", default="EPSG:32643")
args = parser.parse_args()

AOI_PATH   = f"data/raw/boundaries/{args.aoi}.shp"
RAW_DIR    = Path("data/raw/sentinel2") / args.aoi / args.year
OUT_DIR    = Path("data/processed/sentinel2_clipped") / args.aoi / args.year

BANDS      = ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12", "SCL"]
TARGET_CRS = args.target_crs

# Sentinel-2 L2A: 0 is the true nodata sentinel
# We keep it as the nodata value in the GeoTIFF metadata so rasterio
# and downstream scripts can detect it, but we do NOT use it as the
# merge fill value so valid dark pixels (water, shadow) are preserved.
S2_NODATA = 0

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nAOI  : {args.aoi}")
print(f"Year : {args.year}")
print(f"CRS  : {TARGET_CRS}")

# -----------------------------------------
# LOAD AOI
# -----------------------------------------
aoi = gpd.read_file(AOI_PATH)
if aoi.crs is None:
    raise ValueError("AOI has no CRS")

# -----------------------------------------
# REPROJECT FUNCTION
# -----------------------------------------
def reproject_tile(src, target_crs):
    transform, width, height = calculate_default_transform(
        src.crs, target_crs,
        src.width, src.height,
        *src.bounds
    )

    kwargs = src.meta.copy()
    kwargs.update({
        "crs": target_crs,
        "transform": transform,
        "width": width,
        "height": height,
    })

    data = src.read()
    dst_array = np.zeros((data.shape[0], height, width), dtype=data.dtype)

    for i in range(data.shape[0]):
        reproject(
            source=data[i],
            destination=dst_array[i],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=S2_NODATA,
            dst_nodata=S2_NODATA,
        )

    memfile = rasterio.io.MemoryFile()
    dataset = memfile.open(**kwargs)
    dataset.write(dst_array)
    return dataset

# -----------------------------------------
# PROCESS BANDS
# -----------------------------------------
for band in BANDS:
    band_files = sorted(RAW_DIR.glob(f"**/*{band}.tif"))

    if not band_files:
        print(f"\u274c No files for {band}")
        continue

    print(f"\nProcessing {band} ({len(band_files)} tiles)")

    reprojected_srcs = []
    for f in band_files:
        src = rasterio.open(f)
        if str(src.crs) != TARGET_CRS:
            print(f"   Reprojecting {f.name}")
            reproj = reproject_tile(src, TARGET_CRS)
            reprojected_srcs.append(reproj)
        else:
            reprojected_srcs.append(src)

    # -----------------------------------------
    # MERGE
    # FIX: use nodata=S2_NODATA so tiles are stitched correctly;
    #      valid dark pixels (DN > 0) are never masked out.
    # -----------------------------------------
    mosaic, transform = merge(reprojected_srcs, nodata=S2_NODATA)

    # Report valid pixel coverage after merge
    valid_px = int(np.sum(mosaic[0] != S2_NODATA))
    total_px = mosaic[0].size
    print(f"   Mosaic valid   : {valid_px:,} / {total_px:,} ({100*valid_px/total_px:.1f}%)")
    log.info(f"{band} mosaic valid={valid_px}/{total_px}")

    meta = reprojected_srcs[0].meta.copy()
    meta.update({
        "transform": transform,
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "nodata": S2_NODATA,
        "compress": "lzw",
    })

    # -----------------------------------------
    # CLIP
    # FIX: same nodata value; crop=True, filled=True so outside-AOI
    #      pixels are set to S2_NODATA (not 0 overwriting valid data)
    # -----------------------------------------
    aoi_proj = aoi.to_crs(meta["crs"])
    geoms    = list(aoi_proj.geometry)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**meta) as tmp:
            tmp.write(mosaic)

            clipped, clipped_transform = mask(
                tmp,
                geoms,
                crop=True,
                nodata=S2_NODATA,
                filled=True,
            )

    # Report valid pixel coverage after clip
    valid_clip = int(np.sum(clipped[0] != S2_NODATA))
    total_clip = clipped[0].size
    print(f"   Clipped valid  : {valid_clip:,} / {total_clip:,} ({100*valid_clip/total_clip:.1f}%)")
    log.info(f"{band} clipped valid={valid_clip}/{total_clip}")

    meta.update({
        "transform": clipped_transform,
        "height": clipped.shape[1],
        "width": clipped.shape[2],
    })

    # -----------------------------------------
    # SAVE
    # -----------------------------------------
    out_path = OUT_DIR / f"{band}.tif"
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(clipped)

    print(f"\u2714 Saved {band}")

# -----------------------------------------
# DONE
# -----------------------------------------
print("\n\u2705 Preprocessing complete")
print("Output:", OUT_DIR)
