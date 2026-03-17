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
import warnings

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
parser = argparse.ArgumentParser(description="Merge, clip and reproject Sentinel-2 tiles to AOI")
parser.add_argument("--aoi",    required=True, help="AOI name (without path/extension)")
parser.add_argument("--year",   required=True, help="Year to process")
parser.add_argument("--target_crs", default="EPSG:32644", help="Target CRS for output (default: UTM Zone 44N for Chennai)")
args = parser.parse_args()

AOI_PATH = f"data/raw/boundaries/{args.aoi}.shp"
RAW_DIR  = Path("data/raw/sentinel2")         / args.aoi / args.year
OUT_DIR  = Path("data/processed/sentinel2_clipped") / args.aoi / args.year   # ✅ moved to processed/

# SCL added for cloud masking support
BANDS       = ["B02", "B03", "B04", "B08", "B11", "SCL"]
TARGET_CRS  = args.target_crs

OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n📍 AOI  : {args.aoi}")
print(f"📅 Year : {args.year}")
print(f"🗺  CRS  : {TARGET_CRS}")
log.info(f"Starting preprocess: AOI={args.aoi}, year={args.year}, crs={TARGET_CRS}")

# -----------------------------------------
# LOAD AOI
# -----------------------------------------
aoi = gpd.read_file(AOI_PATH)
if aoi.crs is None:
    raise ValueError("AOI shapefile has no CRS defined. Set it before running.")
log.info(f"AOI loaded: {AOI_PATH}, CRS={aoi.crs}")

# -----------------------------------------
# HELPER: Reproject raster array to target CRS
# -----------------------------------------
def reproject_to_target(data, src_meta, target_crs):
    transform, width, height = calculate_default_transform(
        src_meta["crs"], target_crs,
        src_meta["width"], src_meta["height"],
        *rasterio.transform.array_bounds(src_meta["height"], src_meta["width"], src_meta["transform"])
    )
    kwargs = src_meta.copy()
    kwargs.update(crs=target_crs, transform=transform, width=width, height=height)

    reprojected = np.zeros((data.shape[0], height, width), dtype=data.dtype)
    for i in range(data.shape[0]):
        reproject(
            source=data[i],
            destination=reprojected[i],
            src_transform=src_meta["transform"],
            src_crs=src_meta["crs"],
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
    return reprojected, kwargs

# -----------------------------------------
# PROCESS EACH BAND
# -----------------------------------------
skipped = []
failed  = []

for band in BANDS:
    band_files = sorted(RAW_DIR.glob(f"**/*{band}.tif"))

    if not band_files:
        msg = f"No files found for {band} — skipping"
        print(f"⚠️  {msg}")
        log.warning(msg)
        skipped.append(band)
        continue

    print(f"\n🔧 Processing band: {band} ({len(band_files)} tile(s))")
    log.info(f"Processing {band}: {[str(f) for f in band_files]}")

    try:
        srcs = [rasterio.open(f) for f in band_files]

        # -----------------------------------------
        # MERGE TILES
        # -----------------------------------------
        mosaic, transform = merge(srcs, nodata=0, method="first")  # ✅ explicit method

        meta = srcs[0].meta.copy()
        meta.update(
            driver="GTiff",
            transform=transform,
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            nodata=0,
            compress="lzw",          # ✅ compress output TIFs
            tiled=True,              # ✅ cloud-optimized tiling
            blockxsize=256,
            blockysize=256,
        )

        for s in srcs:
            s.close()

        # -----------------------------------------
        # REPROJECT TO TARGET CRS
        # -----------------------------------------
        if str(meta["crs"]) != TARGET_CRS:
            print(f"   🔄 Reprojecting from {meta['crs']} → {TARGET_CRS}")
            mosaic, meta = reproject_to_target(mosaic, meta, TARGET_CRS)
            log.info(f"{band}: reprojected to {TARGET_CRS}")

        # -----------------------------------------
        # REPROJECT AOI TO MATCH RASTER CRS
        # -----------------------------------------
        aoi_proj = aoi.to_crs(meta["crs"])
        geoms    = list(aoi_proj.geometry)

        # -----------------------------------------
        # CLIP TO AOI (via MemoryFile — avoids temp disk writes)
        # -----------------------------------------
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**meta) as tmp:
                tmp.write(mosaic)
                clipped, clipped_transform = mask(
                    tmp,
                    geoms,
                    crop=True,
                    nodata=0,
                    all_touched=True,   # ✅ include edge pixels touching AOI
                )

        meta.update(
            transform=clipped_transform,
            height=clipped.shape[1],
            width=clipped.shape[2],
        )

        # -----------------------------------------
        # WRITE OUTPUT
        # -----------------------------------------
        out_path = OUT_DIR / f"{band}.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(clipped)

        size_mb = out_path.stat().st_size / 1_000_000
        print(f"   ✅ {band} saved → {out_path} ({size_mb:.1f} MB)")
        log.info(f"{band} saved: {out_path} ({size_mb:.1f} MB)")

    except Exception as e:
        msg = f"Failed processing {band}: {e}"
        print(f"   ❌ {msg}")
        log.error(msg)
        failed.append(band)

# -----------------------------------------
# SUMMARY
# -----------------------------------------
print(f"\n{'='*50}")
print(f"✅ Done    : {[b for b in BANDS if b not in skipped and b not in failed]}")
print(f"⚠️  Skipped : {skipped or 'None'}")
print(f"❌ Failed  : {failed or 'None'}")
print(f"📁 Output  : {OUT_DIR}")
print(f"{'='*50}")
log.info(f"Preprocess complete. Skipped={skipped}, Failed={failed}")
