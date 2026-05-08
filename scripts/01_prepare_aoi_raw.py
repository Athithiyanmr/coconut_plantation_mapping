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
# SCL CLOUD CLASSES
# Sentinel-2 L2A Scene Classification Layer values to mask:
#   0  = No Data
#   1  = Saturated / Defective
#   3  = Cloud Shadow
#   8  = Cloud Medium Probability
#   9  = Cloud High Probability
#   10 = Thin Cirrus
# -----------------------------------------
SCL_MASK_CLASSES = {0, 1, 3, 8, 9, 10}

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

SPECTRAL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12"]
TARGET_CRS     = args.target_crs
S2_NODATA      = 0

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
        src.crs, target_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({"crs": target_crs, "transform": transform,
                   "width": width, "height": height})

    data      = src.read()
    dst_array = np.zeros((data.shape[0], height, width), dtype=data.dtype)

    for i in range(data.shape[0]):
        reproject(
            source=data[i], destination=dst_array[i],
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=S2_NODATA, dst_nodata=S2_NODATA,
        )

    valid_frac = float(np.sum(dst_array[0] != S2_NODATA)) / dst_array[0].size
    memfile    = rasterio.io.MemoryFile()
    dataset    = memfile.open(**kwargs)
    dataset.write(dst_array)
    return dataset, valid_frac

# -----------------------------------------
# BUILD PER-TILE SCL CLOUD MASKS
# One mask per tile directory: True = cloudy/bad pixel
# SCL is at 20m so we reproject it to TARGET_CRS at 10m to match bands.
# -----------------------------------------
print("\nBuilding SCL cloud masks...")

tile_dirs  = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
scl_masks  = {}   # tile_dir -> np.ndarray (H, W) bool  (True = mask out)
scl_ref    = {}   # tile_dir -> (transform, crs, shape) for later lookup

for tile_dir in tile_dirs:
    scl_path = tile_dir / "SCL.tif"
    if not scl_path.exists():
        print(f"   {tile_dir.name}: SCL not found -- cloud masking skipped for this tile")
        log.warning(f"{tile_dir.name}: SCL missing, no cloud masking")
        scl_masks[tile_dir] = None
        continue

    with rasterio.open(scl_path) as scl_src:
        # Reproject SCL to target CRS at 10m
        scl_ds, _ = reproject_tile(scl_src, TARGET_CRS)
        scl_arr   = scl_ds.read(1)
        scl_transform = scl_ds.transform
        scl_crs       = scl_ds.crs

    cloud_mask = np.isin(scl_arr, list(SCL_MASK_CLASSES))
    masked_pct = 100.0 * cloud_mask.sum() / cloud_mask.size
    print(f"   {tile_dir.name}: SCL cloud mask = {masked_pct:.1f}%% of tile pixels masked")
    log.info(f"{tile_dir.name}: SCL masked {masked_pct:.1f}%%")

    scl_masks[tile_dir] = cloud_mask
    scl_ref[tile_dir]   = (scl_transform, scl_crs, scl_arr.shape)

# -----------------------------------------
# PROCESS SPECTRAL BANDS
# -----------------------------------------
for band in SPECTRAL_BANDS:
    band_files = sorted(RAW_DIR.glob(f"**/*{band}.tif"))

    if not band_files:
        print(f"\n\u274c No files for {band}")
        continue

    print(f"\nProcessing {band} ({len(band_files)} tiles)")

    reprojected_srcs = []

    for f in band_files:
        tile_dir = f.parent
        src      = rasterio.open(f)

        if str(src.crs) != TARGET_CRS:
            print(f"   Reprojecting {tile_dir.name}/{f.name} ...")
            reproj, valid_frac = reproject_tile(src, TARGET_CRS)
            print(f"   Valid after reproject : {100*valid_frac:.1f}%%  ({tile_dir.name})")
        else:
            reproj     = src
            arr        = src.read(1)
            valid_frac = float(np.sum(arr != S2_NODATA)) / arr.size
            print(f"   Valid (native CRS)    : {100*valid_frac:.1f}%%  ({tile_dir.name})")

        # Apply SCL cloud mask if available for this tile
        cloud_mask = scl_masks.get(tile_dir)
        if cloud_mask is not None:
            data = reproj.read()
            # Resize cloud mask to match band shape if needed (SCL is 20m, band may be 10m)
            if cloud_mask.shape != data.shape[1:]:
                from skimage.transform import resize as sk_resize
                cloud_mask_resized = sk_resize(
                    cloud_mask.astype("float32"),
                    data.shape[1:],
                    order=0,           # nearest neighbour
                    preserve_range=True,
                    anti_aliasing=False,
                ).astype(bool)
            else:
                cloud_mask_resized = cloud_mask

            data[0][cloud_mask_resized] = S2_NODATA

            # Write masked data back into a new MemoryFile
            meta = reproj.meta.copy()
            mf2  = rasterio.io.MemoryFile()
            ds2  = mf2.open(**meta)
            ds2.write(data)
            reproj = ds2

            masked_px  = int(cloud_mask_resized.sum())
            total_px   = cloud_mask_resized.size
            print(f"   SCL masked     : {masked_px:,} / {total_px:,} "
                  f"({100*masked_px/total_px:.1f}%%) cloud/shadow pixels zeroed")

        reprojected_srcs.append(reproj)

    if not reprojected_srcs:
        print(f"   \u274c No tiles available for {band}.")
        continue

    # Sort best-coverage tile first so merge(method='first') picks it
    def get_valid_frac(ds):
        arr = ds.read(1)
        return float(np.sum(arr != S2_NODATA)) / arr.size

    reprojected_srcs.sort(key=get_valid_frac, reverse=True)

    # -----------------------------------------
    # MERGE: best tile first, others fill gaps
    # -----------------------------------------
    mosaic, transform = merge(reprojected_srcs, nodata=S2_NODATA, method="first")

    valid_px = int(np.sum(mosaic[0] != S2_NODATA))
    total_px = mosaic[0].size
    print(f"   Mosaic valid   : {valid_px:,} / {total_px:,} ({100*valid_px/total_px:.1f}%)")
    log.info(f"{band} mosaic valid={valid_px}/{total_px}")

    meta = reprojected_srcs[0].meta.copy()
    meta.update({
        "transform": transform,
        "height":    mosaic.shape[1],
        "width":     mosaic.shape[2],
        "nodata":    S2_NODATA,
        "compress":  "lzw",
    })

    # -----------------------------------------
    # CLIP TO AOI
    # -----------------------------------------
    aoi_proj = aoi.to_crs(meta["crs"])
    geoms    = list(aoi_proj.geometry)

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**meta) as tmp:
            tmp.write(mosaic)
            clipped, clipped_transform = mask(
                tmp, geoms, crop=True, nodata=S2_NODATA, filled=True
            )

    valid_clip = int(np.sum(clipped[0] != S2_NODATA))
    total_clip = clipped[0].size
    print(f"   Clipped valid  : {valid_clip:,} / {total_clip:,} ({100*valid_clip/total_clip:.1f}%)")
    log.info(f"{band} clipped valid={valid_clip}/{total_clip}")

    meta.update({
        "transform": clipped_transform,
        "height":    clipped.shape[1],
        "width":     clipped.shape[2],
    })

    out_path = OUT_DIR / f"{band}.tif"
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(clipped)

    print(f"\u2714 Saved {band}")

print("\n\u2705 Preprocessing complete")
print("Output:", OUT_DIR)
