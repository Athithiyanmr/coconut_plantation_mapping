# scripts/00_download_canopy_height_tn.py
#
# ONE-TIME download of WRI/Meta canopy height tiles covering Tamil Nadu.
# Downloads all required 1°x1° tiles and merges them into a single
# Tamil Nadu mosaic:  data/raw/canopy_height_tamilnadu.tif
#
# Run ONCE, then use --canopy_tn in run.py for any AOI:
#   python scripts/00_download_canopy_height_tn.py
#
# Requirements:
#   pip install boto3 rasterio tqdm
#
# Dataset: Meta & WRI High Resolution Canopy Height Maps (2024)
#   Resolution : 1 m (native)
#   Coverage   : Tamil Nadu ~8-14°N, 76-80°E
#   License    : CC BY 4.0
#   Citation   : Tolan et al. (2024) Remote Sensing of Environment 300, 113888
#   AWS S3     : s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/

import os
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
import rasterio
from rasterio.merge import merge

# ------------------------------------------------------------------
# Tamil Nadu bounding box:  lat 8–14°N,  lon 76–80°E
# Each tile is named  N{lat:02d}E{lon:03d}.tif
# ------------------------------------------------------------------
TILES = [
    f"N{lat:02d}E{lon:03d}.tif"
    for lat in range(8, 14)       # 8,9,10,11,12,13
    for lon in range(76, 80)      # 76,77,78,79
]

TILE_DIR = Path("data/raw/canopy_height_tiles")
OUT_PATH = Path("data/raw/canopy_height_tamilnadu.tif")
S3_BUCKET = "dataforgood-fb-data"
S3_PREFIX = "forests/v1/alsgedi_global_v6_float/global"

TILE_DIR.mkdir(parents=True, exist_ok=True)

if OUT_PATH.exists():
    print(f"✔  TN mosaic already exists: {OUT_PATH}")
    print("   Delete it and re-run if you want to re-download.")
    raise SystemExit(0)

# ------------------------------------------------------------------
# DOWNLOAD TILES
# ------------------------------------------------------------------
print(f"Connecting to S3 (no credentials needed)...")
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

print(f"\nDownloading {len(TILES)} tiles covering Tamil Nadu (8-13°N, 76-79°E)")
print(f"Tile directory: {TILE_DIR}\n")

downloaded = []
for tile_name in TILES:
    local = TILE_DIR / tile_name
    if local.exists():
        print(f"   skip  {tile_name}  (already downloaded)")
        downloaded.append(local)
        continue

    s3_key = f"{S3_PREFIX}/{tile_name}"
    try:
        # Get file size for progress bar
        head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        total = head["ContentLength"]

        with tqdm(total=total, unit="B", unit_scale=True,
                  desc=f"  {tile_name}", leave=False) as bar:
            def _cb(bytes_transferred):
                bar.update(bytes_transferred)
            s3.download_file(S3_BUCKET, s3_key, str(local),
                             Callback=_cb)

        print(f"   ✔  {tile_name}")
        downloaded.append(local)

    except s3.exceptions.ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            print(f"   --  {tile_name}  (not in dataset, skipping)")
        else:
            print(f"   ✗  {tile_name}  ERROR: {e}")

if not downloaded:
    raise RuntimeError("No tiles downloaded — check your internet connection.")

print(f"\nDownloaded {len(downloaded)} tiles.")

# ------------------------------------------------------------------
# MERGE ALL TILES INTO ONE TN MOSAIC
# ------------------------------------------------------------------
print(f"\nMerging tiles into TN mosaic...")
srcs = [rasterio.open(p) for p in sorted(downloaded)]
mosaic, transform = merge(srcs)

meta = srcs[0].meta.copy()
meta.update(
    driver="GTiff",
    height=mosaic.shape[1],
    width=mosaic.shape[2],
    transform=transform,
    compress="lzw",
    tiled=True,
    blockxsize=512,
    blockysize=512,
    bigtiff="IF_SAFER",
)

with rasterio.open(OUT_PATH, "w", **meta) as dst:
    dst.write(mosaic)
    dst.update_tags(
        source="Meta & WRI High Resolution Canopy Height Maps (2024)",
        coverage="Tamil Nadu mosaic (8-13°N, 76-79°E)",
        resolution_native="1m",
        units="metres above ground",
        temporal_coverage="2018-2020",
        citation="Tolan et al. (2024) RSE 300, 113888. CC BY 4.0.",
        tiles=str([t.name for t in sorted(downloaded)]),
    )

for src in srcs:
    src.close()

size_gb = OUT_PATH.stat().st_size / 1e9
print(f"\n✔  TN mosaic saved: {OUT_PATH}")
print(f"   Size   : {size_gb:.2f} GB")
print(f"   Tiles  : {len(downloaded)}")
print(f"   Shape  : {mosaic.shape[1]} x {mosaic.shape[2]} px")
print(f"\nDone! Now run the pipeline for any TN district:")
print(f"  python run.py --year 2022 --aoi dindigul --canopy_tn")
print(f"  python run.py --year 2022 --aoi puducherry --canopy_tn")
