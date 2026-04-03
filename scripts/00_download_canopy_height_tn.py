# scripts/00_download_canopy_height_tn.py
#
# ONE-TIME download of WRI/Meta canopy height tiles covering Tamil Nadu.
#
# How it works:
#   1. Downloads tiles.geojson index from S3 (lists all available QuadKey tiles)
#   2. Finds which QuadKey tiles intersect the Tamil Nadu bounding box
#   3. Downloads those CHM tiles from  s3://.../chm/{quadkey}.tif
#   4. Merges into a single TN mosaic:  data/raw/canopy_height_tamilnadu.tif
#
# Run ONCE, then use --canopy_tn in run.py for any TN district:
#   python scripts/00_download_canopy_height_tn.py
#   python run.py --year 2022 --aoi dindigul   --canopy_tn
#   python run.py --year 2022 --aoi puducherry --canopy_tn
#
# Requirements:
#   pip install boto3 rasterio tqdm shapely
#
# Dataset: Meta & WRI High Resolution Canopy Height Maps (2024)
#   Tile naming : QuadKey integers (NOT lat/lon filenames)
#   Resolution  : 1 m native
#   License     : CC BY 4.0
#   Citation    : Tolan et al. (2024) Remote Sensing of Environment 300, 113888
#   AWS S3      : s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/

import json
import sys
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm
from shapely.geometry import shape, box
import rasterio
from rasterio.merge import merge

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
S3_BUCKET  = "dataforgood-fb-data"
S3_BASE    = "forests/v1/alsgedi_global_v6_float"
TILE_DIR   = Path("data/raw/canopy_height_tiles")
OUT_PATH   = Path("data/raw/canopy_height_tamilnadu.tif")

# Tamil Nadu + Puducherry bounding box  (EPSG:4326)
# Slightly generous margins to ensure full coverage
TN_BBOX = box(75.8, 7.8, 80.4, 13.6)   # (minx, miny, maxx, maxy)

TILE_DIR.mkdir(parents=True, exist_ok=True)

if OUT_PATH.exists():
    print(f"\u2714  TN mosaic already exists: {OUT_PATH}")
    print("   Delete it and re-run if you want to re-download.")
    sys.exit(0)

# ------------------------------------------------------------------
# CONNECT TO S3
# ------------------------------------------------------------------
print("Connecting to S3 (no credentials needed)...")
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

# ------------------------------------------------------------------
# STEP 1 -- Download tiles.geojson index
# ------------------------------------------------------------------
geojson_local = TILE_DIR / "tiles.geojson"
if not geojson_local.exists():
    print("\nDownloading tile index (tiles.geojson)...")
    s3.download_file(
        S3_BUCKET,
        f"{S3_BASE}/tiles.geojson",
        str(geojson_local),
    )
    print(f"   \u2714  tiles.geojson saved")
else:
    print(f"   skip  tiles.geojson  (already downloaded)")

with open(geojson_local) as f:
    tiles_fc = json.load(f)

print(f"   Total tiles in global dataset: {len(tiles_fc['features'])}")

# ------------------------------------------------------------------
# STEP 2 -- Find QuadKey tiles that intersect Tamil Nadu bbox
# ------------------------------------------------------------------
print(f"\nFinding tiles intersecting Tamil Nadu (bbox: {TN_BBOX.bounds})...")

tn_tiles = []
for feat in tiles_fc["features"]:
    geom  = shape(feat["geometry"])
    qk    = str(feat["properties"]["tile"])   # QuadKey integer as string
    if geom.intersects(TN_BBOX):
        tn_tiles.append(qk)

if not tn_tiles:
    print("ERROR: No tiles found intersecting Tamil Nadu bbox.")
    print("Check that tiles.geojson downloaded correctly.")
    sys.exit(1)

print(f"   Found {len(tn_tiles)} tiles covering Tamil Nadu")
print(f"   QuadKeys: {tn_tiles[:5]}{'...' if len(tn_tiles) > 5 else ''}")

# ------------------------------------------------------------------
# STEP 3 -- Download CHM tiles
# ------------------------------------------------------------------
print(f"\nDownloading {len(tn_tiles)} CHM tiles...")

downloaded = []
for qk in tn_tiles:
    local = TILE_DIR / f"{qk}.tif"
    if local.exists():
        print(f"   skip  {qk}.tif  (already downloaded)")
        downloaded.append(local)
        continue

    s3_key = f"{S3_BASE}/chm/{qk}.tif"
    try:
        head  = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
        total = head["ContentLength"]

        with tqdm(total=total, unit="B", unit_scale=True,
                  desc=f"  {qk}.tif", leave=False) as bar:
            def _cb(n, bar=bar):
                bar.update(n)
            s3.download_file(S3_BUCKET, s3_key, str(local), Callback=_cb)

        size_mb = local.stat().st_size / 1e6
        print(f"   \u2714  {qk}.tif  ({size_mb:.0f} MB)")
        downloaded.append(local)

    except Exception as e:
        code = getattr(getattr(e, "response", {}), "get", lambda k, d=None: d)("Error", {}).get("Code", "?")
        if code in ("404", "NoSuchKey"):
            print(f"   --  {qk}.tif  (not found, skipping)")
        else:
            print(f"   \u2717  {qk}.tif  ERROR: {e}")

if not downloaded:
    print("\nERROR: No tiles downloaded.")
    print("Check your internet connection or try: pip install awscli && aws s3 ls --no-sign-request s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/")
    sys.exit(1)

print(f"\nDownloaded {len(downloaded)} / {len(tn_tiles)} tiles.")

# ------------------------------------------------------------------
# STEP 4 -- Merge into TN mosaic
# ------------------------------------------------------------------
print("\nMerging tiles into TN mosaic (this may take a few minutes)...")

srcs    = [rasterio.open(p) for p in sorted(downloaded)]
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
        coverage="Tamil Nadu + Puducherry mosaic",
        bbox=str(TN_BBOX.bounds),
        quadkeys=str([p.stem for p in sorted(downloaded)]),
        resolution_native="1m",
        units="metres above ground",
        temporal_coverage="2018-2020",
        citation="Tolan et al. (2024) RSE 300, 113888. CC BY 4.0.",
    )

for src in srcs:
    src.close()

size_gb = OUT_PATH.stat().st_size / 1e9
print(f"\n\u2714  TN mosaic saved : {OUT_PATH}")
print(f"   Size   : {size_gb:.2f} GB")
print(f"   Tiles  : {len(downloaded)}")
print(f"   Shape  : {mosaic.shape[1]:,} x {mosaic.shape[2]:,} px")
print(f"\nDone! Now run the pipeline for any TN / Puducherry district:")
print(f"  python run.py --year 2022 --aoi dindigul   --canopy_tn")
print(f"  python run.py --year 2022 --aoi puducherry --canopy_tn")
