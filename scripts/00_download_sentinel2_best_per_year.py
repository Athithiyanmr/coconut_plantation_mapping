import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import planetary_computer
import pystac_client
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="download.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Download Sentinel-2 scenes via Planetary Computer STAC")
parser.add_argument("--aoi",   required=True,            help="AOI name (matches shapefile in data/raw/boundaries/)")
parser.add_argument("--year",  type=int, required=True,  help="Year to download")
parser.add_argument("--cloud", type=int, default=40,     help="Max cloud cover %% (default: 40)")
args = parser.parse_args()

AOI   = f"data/raw/boundaries/{args.aoi}.shp"
YEAR  = args.year
CLOUD = args.cloud

OUT = Path("data/raw/sentinel2") / args.aoi / str(YEAR)
OUT.mkdir(parents=True, exist_ok=True)

# SCL included for cloud masking in 01_prepare_aoi_raw.py
BANDS = ["B02", "B03", "B04", "B05", "B06", "B08", "B11", "B12", "SCL"]

# -----------------------------------------
# SAFE DOWNLOAD
# -----------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30))
def download(url, out_path):
    tmp = out_path.with_suffix(".tmp")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=out_path.name,
                leave=False,
            ) as bar:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

        if tmp.stat().st_size < 1_000_000:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Corrupted download: {out_path.name}")

        tmp.rename(out_path)
        log.info(f"Downloaded: {out_path}")

    except Exception as e:
        log.error(f"Failed: {out_path.name} -- {e}")
        tmp.unlink(missing_ok=True)
        raise

# -----------------------------------------
# SCENE SCORING
#
# Problem: all full-tile scenes have nodata between 0.9-1.9%% -- tiny
# differences that cause wrong ordering when sorting by raw nodata%%.
# Example: July (nodata=0.90%%, cloud=11%%) beats October (nodata=1.16%%, cloud=0.02%%)
# even though October is clearly better.
#
# Fix: bucket nodata into 5%% bins so any scene with nodata < 5%% is
# treated as "full tile" and sorted purely by cloud cover within that group.
#   bucket 0  = nodata  0- 5%%  -> full tile, sort by cloud only
#   bucket 1  = nodata  5-10%%  -> partial, deprioritised
#   bucket 10 = nodata 50-55%%  -> very partial, last resort
# -----------------------------------------
BUCKET_SIZE = 5.0

def scene_score(item):
    nodata_pct    = item.properties.get("s2:nodata_pixel_percentage", 100.0)
    cloud_pct     = item.properties.get("eo:cloud_cover",             100.0)
    nodata_bucket = int(nodata_pct / BUCKET_SIZE)
    return (nodata_bucket, cloud_pct)

# -----------------------------------------
# LOAD AOI
# -----------------------------------------
print("Loading AOI...")
aoi  = gpd.read_file(AOI).to_crs("EPSG:4326")
geom = aoi.geometry.iloc[0].__geo_interface__
log.info(f"AOI loaded: {AOI}")

# -----------------------------------------
# OPEN STAC
# -----------------------------------------
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

print(f"\nSearching Sentinel-2 {YEAR} | Cloud < {CLOUD}%%")
log.info(f"Searching: year={YEAR}, cloud<{CLOUD}, AOI={args.aoi}")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=geom,
    datetime=f"{YEAR}-01-01/{YEAR}-12-31",
    query={"eo:cloud_cover": {"lt": CLOUD}},
)

items = search.item_collection()

if not items:
    raise RuntimeError("No scenes found -- try relaxing --cloud threshold.")

print(f"   Found {len(items)} scenes")
log.info(f"Found {len(items)} scenes")

# -----------------------------------------
# GROUP BY TILE
# -----------------------------------------
by_tile = defaultdict(list)
for item in items:
    tile = item.properties.get("s2:mgrs_tile")
    if tile:
        by_tile[tile].append(item)

print("Tiles intersecting AOI:", list(by_tile.keys()))

# -----------------------------------------
# DOWNLOAD BEST SCENE PER TILE
# -----------------------------------------
manifest = []

for tile, tile_items in by_tile.items():

    best       = sorted(tile_items, key=scene_score)[0]
    nodata_pct = best.properties.get("s2:nodata_pixel_percentage", "n/a")
    cloud_pct  = best.properties.get("eo:cloud_cover", "n/a")

    tile_dir = OUT / f"T{tile}"
    tile_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n   Tile   : {tile}")
    print(f"   Date   : {best.datetime}")
    print(f"   Cloud  : {cloud_pct}%%")
    print(f"   NoData : {nodata_pct}%%")

    if isinstance(nodata_pct, float) and nodata_pct > 50.0:
        print(f"   WARNING: {nodata_pct:.1f}%% nodata -- partial orbit pass.")
        log.warning(f"Tile {tile}: high nodata {nodata_pct:.1f}%%")

    downloaded_bands = []

    for band in BANDS:
        asset = best.assets.get(band)
        if not asset:
            log.warning(f"Band {band} not found in tile {tile}")
            continue

        out_file = tile_dir / f"{band}.tif"

        if out_file.exists():
            print(f"   Skipping {band} (already exists)")
            log.info(f"Skipped (exists): {out_file}")
            downloaded_bands.append(band)
            continue

        print(f"   Downloading: {band}")
        download(asset.href, out_file)
        downloaded_bands.append(band)

    manifest.append({
        "tile":          tile,
        "date":          str(best.datetime),
        "cloud_cover":   cloud_pct,
        "nodata_pct":    nodata_pct,
        "bands":         downloaded_bands,
        "downloaded_at": datetime.utcnow().isoformat(),
    })

# -----------------------------------------
# SAVE MANIFEST
# -----------------------------------------
manifest_path = OUT / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest saved -> {manifest_path}")
log.info(f"Manifest saved: {manifest_path}")
print("\nSentinel-2 download complete")
