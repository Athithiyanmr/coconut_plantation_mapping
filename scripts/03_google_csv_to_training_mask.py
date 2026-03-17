# scripts/03_google_csv_to_training_mask.py

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely import wkt
from shapely.validation import make_valid  # ✅ safer than buffer(0)

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="label_buildings.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Generate building raster labels from Google Open Buildings CSV")
parser.add_argument("--year",   required=True)
parser.add_argument("--aoi",    required=True)
parser.add_argument("--csv",    required=True, help="Local Google Open Buildings CSV or CSV.GZ (required)")
parser.add_argument("--conf",   type=float, default=0.7,  help="Minimum confidence score (default: 0.7)")
parser.add_argument("--buffer", type=float, default=0.0,  help="Optional buffer in metres to dilate buildings")
args = parser.parse_args()

YEAR     = args.year
AOI_NAME = args.aoi
CSV      = args.csv
CONF     = args.conf
BUFFER_M = args.buffer

AOI_PATH = f"data/raw/boundaries/{AOI_NAME}.shp"
STACK    = f"data/processed/{AOI_NAME}/stack_{YEAR}.tif"

OUT_DIR  = Path("data/processed/training")   # ✅ consistent with pipeline
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT      = OUT_DIR / f"labels_google_{YEAR}_{AOI_NAME}.tif"

log.info(f"Start: AOI={AOI_NAME}, year={YEAR}, csv={CSV}, conf={CONF}, buffer={BUFFER_M}m")

# -----------------------------------------
# STEP 1 — LOAD AOI
# -----------------------------------------
print("\n📍 Loading AOI...")
if not Path(AOI_PATH).exists():
    raise FileNotFoundError(f"AOI shapefile not found: {AOI_PATH}")

aoi      = gpd.read_file(AOI_PATH).to_crs("EPSG:4326")
aoi_geom = aoi.geometry.iloc[0]
bbox     = aoi.total_bounds
print(f"   BBox: {bbox.round(4)}")
log.info(f"AOI loaded. BBox={bbox}")

# -----------------------------------------
# STEP 2 — LOAD CSV
# ✅ auto-detect .gz compression
# -----------------------------------------
print(f"\n📂 Reading CSV: {CSV}")
if not Path(CSV).exists():
    raise FileNotFoundError(f"CSV not found: {CSV}")

compression = "gzip" if str(CSV).endswith(".gz") else None
df = pd.read_csv(CSV, compression=compression)
print(f"   Total rows: {len(df):,}")
log.info(f"CSV loaded: {CSV}, rows={len(df)}")

if "geometry" not in df.columns:
    raise ValueError(
        "CSV must contain a 'geometry' column in WKT format.\n"
        "Google Open Buildings CSVs use WKT polygon strings in 'geometry'."
    )

# -----------------------------------------
# STEP 3 — PARSE GEOMETRIES (chunked)
# ✅ avoids memory spike on large CSVs
# -----------------------------------------
print("\n🔄 Parsing geometries...")
CHUNK = 100_000
geoms = []
for i in range(0, len(df), CHUNK):
    geoms.extend(df["geometry"].iloc[i:i+CHUNK].apply(wkt.loads).tolist())
    print(f"   Parsed {min(i+CHUNK, len(df)):,} / {len(df):,}", end="\r")

df["geometry"] = geoms
gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
print(f"\n   GeoDataFrame ready: {len(gdf):,} features")

# -----------------------------------------
# STEP 4 — CONFIDENCE FILTER
# -----------------------------------------
if "confidence" in gdf.columns:
    before = len(gdf)
    gdf    = gdf[gdf["confidence"] >= CONF].copy()
    print(f"   Confidence ≥ {CONF}: {before:,} → {len(gdf):,}")
    log.info(f"Confidence filter: {before} → {len(gdf)}")
else:
    print("   ⚠️  No 'confidence' column — skipping filter")
    log.warning("No confidence column in CSV")

# -----------------------------------------
# STEP 5 — FIX INVALID GEOMETRIES
# ✅ make_valid() handles complex self-intersections better than buffer(0)
# -----------------------------------------
invalid = (~gdf.geometry.is_valid).sum()
if invalid > 0:
    print(f"   🔧 Fixing {invalid:,} invalid geometries...")
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    log.warning(f"Fixed {invalid} invalid geometries")

# Remove empty/null geometries after repair
before = len(gdf)
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
removed = before - len(gdf)
if removed > 0:
    print(f"   🗑  Removed {removed:,} empty/null geometries after repair")
    log.info(f"Removed {removed} empty/null geometries")

# -----------------------------------------
# STEP 6 — CLIP TO AOI
# -----------------------------------------
print("\n✂️  Clipping to AOI...")
gdf = gdf[gdf.intersects(aoi_geom)].copy()
print(f"   Buildings after clip: {len(gdf):,}")
log.info(f"After AOI clip: {len(gdf)}")

if gdf.empty:
    raise RuntimeError(
        "No buildings found inside AOI.\n"
        "Possible causes:\n"
        "  • Confidence threshold too high  → try --conf 0.5\n"
        "  • Wrong CSV tile for this region → check bbox coverage\n"
        "  • AOI shapefile CRS mismatch"
    )

# -----------------------------------------
# STEP 7 — ALIGN WITH STACK CRS
# -----------------------------------------
print("\n🛰  Aligning with Sentinel stack...")
if not Path(STACK).exists():
    raise FileNotFoundError(
        f"Stack not found: {STACK}\n"
        "Run stack.py first before generating labels."
    )

with rasterio.open(STACK) as src:
    meta      = src.meta.copy()
    transform = src.transform
    height    = src.height
    width     = src.width
    crs       = src.crs

gdf = gdf.to_crs(crs)
log.info(f"GDF reprojected to: {crs}")

# -----------------------------------------
# STEP 8 — OPTIONAL BUFFER
# -----------------------------------------
if BUFFER_M > 0:
    print(f"   Buffering polygons by {BUFFER_M}m...")
    gdf["geometry"] = gdf.geometry.buffer(BUFFER_M)
    log.info(f"Buffer applied: {BUFFER_M}m")

# -----------------------------------------
# STEP 9 — RASTERIZE
# -----------------------------------------
print("\n🧱 Rasterizing buildings...")

valid_geoms = [
    (geom, 1) for geom in gdf.geometry
    if geom is not None and geom.is_valid and not geom.is_empty
]
print(f"   Valid geometries for rasterize: {len(valid_geoms):,}")

label_mask = rasterize(
    valid_geoms,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=False,  # ✅ centre-point rasterization — avoids over-labelling
)

built = int(label_mask.sum())
total = height * width
pct   = 100 * built / total
print(f"   Built pixels : {built:,} / {total:,} ({pct:.2f}%)")
log.info(f"Rasterized: built={built}, total={total}, pct={pct:.2f}%")

if pct < 0.01:
    print("⚠️  Very few built pixels — verify CRS alignment or lower --conf")
    log.warning(f"Low built pixel ratio: {pct:.4f}%")

# -----------------------------------------
# STEP 10 — SAVE OUTPUT
# -----------------------------------------
meta.update(
    count=1,
    dtype="uint8",
    nodata=0,
    compress="lzw",    # ✅ compress binary mask
    tiled=True,
    blockxsize=256,
    blockysize=256,
)

with rasterio.open(OUT, "w", **meta) as dst:
    dst.write(label_mask, 1)
    dst.update_tags(
        source="Google Open Buildings v3",
        confidence_threshold=str(CONF),
        aoi=AOI_NAME,
        year=YEAR,
        buffer_m=str(BUFFER_M),
        csv=str(CSV),
    )  # ✅ embed provenance in GeoTIFF tags

size_mb = OUT.stat().st_size / 1_000_000
print(f"\n✅ Training mask saved → {OUT} ({size_mb:.1f} MB)")
log.info(f"Saved: {OUT}, size={size_mb:.1f}MB")
