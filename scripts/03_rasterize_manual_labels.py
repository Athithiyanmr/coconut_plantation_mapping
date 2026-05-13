# scripts/03_rasterize_manual_labels.py
#
# Rasterize manually digitized coconut plantation polygons (shapefile)
# onto the Sentinel-2 stack grid.
#
# Supports a 'class' column in the shapefile:
#   class = 1  -> confirmed coconut   -> pixel = 1
#   class = 0  -> confirmed NOT coconut -> pixel = 0
#   (no polygon) -> unlabeled          -> pixel = 255 (IGNORE)
#
# Usage:
#   python scripts/03_rasterize_manual_labels.py \
#       --year 2025 --aoi villupuram \
#       --shp data/raw/training/villupuram_verified_final.shp
#
# Output:
#   data/processed/training/labels_coconut_{year}_{aoi}.tif
#   uint8: 1=coconut, 0=confirmed not-coconut, 255=ignore/unlabeled

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="label_manual.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(
    description="Rasterize manually digitized coconut polygons to match Sentinel-2 stack"
)
parser.add_argument("--year",        required=True, help="Year (must match stack)")
parser.add_argument("--aoi",         required=True, help="AOI name (must match stack)")
parser.add_argument("--shp",         required=True, help="Path to polygon shapefile with 'class' column")
parser.add_argument("--class_col",   default="class", help="Column name for class label (default: 'class')")
parser.add_argument("--all_touched", action="store_true",
                    help="Burn pixels that touch polygon edges (useful for small polygons)")
args = parser.parse_args()

YEAR      = args.year
AOI       = args.aoi
SHP       = Path(args.shp)
CLASS_COL = args.class_col

STACK   = Path(f"data/processed/{AOI}/stack_{YEAR}.tif")
OUT_DIR = Path("data/processed/training")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT     = OUT_DIR / f"labels_coconut_{YEAR}_{AOI}.tif"

# -----------------------------------------
# VALIDATE INPUTS
# -----------------------------------------
if not SHP.exists():
    raise FileNotFoundError(f"Shapefile not found: {SHP}")

if not STACK.exists():
    raise FileNotFoundError(
        f"Stack not found: {STACK}\n"
        "Run 02_build_stack.py before generating labels."
    )

log.info(f"Start: AOI={AOI}, year={YEAR}, shp={SHP}, class_col={CLASS_COL}")

# -----------------------------------------
# STEP 1 -- LOAD REFERENCE STACK GRID
# -----------------------------------------
print("\nLoading Sentinel-2 stack grid...")
with rasterio.open(STACK) as ref:
    ref_transform = ref.transform
    ref_crs       = ref.crs
    ref_height    = ref.height
    ref_width     = ref.width
    ref_meta      = ref.meta.copy()

print(f"   Grid       : {ref_height} x {ref_width} px")
print(f"   CRS        : {ref_crs}")
print(f"   Transform  : {ref_transform}")
log.info(f"Stack grid: {ref_height}x{ref_width}, CRS={ref_crs}")

# -----------------------------------------
# STEP 2 -- LOAD & REPROJECT SHAPEFILE
# -----------------------------------------
print(f"\nLoading shapefile: {SHP}")
gdf = gpd.read_file(SHP)

if gdf.crs is None:
    raise ValueError(
        "Shapefile has no CRS defined.\n"
        "Set the CRS in your GIS software before exporting."
    )

print(f"   Features   : {len(gdf)}")
print(f"   Source CRS : {gdf.crs}")
print(f"   Columns    : {gdf.columns.tolist()}")

# Check class column
if CLASS_COL not in gdf.columns:
    raise ValueError(
        f"Column '{CLASS_COL}' not found in shapefile.\n"
        f"Available columns: {gdf.columns.tolist()}\n"
        f"Use --class_col to specify the correct column name."
    )

class_counts = gdf[CLASS_COL].value_counts().to_dict()
print(f"   Class dist : {class_counts}")
log.info(f"Class distribution: {class_counts}")

# Reproject to match stack CRS
if str(gdf.crs) != str(ref_crs):
    print(f"   Reprojecting to {ref_crs}...")
    gdf = gdf.to_crs(ref_crs)
    log.info(f"Reprojected shapefile to {ref_crs}")

# Drop any empty or null geometries
before = len(gdf)
gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
dropped = before - len(gdf)
if dropped:
    print(f"   Dropped {dropped} null/empty geometries")
    log.warning(f"Dropped {dropped} null/empty geometries")

if gdf.empty:
    raise RuntimeError("No valid geometries in shapefile after cleaning.")

# -----------------------------------------
# STEP 3 -- SPATIAL OVERLAP CHECK
# -----------------------------------------
stack_bounds = rasterio.transform.array_bounds(ref_height, ref_width, ref_transform)
shp_bounds   = gdf.total_bounds

print(f"\nBounds check:")
print(f"   Stack  : {[round(v,2) for v in stack_bounds]}")
print(f"   Labels : {[round(v,2) for v in shp_bounds]}")

sx_min, sy_min, sx_max, sy_max = stack_bounds
lx_min, ly_min, lx_max, ly_max = shp_bounds

overlap = (lx_max > sx_min and lx_min < sx_max and
           ly_max > sy_min and ly_min < sy_max)

if not overlap:
    raise RuntimeError(
        "Shapefile does not overlap with the Sentinel-2 stack extent.\n"
        "Check that your shapefile covers the same area as the AOI."
    )
print("   Overlap : OK")

# -----------------------------------------
# STEP 4 -- RASTERIZE WITH CLASS SUPPORT
#
# Strategy:
#   1. Start with background = 255 (IGNORE / unlabeled)
#   2. Burn class=0 polygons (confirmed not-coconut) -> 0
#   3. Burn class=1 polygons (confirmed coconut)     -> 1
#      (class=1 burns last so it wins if polygons overlap)
# -----------------------------------------
print("\nRasterizing polygons with class support...")
print(f"   255 = unlabeled/ignore  (all pixels not covered by any polygon)")
print(f"     0 = confirmed NOT coconut")
print(f"     1 = confirmed coconut")

# Start: all pixels = 255 (ignore)
label_arr = np.full((ref_height, ref_width), 255, dtype="uint8")

# Burn class=0 (not-coconut) first
gdf_neg = gdf[gdf[CLASS_COL] == 0]
if len(gdf_neg) > 0:
    shapes_neg = (
        (geom, 0)
        for geom in gdf_neg.geometry
        if geom is not None and not geom.is_empty
    )
    neg_arr = rasterize(
        shapes=shapes_neg,
        out_shape=(ref_height, ref_width),
        transform=ref_transform,
        fill=255,
        dtype="uint8",
        all_touched=args.all_touched,
    )
    label_arr[neg_arr == 0] = 0
    print(f"   Not-coconut polygons : {len(gdf_neg)}  -> {int((neg_arr==0).sum()):,} pixels")
    log.info(f"Negative polygons: {len(gdf_neg)}, pixels={int((neg_arr==0).sum())}")
else:
    print(f"   Not-coconut polygons : 0  (none in shapefile with class=0)")

# Burn class=1 (coconut) second — overwrites any overlap
gdf_pos = gdf[gdf[CLASS_COL] == 1]
if len(gdf_pos) > 0:
    shapes_pos = (
        (geom, 1)
        for geom in gdf_pos.geometry
        if geom is not None and not geom.is_empty
    )
    pos_arr = rasterize(
        shapes=shapes_pos,
        out_shape=(ref_height, ref_width),
        transform=ref_transform,
        fill=255,
        dtype="uint8",
        all_touched=args.all_touched,
    )
    label_arr[pos_arr == 1] = 1
    coconut_px = int((pos_arr == 1).sum())
    print(f"   Coconut polygons     : {len(gdf_pos)}  -> {coconut_px:,} pixels")
    log.info(f"Positive polygons: {len(gdf_pos)}, pixels={coconut_px}")
else:
    coconut_px = 0
    print(f"   Coconut polygons : 0  (none with class=1) -- WARNING: no positive labels!")

neg_px    = int((label_arr == 0).sum())
ignore_px = int((label_arr == 255).sum())
total_px  = ref_height * ref_width
pct_coco  = 100 * coconut_px / total_px
pct_neg   = 100 * neg_px / total_px
pct_ign   = 100 * ignore_px / total_px

print(f"\n   Label summary:")
print(f"     Coconut  (1) : {coconut_px:>12,} px  ({pct_coco:.3f}%)")
print(f"     Not-coco (0) : {neg_px:>12,} px  ({pct_neg:.3f}%)")
print(f"     Ignore (255) : {ignore_px:>12,} px  ({pct_ign:.1f}%)")
log.info(f"Label summary: coconut={coconut_px}, neg={neg_px}, ignore={ignore_px}")

if coconut_px == 0:
    raise RuntimeError(
        "No coconut pixels were rasterized (class=1).\n"
        "Check that your shapefile has polygons with class=1."
    )

if pct_coco > 80:
    print(f"   WARNING: {pct_coco:.1f}% coconut coverage seems very high -- check your polygons")
    log.warning(f"High coconut coverage: {pct_coco:.1f}%")

# -----------------------------------------
# STEP 5 -- SAVE OUTPUT
# -----------------------------------------
out_meta = ref_meta.copy()
out_meta.update(
    count=1,
    dtype="uint8",
    nodata=255,
    compress="lzw",
    tiled=True,
    blockxsize=256,
    blockysize=256,
)

with rasterio.open(OUT, "w", **out_meta) as dst:
    dst.write(label_arr, 1)
    dst.update_tags(
        source="Manually digitized coconut plantation polygons",
        shapefile=str(SHP.resolve()),
        aoi=AOI,
        year=YEAR,
        class_col=CLASS_COL,
        coconut_pixels=str(coconut_px),
        neg_pixels=str(neg_px),
        ignore_pixels=str(ignore_px),
        coconut_pct=f"{pct_coco:.3f}",
        all_touched=str(args.all_touched),
    )

size_mb = OUT.stat().st_size / 1_000_000
print(f"\nLabel raster saved -> {OUT} ({size_mb:.1f} MB)")
log.info(f"Saved: {OUT}, size={size_mb:.1f}MB")

# -----------------------------------------
# SUMMARY
# -----------------------------------------
print(f"\n{'='*52}")
print(f"Manual label rasterization complete")
print(f"   Shapefile     : {SHP}")
print(f"   Total polys   : {len(gdf)}")
print(f"   Coconut (1)   : {len(gdf_pos)} polys -> {coconut_px:,} px ({pct_coco:.3f}%)")
print(f"   Not-coco (0)  : {len(gdf_neg)} polys -> {neg_px:,} px ({pct_neg:.3f}%)")
print(f"   Ignore (255)  : {ignore_px:,} px ({pct_ign:.1f}%)")
print(f"   Output        : {OUT}")
print(f"{'='*52}")
