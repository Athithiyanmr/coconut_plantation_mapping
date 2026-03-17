# scripts/03_download_coconut_labels.py
#
# Download Descals et al. (2023) global coconut palm tiles from Zenodo,
# clip to Coimbatore AOI, and resample from 20m to 10m (nearest-neighbor).

import argparse
import logging
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="label_coconut.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(
    description="Prepare coconut plantation labels from Descals et al. (2023) global coconut layer"
)
parser.add_argument("--year",   required=True)
parser.add_argument("--aoi",    required=True)
parser.add_argument("--label_dir", required=True,
                    help="Directory containing extracted Descals coconut GeoTIFF tiles "
                         "(from Zenodo record 8128183)")
parser.add_argument("--bbox", nargs=4, type=float, default=[76.5, 10.8, 77.3, 11.5],
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                    help="Bounding box for clipping (default: Coimbatore district)")
args = parser.parse_args()

YEAR      = args.year
AOI_NAME  = args.aoi
LABEL_DIR = Path(args.label_dir)
BBOX      = args.bbox  # [lon_min, lat_min, lon_max, lat_max]

AOI_PATH = f"data/raw/boundaries/{AOI_NAME}.shp"
STACK    = f"data/processed/{AOI_NAME}/stack_{YEAR}.tif"

OUT_DIR  = Path("data/processed/training")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT      = OUT_DIR / f"labels_coconut_{YEAR}_{AOI_NAME}.tif"

log.info(f"Start: AOI={AOI_NAME}, year={YEAR}, label_dir={LABEL_DIR}, bbox={BBOX}")

# -----------------------------------------
# STEP 1 -- LOAD AOI
# -----------------------------------------
print("\nLoading AOI...")
if Path(AOI_PATH).exists():
    aoi      = gpd.read_file(AOI_PATH).to_crs("EPSG:4326")
    aoi_geom = aoi.geometry.iloc[0]
    bbox     = aoi.total_bounds
    print(f"   BBox from shapefile: {bbox.round(4)}")
else:
    from shapely.geometry import box
    aoi_geom = box(BBOX[0], BBOX[1], BBOX[2], BBOX[3])
    aoi = gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326")
    bbox = BBOX
    print(f"   Using provided BBox: {BBOX}")

log.info(f"AOI loaded. BBox={bbox}")

# -----------------------------------------
# STEP 2 -- FIND COCONUT LABEL TILES
# -----------------------------------------
print(f"\nSearching for Descals coconut tiles in: {LABEL_DIR}")
tif_files = sorted(LABEL_DIR.glob("*.tif"))
if not tif_files:
    tif_files = sorted(LABEL_DIR.glob("**/*.tif"))

if not tif_files:
    raise FileNotFoundError(
        f"No GeoTIFF files found in {LABEL_DIR}.\n"
        "Download the Descals et al. (2023) coconut layer from:\n"
        "  https://zenodo.org/records/8128183\n"
        "Extract the ZIP and provide the directory path via --label_dir."
    )

print(f"   Found {len(tif_files)} tile(s)")
log.info(f"Found {len(tif_files)} label tiles")

# -----------------------------------------
# STEP 3 -- FILTER TILES INTERSECTING AOI
# -----------------------------------------
print("\nFiltering tiles that intersect AOI...")
from shapely.geometry import box as shapely_box

intersecting = []
for tif in tif_files:
    with rasterio.open(tif) as src:
        tile_bounds = src.bounds
        tile_box = shapely_box(tile_bounds.left, tile_bounds.bottom,
                               tile_bounds.right, tile_bounds.top)
        if tile_box.intersects(aoi_geom):
            intersecting.append(tif)

if not intersecting:
    raise RuntimeError(
        "No coconut label tiles intersect the AOI bounding box.\n"
        "Verify the --label_dir contains tiles covering the Coimbatore region."
    )

print(f"   {len(intersecting)} tile(s) intersect AOI")
log.info(f"Intersecting tiles: {[t.name for t in intersecting]}")

# -----------------------------------------
# STEP 4 -- MERGE & CLIP TILES TO AOI
# -----------------------------------------
print("\nMerging and clipping coconut label tiles...")
from rasterio.merge import merge as rasterio_merge

if len(intersecting) == 1:
    src_file = intersecting[0]
else:
    srcs = [rasterio.open(f) for f in intersecting]
    mosaic, mosaic_transform = rasterio_merge(srcs, nodata=0)
    mosaic_meta = srcs[0].meta.copy()
    mosaic_meta.update(
        transform=mosaic_transform,
        height=mosaic.shape[1],
        width=mosaic.shape[2],
    )
    for s in srcs:
        s.close()

# Clip to AOI
aoi_proj = aoi.to_crs("EPSG:4326")
geoms = list(aoi_proj.geometry)

if len(intersecting) == 1:
    with rasterio.open(src_file) as src:
        clipped, clipped_transform = mask(src, geoms, crop=True, nodata=0, all_touched=True)
        clip_meta = src.meta.copy()
        clip_meta.update(
            transform=clipped_transform,
            height=clipped.shape[1],
            width=clipped.shape[2],
        )
else:
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**mosaic_meta) as tmp:
            tmp.write(mosaic)
            clipped, clipped_transform = mask(tmp, geoms, crop=True, nodata=0, all_touched=True)
    clip_meta = mosaic_meta.copy()
    clip_meta.update(
        transform=clipped_transform,
        height=clipped.shape[1],
        width=clipped.shape[2],
    )

label_data = clipped[0]  # single band
print(f"   Clipped label shape: {label_data.shape}")
coconut_px = int((label_data > 0).sum())
total_px = label_data.size
print(f"   Coconut pixels: {coconut_px:,} / {total_px:,} ({100*coconut_px/total_px:.2f}%%)")
log.info(f"Clipped: coconut={coconut_px}, total={total_px}")

# -----------------------------------------
# STEP 5 -- RESAMPLE TO 10m (match Sentinel-2 stack)
# -----------------------------------------
print("\nAligning with Sentinel-2 stack (resampling 20m -> 10m)...")
if not Path(STACK).exists():
    raise FileNotFoundError(
        f"Stack not found: {STACK}\n"
        "Run 02_build_stack.py first before generating labels."
    )

with rasterio.open(STACK) as ref_src:
    ref_meta      = ref_src.meta.copy()
    ref_transform = ref_src.transform
    ref_height    = ref_src.height
    ref_width     = ref_src.width
    ref_crs       = ref_src.crs

# Reproject and resample coconut labels to match stack grid
label_aligned = np.zeros((ref_height, ref_width), dtype="uint8")

reproject(
    source=label_data.astype("uint8"),
    destination=label_aligned,
    src_transform=clipped_transform,
    src_crs=clip_meta["crs"],
    dst_transform=ref_transform,
    dst_crs=ref_crs,
    resampling=Resampling.nearest,  # nearest-neighbor for categorical labels
)

coconut_aligned = int((label_aligned > 0).sum())
total_aligned = ref_height * ref_width
pct = 100 * coconut_aligned / total_aligned
print(f"   Aligned label shape: {label_aligned.shape}")
print(f"   Coconut pixels (10m): {coconut_aligned:,} / {total_aligned:,} ({pct:.2f}%%)")
log.info(f"Aligned: coconut={coconut_aligned}, total={total_aligned}, pct={pct:.2f}%%")

if coconut_aligned == 0:
    print("WARNING: No coconut pixels after alignment -- verify CRS and spatial extent")
    log.warning(f"Zero coconut pixels after alignment")

# -----------------------------------------
# STEP 6 -- SAVE OUTPUT
# -----------------------------------------
out_meta = ref_meta.copy()
out_meta.update(
    count=1,
    dtype="uint8",
    nodata=0,
    compress="lzw",
    tiled=True,
    blockxsize=256,
    blockysize=256,
)

with rasterio.open(OUT, "w", **out_meta) as dst:
    dst.write(label_aligned, 1)
    dst.update_tags(
        source="Descals et al. (2023) Global Coconut Palm Layer v1.2",
        zenodo_doi="10.5281/zenodo.8128183",
        original_resolution="20m",
        resampled_resolution="10m",
        resampling_method="nearest",
        aoi=AOI_NAME,
        year=YEAR,
    )

size_mb = OUT.stat().st_size / 1_000_000
print(f"\nCoconut label mask saved -> {OUT} ({size_mb:.1f} MB)")
log.info(f"Saved: {OUT}, size={size_mb:.1f}MB")
