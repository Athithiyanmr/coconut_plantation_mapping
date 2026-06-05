import subprocess
import argparse
import os
from pathlib import Path


# --------------------------------
# ARGUMENTS
# --------------------------------
parser = argparse.ArgumentParser(
    description="End-to-end coconut plantation mapping pipeline"
)

# Core
parser.add_argument("--year",  required=True,  help="Sentinel-2 year (e.g. 2025)")
parser.add_argument("--aoi",   required=True,  help="AOI name matching your shapefile stem (e.g. dindigul)")

# Label source — accepts .shp, .geojson, or Descals directory
parser.add_argument("--label_dir", default=None,
                    help="Coconut label source: path to a .shp/.geojson file (manual polygons) "
                         "or a directory containing Descals GeoTIFF tiles")

# Dataset prep inputs (used only when --label_dir is a raw unprocessed GeoJSON)
parser.add_argument("--verified_geojson", default=None,
                    help="Path to raw verified coconut GeoJSON (e.g. dindigul_verified.geojson). "
                         "If provided, runs STEP 0 to prep labels + AOI before the pipeline.")
parser.add_argument("--aoi_geojson", default=None,
                    help="Path to district boundary GeoJSON for STEP 0. "
                         "If not provided, auto-looked up at data/raw/boundaries/<aoi>.geojson")

# Canopy height
parser.add_argument("--canopy_tiles_dir", default=None,
                    help="Path to local WRI/Meta canopy height tiles folder.")
parser.add_argument("--canopy_height", default=None,
                    help="Path to an already-clipped canopy height .tif for the AOI.")
parser.add_argument("--skip_canopy", action="store_true",
                    help="Skip canopy height (builds 11-band stack instead of 12)")

# ---- make_patches defaults ----
parser.add_argument("--patch",         type=int,   default=128,  help="Patch size in pixels (default: 128)")
parser.add_argument("--stride",        type=int,   default=32,   help="Stride between patches (default: 32)")
parser.add_argument("--pos_ratio",     type=float, default=0.005,help="Min coconut fraction to keep positive patch (default: 0.005)")
parser.add_argument("--neg_sample",    type=float, default=0.50, help="Background patch keep probability (default: 0.50)")
parser.add_argument("--neg_min_ratio", type=float, default=0.01, help="Min gt==0 fraction required for background patch (default: 0.01)")
parser.add_argument("--nodata_tol",    type=float, default=0.40, help="Max NaN fraction in a patch (default: 0.40)")
parser.add_argument("--ignore_tol",    type=float, default=0.95, help="Skip patch if >X fraction is ignore/255 (default: 0.95)")
parser.add_argument("--dilate",        type=int,   default=2,    help="Coconut mask dilation iterations (default: 2)")

# ---- train_unet defaults ----
parser.add_argument("--epochs",           type=int,   default=60,   help="Max training epochs (default: 60)")
parser.add_argument("--batch",            type=int,   default=8,    help="Batch size (default: 8)")
parser.add_argument("--lr",               type=float, default=1e-4, help="Learning rate (default: 1e-4)")
parser.add_argument("--patience",         type=int,   default=10,   help="Early stopping patience (default: 10)")
parser.add_argument("--threshold",        type=float, default=0.35, help="Prediction binarization threshold (default: 0.35)")
parser.add_argument("--pos_weight_floor", type=float, default=1.0,  help="Minimum pos_weight for BCE loss (default: 1.0)")
parser.add_argument("--pretrained_ckpt",  type=str,   default=None, help="Pretrained model checkpoint for transfer learning")

# Optional skip flags
parser.add_argument("--all_touched",   action="store_true", help="(Shapefile mode) Burn pixels touching polygon edges")
parser.add_argument("--skip_download", action="store_true", help="Skip Sentinel-2 download step")
parser.add_argument("--skip_train",    action="store_true", help="Skip model training step")
parser.add_argument("--skip_prep",     action="store_true", help="Skip STEP 0 dataset prep (use if already prepared)")

args = parser.parse_args()

YEAR      = args.year
AOI       = args.aoi
PATCH     = args.patch
STRIDE    = args.stride
THRESHOLD = args.threshold

# --------------------------------
# Resolve label path
# After STEP 0, the pipeline always uses the .shp output
# --------------------------------
PREP_OUT_SHP     = Path(f"data/raw/training/{AOI}_verified_final.shp")
PREP_OUT_GEOJSON = Path(f"data/raw/training/{AOI}_verified_final.geojson")
AOI_SHP          = Path(f"data/raw/boundaries/{AOI}.shp")

# Determine effective label path for STEP 5
if args.label_dir:
    LABEL_DIR = args.label_dir
elif PREP_OUT_SHP.exists():
    LABEL_DIR = str(PREP_OUT_SHP)
else:
    LABEL_DIR = None

# Resolve canopy height path
AUTO_CANOPY_PATH = Path(f"data/raw/canopy_height/{AOI}.tif")
if args.skip_canopy:
    CANOPY_HEIGHT = None
elif args.canopy_height:
    CANOPY_HEIGHT = args.canopy_height
else:
    CANOPY_HEIGHT = str(AUTO_CANOPY_PATH)


# --------------------------------
# Fix OpenMP crash (Mac)
# --------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --------------------------------
# Helper
# --------------------------------
def run(cmd):
    print(f"\n{'='*65}")
    print(f"CMD: {cmd}")
    print(f"{'='*65}\n")
    subprocess.run(cmd, shell=True, check=True)


# --------------------------------
# Detect label mode
# .shp or .geojson  -> manual rasterize mode
# directory         -> Descals tile mode
# --------------------------------
def is_manual_labels(path):
    if path is None:
        return False
    p = str(path).lower()
    return p.endswith(".shp") or p.endswith(".geojson")


# --------------------------------
# CLEAN macOS hidden files
# --------------------------------
run('find . -name "._*" -type f -delete')


# ================================
# PIPELINE
# ================================

# STEP 0 -- Dataset Preparation (NEW)
# Runs when --verified_geojson is provided OR when prep output doesn't exist yet
print("\nSTEP 0/8  --  Dataset Preparation")

if args.skip_prep:
    print("  [skipped] --skip_prep was set")

elif args.verified_geojson:
    # User explicitly provided raw verified GeoJSON — run prep
    verified_path = args.verified_geojson

    # Resolve AOI boundary: use --aoi_geojson if given, else look for default locations
    if args.aoi_geojson:
        aoi_boundary = args.aoi_geojson
    elif Path(f"data/raw/boundaries/{AOI}.geojson").exists():
        aoi_boundary = f"data/raw/boundaries/{AOI}.geojson"
    elif AOI_SHP.exists():
        aoi_boundary = str(AOI_SHP)
    else:
        raise FileNotFoundError(
            f"District boundary not found for AOI '{AOI}'.\n"
            f"Provide it via --aoi_geojson /path/to/{AOI}_boundary.geojson\n"
            f"or place it at data/raw/boundaries/{AOI}.geojson"
        )

    print(f"  Verified GeoJSON : {verified_path}")
    print(f"  AOI boundary     : {aoi_boundary}")
    run(
        f"python scripts/00_prepare_training_dataset.py "
        f"--verified \"{verified_path}\" "
        f"--aoi \"{aoi_boundary}\" "
        f"--name {AOI}"
    )
    # After prep, always use the .shp output for STEP 5
    LABEL_DIR = str(PREP_OUT_SHP)
    print(f"  Labels ready     : {LABEL_DIR}")

elif PREP_OUT_SHP.exists():
    print(f"  [cached] {PREP_OUT_SHP} already exists — skipping prep")
    LABEL_DIR = str(PREP_OUT_SHP)

elif PREP_OUT_GEOJSON.exists():
    # GeoJSON prep output exists but no .shp yet — still fine, rasterize accepts geojson
    print(f"  [cached] {PREP_OUT_GEOJSON} found — using for labels")
    LABEL_DIR = str(PREP_OUT_GEOJSON)

else:
    print(f"  WARNING: No --verified_geojson provided and no prepared labels found.")
    print(f"  Expected: {PREP_OUT_SHP}")
    print(f"  Provide --verified_geojson /path/to/{AOI}_verified.geojson to auto-prepare.")
    print(f"  Continuing — labels must exist at:")
    print(f"  data/processed/training/labels_coconut_{YEAR}_{AOI}.tif")

run('find . -name "._*" -type f -delete')


# STEP 1 -- Download Sentinel-2
print("\nSTEP 1/8  --  Download Sentinel-2")
if not args.skip_download:
    run(f"python scripts/00_download_sentinel2_best_per_year.py --year {YEAR} --aoi {AOI}")
else:
    print("  [skipped] --skip_download was set")

run('find . -name "._*" -type f -delete')


# STEP 2 -- Canopy Height
print("\nSTEP 2/8  --  Canopy Height")
if CANOPY_HEIGHT is None:
    print("  [skipped] --skip_canopy was set  ->  will build 11-band stack")
elif Path(CANOPY_HEIGHT).exists():
    print(f"  [cached]  {CANOPY_HEIGHT} already exists, skipping tile selection")
else:
    if args.canopy_tiles_dir:
        print(f"  Selecting tiles from: {args.canopy_tiles_dir}")
        run(
            f"python scripts/00b_download_canopy_height.py "
            f"--aoi {AOI} "
            f"--tiles_dir \"{args.canopy_tiles_dir}\""
        )
    else:
        print("  WARNING: canopy height .tif not found and --canopy_tiles_dir not provided.")
        print(f"           Expected: {CANOPY_HEIGHT}")
        print("  Options:")
        print("    a) Pass --canopy_tiles_dir /path/to/your/tiles")
        print("    b) Pass --skip_canopy to build an 11-band stack")
        print("  Continuing without canopy height...")
        CANOPY_HEIGHT = None


# STEP 3 -- AOI clip
print("\nSTEP 3/8  --  Prepare AOI")
run(f"python scripts/01_prepare_aoi_raw.py --year {YEAR} --aoi {AOI}")

run('find . -name "._*" -type f -delete')


# STEP 4 -- Build Stack
print("\nSTEP 4/8  --  Build Stack")
stack_cmd = f"python scripts/02_build_stack.py --year {YEAR} --aoi {AOI}"
if CANOPY_HEIGHT and Path(CANOPY_HEIGHT).exists():
    stack_cmd += f" --canopy_height \"{CANOPY_HEIGHT}\""
    print(f"  12-band stack  (canopy height: {CANOPY_HEIGHT})")
else:
    print("  11-band stack  (no canopy height)")
run(stack_cmd)

run('find . -name "._*" -type f -delete')


# STEP 5 -- Coconut Labels
print("\nSTEP 5/8  --  Coconut Labels")
if LABEL_DIR:
    if is_manual_labels(LABEL_DIR):
        print(f"  [Label mode] Manual polygons -> rasterizing  ({LABEL_DIR})")
        all_touched_flag = "--all_touched" if args.all_touched else ""
        run(
            f"python scripts/03_rasterize_manual_labels.py "
            f"--year {YEAR} --aoi {AOI} "
            f"--shp \"{LABEL_DIR}\" {all_touched_flag}"
        )
    else:
        print("  [Label mode] Directory -> using Descals et al. (2023) tiles")
        run(
            f"python scripts/03_download_coconut_labels.py "
            f"--year {YEAR} --aoi {AOI} --label_dir \"{LABEL_DIR}\""
        )
else:
    print("  [Label mode] No labels found -- skipping label step")
    print(f"               Labels must exist at:")
    print(f"               data/processed/training/labels_coconut_{YEAR}_{AOI}.tif")

run('find . -name "._*" -type f -delete')


# STEP 6 -- Create Patches
print("\nSTEP 6/8  --  Create Patches")
run(
    f"python -m scripts.dl.make_patches "
    f"--year {YEAR} --aoi {AOI} "
    f"--patch {PATCH} "
    f"--stride {STRIDE} "
    f"--pos_ratio {args.pos_ratio} "
    f"--neg_sample {args.neg_sample} "
    f"--neg_min_ratio {args.neg_min_ratio} "
    f"--nodata_tol {args.nodata_tol} "
    f"--ignore_tol {args.ignore_tol} "
    f"--dilate {args.dilate} "
    f"--clean"
)
run('find data/dl -name "._*" -type f -delete')


# STEP 7 -- Train
print("\nSTEP 7/8  --  Train")
if not args.skip_train:
    train_cmd = (
        f"python -m scripts.dl.train_unet "
        f"--year {YEAR} --aoi {AOI} "
        f"--epochs {args.epochs} "
        f"--batch {args.batch} "
        f"--lr {args.lr} "
        f"--patience {args.patience} "
        f"--threshold {THRESHOLD} "
        f"--pos_weight_floor {args.pos_weight_floor}"
    )
    if args.pretrained_ckpt:
        train_cmd += f" --pretrained_ckpt \"{args.pretrained_ckpt}\""
    run(train_cmd)
else:
    print("  [skipped] --skip_train was set")


# STEP 8 -- Predict
print("\nSTEP 8/8  --  Predict")
run(
    f"python -m scripts.dl.predict_unet "
    f"--year {YEAR} --aoi {AOI} "
    f"--patch {PATCH} "
    f"--stride {STRIDE} "
    f"--threshold {THRESHOLD}"
)


# FINAL -- Evaluate
print("\nFINAL STEP  --  Evaluate")
run(f"python scripts/evaluate_iou.py --year {YEAR} --aoi {AOI}")


print("\n" + "="*65)
print("FULL PIPELINE COMPLETE")
print("="*65)
