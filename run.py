import subprocess
import argparse
import os
from pathlib import Path


# --------------------------------
# ARGUMENTS
# --------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--year",  required=True)
parser.add_argument("--aoi",   required=True)

# Coconut label source:
#   Pass a .shp file  -> rasterize manually digitized polygons
#   Pass a directory  -> use Descals et al. (2023) tiles
parser.add_argument("--label_dir", default=None,
                    help="Coconut label source: path to a .shp file (manual polygons) "
                         "or a directory containing Descals GeoTIFF tiles")

# WRI / Meta canopy height
# Three modes:
#   1. --canopy_tiles_dir  -> you have the raw tiles locally; script selects + clips them
#   2. --canopy_height     -> you already have the final clipped .tif ready
#   3. --skip_canopy       -> skip entirely (11-band stack)
parser.add_argument("--canopy_tiles_dir", default=None,
                    help="Path to your local folder of WRI/Meta canopy height .tif tiles. "
                         "The pipeline will auto-select tiles intersecting the AOI, "
                         "merge them, clip to AOI and save to data/raw/canopy_height/{aoi}.tif")
parser.add_argument("--canopy_height", default=None,
                    help="Path to an already-clipped canopy height .tif for the AOI. "
                         "Use this if you've already run the canopy step before.")
parser.add_argument("--skip_canopy", action="store_true",
                    help="Skip canopy height entirely (builds 11-band stack instead of 12)")

# DL tuning
parser.add_argument("--patch",     type=int,   default=64)
parser.add_argument("--stride",    type=int,   default=32)
parser.add_argument("--threshold", type=float, default=0.35)
parser.add_argument("--all_touched", action="store_true",
                    help="(Shapefile mode only) Burn pixels touching polygon edges")

# optional skip flags
parser.add_argument("--skip_download", action="store_true")
parser.add_argument("--skip_train",    action="store_true")

args = parser.parse_args()

YEAR      = args.year
AOI       = args.aoi
LABEL_DIR = args.label_dir
PATCH     = args.patch
STRIDE    = args.stride
THRESHOLD = args.threshold

# Resolve final canopy height .tif path
# Priority: --skip_canopy > --canopy_height (explicit) > auto path (from tiles or cached)
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
# --------------------------------
def is_shapefile(path):
    return path is not None and str(path).lower().endswith(".shp")


# --------------------------------
# CLEAN macOS hidden files
# --------------------------------
run('find . -name "._*" -type f -delete')


# --------------------------------
# PIPELINE
# --------------------------------

# STEP 1 — Download Sentinel-2
print("\nSTEP 1/8  --  Download Sentinel-2")
if not args.skip_download:
    run(f"python scripts/00_download_sentinel2_best_per_year.py --year {YEAR} --aoi {AOI}")
else:
    print("  [skipped] --skip_download was set")

run('find . -name "._*" -type f -delete')

# STEP 2 — Canopy height
print("\nSTEP 2/8  --  Canopy Height")
if CANOPY_HEIGHT is None:
    print("  [skipped] --skip_canopy was set  ->  will build 11-band stack")
elif Path(CANOPY_HEIGHT).exists():
    print(f"  [cached]  {CANOPY_HEIGHT} already exists, skipping tile selection")
else:
    # Need to build the clipped .tif from raw tiles
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

# STEP 3 — AOI clip
print("\nSTEP 3/8  --  Prepare AOI")
run(f"python scripts/01_prepare_aoi_raw.py --year {YEAR} --aoi {AOI}")

run('find . -name "._*" -type f -delete')

# STEP 4 — Build stack
print("\nSTEP 4/8  --  Build Stack")
stack_cmd = f"python scripts/02_build_stack.py --year {YEAR} --aoi {AOI}"
if CANOPY_HEIGHT and Path(CANOPY_HEIGHT).exists():
    stack_cmd += f" --canopy_height \"{CANOPY_HEIGHT}\""
    print(f"  12-band stack  (including canopy height: {CANOPY_HEIGHT})")
else:
    print("  11-band stack  (no canopy height)")
run(stack_cmd)

run('find . -name "._*" -type f -delete')

# STEP 5 — Coconut labels
print("\nSTEP 5/8  --  Coconut Labels")
if LABEL_DIR:
    if is_shapefile(LABEL_DIR):
        print("  [Label mode] Shapefile -> rasterizing manual polygons")
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
    print("  [Label mode] --label_dir not provided -- skipping label step")
    print(f"               Labels must exist at:")
    print(f"               data/processed/training/labels_coconut_{YEAR}_{AOI}.tif")

run('find . -name "._*" -type f -delete')

# STEP 6 — Create patches
print("\nSTEP 6/8  --  Create Patches")
run(
    f"python -m scripts.dl.make_patches "
    f"--year {YEAR} --aoi {AOI} "
    f"--patch {PATCH} --stride {STRIDE}"
)
run('find data/dl -name "._*" -type f -delete')

# STEP 7 — Train
print("\nSTEP 7/8  --  Train")
if not args.skip_train:
    run(f"python -m scripts.dl.train_unet --year {YEAR} --aoi {AOI}")
else:
    print("  [skipped] --skip_train was set")

# STEP 8 — Predict
print("\nSTEP 8/8  --  Predict")
run(
    f"python -m scripts.dl.predict_unet "
    f"--year {YEAR} --aoi {AOI} "
    f"--patch {PATCH} --stride {STRIDE} "
    f"--threshold {THRESHOLD}"
)

# STEP 9 — Evaluate
print("\nFINAL STEP  --  Evaluate")
run(f"python scripts/evaluate_iou.py --year {YEAR} --aoi {AOI}")


print("\n" + "="*65)
print("FULL PIPELINE COMPLETE")
print("="*65)
