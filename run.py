import subprocess
import argparse
import os


# --------------------------------
# ARGUMENTS
# --------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("--year", required=True)
parser.add_argument("--aoi", required=True)

# Coconut label source:
#   Pass a .shp file  -> rasterize manually digitized polygons
#   Pass a directory  -> use Descals et al. (2023) tiles
parser.add_argument("--label_dir", default=None,
                    help="Coconut label source: path to a .shp file (manual polygons) "
                         "or a directory containing Descals GeoTIFF tiles")

# DL tuning
parser.add_argument("--patch", type=int, default=64)
parser.add_argument("--stride", type=int, default=32)
parser.add_argument("--threshold", type=float, default=0.35)
parser.add_argument("--all_touched", action="store_true",
                    help="(Shapefile mode only) Burn pixels touching polygon edges")

# optional skip flags
parser.add_argument("--skip_download", action="store_true")
parser.add_argument("--skip_train", action="store_true")

args = parser.parse_args()

YEAR      = args.year
AOI       = args.aoi
LABEL_DIR = args.label_dir
PATCH     = args.patch
STRIDE    = args.stride
THRESHOLD = args.threshold


# --------------------------------
# Fix OpenMP crash (Mac)
# --------------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# --------------------------------
# Helper
# --------------------------------
def run(cmd):
    print("\nRunning:", cmd)
    subprocess.run(cmd, shell=True, check=True)


# --------------------------------
# Detect label mode
# --------------------------------
def is_shapefile(path):
    return path is not None and str(path).lower().endswith(".shp")


# --------------------------------
# CLEAN hidden files
# --------------------------------
run('find . -name "._*" -type f -delete')


# --------------------------------
# PIPELINE
# --------------------------------

# 1. Download Sentinel-2
if not args.skip_download:
    run(f"python scripts/00_download_sentinel2_best_per_year.py --year {YEAR} --aoi {AOI}")

run('find . -name "._*" -type f -delete')

# 2. AOI clip
run(f"python scripts/01_prepare_aoi_raw.py --year {YEAR} --aoi {AOI}")

run('find . -name "._*" -type f -delete')

# 3. Build stack
run(f"python scripts/02_build_stack.py --year {YEAR} --aoi {AOI}")

run('find . -name "._*" -type f -delete')

# 4. Coconut labels
if LABEL_DIR:
    if is_shapefile(LABEL_DIR):
        # --- Manual polygon mode ---
        print(f"\n[Label mode] Shapefile detected -> rasterizing manual polygons")
        all_touched_flag = "--all_touched" if args.all_touched else ""
        run(
            f"python scripts/03_rasterize_manual_labels.py "
            f"--year {YEAR} --aoi {AOI} "
            f"--shp {LABEL_DIR} "
            f"{all_touched_flag}"
        )
    else:
        # --- Descals tiles mode ---
        print(f"\n[Label mode] Directory detected -> using Descals et al. (2023) tiles")
        run(
            f"python scripts/03_download_coconut_labels.py "
            f"--year {YEAR} --aoi {AOI} --label_dir {LABEL_DIR}"
        )
else:
    print("\n[Label mode] --label_dir not provided -- skipping label step")
    print("             Labels must already exist at:")
    print(f"             data/processed/training/labels_coconut_{YEAR}_{AOI}.tif")

run('find . -name "._*" -type f -delete')

# 5. Create patches
run(
    f"python -m scripts.dl.make_patches "
    f"--year {YEAR} --aoi {AOI} "
    f"--patch {PATCH} --stride {STRIDE}"
)

# clean macOS hidden patch files
run('find data/dl -name "._*" -type f -delete')


# 6. Train
if not args.skip_train:
    run(f"python -m scripts.dl.train_unet --year {YEAR} --aoi {AOI}")


# 7. Predict
run(
    f"python -m scripts.dl.predict_unet "
    f"--year {YEAR} --aoi {AOI} "
    f"--patch {PATCH} --stride {STRIDE} "
    f"--threshold {THRESHOLD}"
)


# 8. Evaluate
run(f"python scripts/evaluate_iou.py --year {YEAR} --aoi {AOI}")


print("\nFULL PIPELINE COMPLETE")