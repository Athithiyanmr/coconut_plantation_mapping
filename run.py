
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Coconut Plantation Mapping Pipeline")

# --- Stack ---
parser.add_argument("--year",           required=True,  help="Year (e.g. 2022)")
parser.add_argument("--aoi",            required=True,  help="AOI name (e.g. puducherry)")
parser.add_argument("--canopy_height",  default=None,   help="Path to canopy height GeoTIFF (optional)")

# --- Patches ---
parser.add_argument("--patch",          type=int,   default=128,   help="Patch size (default: 128)")
parser.add_argument("--stride",         type=int,   default=64,    help="Patch stride (default: 64)")
parser.add_argument("--pos_ratio",      type=float, default=0.02,  help="Min coconut ratio for positive patches (default: 0.02)")
parser.add_argument("--neg_sample",     type=float, default=0.25,  help="Keep rate for background patches (default: 0.25)")
parser.add_argument("--dilate",         type=int,   default=1,     help="Label dilation iterations (default: 1)")

# --- Training ---
parser.add_argument("--epochs",         type=int,   default=40,    help="Training epochs (default: 40)")
parser.add_argument("--batch",          type=int,   default=8,     help="Batch size (default: 8)")
parser.add_argument("--lr",             type=float, default=1e-4,  help="Learning rate (default: 1e-4)")
parser.add_argument("--val_split",      type=float, default=0.2,   help="Validation split (default: 0.2)")
parser.add_argument("--patience",       type=int,   default=6,     help="Early stopping patience (default: 6)")

<<<<<<< Updated upstream
# WRI / Meta canopy height layer (optional)
# Dataset: Meta & WRI High Resolution Canopy Height Maps (2024), 1 m resolution
# Download options:
#   GEE  : ee.ImageCollection('projects/sat-io/open-datasets/facebook/meta-canopy-height')
#   AWS  : aws s3 cp --no-sign-request s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/ .
#   Meta : https://ai.meta.com/ai-for-good/datasets/canopy-height-maps/
parser.add_argument("--canopy_height", default=None,
                    help="Path to WRI/Meta canopy height GeoTIFF (.tif). "
                         "When provided, a CanopyHeight_m band is added to the stack "
                         "(Band 12). Coconut palms (15-30 m) are clearly separated "
                         "from low-canopy crops using this structural layer.")

# DL tuning
parser.add_argument("--patch", type=int, default=64)
parser.add_argument("--stride", type=int, default=32)
parser.add_argument("--threshold", type=float, default=0.35)
parser.add_argument("--all_touched", action="store_true",
                    help="(Shapefile mode only) Burn pixels touching polygon edges")
=======
# --- Threshold search ---
parser.add_argument("--threshold",      type=float, default=None,
                    help="Fixed threshold for binarisation. If omitted, auto-search optimal threshold.")
parser.add_argument("--t_min",          type=float, default=0.20,  help="Threshold search min (default: 0.20)")
parser.add_argument("--t_max",          type=float, default=0.80,  help="Threshold search max (default: 0.80)")
parser.add_argument("--t_step",         type=float, default=0.02,  help="Coarse threshold step (default: 0.02)")
parser.add_argument("--t_fine_step",    type=float, default=0.005, help="Fine threshold step (default: 0.005)")
parser.add_argument("--thr_metric",     choices=["f1", "iou"], default="f1",
                    help="Metric for threshold selection (default: f1)")
>>>>>>> Stashed changes

# --- Prediction ---
parser.add_argument("--pred_stride",    type=int,   default=32,    help="Prediction sliding window stride (default: 32)")
parser.add_argument("--pred_batch",     type=int,   default=8,     help="Prediction batch size (default: 8)")

# --- Workflow control ---
parser.add_argument("--skip_stack",     action="store_true",       help="Skip stack building")
parser.add_argument("--skip_labels",    action="store_true",       help="Skip label download")
parser.add_argument("--skip_patches",   action="store_true",       help="Skip patch generation")
parser.add_argument("--skip_train",     action="store_true",       help="Skip training")
parser.add_argument("--skip_predict",   action="store_true",       help="Skip prediction")
parser.add_argument("--skip_evaluate",  action="store_true",       help="Skip evaluation")
parser.add_argument("--workers",        type=int,   default=0,     help="DataLoader workers (default: 0)")
parser.add_argument("--seed",           type=int,   default=42,    help="Random seed (default: 42)")
parser.add_argument("--clean_patches",  action="store_true",       help="Delete old patches before making new ones")

args = parser.parse_args()

<<<<<<< Updated upstream
YEAR          = args.year
AOI           = args.aoi
LABEL_DIR     = args.label_dir
CANOPY_HEIGHT = args.canopy_height
PATCH         = args.patch
STRIDE        = args.stride
THRESHOLD     = args.threshold
=======
>>>>>>> Stashed changes

def run(cmd):
    """Run a subprocess command and exit on failure."""
    print(f"\n{'='*60}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: Command exited with code {result.returncode}")
        sys.exit(result.returncode)


# STEP 1 -- BUILD STACK
if not args.skip_stack:
    cmd = [
        sys.executable, "scripts/02_build_stack.py",
        "--year", args.year,
        "--aoi",  args.aoi,
    ]
    if args.canopy_height:
        cmd += ["--canopy_height", args.canopy_height]
    run(cmd)

# STEP 2 -- DOWNLOAD / PREPARE LABELS
if not args.skip_labels:
    run([
        sys.executable, "scripts/03_download_coconut_labels.py",
        "--year", args.year,
        "--aoi",  args.aoi,
    ])

# STEP 3 -- GENERATE PATCHES
if not args.skip_patches:
    cmd = [
        sys.executable, "scripts/dl/make_patches.py",
        "--year",       args.year,
        "--aoi",        args.aoi,
        "--patch",      str(args.patch),
        "--stride",     str(args.stride),
        "--pos_ratio",  str(args.pos_ratio),
        "--neg_sample", str(args.neg_sample),
        "--dilate",     str(args.dilate),
        "--seed",       str(args.seed),
    ]
    if args.clean_patches:
        cmd.append("--clean")
    run(cmd)

<<<<<<< Updated upstream

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

# 3. Build stack (optionally with canopy height)
stack_cmd = f"python scripts/02_build_stack.py --year {YEAR} --aoi {AOI}"
if CANOPY_HEIGHT:
    stack_cmd += f" --canopy_height {CANOPY_HEIGHT}"
run(stack_cmd)

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
=======
# STEP 4 -- TRAIN
>>>>>>> Stashed changes
if not args.skip_train:
    cmd = [
        sys.executable, "scripts/dl/train_unet.py",
        "--year",        args.year,
        "--aoi",         args.aoi,
        "--epochs",      str(args.epochs),
        "--batch",       str(args.batch),
        "--lr",          str(args.lr),
        "--val_split",   str(args.val_split),
        "--patience",    str(args.patience),
        "--workers",     str(args.workers),
        "--t_min",       str(args.t_min),
        "--t_max",       str(args.t_max),
        "--t_step",      str(args.t_step),
        "--t_fine_step", str(args.t_fine_step),
        "--metric",      args.thr_metric,
    ]
    if args.threshold is not None:
        cmd += ["--threshold", str(args.threshold)]
    run(cmd)

# STEP 5 -- PREDICT
if not args.skip_predict:
    cmd = [
        sys.executable, "scripts/dl/predict_unet.py",
        "--year",       args.year,
        "--aoi",        args.aoi,
        "--patch",      str(args.patch),
        "--stride",     str(args.pred_stride),
        "--batch_size", str(args.pred_batch),
    ]
    if args.threshold is not None:
        cmd += ["--threshold", str(args.threshold)]
    run(cmd)

# STEP 6 -- EVALUATE
if not args.skip_evaluate:
    cmd = [
        sys.executable, "scripts/evaluate_iou.py",
        "--year",       args.year,
        "--aoi",        args.aoi,
        "--t_min",      str(args.t_min),
        "--t_max",      str(args.t_max),
        "--t_step",     str(args.t_step),
        "--t_fine_step",str(args.t_fine_step),
        "--metric",     args.thr_metric,
    ]
    if args.threshold is not None:
        cmd += ["--threshold", str(args.threshold)]
    run(cmd)

<<<<<<< Updated upstream

# 8. Evaluate
run(f"python scripts/evaluate_iou.py --year {YEAR} --aoi {AOI}")


print("\nFULL PIPELINE COMPLETE")
=======
# DONE
print(f"\n{'='*60}")
print("PIPELINE COMPLETE")
print(f"   AOI   : {args.aoi}")
print(f"   Year  : {args.year}")
print(f"{'='*60}")
>>>>>>> Stashed changes
