# run.py  --  Full coconut plantation mapping pipeline
#
# Usage (11-band, no canopy height):
#   python run.py --year 2022 --aoi puducherry
#
# Usage (12-band, WITH canopy height):
#   python run.py --year 2022 --aoi puducherry \
#       --canopy_height data/raw/canopy_height_puducherry.tif
#
# WRI/Meta canopy height download options:
#   GEE : ee.ImageCollection('projects/sat-io/open-datasets/facebook/meta-canopy-height')
#   AWS : aws s3 cp --no-sign-request \
#           s3://dataforgood-fb-data/forests/v1/alsgedi_global_v6_float/ . --recursive
#   Web : https://ai.meta.com/ai-for-good/datasets/canopy-height-maps/

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Coconut Plantation Mapping Pipeline")

# --- Core ---
parser.add_argument("--year",           required=True,  help="Year  (e.g. 2022)")
parser.add_argument("--aoi",            required=True,  help="AOI name (e.g. puducherry)")

# --- Canopy height (optional) ---
parser.add_argument("--canopy_height",  default=None,
                    help="Path to WRI/Meta canopy height GeoTIFF (.tif). "
                         "When provided, a CanopyHeight_m band is added as Band 12. "
                         "Coconut palms (15-30 m) are clearly separated from low crops.")

# --- Patches ---
parser.add_argument("--patch",          type=int,   default=128,   help="Patch size in pixels (default: 128)")
parser.add_argument("--stride",         type=int,   default=64,    help="Patch stride during training (default: 64)")
parser.add_argument("--pos_ratio",      type=float, default=0.02,  help="Min coconut ratio for positive patch (default: 0.02)")
parser.add_argument("--neg_sample",     type=float, default=0.25,  help="Keep rate for background patches (default: 0.25)")
parser.add_argument("--dilate",         type=int,   default=1,     help="Label dilation iterations (default: 1)")

# --- Training ---
parser.add_argument("--epochs",         type=int,   default=40,    help="Training epochs (default: 40)")
parser.add_argument("--batch",          type=int,   default=8,     help="Batch size (default: 8)")
parser.add_argument("--lr",             type=float, default=1e-4,  help="Learning rate (default: 1e-4)")
parser.add_argument("--val_split",      type=float, default=0.2,   help="Validation split (default: 0.2)")
parser.add_argument("--patience",       type=int,   default=6,     help="Early stopping patience (default: 6)")

# --- Threshold search ---
parser.add_argument("--threshold",      type=float, default=None,
                    help="Fixed binarisation threshold. If omitted, auto-search optimal threshold.")
parser.add_argument("--t_min",          type=float, default=0.20,  help="Threshold search min (default: 0.20)")
parser.add_argument("--t_max",          type=float, default=0.80,  help="Threshold search max (default: 0.80)")
parser.add_argument("--t_step",         type=float, default=0.02,  help="Coarse threshold step (default: 0.02)")
parser.add_argument("--t_fine_step",    type=float, default=0.005, help="Fine threshold step (default: 0.005)")
parser.add_argument("--thr_metric",     choices=["f1", "iou"], default="f1",
                    help="Metric for threshold selection (default: f1)")

# --- Prediction ---
parser.add_argument("--pred_stride",    type=int,   default=32,    help="Prediction sliding window stride (default: 32)")
parser.add_argument("--pred_batch",     type=int,   default=8,     help="Prediction batch size (default: 8)")

# --- Workflow control ---
parser.add_argument("--skip_stack",     action="store_true", help="Skip stack building")
parser.add_argument("--skip_labels",    action="store_true", help="Skip label download")
parser.add_argument("--skip_patches",   action="store_true", help="Skip patch generation")
parser.add_argument("--skip_train",     action="store_true", help="Skip training")
parser.add_argument("--skip_predict",   action="store_true", help="Skip prediction")
parser.add_argument("--skip_evaluate",  action="store_true", help="Skip evaluation")
parser.add_argument("--workers",        type=int,   default=0,     help="DataLoader workers (default: 0)")
parser.add_argument("--seed",           type=int,   default=42,    help="Random seed (default: 42)")
parser.add_argument("--clean_patches",  action="store_true", help="Delete old patches before making new ones")

args = parser.parse_args()


def run(cmd):
    """Run a subprocess command, print it, and exit on failure."""
    print(f"\n{'='*60}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: Command exited with code {result.returncode}")
        sys.exit(result.returncode)


# ----------------------------------------------------------
# STEP 1 -- BUILD STACK
#   Adds Band 12 (CanopyHeight_m) when --canopy_height is set
# ----------------------------------------------------------
if not args.skip_stack:
    cmd = [
        sys.executable, "scripts/02_build_stack.py",
        "--year", args.year,
        "--aoi",  args.aoi,
    ]
    if args.canopy_height:
        cmd += ["--canopy_height", args.canopy_height]
    run(cmd)


# ----------------------------------------------------------
# STEP 2 -- DOWNLOAD / PREPARE LABELS
# ----------------------------------------------------------
if not args.skip_labels:
    run([
        sys.executable, "scripts/03_download_coconut_labels.py",
        "--year", args.year,
        "--aoi",  args.aoi,
    ])


# ----------------------------------------------------------
# STEP 3 -- GENERATE PATCHES
#   make_patches.py auto-detects band count from the stack,
#   so 11 or 12 bands are handled transparently.
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# STEP 4 -- TRAIN
#   train_unet.py auto-detects in_channels from first patch.
#   Threshold auto-searched unless --threshold is fixed.
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# STEP 5 -- PREDICT
#   predict_unet.py reads in_channels + best_threshold
#   automatically from the saved checkpoint.
# ----------------------------------------------------------
if not args.skip_predict:
    cmd = [
        sys.executable, "scripts/dl/predict_unet.py",
        "--year",   args.year,
        "--aoi",    args.aoi,
        "--patch",  str(args.patch),
        "--stride", str(args.pred_stride),
    ]
    if args.threshold is not None:
        cmd += ["--threshold", str(args.threshold)]
    run(cmd)


# ----------------------------------------------------------
# STEP 6 -- EVALUATE
# ----------------------------------------------------------
if not args.skip_evaluate:
    cmd = [
        sys.executable, "scripts/evaluate_iou.py",
        "--year",        args.year,
        "--aoi",         args.aoi,
        "--t_min",       str(args.t_min),
        "--t_max",       str(args.t_max),
        "--t_step",      str(args.t_step),
        "--t_fine_step", str(args.t_fine_step),
        "--metric",      args.thr_metric,
    ]
    if args.threshold is not None:
        cmd += ["--threshold", str(args.threshold)]
    run(cmd)


# ----------------------------------------------------------
# DONE
# ----------------------------------------------------------
print(f"\n{'='*60}")
print("PIPELINE COMPLETE")
print(f"   AOI          : {args.aoi}")
print(f"   Year         : {args.year}")
if args.canopy_height:
    print(f"   Canopy Height: {args.canopy_height}  (Band 12 added)")
print(f"{'='*60}")
