# run.py  --  Full coconut plantation mapping pipeline
#
# Usage (11-band, no canopy height):
#   python run.py --year 2022 --aoi puducherry
#
# Usage (12-band, TN-wide mosaic — RECOMMENDED):
#   # Download once for all TN districts:
#   python scripts/00_download_canopy_height_tn.py
#
#   # Then run any district:
#   python run.py --year 2022 --aoi puducherry  --canopy_tn
#   python run.py --year 2022 --aoi dindigul    --canopy_tn
#   python run.py --year 2022 --aoi coimbatore  --canopy_tn
#
# Usage (12-band, AOI-specific file):
#   python run.py --year 2022 --aoi puducherry \
#       --canopy_height data/raw/canopy_height_puducherry.tif

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description="Coconut Plantation Mapping Pipeline")

# --- Core ---
parser.add_argument("--year",           required=True,  help="Year (e.g. 2022)")
parser.add_argument("--aoi",            required=True,  help="AOI name (e.g. puducherry, dindigul)")

# --- Canopy height (mutually exclusive) ---
canopy_group = parser.add_mutually_exclusive_group()
canopy_group.add_argument(
    "--canopy_tn",
    action="store_true",
    default=False,
    help=(
        "Use the TN-wide canopy height mosaic (data/raw/canopy_height_tamilnadu.tif). "
        "Auto-clips to the AOI boundary at stack-build time. "
        "Run scripts/00_download_canopy_height_tn.py once before using this flag."
    ),
)
canopy_group.add_argument(
    "--canopy_height",
    default=None,
    metavar="PATH",
    help="Path to a pre-clipped AOI-specific canopy height GeoTIFF.",
)

# --- Patches ---
parser.add_argument("--patch",          type=int,   default=128)
parser.add_argument("--stride",         type=int,   default=64)
parser.add_argument("--pos_ratio",      type=float, default=0.02)
parser.add_argument("--neg_sample",     type=float, default=0.25)
parser.add_argument("--dilate",         type=int,   default=1)

# --- Training ---
parser.add_argument("--epochs",         type=int,   default=40)
parser.add_argument("--batch",          type=int,   default=8)
parser.add_argument("--lr",             type=float, default=1e-4)
parser.add_argument("--val_split",      type=float, default=0.2)
parser.add_argument("--patience",       type=int,   default=6)

# --- Threshold search ---
parser.add_argument("--threshold",      type=float, default=None,
                    help="Fixed threshold. If omitted, auto-search.")
parser.add_argument("--t_min",          type=float, default=0.20)
parser.add_argument("--t_max",          type=float, default=0.80)
parser.add_argument("--t_step",         type=float, default=0.02)
parser.add_argument("--t_fine_step",    type=float, default=0.005)
parser.add_argument("--thr_metric",     choices=["f1", "iou"], default="f1")

# --- Prediction ---
parser.add_argument("--pred_stride",    type=int,   default=32)
parser.add_argument("--pred_batch",     type=int,   default=8)

# --- Workflow control ---
parser.add_argument("--skip_stack",     action="store_true")
parser.add_argument("--skip_labels",    action="store_true")
parser.add_argument("--skip_patches",   action="store_true")
parser.add_argument("--skip_train",     action="store_true")
parser.add_argument("--skip_predict",   action="store_true")
parser.add_argument("--skip_evaluate",  action="store_true")
parser.add_argument("--workers",        type=int,   default=0)
parser.add_argument("--seed",           type=int,   default=42)
parser.add_argument("--clean_patches",  action="store_true")

args = parser.parse_args()


def run(cmd):
    print(f"\n{'='*60}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: Command exited with code {result.returncode}")
        sys.exit(result.returncode)


# ----------------------------------------------------------
# STEP 1 -- BUILD STACK
# ----------------------------------------------------------
if not args.skip_stack:
    cmd = [
        sys.executable, "scripts/02_build_stack.py",
        "--year", args.year,
        "--aoi",  args.aoi,
    ]
    if args.canopy_tn:
        cmd.append("--canopy_tn")
    elif args.canopy_height:
        cmd += ["--canopy_height", args.canopy_height]
    run(cmd)


# ----------------------------------------------------------
# STEP 2 -- LABELS
# ----------------------------------------------------------
if not args.skip_labels:
    run([
        sys.executable, "scripts/03_download_coconut_labels.py",
        "--year", args.year,
        "--aoi",  args.aoi,
    ])


# ----------------------------------------------------------
# STEP 3 -- PATCHES
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
print(f"   AOI    : {args.aoi}")
print(f"   Year   : {args.year}")
if args.canopy_tn:
    print(f"   Canopy : TN mosaic (auto-clipped to {args.aoi})")
elif args.canopy_height:
    print(f"   Canopy : {args.canopy_height}")
else:
    print(f"   Canopy : not included (11-band stack)")
print(f"{'='*60}")
