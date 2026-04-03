# run.py  --  Full coconut plantation mapping pipeline
#
# --------------------------------------------------------
#  ONE-SHOT USAGE (recommended)
# --------------------------------------------------------
#
#  Minimal (no canopy height, Descals labels):
#
#    python run.py --year 2022 --aoi puducherry \
#        --label_dir data/raw/training/GlobalCoconutLayer_2020_v1-2
#
#  With WRI/Meta canopy height (recommended):
#
#    python run.py --year 2022 --aoi puducherry \
#        --canopy_tn \
#        --label_dir data/raw/training/GlobalCoconutLayer_2020_v1-2
#
#  Custom AOI with all best-practice settings:
#
#    python run.py \
#        --year        2022 \
#        --aoi         dindigul \
#        --canopy_tn \
#        --label_mode  descals \
#        --label_dir   data/raw/training/GlobalCoconutLayer_2020_v1-2 \
#        --patch       256 \
#        --stride      128 \
#        --min_pos_px  10 \
#        --neg_sample  0.15 \
#        --pos_oversample 5.0 \
#        --epochs      40 \
#        --batch       8 \
#        --workers     4 \
#        --clean_patches
#
#  Manual polygon labels (field survey / hand-digitised):
#
#    python run.py --year 2022 --aoi thanjavur \
#        --label_mode manual \
#        --label_dir  data/raw/training/my_coconut_polygons.shp
#
# --------------------------------------------------------
#  SKIP FLAGS  (resume from any stage)
# --------------------------------------------------------
#
#  Already built the stack?     add  --skip_stack
#  Already prepared labels?     add  --skip_labels
#  Already generated patches?   add  --skip_patches
#  Already trained the model?   add  --skip_train
#  Already ran inference?       add  --skip_predict
#
#  Example -- resume from training only:
#    python run.py --year 2022 --aoi puducherry \
#        --skip_stack --skip_labels --skip_patches
#
# --------------------------------------------------------
#  CANOPY HEIGHT MODES
# --------------------------------------------------------
#
#  --canopy_tn          TN-wide mosaic (download once, reuse forever).
#                       Run scripts/00_download_canopy_height_tn.py first.
#  --canopy_height PATH Pre-clipped AOI-specific GeoTIFF.
#  (neither)            11-band stack, no canopy height.
#
# --------------------------------------------------------

import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(
    description="Coconut Plantation Mapping -- one-shot pipeline runner",
    formatter_class=argparse.RawTextHelpFormatter,
)

# --- Core ---
parser.add_argument("--year", required=True, help="Year to process (e.g. 2022)")
parser.add_argument("--aoi",  required=True,
                    help="AOI name matching data/raw/boundaries/{aoi}.shp\n"
                         "Examples: puducherry, coimbatore, dindigul, thanjavur")

# --- Label mode ---
parser.add_argument(
    "--label_mode",
    choices=["descals", "manual"],
    default="descals",
    help=(
        "Label source mode:\n"
        "  descals (default) : Descals et al. (2023) global coconut tiles\n"
        "                      Pass tile folder via --label_dir\n"
        "  manual            : Your own polygon shapefile\n"
        "                      Pass .shp path via --label_dir"
    ),
)
parser.add_argument(
    "--label_dir",
    default=None,
    metavar="PATH",
    help=(
        "descals mode : path to folder of Descals .tif tiles\n"
        "manual  mode : path to your coconut polygon .shp file"
    ),
)

# --- Canopy height (mutually exclusive) ---
canopy_group = parser.add_mutually_exclusive_group()
canopy_group.add_argument(
    "--canopy_tn",
    action="store_true",
    default=False,
    help="Use TN-wide canopy height mosaic, auto-clipped to AOI boundary.\n"
         "Run scripts/00_download_canopy_height_tn.py once first.",
)
canopy_group.add_argument(
    "--canopy_height",
    default=None,
    metavar="PATH",
    help="Path to a pre-clipped AOI-specific canopy height GeoTIFF.",
)

# --- Patch generation ---
parser.add_argument("--patch",      type=int,   default=256,
                    help="Patch size in pixels (default: 256)")
parser.add_argument("--stride",     type=int,   default=128,
                    help="Stride between patches (default: 128)")
parser.add_argument("--pos_ratio",  type=float, default=0.02,
                    help="Min coconut ratio to keep as positive patch (default: 0.02)")
parser.add_argument("--min_pos_px", type=int,   default=10,
                    help="Min coconut pixels to force-keep patch as positive\n"
                         "(hard-positive mining, default: 10)")
parser.add_argument("--neg_sample", type=float, default=0.15,
                    help="Keep probability for background patches (default: 0.15)")
parser.add_argument("--dilate",     type=int,   default=1,
                    help="Label mask dilation iterations (default: 1)")

# --- Training ---
parser.add_argument("--epochs",        type=int,   default=40,
                    help="Training epochs (default: 40)")
parser.add_argument("--batch",         type=int,   default=8,
                    help="Batch size (default: 8)")
parser.add_argument("--lr",            type=float, default=1e-4,
                    help="Learning rate (default: 1e-4)")
parser.add_argument("--val_split",     type=float, default=0.2,
                    help="Validation fraction, stratified (default: 0.2)")
parser.add_argument("--patience",      type=int,   default=6,
                    help="Early stopping patience (default: 6)")
parser.add_argument("--pos_oversample", type=float, default=5.0,
                    help="Positive patch oversampling multiplier for\n"
                         "WeightedRandomSampler (default: 5.0)")

# --- Threshold search ---
parser.add_argument("--threshold",   type=float, default=None,
                    help="Fixed decision threshold (0-1). If omitted, auto-searched.")
parser.add_argument("--t_min",       type=float, default=0.20)
parser.add_argument("--t_max",       type=float, default=0.80)
parser.add_argument("--t_step",      type=float, default=0.02)
parser.add_argument("--t_fine_step", type=float, default=0.005)
parser.add_argument("--thr_metric",  choices=["f1", "iou"], default="f1",
                    help="Metric to optimise during threshold search (default: f1)")

# --- Prediction ---
parser.add_argument("--pred_stride", type=int, default=32,
                    help="Sliding-window stride for inference (default: 32)")
parser.add_argument("--pred_batch",  type=int, default=8,
                    help="Batch size for inference (default: 8)")

# --- Workflow control ---
parser.add_argument("--skip_stack",    action="store_true", help="Skip Step 1 (stack build)")
parser.add_argument("--skip_labels",   action="store_true", help="Skip Step 2 (label prep)")
parser.add_argument("--skip_patches",  action="store_true", help="Skip Step 3 (patch generation)")
parser.add_argument("--skip_train",    action="store_true", help="Skip Step 4 (training)")
parser.add_argument("--skip_predict",  action="store_true", help="Skip Step 5 (inference)")
parser.add_argument("--skip_evaluate", action="store_true", help="Skip Step 6 (evaluation)")
parser.add_argument("--workers",       type=int, default=0,
                    help="DataLoader workers (0=main process, default: 0)")
parser.add_argument("--seed",          type=int, default=42)
parser.add_argument("--clean_patches", action="store_true",
                    help="Delete old patches before regenerating")

args = parser.parse_args()

# Validate
if not args.skip_labels and args.label_dir is None:
    print(
        "WARNING: --label_dir not provided.\n"
        "  descals mode: pass folder of Descals .tif tiles with --label_dir\n"
        "  manual  mode: pass your .shp polygon file with --label_dir\n"
        "  Use --skip_labels to suppress this warning."
    )


def run(cmd):
    print(f"\n{'='*65}")
    print("CMD: " + " ".join(cmd))
    print(f"{'='*65}\n")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: Command exited with code {result.returncode}")
        sys.exit(result.returncode)


# ============================================================
# STEP 1  --  BUILD FEATURE STACK
# Output: data/processed/{aoi}/stack_{year}.tif
# Bands:  11 (spectral) or 12 (+ canopy height)
# ============================================================
if not args.skip_stack:
    print("\nSTEP 1/6  --  Build feature stack")
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
else:
    print("STEP 1/6  --  Skipped (--skip_stack)")


# ============================================================
# STEP 2  --  PREPARE LABELS
# Output: data/processed/training/labels_coconut_{year}_{aoi}.tif
# ============================================================
if not args.skip_labels:
    print("\nSTEP 2/6  --  Prepare labels")
    if args.label_mode == "descals":
        if args.label_dir is None:
            print("ERROR: --label_dir required for --label_mode descals")
            sys.exit(1)
        run([
            sys.executable, "scripts/03_download_coconut_labels.py",
            "--year",      args.year,
            "--aoi",       args.aoi,
            "--label_dir", args.label_dir,
        ])
    elif args.label_mode == "manual":
        if args.label_dir is None:
            print("ERROR: --label_dir required for --label_mode manual")
            sys.exit(1)
        run([
            sys.executable, "scripts/03_rasterize_manual_labels.py",
            "--year", args.year,
            "--aoi",  args.aoi,
            "--shp",  args.label_dir,
        ])
else:
    print("STEP 2/6  --  Skipped (--skip_labels)")


# ============================================================
# STEP 3  --  GENERATE PATCHES
# Output: data/dl/{year}_{aoi}/images/*.npy  +  masks/*.npy
# ============================================================
if not args.skip_patches:
    print("\nSTEP 3/6  --  Generate patches")
    cmd = [
        sys.executable, "scripts/dl/make_patches.py",
        "--year",       args.year,
        "--aoi",        args.aoi,
        "--patch",      str(args.patch),
        "--stride",     str(args.stride),
        "--pos_ratio",  str(args.pos_ratio),
        "--min_pos_px", str(args.min_pos_px),
        "--neg_sample", str(args.neg_sample),
        "--dilate",     str(args.dilate),
        "--seed",       str(args.seed),
    ]
    if args.clean_patches:
        cmd.append("--clean")
    run(cmd)
else:
    print("STEP 3/6  --  Skipped (--skip_patches)")


# ============================================================
# STEP 4  --  TRAIN
# Output: models/{aoi}_{year}_best.pth  +  training_history.json
# ============================================================
if not args.skip_train:
    print("\nSTEP 4/6  --  Train UNet-Transformer")
    cmd = [
        sys.executable, "scripts/dl/train_unet.py",
        "--year",          args.year,
        "--aoi",           args.aoi,
        "--epochs",        str(args.epochs),
        "--batch",         str(args.batch),
        "--lr",            str(args.lr),
        "--val_split",     str(args.val_split),
        "--patience",      str(args.patience),
        "--pos_oversample",str(args.pos_oversample),
        "--workers",       str(args.workers),
        "--t_min",         str(args.t_min),
        "--t_max",         str(args.t_max),
        "--t_step",        str(args.t_step),
        "--t_fine_step",   str(args.t_fine_step),
        "--metric",        str(args.thr_metric),
        "--seed",          str(args.seed),
    ]
    if args.threshold is not None:
        cmd += ["--threshold", str(args.threshold)]
    run(cmd)
else:
    print("STEP 4/6  --  Skipped (--skip_train)")


# ============================================================
# STEP 5  --  INFERENCE
# Output: outputs/{aoi}_{year}_prediction.tif
# ============================================================
if not args.skip_predict:
    print("\nSTEP 5/6  --  Run inference")
    cmd = [
        sys.executable, "scripts/dl/predict_unet.py",
        "--year",   args.year,
        "--aoi",    args.aoi,
        "--patch",  str(args.patch),
        "--stride", str(args.pred_stride),
        "--batch",  str(args.pred_batch),
    ]
    if args.threshold is not None:
        cmd += ["--threshold", str(args.threshold)]
    run(cmd)
else:
    print("STEP 5/6  --  Skipped (--skip_predict)")


# ============================================================
# STEP 6  --  EVALUATE
# Output: outputs/{aoi}_{year}_metrics.json  +  confusion matrix
# ============================================================
if not args.skip_evaluate:
    print("\nSTEP 6/6  --  Evaluate")
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
else:
    print("STEP 6/6  --  Skipped (--skip_evaluate)")


# ============================================================
# DONE
# ============================================================
print(f"\n{'='*65}")
print("PIPELINE COMPLETE")
print(f"   AOI          : {args.aoi}")
print(f"   Year         : {args.year}")
print(f"   Labels       : {args.label_mode}" +
      (f"  ({args.label_dir})" if args.label_dir else ""))
if args.canopy_tn:
    print(f"   Canopy       : TN mosaic  (auto-clipped to {args.aoi})")
elif args.canopy_height:
    print(f"   Canopy       : {args.canopy_height}")
else:
    print(f"   Canopy       : not included  (11-band stack)")
print(f"   Patch size   : {args.patch}px  stride={args.stride}")
print(f"   min_pos_px   : {args.min_pos_px}")
print(f"   pos_oversample: {args.pos_oversample}x")
print(f"   Epochs       : {args.epochs}  batch={args.batch}  lr={args.lr}")
print(f"   Outputs in   : outputs/{args.aoi}_{args.year}_*")
print(f"{'='*65}")
