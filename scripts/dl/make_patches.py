# scripts/dl/make_patches.py
#
# Generates image/mask patch pairs for DL training.
#
# Label values from rasterize step:
#   1   = confirmed coconut   -> positive patch
#   0   = confirmed not-coconut -> negative patch (confirmed)
#   255 = unlabeled / ignore  -> SKIP (do not train on these)

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation
from tqdm import tqdm

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="make_patches.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Generate image/mask patch pairs for DL training")
parser.add_argument("--year",          required=True)
parser.add_argument("--aoi",           required=True)
parser.add_argument("--patch",         type=int,   default=128,   help="Patch size in pixels (default: 128)")
parser.add_argument("--stride",        type=int,   default=64,    help="Stride between patches (default: 64)")
parser.add_argument("--pos_ratio",     type=float, default=0.005, help="Min coconut ratio to keep as positive patch (default: 0.005 = 0.5%%)")
parser.add_argument("--neg_sample",    type=float, default=0.50,  help="Keep probability for confirmed-negative patches (default: 0.50)")
parser.add_argument("--neg_min_ratio", type=float, default=0.01,  help="Min fraction of gt==0 pixels required to count as background patch (default: 0.01 = 1%%)")
parser.add_argument("--nodata_tol",    type=float, default=0.05,  help="Max nodata fraction allowed in a patch (default: 0.05 = 5%%)")
parser.add_argument("--ignore_tol",    type=float, default=0.80,  help="Skip patch if >X fraction of mask is 255/ignore (default: 0.80)")
parser.add_argument("--dilate",        type=int,   default=2,     help="Dilation iterations on coconut label mask (default: 2)")
parser.add_argument("--seed",          type=int,   default=42,    help="Random seed for reproducibility (default: 42)")
parser.add_argument("--clean",         action="store_true",       help="Delete old patches before running")
args = parser.parse_args()

YEAR          = args.year
AOI           = args.aoi
PATCH         = args.patch
STRIDE        = args.stride
POS_RATIO     = args.pos_ratio
NEG_SAMPLE    = args.neg_sample
NEG_MIN_RATIO = args.neg_min_ratio
NODATA_TOL    = args.nodata_tol
IGNORE_TOL    = args.ignore_tol
SEED          = args.seed

np.random.seed(SEED)

STACK = f"data/processed/{AOI}/stack_{YEAR}.tif"
LABEL = f"data/processed/training/labels_coconut_{YEAR}_{AOI}.tif"

OUT_BASE = Path(f"data/dl/{YEAR}_{AOI}")
OUT_IMG  = OUT_BASE / "images"
OUT_MSK  = OUT_BASE / "masks"

# -----------------------------------------
# CLEAN OLD PATCHES
# -----------------------------------------
if args.clean:
    print("Cleaning old patches...")
    shutil.rmtree(OUT_BASE, ignore_errors=True)
    log.info("Cleaned old patch directory")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MSK.mkdir(parents=True, exist_ok=True)

log.info(f"Start: AOI={AOI}, year={YEAR}, patch={PATCH}, stride={STRIDE}, "
         f"pos_ratio={POS_RATIO}, neg_sample={NEG_SAMPLE}, neg_min_ratio={NEG_MIN_RATIO}, "
         f"nodata_tol={NODATA_TOL}, seed={SEED}")

# -----------------------------------------
# STEP 1 -- LOAD STACK
# -----------------------------------------
print("\nLoading stack...")
if not Path(STACK).exists():
    raise FileNotFoundError(f"Stack not found: {STACK}\nRun 02_build_stack.py first.")

with rasterio.open(STACK) as src:
    img      = src.read().astype("float32")
    ref_meta = src.meta.copy()
    H, W     = src.height, src.width
    nodata   = src.nodata

print(f"   Stack shape : {img.shape}  (bands x H x W)")
log.info(f"Stack loaded: shape={img.shape}, nodata={nodata}")

# -----------------------------------------
# STEP 2 -- NODATA -> NaN
# -----------------------------------------
if nodata is not None and not np.isnan(nodata):
    img[img == nodata] = np.nan

nan_frac = np.isnan(img[0]).mean()
print(f"   NaN fraction  : {100*nan_frac:.1f}%  (band 0 / B02)")
if nan_frac > 0.50:
    print("   WARNING: >50% of stack is NaN. Re-run 01_prepare_aoi_raw.py + 02_build_stack.py.")

# -----------------------------------------
# STEP 3 -- PER-BAND NORMALISATION
# z-score on valid pixels only
# -----------------------------------------
print("Normalising bands...")
for b in range(img.shape[0]):
    band  = img[b]
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        continue
    mu, sigma = valid.mean(), valid.std()
    if sigma > 0:
        img[b] = (band - mu) / sigma
    log.info(f"Band {b}: mean={mu:.4f}, std={sigma:.4f}")

# -----------------------------------------
# STEP 4 -- LOAD & ALIGN LABEL MASK
# Values: 1=coconut, 0=not-coconut, 255=ignore
# -----------------------------------------
print("\nLoading labels...")
if not Path(LABEL).exists():
    raise FileNotFoundError(
        f"Label mask not found: {LABEL}\n"
        "Run 03_rasterize_manual_labels.py first."
    )

# Use 255 as default (ignore) for unlabeled areas
label_mask = np.full((H, W), 255, dtype="uint8")

with rasterio.open(LABEL) as src:
    reproject(
        source=src.read(1),
        destination=label_mask,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=ref_meta["transform"],
        dst_crs=ref_meta["crs"],
        resampling=Resampling.nearest,
    )

coconut_total = int((label_mask == 1).sum())
neg_total     = int((label_mask == 0).sum())
ignore_total  = int((label_mask == 255).sum())
print(f"   Labels aligned   : {label_mask.shape}")
print(f"   Coconut (1)      : {coconut_total:,} / {H*W:,} ({100*coconut_total/(H*W):.3f}%%)")
print(f"   Not-coconut (0)  : {neg_total:,} / {H*W:,} ({100*neg_total/(H*W):.3f}%%)")
print(f"   Ignore (255)     : {ignore_total:,} / {H*W:,} ({100*ignore_total/(H*W):.1f}%%)")
log.info(f"Label aligned: coconut={coconut_total}, neg={neg_total}, ignore={ignore_total}")

if coconut_total == 0:
    raise RuntimeError(
        "Label mask has no coconut pixels (class=1).\n"
        "Check that 03_rasterize_manual_labels.py ran correctly."
    )

# -----------------------------------------
# STEP 5 -- EDGE DILATION on coconut mask only
# Only dilate class=1 into ignore areas (255).
# Never overwrite confirmed negatives (0).
# -----------------------------------------
if args.dilate > 0:
    print(f"   Dilating coconut mask ({args.dilate} iteration(s))...")
    coconut_binary = (label_mask == 1).astype("uint8")
    dilated        = binary_dilation(coconut_binary, iterations=args.dilate).astype("uint8")
    # Only expand into ignore areas (255), never overwrite confirmed negatives (0)
    expand_mask = (dilated == 1) & (label_mask == 255)
    label_mask[expand_mask] = 1
    dilated_total = int((label_mask == 1).sum())
    print(f"   After dilation   : {dilated_total:,} coconut pixels")
    log.info(f"Dilation: iterations={args.dilate}, pixels after={dilated_total}")

# -----------------------------------------
# STEP 6 -- PATCH GENERATION
#
# Patch classification (mutually exclusive, checked in order):
#
#   POSITIVE patch:
#     - coconut_ratio (gt==1 / total) >= POS_RATIO
#     - Always saved
#
#   BACKGROUND patch (confirmed negative):
#     - coconut_ratio < POS_RATIO
#     - neg_ratio (gt==0 / total) >= NEG_MIN_RATIO  <-- KEY FIX
#     - Randomly sampled at NEG_SAMPLE probability
#
#   SKIP (pure ignore):
#     - No coconut AND no meaningful gt==0 pixels
#     - Also: >IGNORE_TOL fraction is 255
#     - Also: >NODATA_TOL fraction is NaN
# -----------------------------------------
total_count    = 0
coconut_count  = 0
confirmed_neg  = 0
skipped_nodata = 0
skipped_ignore = 0

total_i = len(range(0, H - PATCH + 1, STRIDE))
total_j = len(range(0, W - PATCH + 1, STRIDE))

print(f"\nPatch config : {PATCH}x{PATCH} px, stride={STRIDE}")
print(f"   Grid          : {total_i} x {total_j} = {total_i * total_j:,} candidates")
print(f"   Pos ratio     : >= {POS_RATIO*100:.2f}%% coconut -> always keep")
print(f"   Neg min ratio : >= {NEG_MIN_RATIO*100:.1f}%% gt==0 pixels required for bg patch")
print(f"   Neg sample    : {NEG_SAMPLE*100:.0f}%% of confirmed-negative patches kept")
print(f"   NoData tol    : skip patch if >{NODATA_TOL*100:.0f}%% NaN pixels")
print(f"   Ignore tol    : skip patch if >{IGNORE_TOL*100:.0f}%% ignore pixels")

pbar = tqdm(total=total_i * total_j, unit="patch")

for i in range(0, H - PATCH + 1, STRIDE):
    for j in range(0, W - PATCH + 1, STRIDE):
        pbar.update(1)

        x = img[:, i:i+PATCH, j:j+PATCH]
        y = label_mask[i:i+PATCH, j:j+PATCH]

        if x.shape[1:] != (PATCH, PATCH):
            continue

        # Skip nodata patches
        nan_ratio = np.isnan(x).mean()
        if nan_ratio > NODATA_TOL:
            skipped_nodata += 1
            continue

        # Skip patches that are mostly unlabeled/ignore
        ignore_ratio = (y == 255).mean()
        if ignore_ratio > IGNORE_TOL:
            skipped_ignore += 1
            continue

        x = np.nan_to_num(x, nan=0.0)

        # -----------------------------------------
        # PATCH CLASSIFICATION
        # -----------------------------------------
        patch_pixels  = PATCH * PATCH
        coconut_ratio = (y == 1).sum() / patch_pixels
        neg_ratio     = (y == 0).sum() / patch_pixels

        if coconut_ratio >= POS_RATIO:
            # POSITIVE patch -- always save
            coconut_count += 1

        elif neg_ratio >= NEG_MIN_RATIO:
            # CONFIRMED NEGATIVE patch -- has real gt==0 pixels
            # Random subsample to balance classes
            if np.random.rand() >= NEG_SAMPLE:
                skipped_ignore += 1
                continue
            confirmed_neg += 1

        else:
            # Pure ignore patch -- no useful labels at all, skip
            skipped_ignore += 1
            continue

        # Save patch and mask (mask keeps 0/1/255 values intact)
        np.save(OUT_IMG / f"img_{total_count:06d}.npy",  x.astype("float32"))
        np.save(OUT_MSK / f"mask_{total_count:06d}.npy", y.astype("uint8"))
        total_count += 1

pbar.close()

# -----------------------------------------
# STEP 7 -- SUMMARY REPORT
# -----------------------------------------
pos_pct = 100 * coconut_count / max(total_count, 1)
neg_pct = 100 * confirmed_neg  / max(total_count, 1)

print(f"\n{'='*52}")
print(f"Patch generation complete")
print(f"   Total saved          : {total_count:,}")
print(f"   Coconut patches      : {coconut_count:,}  ({pos_pct:.1f}%%)")
print(f"   Background patches   : {confirmed_neg:,}  ({neg_pct:.1f}%%)")
print(f"   Pos:Neg ratio        : 1:{confirmed_neg // max(coconut_count, 1)}")
print(f"   Skipped (nodata)     : {skipped_nodata:,}")
print(f"   Skipped (ignore/bg)  : {skipped_ignore:,}")
print(f"   Output               : {OUT_BASE}")
print(f"{'='*52}")

if confirmed_neg == 0:
    print("\n  WARNING: No background patches saved!")
    print("  Your gt==0 labels may be too sparse.")
    print(f"  Try reducing --neg_min_ratio below {NEG_MIN_RATIO} (e.g. 0.005)")
    print("  or adding more not-coconut polygons in your label file.")

log.info(f"Done: total={total_count}, coconut={coconut_count}, "
         f"confirmed_neg={confirmed_neg}, nodata_skip={skipped_nodata}, ignore_skip={skipped_ignore}")
