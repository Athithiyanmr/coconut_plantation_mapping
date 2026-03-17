# scripts/dl/make_patches.py

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
parser.add_argument("--year",       required=True)
parser.add_argument("--aoi",        required=True)
parser.add_argument("--patch",      type=int,   default=128,  help="Patch size in pixels (default: 128)")
parser.add_argument("--stride",     type=int,   default=64,   help="Stride between patches (default: 64)")
parser.add_argument("--pos_ratio",  type=float, default=0.02, help="Min coconut ratio to keep as positive patch (default: 0.02)")
parser.add_argument("--neg_sample", type=float, default=0.25, help="Keep probability for background patches (default: 0.25)")
parser.add_argument("--dilate",     type=int,   default=1,    help="Dilation iterations on label mask (default: 1, 0=off)")
parser.add_argument("--seed",       type=int,   default=42,   help="Random seed for reproducibility (default: 42)")
parser.add_argument("--clean",      action="store_true",      help="Delete old patches before running")
args = parser.parse_args()

YEAR       = args.year
AOI        = args.aoi
PATCH      = args.patch
STRIDE     = args.stride
POS_RATIO  = args.pos_ratio
NEG_SAMPLE = args.neg_sample
SEED       = args.seed

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
         f"pos_ratio={POS_RATIO}, neg_sample={NEG_SAMPLE}, seed={SEED}")

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
if nodata is not None:
    img[img == nodata] = np.nan

# -----------------------------------------
# STEP 3 -- PER-BAND NORMALISATION
# z-score on valid pixels only -- model-ready patches
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
# -----------------------------------------
print("\nLoading labels...")
if not Path(LABEL).exists():
    raise FileNotFoundError(
        f"Label mask not found: {LABEL}\n"
        "Run 03_download_coconut_labels.py first."
    )

label_mask = np.zeros((H, W), dtype="uint8")

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

coconut_total = label_mask.sum()
print(f"   Labels aligned : {label_mask.shape}")
print(f"   Coconut pixels : {coconut_total:,} / {H*W:,} ({100*label_mask.mean():.2f}%%)")
log.info(f"Label aligned: coconut={coconut_total}, total={H*W}")

if coconut_total == 0:
    raise RuntimeError(
        "Label mask is empty (all zeros).\n"
        "Check that label and stack share the same CRS and spatial extent."
    )

# -----------------------------------------
# STEP 5 -- EDGE DILATION
# -----------------------------------------
if args.dilate > 0:
    print(f"   Dilating mask ({args.dilate} iteration(s))...")
    label_mask = binary_dilation(label_mask, iterations=args.dilate).astype("uint8")
    log.info(f"Dilation: iterations={args.dilate}")

# -----------------------------------------
# STEP 6 -- PATCH GENERATION
# -----------------------------------------
total_count    = 0
coconut_count  = 0
empty_kept     = 0
skipped_nodata = 0
skipped_empty  = 0

total_i = len(range(0, H - PATCH + 1, STRIDE))
total_j = len(range(0, W - PATCH + 1, STRIDE))

print(f"\nPatch config : {PATCH}x{PATCH} px, stride={STRIDE}")
print(f"   Grid         : {total_i} x {total_j} = {total_i * total_j:,} candidates")
print(f"   Pos ratio    : >= {POS_RATIO*100:.1f}%% coconut -> always keep")
print(f"   Neg sample   : {NEG_SAMPLE*100:.0f}%% of background patches kept")

pbar = tqdm(total=total_i * total_j, unit="patch")

for i in range(0, H - PATCH + 1, STRIDE):
    for j in range(0, W - PATCH + 1, STRIDE):
        pbar.update(1)

        x = img[:, i:i+PATCH, j:j+PATCH]
        y = label_mask[i:i+PATCH, j:j+PATCH]

        # Shape guard
        if x.shape[1:] != (PATCH, PATCH):
            continue

        # Skip patches with >10%% nodata
        nan_ratio = np.isnan(x).mean()
        if nan_ratio > 0.10:
            skipped_nodata += 1
            continue

        # Fill remaining NaN with 0 before saving
        x = np.nan_to_num(x, nan=0.0)

        # -----------------------------------------
        # BALANCED SAMPLING
        # -----------------------------------------
        coconut_ratio = y.sum() / (PATCH * PATCH)

        if coconut_ratio >= POS_RATIO:
            coconut_count += 1
        else:
            if np.random.rand() >= NEG_SAMPLE:
                skipped_empty += 1
                continue
            empty_kept += 1

        # -----------------------------------------
        # SAVE
        # -----------------------------------------
        np.save(OUT_IMG / f"img_{total_count:06d}.npy",  x.astype("float32"))
        np.save(OUT_MSK / f"mask_{total_count:06d}.npy", y.astype("uint8"))
        total_count += 1

pbar.close()

# -----------------------------------------
# STEP 7 -- SUMMARY REPORT
# -----------------------------------------
pos_pct = 100 * coconut_count / max(total_count, 1)
neg_pct = 100 * empty_kept     / max(total_count, 1)

print(f"\n{'='*52}")
print(f"Patch generation complete")
print(f"   Total saved       : {total_count:,}")
print(f"   Coconut patches   : {coconut_count:,}  ({pos_pct:.1f}%%)")
print(f"   Empty kept        : {empty_kept:,}  ({neg_pct:.1f}%%)")
print(f"   Skipped (nodata)  : {skipped_nodata:,}")
print(f"   Skipped (empty)   : {skipped_empty:,}")
print(f"   Output            : {OUT_BASE}")
print(f"{'='*52}")
log.info(f"Done: total={total_count}, coconut={coconut_count}, "
         f"empty_kept={empty_kept}, nodata_skip={skipped_nodata}, empty_skip={skipped_empty}")
