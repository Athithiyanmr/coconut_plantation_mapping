# scripts/dl/make_patches.py
#
# Change for 1% coconut class imbalance
# ----------------------------------------
# Hard-positive mining: a patch is only counted as "positive" if it
# contains at least --min_coconut_px coconut pixels (default: 10).
# Patches with 1-9 coconut pixels are treated as background and
# subject to the usual neg_sample probability.  This prevents the
# model from wasting capacity learning from near-empty patches that
# contribute almost no gradient signal for the coconut class.
#
# New argument:
#   --min_coconut_px  INT   Min coconut pixels to count as positive (default: 10)

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation
from tqdm import tqdm

logging.basicConfig(
    filename="make_patches.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Generate image/mask patch pairs for DL training")
parser.add_argument("--year",            required=True)
parser.add_argument("--aoi",             required=True)
parser.add_argument("--patch",           type=int,   default=128,  help="Patch size in pixels (default: 128)")
parser.add_argument("--stride",          type=int,   default=64,   help="Stride between patches (default: 64)")
parser.add_argument("--pos_ratio",       type=float, default=0.02, help="Min coconut ratio to keep as positive patch (legacy, overridden by --min_coconut_px if set)")
parser.add_argument("--min_coconut_px",  type=int,   default=10,
                    help="Min absolute coconut pixel count to treat patch as positive (default: 10). "
                         "Overrides --pos_ratio when > 0.")
parser.add_argument("--neg_sample",      type=float, default=0.25, help="Keep probability for background patches")
parser.add_argument("--dilate",          type=int,   default=1,    help="Dilation iterations on label mask (0=off)")
parser.add_argument("--seed",            type=int,   default=42,   help="Random seed")
parser.add_argument("--clean",           action="store_true",      help="Delete old patches before running")
args = parser.parse_args()

YEAR            = args.year
AOI             = args.aoi
PATCH           = args.patch
STRIDE          = args.stride
POS_RATIO       = args.pos_ratio
MIN_COCONUT_PX  = args.min_coconut_px   # hard-positive threshold
NEG_SAMPLE      = args.neg_sample
SEED            = args.seed

np.random.seed(SEED)

STACK = f"data/processed/{AOI}/stack_{YEAR}.tif"
LABEL = f"data/processed/training/labels_coconut_{YEAR}_{AOI}.tif"

OUT_BASE = Path(f"data/dl/{YEAR}_{AOI}")
OUT_IMG  = OUT_BASE / "images"
OUT_MSK  = OUT_BASE / "masks"
CFG_PATH = OUT_BASE / "patch_config.json"

if args.clean:
    print("Cleaning old patches...")
    shutil.rmtree(OUT_BASE, ignore_errors=True)
    log.info("Cleaned old patch directory")

OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_MSK.mkdir(parents=True, exist_ok=True)

log.info(
    f"Start: AOI={AOI}, year={YEAR}, patch={PATCH}, stride={STRIDE}, "
    f"pos_ratio={POS_RATIO}, min_coconut_px={MIN_COCONUT_PX}, "
    f"neg_sample={NEG_SAMPLE}, seed={SEED}"
)

# STEP 1 -- LOAD STACK (auto-detect band count)
print("\nLoading stack...")
if not Path(STACK).exists():
    raise FileNotFoundError(f"Stack not found: {STACK}\nRun 02_build_stack.py first.")

with rasterio.open(STACK) as src:
    img      = src.read().astype("float32")
    ref_meta = src.meta.copy()
    H, W     = src.height, src.width
    nodata   = src.nodata
    band_names = []
    for i in range(1, src.count + 1):
        tags = src.tags(i)
        band_names.append(tags.get("name", f"band_{i}"))

n_bands = img.shape[0]
print(f"   Stack shape : {img.shape}  (bands x H x W)")
print(f"   Bands ({n_bands}): {band_names}")
log.info(f"Stack loaded: shape={img.shape}, nodata={nodata}, bands={band_names}")

# STEP 2 -- NODATA -> NaN
if nodata is not None:
    img[img == nodata] = np.nan

# STEP 3 -- PER-BAND NORMALISATION (z-score on valid pixels)
print("Normalising bands...")
band_stats = []
for b in range(n_bands):
    band  = img[b]
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        band_stats.append({"band": band_names[b], "mean": None, "std": None})
        continue
    mu, sigma = float(valid.mean()), float(valid.std())
    if sigma > 0:
        img[b] = (band - mu) / sigma
    band_stats.append({"band": band_names[b], "mean": mu, "std": sigma})
    log.info(f"Band {b} ({band_names[b]}): mean={mu:.4f}, std={sigma:.4f}")

# STEP 4 -- LOAD & ALIGN LABEL MASK
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

coconut_total = int(label_mask.sum())
print(f"   Labels aligned : {label_mask.shape}")
print(f"   Coconut pixels : {coconut_total:,} / {H*W:,} ({100*label_mask.mean():.2f}%)")
log.info(f"Label aligned: coconut={coconut_total}, total={H*W}")

if coconut_total == 0:
    raise RuntimeError(
        "Label mask is empty (all zeros).\n"
        "Check that label and stack share the same CRS and spatial extent."
    )

# STEP 5 -- EDGE DILATION
if args.dilate > 0:
    print(f"   Dilating mask ({args.dilate} iteration(s))...")
    label_mask = binary_dilation(label_mask, iterations=args.dilate).astype("uint8")
    log.info(f"Dilation: iterations={args.dilate}")

# STEP 6 -- PATCH GENERATION
total_count    = 0
coconut_count  = 0   # patches with >= MIN_COCONUT_PX coconut pixels
near_pos_kept  = 0   # patches with 1..(MIN_COCONUT_PX-1) pixels, kept by neg_sample
empty_kept     = 0   # patches with 0 coconut pixels, kept by neg_sample
skipped_nodata = 0
skipped_empty  = 0

rows = list(range(0, H - PATCH + 1, STRIDE))
cols = list(range(0, W - PATCH + 1, STRIDE))
total_i = len(rows)
total_j = len(cols)

# Decide positive threshold: use absolute pixel count if --min_coconut_px > 0,
# otherwise fall back to the ratio-based threshold (legacy behaviour).
use_abs_threshold = MIN_COCONUT_PX > 0
if use_abs_threshold:
    pos_threshold_desc = f">= {MIN_COCONUT_PX} coconut pixels (hard-positive mining)"
else:
    pos_threshold_desc = f">= {POS_RATIO*100:.1f}% coconut ratio (legacy)"

print(f"\nPatch config : {PATCH}x{PATCH} px, stride={STRIDE}")
print(f"   Grid         : {total_i} x {total_j} = {total_i * total_j:,} candidates")
print(f"   Positive def : {pos_threshold_desc}")
print(f"   Neg sample   : {NEG_SAMPLE*100:.0f}% of non-positive patches kept")

pbar = tqdm(total=total_i * total_j, unit="patch")

for i in rows:
    for j in cols:
        pbar.update(1)
        x = img[:, i:i+PATCH, j:j+PATCH]
        y = label_mask[i:i+PATCH, j:j+PATCH]

        if x.shape[1:] != (PATCH, PATCH):
            continue

        nan_ratio = float(np.isnan(x).mean())
        if nan_ratio > 0.10:
            skipped_nodata += 1
            continue

        x = np.nan_to_num(x, nan=0.0)
        coconut_px    = int(y.sum())
        coconut_ratio = coconut_px / (PATCH * PATCH)

        # Determine whether this is a positive patch
        if use_abs_threshold:
            is_positive = coconut_px >= MIN_COCONUT_PX
        else:
            is_positive = coconut_ratio >= POS_RATIO

        if is_positive:
            # Always keep hard-positive patches
            coconut_count += 1
        else:
            # Non-positive: keep with neg_sample probability
            if np.random.rand() >= NEG_SAMPLE:
                skipped_empty += 1
                continue
            if coconut_px > 0:
                near_pos_kept += 1  # near-miss: has some coconut but below threshold
            else:
                empty_kept += 1

        np.save(OUT_IMG / f"img_{total_count:06d}.npy",  x.astype("float32"))
        np.save(OUT_MSK / f"mask_{total_count:06d}.npy", y.astype("uint8"))
        total_count += 1

pbar.close()

# STEP 7 -- SUMMARY + SAVE CONFIG
pos_pct      = 100 * coconut_count  / max(total_count, 1)
near_pct     = 100 * near_pos_kept  / max(total_count, 1)
neg_pct      = 100 * empty_kept     / max(total_count, 1)

print(f"\n{'='*56}")
print("Patch generation complete")
print(f"   Total saved          : {total_count:,}")
print(f"   Hard-positive patches: {coconut_count:,}  ({pos_pct:.1f}%)  [>= {MIN_COCONUT_PX} px]")
print(f"   Near-miss kept       : {near_pos_kept:,}  ({near_pct:.1f}%)  [1-{MIN_COCONUT_PX-1} px, sampled]")
print(f"   Empty kept           : {empty_kept:,}  ({neg_pct:.1f}%)  [0 px, sampled]")
print(f"   Skipped (nodata)     : {skipped_nodata:,}")
print(f"   Skipped (empty/ratio): {skipped_empty:,}")
print(f"   Output               : {OUT_BASE}")
print(f"{'='*56}")
log.info(
    f"Done: total={total_count}, hard_pos={coconut_count}, near_pos={near_pos_kept}, "
    f"empty={empty_kept}, nodata_skip={skipped_nodata}, empty_skip={skipped_empty}"
)

patch_meta = {
    "year": YEAR,
    "aoi": AOI,
    "stack": STACK,
    "label": LABEL,
    "patch": PATCH,
    "stride": STRIDE,
    "bands": band_names,
    "n_bands": n_bands,
    "pos_ratio": POS_RATIO,
    "min_coconut_px": MIN_COCONUT_PX,
    "neg_sample": NEG_SAMPLE,
    "dilate": args.dilate,
    "seed": SEED,
    "total_candidates": total_i * total_j,
    "total_saved": total_count,
    "hard_positive_patches": coconut_count,
    "near_pos_kept": near_pos_kept,
    "empty_kept": empty_kept,
    "skipped_nodata": skipped_nodata,
    "skipped_empty": skipped_empty,
    "band_stats": band_stats,
}

with open(CFG_PATH, "w") as f:
    json.dump(patch_meta, f, indent=2)

print(f"Patch config saved -> {CFG_PATH}")
log.info(f"Patch config saved: {CFG_PATH}")
