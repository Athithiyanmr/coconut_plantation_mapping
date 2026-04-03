# scripts/dl/make_patches.py
#
# Import fix: sys.path injection so script runs from any working directory.
# Hard-positive mining via --min_pos_px.

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent   # scripts/dl/
_ROOT = _HERE.parent.parent               # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import logging
import shutil

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
parser.add_argument("--year",        required=True)
parser.add_argument("--aoi",         required=True)
parser.add_argument("--patch",       type=int,   default=256,  help="Patch size in pixels (default: 256)")
parser.add_argument("--stride",      type=int,   default=128,  help="Stride between patches (default: 128)")
parser.add_argument("--pos_ratio",   type=float, default=0.02, help="Min coconut ratio to keep as positive")
parser.add_argument("--neg_sample",  type=float, default=0.15, help="Keep probability for background patches")
parser.add_argument("--min_pos_px",  type=int,   default=10,
                    help="Min coconut pixels to force-keep a patch as positive (hard-positive mining)")
parser.add_argument("--dilate",      type=int,   default=1,    help="Label dilation iterations (0=off)")
parser.add_argument("--seed",        type=int,   default=42)
parser.add_argument("--clean",       action="store_true",      help="Delete old patches before running")
args = parser.parse_args()

YEAR       = args.year
AOI        = args.aoi
PATCH      = args.patch
STRIDE     = args.stride
POS_RATIO  = args.pos_ratio
NEG_SAMPLE = args.neg_sample
MIN_POS_PX = args.min_pos_px
SEED       = args.seed

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

log.info(f"Start: AOI={AOI}, year={YEAR}, patch={PATCH}, stride={STRIDE}, "
         f"pos_ratio={POS_RATIO}, neg_sample={NEG_SAMPLE}, min_pos_px={MIN_POS_PX}, seed={SEED}")

# STEP 1 -- LOAD STACK
print("\nLoading stack...")
if not Path(STACK).exists():
    raise FileNotFoundError(f"Stack not found: {STACK}\nRun 02_build_stack.py first.")

with rasterio.open(STACK) as src:
    img      = src.read().astype("float32")
    ref_meta = src.meta.copy()
    H, W     = src.height, src.width
    nodata   = src.nodata
    band_names = [src.tags(i).get("name", f"band_{i}") for i in range(1, src.count + 1)]

n_bands = img.shape[0]
print(f"   Stack  : {img.shape}   bands: {band_names}")
log.info(f"Stack: shape={img.shape}, nodata={nodata}, bands={band_names}")

# STEP 2 -- NODATA -> NaN
if nodata is not None:
    img[img == nodata] = np.nan

# STEP 3 -- PER-BAND Z-SCORE
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
        f"Label mask not found: {LABEL}\nRun 03_download_coconut_labels.py first."
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
print(f"   Labels : {label_mask.shape}  |  coconut pixels: {coconut_total:,} / {H*W:,} "
      f"({100*label_mask.mean():.2f}%)")
log.info(f"Label aligned: coconut={coconut_total}, total={H*W}")

if coconut_total == 0:
    raise RuntimeError(
        "Label mask is all zeros.\n"
        "Check that label and stack share the same CRS and spatial extent."
    )

# STEP 5 -- EDGE DILATION
if args.dilate > 0:
    print(f"   Dilating mask ({args.dilate} iteration(s))...")
    label_mask = binary_dilation(label_mask, iterations=args.dilate).astype("uint8")
    log.info(f"Dilation: iterations={args.dilate}")

# STEP 6 -- PATCH GENERATION
positive_count   = 0
borderline_pos   = 0
background_count = 0
total_count      = 0
skipped_nodata   = 0
skipped_empty    = 0

rows = list(range(0, H - PATCH + 1, STRIDE))
cols = list(range(0, W - PATCH + 1, STRIDE))

print(f"\nPatch config : {PATCH}x{PATCH} px | stride={STRIDE}")
print(f"   Grid         : {len(rows)} x {len(cols)} = {len(rows)*len(cols):,} candidates")
print(f"   Pos ratio    : >= {POS_RATIO*100:.1f}% coconut -> always keep")
print(f"   Min pos px   : >= {MIN_POS_PX} coconut pixels -> always keep (hard-positive mining)")
print(f"   Neg sample   : {NEG_SAMPLE*100:.0f}% of background patches kept")

pbar = tqdm(total=len(rows) * len(cols), unit="patch")

for i in rows:
    for j in cols:
        pbar.update(1)
        x = img[:, i:i+PATCH, j:j+PATCH]
        y = label_mask[i:i+PATCH, j:j+PATCH]

        if x.shape[1:] != (PATCH, PATCH):
            continue
        if float(np.isnan(x).mean()) > 0.10:
            skipped_nodata += 1
            continue

        x = np.nan_to_num(x, nan=0.0)
        coconut_px    = int(y.sum())
        coconut_ratio = float(coconut_px / (PATCH * PATCH))

        by_ratio    = (coconut_ratio >= POS_RATIO)
        by_px       = (coconut_px    >= MIN_POS_PX)
        is_positive = by_ratio or by_px

        if is_positive:
            positive_count += 1
            if by_px and not by_ratio:
                borderline_pos += 1
        else:
            if np.random.rand() >= NEG_SAMPLE:
                skipped_empty += 1
                continue
            background_count += 1

        np.save(OUT_IMG / f"img_{total_count:06d}.npy",  x.astype("float32"))
        np.save(OUT_MSK / f"mask_{total_count:06d}.npy", y.astype("uint8"))
        total_count += 1

pbar.close()

# STEP 7 -- SUMMARY
pos_pct = 100 * positive_count   / max(total_count, 1)
neg_pct = 100 * background_count / max(total_count, 1)

print(f"\n{'='*55}")
print("Patch generation complete")
print(f"   Total saved           : {total_count:,}")
print(f"   Positive patches      : {positive_count:,}  ({pos_pct:.1f}%)")
print(f"     borderline (px rule): {borderline_pos:,}")
print(f"   Background kept       : {background_count:,}  ({neg_pct:.1f}%)")
print(f"   Skipped (nodata)      : {skipped_nodata:,}")
print(f"   Skipped (empty)       : {skipped_empty:,}")
print(f"   Output                : {OUT_BASE}")
print(f"{'='*55}")
log.info(f"Done: total={total_count}, positive={positive_count}, borderline={borderline_pos}, "
         f"background={background_count}, nodata_skip={skipped_nodata}, empty_skip={skipped_empty}")

with open(CFG_PATH, "w") as f:
    json.dump({
        "year": YEAR, "aoi": AOI, "stack": STACK, "label": LABEL,
        "patch": PATCH, "stride": STRIDE, "bands": band_names, "n_bands": n_bands,
        "pos_ratio": POS_RATIO, "neg_sample": NEG_SAMPLE, "min_pos_px": MIN_POS_PX,
        "dilate": args.dilate, "seed": SEED,
        "total_saved": total_count, "positive_patches": positive_count,
        "borderline_positive_patches": borderline_pos,
        "background_kept": background_count,
        "skipped_nodata": skipped_nodata, "skipped_empty": skipped_empty,
        "band_stats": band_stats,
    }, f, indent=2)

print(f"Patch config saved -> {CFG_PATH}")
log.info(f"Patch config saved: {CFG_PATH}")
