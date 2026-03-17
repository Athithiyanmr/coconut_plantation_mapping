# scripts/dl/predict.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from scripts.dl.unet_transformer import UNetTransformer

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="predict.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Sliding-window inference for built-up area mapping")
parser.add_argument("--year",        required=True)
parser.add_argument("--aoi",         required=True)
parser.add_argument("--threshold",   type=float, default=0.35,  help="Binarization threshold (default: 0.35)")
parser.add_argument("--patch",       type=int,   default=128,   help="Patch size — must match training (default: 128)")
parser.add_argument("--stride",      type=int,   default=64,    help="Sliding window stride (default: 64)")
parser.add_argument("--in_channels", type=int,   default=11,    help="Input bands — must match training (default: 11)")
parser.add_argument("--batch_size",  type=int,   default=16,    help="Patches per inference batch (default: 16)")
parser.add_argument("--no_binary",   action="store_true",       help="Skip binary output, save probability only")
args = parser.parse_args()

YEAR       = args.year
AOI        = args.aoi
THRESHOLD  = args.threshold
PATCH      = args.patch
STRIDE     = args.stride
IN_CH      = args.in_channels
BATCH_SIZE = args.batch_size

# -----------------------------------------
# PATHS
# -----------------------------------------
STACK     = Path(f"data/processed/{AOI}/stack_{YEAR}.tif")
MODEL     = Path(f"models/unet_{YEAR}_{AOI}_best.pth")    # ✅ use best checkpoint

OUT_DIR   = Path(f"outputs/unet/{YEAR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PROB  = OUT_DIR / f"builtup_prob_{YEAR}_{AOI}.tif"
OUT_BIN   = OUT_DIR / f"builtup_binary_{YEAR}_{AOI}.tif"

# -----------------------------------------
# VALIDATE INPUTS
# -----------------------------------------
if not STACK.exists():
    raise FileNotFoundError(f"Stack not found: {STACK}\nRun stack.py first.")
if not MODEL.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL}\n"
        "Run train.py first, or check --year/--aoi match the saved model."
    )

log.info(f"Start: AOI={AOI}, year={YEAR}, patch={PATCH}, stride={STRIDE}, threshold={THRESHOLD}")

# -----------------------------------------
# DEVICE
# -----------------------------------------
device = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()           else
    torch.device("cpu")
)
print(f"💻 Device : {device}")
log.info(f"Device: {device}")

# -----------------------------------------
# LOAD STACK
# -----------------------------------------
print("\n🛰  Loading stack...")
with rasterio.open(STACK) as src:
    img    = src.read().astype("float32")
    meta   = src.meta.copy()
    H, W   = src.height, src.width
    nodata = src.nodata

bands = img.shape[0]
print(f"   Shape  : {img.shape}  (bands × H × W)")
print(f"   Nodata : {nodata}")
log.info(f"Stack loaded: shape={img.shape}")

# ✅ Nodata → NaN, then per-band z-score (mirrors stack.py + make_patches.py)
if nodata is not None:
    img[img == nodata] = np.nan

print("📐 Normalising bands...")
for b in range(img.shape[0]):
    band  = img[b]
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        continue
    mu, sigma = valid.mean(), valid.std()
    if sigma > 0:
        img[b] = (band - mu) / sigma

img = np.nan_to_num(img, nan=0.0)   # ✅ fill after normalisation

# -----------------------------------------
# LOAD MODEL
# ✅ Load full checkpoint — reads config to validate channel match
# -----------------------------------------
print("\n🧠 Loading model...")
checkpoint = torch.load(MODEL, map_location=device)

# Support both full checkpoint dict and raw state_dict
if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    state_dict  = checkpoint["model_state"]
    saved_cfg   = checkpoint.get("config", {})
    saved_ch    = saved_cfg.get("in_channels", IN_CH)
    saved_thr   = saved_cfg.get("threshold",   THRESHOLD)

    if saved_ch != IN_CH:
        print(f"⚠️  Checkpoint in_channels={saved_ch} — overriding --in_channels {IN_CH}")
        IN_CH = saved_ch

    if args.threshold == 0.35 and saved_thr != 0.35:
        print(f"   Using saved threshold: {saved_thr}")
        THRESHOLD = saved_thr
else:
    state_dict = checkpoint   # raw state_dict fallback

model = UNetTransformer(in_channels=IN_CH).to(device)
model.load_state_dict(state_dict)
model.eval()
torch.set_grad_enabled(False)

total_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters : {total_params:,}")
print(f"   In channels: {IN_CH}")
print(f"   Threshold  : {THRESHOLD}")
log.info(f"Model loaded: in_channels={IN_CH}, threshold={THRESHOLD}")

# -----------------------------------------
# PADDING (reflect — avoids border artifacts)
# -----------------------------------------
pad_h = (PATCH - H % PATCH) % PATCH
pad_w = (PATCH - W % PATCH) % PATCH

if pad_h > 0 or pad_w > 0:
    img = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    print(f"\n📐 Padded: {H}×{W} → {img.shape[1]}×{img.shape[2]}")

H_pad, W_pad = img.shape[1], img.shape[2]

# -----------------------------------------
# OUTPUT ACCUMULATORS
# -----------------------------------------
pred_sum = np.zeros((H_pad, W_pad), dtype="float32")
pred_cnt = np.zeros((H_pad, W_pad), dtype="float32")

# Pre-collect all patch coordinates
coords = [
    (i, j)
    for i in range(0, H_pad - PATCH + 1, STRIDE)
    for j in range(0, W_pad - PATCH + 1, STRIDE)
]

total_patches = len(coords)
print(f"\n🔍 Sliding window inference")
print(f"   Patch    : {PATCH}×{PATCH}  |  Stride : {STRIDE}")
print(f"   Patches  : {total_patches:,}  |  Batch  : {BATCH_SIZE}")
log.info(f"Inference: patches={total_patches}, batch={BATCH_SIZE}")

# -----------------------------------------
# BATCHED INFERENCE  ✅ much faster than one patch at a time
# -----------------------------------------
for batch_start in tqdm(range(0, total_patches, BATCH_SIZE), desc="Inference", unit="batch"):

    batch_coords = coords[batch_start: batch_start + BATCH_SIZE]
    batch_patches = []
    valid_coords  = []

    for (i, j) in batch_coords:
        patch = img[:, i:i+PATCH, j:j+PATCH]
        if np.isnan(patch).any() or patch.shape[1:] != (PATCH, PATCH):
            continue
        batch_patches.append(patch)
        valid_coords.append((i, j))

    if not batch_patches:
        continue

    x    = torch.from_numpy(np.stack(batch_patches)).to(device)  # (B, C, H, W)
    pred = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()       # (B, H, W)

    for idx, (i, j) in enumerate(valid_coords):
        pred_sum[i:i+PATCH, j:j+PATCH] += pred[idx]
        pred_cnt[i:i+PATCH, j:j+PATCH] += 1

# -----------------------------------------
# AVERAGE OVERLAPPING PREDICTIONS
# -----------------------------------------
pred_final = np.divide(
    pred_sum, pred_cnt,
    out=np.zeros_like(pred_sum),
    where=pred_cnt > 0,
)

pred_final = pred_final[:H, :W]   # ✅ crop back to original extent

covered = (pred_cnt[:H, :W] > 0).mean() * 100
print(f"\n   Coverage : {covered:.1f}% of pixels predicted")
log.info(f"Coverage: {covered:.1f}%")

# -----------------------------------------
# SAVE PROBABILITY MAP
# -----------------------------------------
meta.update(
    count=1,
    dtype="float32",
    nodata=0,
    compress="lzw",
    tiled=True,
    blockxsize=256,
    blockysize=256,
)

with rasterio.open(OUT_PROB, "w", **meta) as dst:
    dst.write(pred_final, 1)
    dst.update_tags(
        model=str(MODEL),
        year=YEAR,
        aoi=AOI,
        patch=str(PATCH),
        stride=str(STRIDE),
        threshold=str(THRESHOLD),
    )

prob_mb = OUT_PROB.stat().st_size / 1_000_000
print(f"\n✅ Probability map  → {OUT_PROB} ({prob_mb:.1f} MB)")
log.info(f"Prob saved: {OUT_PROB}, size={prob_mb:.1f}MB")

# -----------------------------------------
# BINARY OUTPUT
# -----------------------------------------
if not args.no_binary:
    binary = (pred_final > THRESHOLD).astype("uint8")
    built  = int(binary.sum())
    total  = H * W
    pct    = 100 * built / total

    meta.update(dtype="uint8", nodata=0)

    with rasterio.open(OUT_BIN, "w", **meta) as dst:
        dst.write(binary, 1)
        dst.update_tags(
            threshold=str(THRESHOLD),
            built_pixels=str(built),
            built_pct=f"{pct:.2f}",
        )

    bin_mb = OUT_BIN.stat().st_size / 1_000_000
    print(f"✅ Binary map       → {OUT_BIN} ({bin_mb:.1f} MB)")
    print(f"   Built pixels     : {built:,} / {total:,} ({pct:.2f}%)")
    log.info(f"Binary saved: {OUT_BIN}, built={built}, pct={pct:.2f}%")

# -----------------------------------------
# FINAL SUMMARY
# -----------------------------------------
print(f"\n{'='*55}")
print(f"✅ Prediction complete : {AOI} {YEAR}")
print(f"   Probability map : {OUT_PROB}")
if not args.no_binary:
    print(f"   Binary map      : {OUT_BIN}")
    print(f"   Built-up area   : {pct:.2f}% of AOI")
print(f"{'='*55}")
log.info(f"Prediction complete: AOI={AOI}, year={YEAR}, built_pct={pct:.2f}%")
