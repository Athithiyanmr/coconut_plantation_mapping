# scripts/dl/predict_unet.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from tqdm import tqdm
import torch

from scripts.dl.unet_transformer import UNetTransformer

logging.basicConfig(
    filename="predict.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Run UNet inference on a stack image")
parser.add_argument("--year",       required=True)
parser.add_argument("--aoi",        required=True)
parser.add_argument("--patch",      type=int,   default=128,  help="Patch size in pixels (default: 128)")
parser.add_argument("--stride",     type=int,   default=32,   help="Stride between patches (default: 32)")
parser.add_argument("--threshold",  type=float, default=None,
                    help="Binarization threshold. If omitted, reads best_threshold from checkpoint.")
parser.add_argument("--in_channels",type=int,   default=None,
                    help="Override input band count. If omitted, read from checkpoint config.")
parser.add_argument("--blend",      choices=["hann", "flat"], default="hann",
                    help="Overlap blending window: hann (smooth) or flat (average). Default: hann")
args = parser.parse_args()

YEAR   = args.year
AOI    = args.aoi
PATCH  = args.patch
STRIDE = args.stride

STACK      = f"data/processed/{AOI}/stack_{YEAR}.tif"
MODEL_DIR  = Path("models")
BEST_CKPT  = MODEL_DIR / f"unet_{YEAR}_{AOI}_best.pth"
OUT_DIR    = Path(f"outputs/{AOI}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PROB   = OUT_DIR / f"coconut_prob_{YEAR}_{AOI}.tif"
OUT_MASK   = OUT_DIR / f"coconut_mask_{YEAR}_{AOI}.tif"

log.info(f"Start: AOI={AOI}, year={YEAR}, patch={PATCH}, stride={STRIDE}")

# -----------------------------------------
# DEVICE
# -----------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Device : {device}")

# -----------------------------------------
# LOAD CHECKPOINT
# -----------------------------------------
if not BEST_CKPT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {BEST_CKPT}\n"
        "Run train_unet.py first."
    )

ckpt = torch.load(BEST_CKPT, map_location=device)
ckpt_cfg = ckpt.get("config", {})

# Resolve in_channels: CLI override > checkpoint config > fallback 11
if args.in_channels is not None:
    IN_CH = args.in_channels
elif "in_channels" in ckpt_cfg:
    IN_CH = int(ckpt_cfg["in_channels"])
else:
    IN_CH = 11
    print("WARNING: in_channels not in checkpoint config, defaulting to 11.")

# Resolve threshold: CLI override > checkpoint best_threshold > fallback 0.35
if args.threshold is not None:
    THRESHOLD = args.threshold
    print(f"Threshold  : {THRESHOLD:.3f}  (from --threshold)")
elif "best_threshold" in ckpt_cfg:
    THRESHOLD = float(ckpt_cfg["best_threshold"])
    print(f"Threshold  : {THRESHOLD:.3f}  (from checkpoint best_threshold)")
else:
    THRESHOLD = 0.35
    print(f"Threshold  : {THRESHOLD:.3f}  (fallback default)")

print(f"Channels   : {IN_CH}")
print(f"Checkpoint : {BEST_CKPT}  (epoch {ckpt.get('epoch', '?')})")
log.info(f"Checkpoint: in_channels={IN_CH}, threshold={THRESHOLD}, epoch={ckpt.get('epoch')}")

# -----------------------------------------
# MODEL
# -----------------------------------------
model = UNetTransformer(in_channels=IN_CH).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters : {total_params:,}")
log.info(f"Model loaded: params={total_params}")

# -----------------------------------------
# LOAD STACK
# -----------------------------------------
if not Path(STACK).exists():
    raise FileNotFoundError(f"Stack not found: {STACK}\nRun 02_build_stack.py first.")

print(f"\nLoading stack: {STACK}")
with rasterio.open(STACK) as src:
    img      = src.read().astype("float32")
    profile  = src.profile.copy()
    nodata   = src.nodata
    n_bands  = src.count

if n_bands != IN_CH:
    raise ValueError(
        f"Stack has {n_bands} bands but model expects {IN_CH}. "
        "Rebuild the stack or pass --in_channels to override."
    )

print(f"   Stack shape : {img.shape}")

# NODATA -> NaN then z-score normalise each band
if nodata is not None:
    img[img == nodata] = np.nan

for b in range(n_bands):
    band  = img[b]
    valid = band[~np.isnan(band)]
    if valid.size == 0:
        continue
    mu, sigma = float(valid.mean()), float(valid.std())
    if sigma > 0:
        img[b] = (band - mu) / sigma

img = np.nan_to_num(img, nan=0.0)
_, H, W = img.shape
print(f"   Normalised  : {H} x {W} px")

# -----------------------------------------
# HANN BLENDING WINDOW
# -----------------------------------------
def make_window(size, mode):
    if mode == "hann":
        win_1d = np.hanning(size).astype("float32")
        return np.outer(win_1d, win_1d)
    return np.ones((size, size), dtype="float32")

window = make_window(PATCH, args.blend)

# -----------------------------------------
# SLIDING WINDOW INFERENCE
# -----------------------------------------
prob_map   = np.zeros((H, W), dtype="float32")
weight_map = np.zeros((H, W), dtype="float32")

rows = list(range(0, H - PATCH + 1, STRIDE))
cols = list(range(0, W - PATCH + 1, STRIDE))

# Add final row/col to cover edges if not already included
if rows[-1] + PATCH < H:
    rows.append(H - PATCH)
if cols[-1] + PATCH < W:
    cols.append(W - PATCH)

total_patches = len(rows) * len(cols)
print(f"\nInference  : {len(rows)} x {len(cols)} = {total_patches:,} patches  (stride={STRIDE}, blend={args.blend})")
log.info(f"Inference: rows={len(rows)}, cols={len(cols)}, total={total_patches}")

with torch.no_grad():
    for i in tqdm(rows, desc="Rows", unit="row"):
        for j in cols:
            patch = img[:, i:i+PATCH, j:j+PATCH]
            if patch.shape[1:] != (PATCH, PATCH):
                continue
            x    = torch.from_numpy(patch).unsqueeze(0).to(device)
            prob = model(x).squeeze().cpu().numpy()   # (PATCH, PATCH)
            prob_map  [i:i+PATCH, j:j+PATCH] += prob * window
            weight_map[i:i+PATCH, j:j+PATCH] += window

# Normalise by accumulated weights
valid = weight_map > 0
prob_map[valid] /= weight_map[valid]

# -----------------------------------------
# WRITE OUTPUTS
# -----------------------------------------
out_profile = profile.copy()
out_profile.update(count=1, dtype="float32", nodata=None, compress="lzw")

print(f"\nSaving probability map  -> {OUT_PROB}")
with rasterio.open(OUT_PROB, "w", **out_profile) as dst:
    dst.write(prob_map[np.newaxis, :, :].astype("float32"))
    dst.update_tags(1,
        name="coconut_probability",
        threshold=str(THRESHOLD),
        model=str(BEST_CKPT),
        blend=args.blend,
    )

mask = (prob_map >= THRESHOLD).astype("uint8")
out_profile.update(dtype="uint8", nodata=255)

print(f"Saving binary mask      -> {OUT_MASK}")
with rasterio.open(OUT_MASK, "w", **out_profile) as dst:
    dst.write(mask[np.newaxis, :, :].astype("uint8"))
    dst.update_tags(1,
        name="coconut_mask",
        threshold=str(THRESHOLD),
    )

# -----------------------------------------
# SUMMARY
# -----------------------------------------
total_px    = H * W
coconut_px  = int(mask.sum())
coverage    = 100.0 * coconut_px / total_px

print(f"\n{'='*55}")
print("Prediction complete")
print(f"   Grid size        : {H} x {W} = {total_px:,} px")
print(f"   Coconut pixels   : {coconut_px:,}  ({coverage:.2f}%)")
print(f"   Threshold used   : {THRESHOLD:.3f}")
print(f"   Probability map  : {OUT_PROB}")
print(f"   Binary mask      : {OUT_MASK}")
print(f"{'='*55}")
log.info(
    f"Prediction complete: coconut_px={coconut_px}, coverage={coverage:.2f}%, "
    f"threshold={THRESHOLD}, out_prob={OUT_PROB}, out_mask={OUT_MASK}"
)
