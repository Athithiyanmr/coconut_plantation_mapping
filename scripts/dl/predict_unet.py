# scripts/dl/predict_unet.py
#
# Import fix: sys.path injection so the script runs from any working directory.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent   # scripts/dl/
_ROOT = _HERE.parent.parent               # project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import logging

import numpy as np
import rasterio
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
parser.add_argument("--year",        required=True)
parser.add_argument("--aoi",         required=True)
parser.add_argument("--patch",       type=int,   default=256,  help="Patch size in pixels (default: 256)")
parser.add_argument("--stride",      type=int,   default=32,   help="Inference stride (default: 32)")
parser.add_argument("--batch",       type=int,   default=8,    help="Inference batch size (default: 8)")
parser.add_argument("--threshold",   type=float, default=None,
                    help="Binarization threshold. If omitted, reads from checkpoint.")
parser.add_argument("--in_channels", type=int,   default=None,
                    help="Override channel count. If omitted, read from checkpoint config.")
parser.add_argument("--blend",       choices=["hann", "flat"], default="hann",
                    help="Overlap blending: hann (smooth) or flat (average). Default: hann")
args = parser.parse_args()

YEAR   = args.year
AOI    = args.aoi
PATCH  = args.patch
STRIDE = args.stride

STACK     = f"data/processed/{AOI}/stack_{YEAR}.tif"
MODEL_DIR = Path("models")
BEST_CKPT = MODEL_DIR / f"unet_{YEAR}_{AOI}_best.pth"
OUT_DIR   = Path(f"outputs/{AOI}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PROB  = OUT_DIR / f"coconut_prob_{YEAR}_{AOI}.tif"
OUT_MASK  = OUT_DIR / f"coconut_mask_{YEAR}_{AOI}.tif"

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
        f"Checkpoint not found: {BEST_CKPT}\nRun train_unet.py first."
    )

ckpt     = torch.load(BEST_CKPT, map_location=device)
ckpt_cfg = ckpt.get("config", {})

IN_CH = (
    args.in_channels if args.in_channels is not None
    else int(ckpt_cfg.get("in_channels", 11))
)
THRESHOLD = (
    args.threshold if args.threshold is not None
    else float(ckpt_cfg.get("best_threshold", 0.35))
)

print(f"Channels   : {IN_CH}")
print(f"Threshold  : {THRESHOLD:.3f}  ({'CLI' if args.threshold else 'checkpoint'})")
print(f"Checkpoint : {BEST_CKPT}  (epoch {ckpt.get('epoch', '?')})")
log.info(f"Checkpoint: in_channels={IN_CH}, threshold={THRESHOLD}")

# -----------------------------------------
# MODEL
# -----------------------------------------
model = UNetTransformer(in_channels=IN_CH).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"Parameters : {sum(p.numel() for p in model.parameters()):,}")

# -----------------------------------------
# LOAD + NORMALISE STACK
# -----------------------------------------
if not Path(STACK).exists():
    raise FileNotFoundError(f"Stack not found: {STACK}\nRun 02_build_stack.py first.")

print(f"\nLoading stack: {STACK}")
with rasterio.open(STACK) as src:
    img     = src.read().astype("float32")
    profile = src.profile.copy()
    nodata  = src.nodata
    n_bands = src.count

if n_bands != IN_CH:
    raise ValueError(
        f"Stack has {n_bands} bands but model expects {IN_CH}.\n"
        "Rebuild the stack or pass --in_channels to override."
    )

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
print(f"   Shape : {n_bands} bands x {H} x {W} px")

# -----------------------------------------
# BLENDING WINDOW
# -----------------------------------------
def make_window(size, mode):
    if mode == "hann":
        win1d = np.hanning(size).astype("float32")
        return np.outer(win1d, win1d)
    return np.ones((size, size), dtype="float32")

window = make_window(PATCH, args.blend)

# -----------------------------------------
# SLIDING WINDOW INFERENCE
# -----------------------------------------
prob_map   = np.zeros((H, W), dtype="float32")
weight_map = np.zeros((H, W), dtype="float32")

rows = list(range(0, H - PATCH + 1, STRIDE))
cols = list(range(0, W - PATCH + 1, STRIDE))
if rows[-1] + PATCH < H:
    rows.append(H - PATCH)
if cols[-1] + PATCH < W:
    cols.append(W - PATCH)

print(f"\nInference: {len(rows)} x {len(cols)} = {len(rows)*len(cols):,} patches  "
      f"(stride={STRIDE}, blend={args.blend})")
log.info(f"Inference: rows={len(rows)}, cols={len(cols)}")

# Batch patches for faster GPU utilisation
with torch.no_grad():
    batch_patches, batch_coords = [], []
    for i in tqdm(rows, desc="Rows", unit="row"):
        for j in cols:
            patch = img[:, i:i+PATCH, j:j+PATCH]
            if patch.shape[1:] != (PATCH, PATCH):
                continue
            batch_patches.append(patch)
            batch_coords.append((i, j))
            if len(batch_patches) == args.batch:
                x     = torch.from_numpy(np.stack(batch_patches)).to(device)
                probs = model(x).squeeze(1).cpu().numpy()  # (B, PATCH, PATCH)
                for prob, (ri, rj) in zip(probs, batch_coords):
                    prob_map  [ri:ri+PATCH, rj:rj+PATCH] += prob * window
                    weight_map[ri:ri+PATCH, rj:rj+PATCH] += window
                batch_patches, batch_coords = [], []
    # Flush remainder
    if batch_patches:
        x     = torch.from_numpy(np.stack(batch_patches)).to(device)
        probs = model(x).squeeze(1).cpu().numpy()
        for prob, (ri, rj) in zip(probs, batch_coords):
            prob_map  [ri:ri+PATCH, rj:rj+PATCH] += prob * window
            weight_map[ri:ri+PATCH, rj:rj+PATCH] += window

valid = weight_map > 0
prob_map[valid] /= weight_map[valid]

# -----------------------------------------
# WRITE OUTPUTS
# -----------------------------------------
out_profile = profile.copy()
out_profile.update(count=1, dtype="float32", nodata=None, compress="lzw")

print(f"\nSaving probability map -> {OUT_PROB}")
with rasterio.open(OUT_PROB, "w", **out_profile) as dst:
    dst.write(prob_map[np.newaxis].astype("float32"))
    dst.update_tags(1, name="coconut_probability",
                    threshold=str(THRESHOLD), model=str(BEST_CKPT), blend=args.blend)

mask = (prob_map >= THRESHOLD).astype("uint8")
out_profile.update(dtype="uint8", nodata=255)

print(f"Saving binary mask     -> {OUT_MASK}")
with rasterio.open(OUT_MASK, "w", **out_profile) as dst:
    dst.write(mask[np.newaxis].astype("uint8"))
    dst.update_tags(1, name="coconut_mask", threshold=str(THRESHOLD))

# -----------------------------------------
# SUMMARY
# -----------------------------------------
coconut_px = int(mask.sum())
coverage   = 100.0 * coconut_px / (H * W)

print(f"\n{'='*55}")
print("Prediction complete")
print(f"   Grid size       : {H} x {W} = {H*W:,} px")
print(f"   Coconut pixels  : {coconut_px:,}  ({coverage:.2f}%)")
print(f"   Threshold used  : {THRESHOLD:.3f}")
print(f"   Probability map : {OUT_PROB}")
print(f"   Binary mask     : {OUT_MASK}")
print(f"{'='*55}")
log.info(f"Done: coconut_px={coconut_px}, coverage={coverage:.2f}%, threshold={THRESHOLD}")
