# scripts/evaluate_iou.py
#
# Evaluate coconut plantation prediction against ground truth.
#
# Label values (from 03_rasterize_manual_labels.py):
#   1   = confirmed coconut  -> positive class
#   0   = confirmed not-coconut -> negative class
#   255 = ignore / unlabeled -> EXCLUDED from all metrics
#
# Metrics are computed ONLY on pixels where gt != 255.

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="evaluate.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Evaluate coconut plantation prediction against ground truth")
parser.add_argument("--year",      required=True)
parser.add_argument("--aoi",       required=True)
parser.add_argument("--threshold", type=float, default=None,
                    help="Fixed threshold. If omitted -> auto-search over [0.1, 0.9]")
parser.add_argument("--t_min",     type=float, default=0.1,  help="Auto-search min threshold (default: 0.1)")
parser.add_argument("--t_max",     type=float, default=0.9,  help="Auto-search max threshold (default: 0.9)")
parser.add_argument("--t_step",    type=float, default=0.05, help="Auto-search step (default: 0.05)")
args = parser.parse_args()

YEAR      = args.year
AOI       = args.aoi
THRESHOLD = args.threshold
IGNORE    = 255

# -----------------------------------------
# PATHS
# -----------------------------------------
PRED      = Path(f"outputs/unet/{YEAR}/coconut_prob_{YEAR}_{AOI}.tif")
LABEL     = Path(f"data/processed/training/labels_coconut_{YEAR}_{AOI}.tif")
OUT_DIR   = Path(f"outputs/unet/{YEAR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_PATH = OUT_DIR / f"confusion_{YEAR}_{AOI}.tif"
PLOT_DIR  = OUT_DIR / "plots"
PLOT_DIR.mkdir(exist_ok=True)

for p in [PRED, LABEL]:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

print(f"\nEvaluating: {AOI} {YEAR}")
log.info(f"Start: AOI={AOI}, year={YEAR}, threshold={THRESHOLD}")

# -----------------------------------------
# LOAD PREDICTION
# -----------------------------------------
print("\nLoading prediction...")
with rasterio.open(PRED) as src:
    pred = src.read(1).astype("float32")
    meta = src.meta.copy()
    H, W = pred.shape

print(f"   Pred shape : {pred.shape}")
print(f"   Pred range : [{pred.min():.3f}, {pred.max():.3f}]")
log.info(f"Pred loaded: shape={pred.shape}, range=[{pred.min():.3f},{pred.max():.3f}]")

# -----------------------------------------
# ALIGN GROUND TRUTH TO PREDICTION GRID
# -----------------------------------------
print("\nAligning ground truth...")
with rasterio.open(LABEL) as gt_src:
    # Default fill = 255 (ignore) for areas outside label extent
    gt_aligned = np.full((H, W), IGNORE, dtype="uint8")
    reproject(
        source=gt_src.read(1),
        destination=gt_aligned,
        src_transform=gt_src.transform,
        src_crs=gt_src.crs,
        dst_transform=meta["transform"],
        dst_crs=meta["crs"],
        resampling=Resampling.nearest,
    )

# CRITICAL: use == 1, NOT > 0 (255 ignore pixels satisfy > 0!)
gt_coconut_mask = (gt_aligned == 1)   # confirmed coconut
gt_neg_mask     = (gt_aligned == 0)   # confirmed not-coconut
valid_mask      = (gt_aligned != IGNORE)  # labeled pixels only

gt_coconut_px = int(gt_coconut_mask.sum())
gt_neg_px     = int(gt_neg_mask.sum())
gt_ignore_px  = int((gt_aligned == IGNORE).sum())
gt_valid_px   = int(valid_mask.sum())

print(f"   GT coconut  (1)  : {gt_coconut_px:,} / {H*W:,} ({100*gt_coconut_px/(H*W):.3f}%)")
print(f"   GT not-coco (0)  : {gt_neg_px:,} / {H*W:,} ({100*gt_neg_px/(H*W):.3f}%)")
print(f"   GT ignore  (255) : {gt_ignore_px:,} / {H*W:,} ({100*gt_ignore_px/(H*W):.1f}%)")
print(f"   GT valid labeled : {gt_valid_px:,} px  <- metrics computed on these only")
log.info(f"GT aligned: coconut={gt_coconut_px}, neg={gt_neg_px}, ignore={gt_ignore_px}, valid={gt_valid_px}")

if gt_coconut_px == 0:
    raise RuntimeError(
        "Ground truth has no coconut pixels (class=1) after alignment.\n"
        "Check that prediction and label share the same CRS and spatial extent,\n"
        "and that 03_rasterize_manual_labels.py ran correctly."
    )

if gt_valid_px == 0:
    raise RuntimeError("All ground truth pixels are marked as ignore (255). Cannot evaluate.")

# -----------------------------------------
# METRICS HELPER
# Only evaluated on valid_mask pixels (gt != 255)
# -----------------------------------------
def calc_metrics(pred_prob, gt_coconut, valid, t):
    pred_bin = (pred_prob > t)
    # Mask to labeled pixels only
    p = pred_bin[valid]
    g = gt_coconut[valid]
    TP = int(np.logical_and( p,  g).sum())
    FP = int(np.logical_and( p, ~g).sum())
    FN = int(np.logical_and(~p,  g).sum())
    TN = int(np.logical_and(~p, ~g).sum())
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = TP / (TP + FP + FN + 1e-6)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    return dict(
        threshold=round(float(t), 4),
        TP=TP, FP=FP, FN=FN, TN=TN,
        precision=round(float(precision), 6),
        recall=round(float(recall),    6),
        f1=round(float(f1),        6),
        iou=round(float(iou),       6),
        accuracy=round(float(accuracy),  6),
        evaluated_pixels=int(valid.sum()),
    )

# -----------------------------------------
# AUTO THRESHOLD SEARCH
# -----------------------------------------
if THRESHOLD is None:
    print(f"\nSearching best threshold [{args.t_min:.2f} -> {args.t_max:.2f}, step={args.t_step}]...")
    print(f"   (Metrics computed on {gt_valid_px:,} labeled pixels, ignore pixels excluded)")
    thresholds     = np.arange(args.t_min, args.t_max + args.t_step, args.t_step)
    search_results = []
    best_iou       = 0
    best_t         = 0.5

    for t in thresholds:
        m = calc_metrics(pred, gt_coconut_mask, valid_mask, t)
        print(f"   t={t:.2f} -> IoU={m['iou']:.4f}  F1={m['f1']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}")
        search_results.append(m)
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            best_t   = t

    THRESHOLD = float(best_t)
    print(f"\n   Best threshold: {THRESHOLD:.2f}  (IoU={best_iou:.4f})")
    log.info(f"Auto threshold: best_t={THRESHOLD}, best_iou={best_iou:.4f}")

    # IoU + F1 vs Threshold plot
    iou_vals = [r["iou"] for r in search_results]
    f1_vals  = [r["f1"]  for r in search_results]
    t_vals   = [r["threshold"] for r in search_results]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(t_vals, iou_vals, "b-o", label="IoU")
    ax1.plot(t_vals, f1_vals,  "g-s", label="F1")
    ax1.axvline(THRESHOLD, color="red", linestyle="--", label=f"Best t={THRESHOLD:.2f}")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title(f"IoU & F1 vs Threshold -- {AOI} {YEAR}\n(evaluated on {gt_valid_px:,} labeled px)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    iou_plot = PLOT_DIR / f"threshold_search_{YEAR}_{AOI}.png"
    plt.savefig(iou_plot, dpi=150)
    plt.close()
    print(f"   Threshold plot saved -> {iou_plot}")

# -----------------------------------------
# FINAL METRICS AT CHOSEN THRESHOLD
# -----------------------------------------
m = calc_metrics(pred, gt_coconut_mask, valid_mask, THRESHOLD)

print(f"\n{'='*50}")
print(f"METRICS -- {AOI} {YEAR}  (t={THRESHOLD:.2f})")
print(f"Evaluated on {gt_valid_px:,} labeled pixels (255 ignored)")
print(f"{'='*50}")
print(f"   Accuracy  : {m['accuracy']:.4f}")
print(f"   Precision : {m['precision']:.4f}")
print(f"   Recall    : {m['recall']:.4f}")
print(f"   F1 Score  : {m['f1']:.4f}")
print(f"   IoU       : {m['iou']:.4f}")
print(f"{'='*50}")
print(f"   TP: {m['TP']:,}  FP: {m['FP']:,}")
print(f"   FN: {m['FN']:,}  TN: {m['TN']:,}")
log.info(f"Metrics: {m}")

# Save metrics to JSON
metrics_path = OUT_DIR / f"metrics_{YEAR}_{AOI}.json"
with open(metrics_path, "w") as f:
    json.dump(m, f, indent=2)
print(f"\nMetrics saved -> {metrics_path}")
log.info(f"Metrics saved: {metrics_path}")

# -----------------------------------------
# CONFUSION MAP RASTER
# 0=TN  1=FP  2=FN  3=TP  255=IGNORE
# Only labeled pixels get a confusion label.
# -----------------------------------------
pred_bin = pred > THRESHOLD
conf     = np.full((H, W), IGNORE, dtype="uint8")  # default: ignore
conf[valid_mask & ~pred_bin & ~gt_coconut_mask] = 0  # TN
conf[valid_mask &  pred_bin & ~gt_coconut_mask] = 1  # FP
conf[valid_mask & ~pred_bin &  gt_coconut_mask] = 2  # FN
conf[valid_mask &  pred_bin &  gt_coconut_mask] = 3  # TP

meta.update(
    count=1,
    dtype="uint8",
    nodata=IGNORE,
    compress="lzw",
    tiled=True,
    blockxsize=256,
    blockysize=256,
)
with rasterio.open(CONF_PATH, "w", **meta) as dst:
    dst.write(conf, 1)
    dst.update_tags(
        legend="0=TN 1=FP 2=FN 3=TP 255=IGNORE",
        threshold=str(THRESHOLD),
        aoi=AOI,
        year=YEAR,
    )
print(f"Confusion raster -> {CONF_PATH}")
log.info(f"Confusion raster saved: {CONF_PATH}")

# -----------------------------------------
# CONFUSION MAP VISUAL (show only labeled pixels)
# -----------------------------------------
cmap_colors = ["#d3d3d3", "#e74c3c", "#3498db", "#2ecc71"]
cmap        = matplotlib.colors.ListedColormap(cmap_colors)
labels      = ["TN (correct background)", "FP (false coconut)",
               "FN (missed coconut)",     "TP (correct coconut)"]
patches     = [mpatches.Patch(color=c, label=l)
               for c, l in zip(cmap_colors, labels)]

# Show conf map; unlabeled pixels (255) shown as white
conf_vis = conf.astype("float32")
conf_vis[conf == IGNORE] = np.nan

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(conf_vis, cmap=cmap, vmin=0, vmax=3, interpolation="none")
ax.legend(handles=patches, loc="lower right", fontsize=9)
ax.set_title(
    f"Confusion Map -- {AOI} {YEAR}  (t={THRESHOLD:.2f})\n"
    f"Evaluated on {gt_valid_px:,} labeled px | IoU={m['iou']:.4f}  F1={m['f1']:.4f}",
    fontsize=11
)
ax.axis("off")
plt.tight_layout()
conf_png = PLOT_DIR / f"confusion_map_{YEAR}_{AOI}.png"
plt.savefig(conf_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Confusion map PNG -> {conf_png}")

# -----------------------------------------
# PROBABILITY HISTOGRAM (split by GT class)
# -----------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
# Histogram for confirmed coconut pixels
ax.hist(pred[gt_coconut_mask], bins=80, color="#2ecc71", alpha=0.7,
        label=f"GT coconut (n={gt_coconut_px:,})", density=True)
# Histogram for confirmed not-coconut pixels
ax.hist(pred[gt_neg_mask], bins=80, color="#e74c3c", alpha=0.7,
        label=f"GT not-coconut (n={gt_neg_px:,})", density=True)
ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"t={THRESHOLD:.2f}")
ax.set_title(f"Prediction Probability by GT Class -- {AOI} {YEAR}")
ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
hist_png = PLOT_DIR / f"prob_histogram_{YEAR}_{AOI}.png"
plt.savefig(hist_png, dpi=150)
plt.close()
print(f"Histogram PNG -> {hist_png}")

# -----------------------------------------
# FINAL OUTPUT SUMMARY
# -----------------------------------------
print(f"\n{'='*50}")
print(f"Evaluation complete : {AOI} {YEAR}")
print(f"   Metrics  : {metrics_path}")
print(f"   Conf TIF : {CONF_PATH}")
print(f"   Conf PNG : {conf_png}")
print(f"   Histogram: {hist_png}")
print(f"{'='*50}")
