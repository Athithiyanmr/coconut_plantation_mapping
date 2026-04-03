import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

logging.basicConfig(filename="evaluate.log", level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Evaluate coconut prediction against ground truth")
parser.add_argument("--year",        required=True)
parser.add_argument("--aoi",         required=True)
parser.add_argument("--threshold",   type=float, default=None,
                    help="Fixed threshold. If omitted, auto-search best threshold.")
parser.add_argument("--t_min",       type=float, default=0.20)
parser.add_argument("--t_max",       type=float, default=0.80)
parser.add_argument("--t_step",      type=float, default=0.02)
parser.add_argument("--t_fine_step", type=float, default=0.005)
parser.add_argument("--metric",      choices=["f1", "iou"], default="f1")
args = parser.parse_args()

YEAR      = args.year
AOI       = args.aoi
FIXED_THR = args.threshold

PROB_PATH  = Path(f"outputs/unet/{YEAR}/coconut_prob_{YEAR}_{AOI}.tif")
LABEL_PATH = Path(f"data/processed/training/labels_coconut_{YEAR}_{AOI}.tif")
OUT_DIR    = Path(f"outputs/unet/{YEAR}/evaluation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV  = OUT_DIR / f"threshold_search_{YEAR}_{AOI}.csv"
OUT_JSON = OUT_DIR / f"metrics_{YEAR}_{AOI}.json"
OUT_CONF = OUT_DIR / f"confusion_{YEAR}_{AOI}.tif"

for p, name in [(PROB_PATH, "Probability map"), (LABEL_PATH, "Label raster")]:
    if not p.exists():
        raise FileNotFoundError(f"{name} not found: {p}")

log.info(f"Start evaluation: AOI={AOI}, year={YEAR}")

# LOAD DATA
print("\nLoading probability map...")
with rasterio.open(PROB_PATH) as src:
    prob          = src.read(1).astype("float32")
    ref_meta      = src.meta.copy()
    H, W          = src.height, src.width
    ref_transform = src.transform
    ref_crs       = src.crs

print("Loading ground truth labels...")
label = np.zeros((H, W), dtype="uint8")
with rasterio.open(LABEL_PATH) as src:
    reproject(
        source=src.read(1), destination=label,
        src_transform=src.transform, src_crs=src.crs,
        dst_transform=ref_transform, dst_crs=ref_crs,
        resampling=Resampling.nearest,
    )

label_binary = (label > 0).astype("uint8")
total_px   = int(H * W)
coconut_px = int(label_binary.sum())
print(f"   Coconut px : {coconut_px:,}  ({100*coconut_px/total_px:.2f}%)")

if coconut_px == 0:
    raise RuntimeError("Label mask is entirely zero.")


def evaluate_threshold(pred, gt, thresholds):
    rows = []
    for thr in thresholds:
        pred_bin = (pred > thr).astype("uint8")
        tp = int(((pred_bin == 1) & (gt == 1)).sum())
        fp = int(((pred_bin == 1) & (gt == 0)).sum())
        fn = int(((pred_bin == 0) & (gt == 1)).sum())
        tn = int(((pred_bin == 0) & (gt == 0)).sum())
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1  = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        iou = tp / (tp + fp + fn + 1e-6)
        accuracy  = (tp + tn) / (tp + fp + fn + tn + 1e-6)
        rows.append({"threshold": round(float(thr), 6), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "precision": round(precision, 6), "recall": round(recall, 6),
                      "f1": round(f1, 6), "iou": round(iou, 6), "accuracy": round(accuracy, 6)})
    return rows

def pick_best(rows, metric):
    if metric == "f1":
        return max(rows, key=lambda r: (r["f1"], r["iou"]))
    return max(rows, key=lambda r: (r["iou"], r["f1"]))


# THRESHOLD SEARCH
if FIXED_THR is not None:
    print(f"\nUsing fixed threshold: {FIXED_THR}")
    all_rows = evaluate_threshold(prob, label_binary, [FIXED_THR])
    best_row = all_rows[0]
else:
    print(f"\nThreshold search: [{args.t_min}, {args.t_max}]")
    coarse = np.arange(args.t_min, args.t_max + 1e-9, args.t_step, dtype=float)
    coarse_rows = evaluate_threshold(prob, label_binary, coarse)
    best_coarse = pick_best(coarse_rows, args.metric)
    print(f"   Coarse best : thr={best_coarse['threshold']:.3f}  F1={best_coarse['f1']:.4f}  IoU={best_coarse['iou']:.4f}")

    margin  = max(args.t_step, 0.05)
    fine_lo = max(args.t_min, best_coarse["threshold"] - margin)
    fine_hi = min(args.t_max, best_coarse["threshold"] + margin)
    fine    = np.arange(fine_lo, fine_hi + 1e-9, args.t_fine_step, dtype=float)
    fine    = np.unique(np.clip(fine, args.t_min, args.t_max))
    fine_rows = evaluate_threshold(prob, label_binary, fine)
    best_row  = pick_best(fine_rows, args.metric)
    all_rows  = sorted(coarse_rows + fine_rows, key=lambda r: r["threshold"])
    print(f"   Fine best   : thr={best_row['threshold']:.3f}  F1={best_row['f1']:.4f}  IoU={best_row['iou']:.4f}")

BEST_THR = best_row["threshold"]
log.info(f"Best threshold: {BEST_THR}, F1={best_row['f1']}, IoU={best_row['iou']}")

# SAVE CSV
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    writer.writeheader()
    writer.writerows(all_rows)
print(f"\nThreshold CSV -> {OUT_CSV}  ({len(all_rows)} rows)")

# SAVE JSON
metrics = {
    "year": YEAR, "aoi": AOI, "best_threshold": BEST_THR,
    "iou": best_row["iou"], "f1": best_row["f1"],
    "precision": best_row["precision"], "recall": best_row["recall"],
    "accuracy": best_row["accuracy"],
    "tp": best_row["tp"], "fp": best_row["fp"],
    "fn": best_row["fn"], "tn": best_row["tn"],
    "total_pixels": total_px, "coconut_pixels": coconut_px,
    "threshold_metric": args.metric,
}
with open(OUT_JSON, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics JSON  -> {OUT_JSON}")

# CONFUSION RASTER (TP=1, FP=2, FN=3, TN=0)
pred_bin  = (prob > BEST_THR).astype("uint8")
confusion = np.zeros((H, W), dtype="uint8")
confusion[(pred_bin == 1) & (label_binary == 1)] = 1
confusion[(pred_bin == 1) & (label_binary == 0)] = 2
confusion[(pred_bin == 0) & (label_binary == 1)] = 3

conf_meta = ref_meta.copy()
conf_meta.update(count=1, dtype="uint8", nodata=255, compress="lzw")
with rasterio.open(OUT_CONF, "w", **conf_meta) as dst:
    dst.write(confusion, 1)
    dst.update_tags(description="Confusion: 0=TN, 1=TP, 2=FP, 3=FN", threshold=str(BEST_THR))
    dst.update_tags(1, CLASS_0="TN", CLASS_1="TP", CLASS_2="FP", CLASS_3="FN")
print(f"Confusion map -> {OUT_CONF}")

# FINAL REPORT
print(f"\n{'='*55}")
print(f"Evaluation : {AOI} {YEAR}")
print(f"   Threshold : {BEST_THR:.4f}")
print(f"   IoU       : {best_row['iou']:.4f}")
print(f"   F1        : {best_row['f1']:.4f}")
print(f"   Precision : {best_row['precision']:.4f}")
print(f"   Recall    : {best_row['recall']:.4f}")
print(f"   Accuracy  : {best_row['accuracy']:.4f}")
print(f"   TP: {best_row['tp']:>10,}  |  FP: {best_row['fp']:>10,}")
print(f"   FN: {best_row['fn']:>10,}  |  TN: {best_row['tn']:>10,}")
print(f"{'='*55}")