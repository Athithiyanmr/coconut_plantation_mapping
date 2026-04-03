# scripts/dl/train_unet.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from scripts.dl.dataset import CoconutDataset
from scripts.dl.unet_transformer import UNetTransformer

logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Train UNet-Transformer for coconut segmentation")
parser.add_argument("--year",        required=True)
parser.add_argument("--aoi",         required=True)
parser.add_argument("--epochs",      type=int,   default=40)
parser.add_argument("--batch",       type=int,   default=8)
parser.add_argument("--lr",          type=float, default=1e-4)
parser.add_argument("--val_split",   type=float, default=0.2)
parser.add_argument("--patience",    type=int,   default=6)
parser.add_argument("--threshold",   type=float, default=None,
                    help="Fixed threshold. If omitted, auto-search on validation set.")
parser.add_argument("--in_channels", type=int,   default=None,
                    help="Override input band count. If omitted, inferred from first patch.")
parser.add_argument("--workers",     type=int,   default=0)
# threshold search range
parser.add_argument("--t_min",       type=float, default=0.20)
parser.add_argument("--t_max",       type=float, default=0.80)
parser.add_argument("--t_step",      type=float, default=0.02)
parser.add_argument("--t_fine_step", type=float, default=0.005)
parser.add_argument("--metric",      choices=["f1", "iou"], default="f1",
                    help="Metric to optimise threshold on (default: f1)")
args = parser.parse_args()

YEAR       = args.year
AOI        = args.aoi
EPOCHS     = args.epochs
BATCH_SIZE = args.batch
LR         = args.lr
VAL_SPLIT  = args.val_split
PATIENCE   = args.patience
FIXED_THR  = args.threshold

MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / f"unet_{YEAR}_{AOI}.pth"
BEST_CKPT  = MODEL_DIR / f"unet_{YEAR}_{AOI}_best.pth"
HIST_PATH  = MODEL_DIR / f"history_{YEAR}_{AOI}.json"

log.info(f"Start: AOI={AOI}, year={YEAR}, epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}")

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
log.info(f"Device: {device}")

# -----------------------------------------
# DATASET  (auto-detect band count)
# -----------------------------------------
img_dir  = Path(f"data/dl/{YEAR}_{AOI}/images")
mask_dir = Path(f"data/dl/{YEAR}_{AOI}/masks")

if not img_dir.exists() or not mask_dir.exists():
    raise FileNotFoundError(
        f"Patch directories not found: {img_dir}\n"
        "Run make_patches.py first."
    )

ds = CoconutDataset(img_dir, mask_dir)
if len(ds) == 0:
    raise RuntimeError("Dataset is empty -- check make_patches.py output.")

# Infer channel count from first patch
sample_x, _ = ds[0]
inferred_ch  = int(sample_x.shape[0])
IN_CH = inferred_ch if args.in_channels is None else args.in_channels
if IN_CH != inferred_ch:
    raise ValueError(
        f"--in_channels={IN_CH} does not match actual patch channels={inferred_ch}. "
        "Re-run make_patches.py with the correct stack."
    )

torch.manual_seed(42)
val_size   = max(1, int(round(len(ds) * VAL_SPLIT)))
train_size = len(ds) - val_size
train_ds, val_ds = random_split(ds, [train_size, val_size])

# -----------------------------------------
# LOSS FUNCTIONS
# -----------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds   = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        return 1 - (2 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=3.0):
        """Focal loss with gamma=3.0 (increased from 2.0 for sparser coconut class)."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce = nn.functional.binary_cross_entropy(preds, targets, reduction="none")
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


focal_loss = FocalLoss()
dice_loss  = DiceLoss()


def combined_loss(pred, target):
    return focal_loss(pred, target) + dice_loss(pred, target)


# -----------------------------------------
# TWO-STAGE THRESHOLD SWEEP
# -----------------------------------------
def sweep_thresholds(preds, targets, thresholds):
    """Vectorised sweep over a list of threshold values."""
    rows = []
    for thr in thresholds:
        pred_bin  = (preds > thr).float()
        tp = (pred_bin * targets).sum().item()
        fp = (pred_bin * (1 - targets)).sum().item()
        fn = ((1 - pred_bin) * targets).sum().item()
        precision = tp / (tp + fp + 1e-6)
        recall    = tp / (tp + fn + 1e-6)
        f1        = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        iou       = tp / (tp + fp + fn + 1e-6)
        rows.append({
            "threshold": round(float(thr), 6),
            "precision": float(precision),
            "recall":    float(recall),
            "f1":        float(f1),
            "iou":       float(iou),
        })
    return rows


def pick_best(rows, metric):
    if metric == "f1":
        return max(rows, key=lambda r: (r["f1"], r["iou"]))
    return max(rows, key=lambda r: (r["iou"], r["f1"]))


def threshold_search(preds, targets):
    """Coarse pass then fine refinement around the best coarse point."""
    if FIXED_THR is not None:
        row = sweep_thresholds(preds, targets, [FIXED_THR])[0]
        return row, [row]

    coarse      = np.arange(args.t_min, args.t_max + 1e-9, args.t_step, dtype=float)
    coarse_rows = sweep_thresholds(preds, targets, coarse)
    best_coarse = pick_best(coarse_rows, args.metric)

    margin  = max(args.t_step, 0.05)
    fine_lo = max(args.t_min, best_coarse["threshold"] - margin)
    fine_hi = min(args.t_max, best_coarse["threshold"] + margin)
    fine    = np.arange(fine_lo, fine_hi + 1e-9, args.t_fine_step, dtype=float)
    fine    = np.unique(np.clip(fine, args.t_min, args.t_max))

    fine_rows = sweep_thresholds(preds, targets, fine)
    best_fine = pick_best(fine_rows, args.metric)
    return best_fine, coarse_rows + fine_rows


# -----------------------------------------
# MAIN GUARD
# -----------------------------------------
if __name__ == "__main__":

    pin = (device.type == "cuda")

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin,
    )

    print(f"\nDataset   : {len(ds):,} patches")
    print(f"   Train     : {len(train_ds):,}")
    print(f"   Val       : {len(val_ds):,}")
    print(f"   Channels  : {IN_CH}")
    log.info(f"Dataset: total={len(ds)}, train={len(train_ds)}, val={len(val_ds)}, channels={IN_CH}")

    # MODEL
    model = UNetTransformer(in_channels=IN_CH).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,}")
    log.info(f"Model: UNetTransformer, params={total_params}, in_channels={IN_CH}")

    # OPTIMIZER + SCHEDULER
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # TRAINING LOOP
    best_score       = float("-inf")
    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []
    best_epoch_info  = None

    print(f"\nTraining : {AOI} {YEAR}")
    print(f"   Epochs   : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}")
    print(f"   Patience : {PATIENCE}  |  Selection : 0.7*F1 + 0.3*IoU")
    if FIXED_THR is not None:
        print(f"   Fixed threshold : {FIXED_THR}")
    else:
        print(
            f"   Threshold search: [{args.t_min}, {args.t_max}] "
            f"coarse={args.t_step}, fine={args.t_fine_step}, metric={args.metric}"
        )
    print()

    for epoch in range(1, EPOCHS + 1):

        # TRAIN
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = combined_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_dl), 1)

        # VALIDATION  – collect all predictions then sweep thresholds once
        model.eval()
        val_loss    = 0.0
        all_preds   = []
        all_targets = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred  = model(x)
                loss  = combined_loss(pred, y)
                val_loss += loss.item()
                all_preds.append(pred.detach().cpu())
                all_targets.append(y.detach().cpu())

        val_loss    /= max(len(val_dl), 1)
        all_preds    = torch.cat(all_preds,   dim=0)
        all_targets  = torch.cat(all_targets, dim=0)

        # THRESHOLD SEARCH
        best_thr_row, _ = threshold_search(all_preds, all_targets)
        avg_iou       = best_thr_row["iou"]
        avg_f1        = best_thr_row["f1"]
        avg_precision = best_thr_row["precision"]
        avg_recall    = best_thr_row["recall"]
        best_thr      = best_thr_row["threshold"]

        score      = 0.7 * avg_f1 + 0.3 * avg_iou
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"IoU: {avg_iou:.4f} | F1: {avg_f1:.4f} | "
            f"P: {avg_precision:.4f} | R: {avg_recall:.4f} | "
            f"t*: {best_thr:.3f} | Score: {score:.4f} | LR: {current_lr:.2e}"
        )
        log.info(
            f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, "
            f"iou={avg_iou:.4f}, f1={avg_f1:.4f}, thr={best_thr:.3f}, score={score:.4f}"
        )

        epoch_info = {
            "epoch":          epoch,
            "train_loss":     round(train_loss,    6),
            "val_loss":       round(val_loss,       6),
            "iou":            round(avg_iou,        6),
            "f1":             round(avg_f1,         6),
            "precision":      round(avg_precision,  6),
            "recall":         round(avg_recall,     6),
            "best_threshold": round(best_thr,       6),
            "score":          round(score,          6),
            "lr":             current_lr,
        }
        history.append(epoch_info)

        # Improved = higher composite score, or same score with lower loss
        improved = (score > best_score + 1e-6) or (
            abs(score - best_score) <= 1e-6 and val_loss < best_val_loss
        )

        if improved:
            best_score       = score
            best_val_loss    = val_loss
            best_epoch_info  = epoch_info
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    val_loss,
                "iou":         avg_iou,
                "f1":          avg_f1,
                "precision":   avg_precision,
                "recall":      avg_recall,
                "score":       score,
                "config": {
                    "in_channels":     IN_CH,
                    "threshold":       FIXED_THR,
                    "best_threshold":  best_thr,
                    "threshold_metric": args.metric,
                    "t_min":           args.t_min,
                    "t_max":           args.t_max,
                    "t_step":          args.t_step,
                    "t_fine_step":     args.t_fine_step,
                    "aoi":             AOI,
                    "year":            YEAR,
                }
            }, BEST_CKPT)
            print(f"  >> Best model saved  (score={score:.4f}, t*={best_thr:.3f})")
            log.info(f"Best checkpoint: epoch={epoch}, score={score:.4f}, thr={best_thr:.3f}")

        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                log.info(f"Early stopping at epoch {epoch}")
                break

        # LATEST CHECKPOINT (every epoch — resume-safe)
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss":    val_loss,
            "config": {
                "in_channels":    IN_CH,
                "best_threshold": best_thr,
                "aoi":            AOI,
                "year":           YEAR,
            }
        }, MODEL_PATH)

    # -----------------------------------------
    # SAVE HISTORY
    # -----------------------------------------
    with open(HIST_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved -> {HIST_PATH}")

    # -----------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------
    best = best_epoch_info if best_epoch_info else max(history, key=lambda x: x["score"])
    print(f"\n{'='*55}")
    print("Training complete")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Best IoU      : {best['iou']:.4f}  (epoch {best['epoch']})")
    print(f"   Best F1       : {best['f1']:.4f}")
    print(f"   Best threshold: {best['best_threshold']:.3f}")
    print(f"   Best score    : {best['score']:.4f}")
    print(f"   Checkpoint    : {BEST_CKPT}")
    print(f"   History       : {HIST_PATH}")
    print(f"{'='*55}")
    log.info(
        f"Training complete. Best epoch={best['epoch']}, IoU={best['iou']:.4f}, "
        f"F1={best['f1']:.4f}, threshold={best['best_threshold']:.3f}, score={best['score']:.4f}"
    )
