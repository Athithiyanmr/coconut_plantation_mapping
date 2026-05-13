# scripts/dl/train_unet.py
#
# Trains UNet-Transformer for coconut plantation segmentation.
# Supports ignore mask (255) — pixels with label=255 are excluded from loss.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.dl.dataset import CoconutDataset
from scripts.dl.unet_transformer import UNetTransformer

# -----------------------------------------
# LOGGING
# -----------------------------------------
logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# -----------------------------------------
# ARGUMENTS
# -----------------------------------------
parser = argparse.ArgumentParser(description="Train UNet-Transformer for coconut plantation segmentation")
parser.add_argument("--year",        required=True)
parser.add_argument("--aoi",         required=True)
parser.add_argument("--epochs",      type=int,   default=40,   help="Max training epochs (default: 40)")
parser.add_argument("--batch",       type=int,   default=8,    help="Batch size (default: 8)")
parser.add_argument("--lr",          type=float, default=1e-4, help="Learning rate (default: 1e-4)")
parser.add_argument("--val_split",   type=float, default=0.2,  help="Validation split (default: 0.2)")
parser.add_argument("--patience",    type=int,   default=6,    help="Early stopping patience (default: 6)")
parser.add_argument("--threshold",   type=float, default=0.35, help="Binarization threshold (default: 0.35)")
parser.add_argument("--in_channels", type=int,   default=None, help="Input bands. Auto-detected from stack if omitted.")
parser.add_argument("--pos_weight",  type=float, default=None, help="BCE pos_weight. Auto-computed from masks if omitted.")
parser.add_argument("--workers",     type=int,   default=0,    help="DataLoader workers (default: 0 for macOS MPS)")
args = parser.parse_args()

YEAR       = args.year
AOI        = args.aoi
EPOCHS     = args.epochs
BATCH_SIZE = args.batch
LR         = args.lr
VAL_SPLIT  = args.val_split
PATIENCE   = args.patience
THRESHOLD  = args.threshold
IGNORE_IDX = 255  # mask value to exclude from loss

MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / f"unet_{YEAR}_{AOI}.pth"
BEST_CKPT  = MODEL_DIR / f"unet_{YEAR}_{AOI}_best.pth"

# Auto-detect in_channels
if args.in_channels is not None:
    IN_CH = args.in_channels
else:
    import rasterio
    stack_path = Path(f"data/processed/{AOI}/stack_{YEAR}.tif")
    if stack_path.exists():
        with rasterio.open(stack_path) as src:
            IN_CH = src.count
        print(f"   Auto-detected in_channels={IN_CH} from {stack_path}")
    else:
        IN_CH = 12
        print(f"   Stack not found, defaulting to in_channels={IN_CH}")

log.info(f"Start: AOI={AOI}, year={YEAR}, epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, in_channels={IN_CH}")

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
# DATASET
# -----------------------------------------
img_dir  = Path(f"data/dl/{YEAR}_{AOI}/images")
mask_dir = Path(f"data/dl/{YEAR}_{AOI}/masks")

if not img_dir.exists() or not mask_dir.exists():
    raise FileNotFoundError(
        f"Patch directories not found: {img_dir}\n"
        "Run make_patches.py first."
    )

ds_full = CoconutDataset(img_dir, mask_dir, augment=False)
n       = len(ds_full)
if n == 0:
    raise RuntimeError("Dataset is empty -- check make_patches.py output.")

torch.manual_seed(42)
all_idx   = torch.randperm(n).tolist()
val_size  = int(n * VAL_SPLIT)
train_idx = all_idx[val_size:]
val_idx   = all_idx[:val_size]

ds_train = CoconutDataset(img_dir, mask_dir, augment=True)
ds_val   = CoconutDataset(img_dir, mask_dir, augment=False)

train_ds = Subset(ds_train, train_idx)
val_ds   = Subset(ds_val,   val_idx)

# -----------------------------------------
# COMPUTE pos_weight FROM TRAINING MASKS
# Only count pixels with label 0 or 1 (ignore 255)
# -----------------------------------------
if args.pos_weight is not None:
    POS_WEIGHT = args.pos_weight
else:
    print("\nComputing pos_weight from training masks...")
    pos_count, neg_count = 0, 0
    for idx in train_idx:
        mask = np.load(mask_dir / f"mask_{idx:06d}.npy").astype(np.float32)
        pos_count += int((mask == 1).sum())
        neg_count += int((mask == 0).sum())  # only confirmed negatives
    if pos_count == 0:
        POS_WEIGHT = 10.0
        print(f"   WARNING: No positive pixels found -- using default pos_weight={POS_WEIGHT}")
    else:
        POS_WEIGHT = min(neg_count / pos_count, 50.0)
        print(f"   pos pixels (coconut)    : {pos_count:,}")
        print(f"   neg pixels (confirmed)  : {neg_count:,}")
        print(f"   pos_weight              : {POS_WEIGHT:.1f}  (capped at 50)")

log.info(f"pos_weight={POS_WEIGHT:.2f}")

# -----------------------------------------
# LOSS FUNCTIONS WITH IGNORE MASK SUPPORT
# Pixels where mask == 255 are excluded from all loss calculations.
# -----------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets, valid_mask):
        preds   = torch.sigmoid(logits)
        # Apply valid mask — only compute Dice on labeled pixels
        preds   = preds[valid_mask]
        targets = targets[valid_mask]
        inter   = (preds * targets).sum()
        return 1 - (2 * inter + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets, valid_mask):
        # Apply valid mask
        logits  = logits[valid_mask]
        targets = targets[valid_mask]
        if logits.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        pw  = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


dice_loss  = DiceLoss()
focal_loss = FocalLoss(pos_weight=POS_WEIGHT, gamma=2.0)

def combined_loss(logits, targets_raw):
    # Build valid mask: only pixels with label 0 or 1
    valid_mask = (targets_raw != IGNORE_IDX)
    # For loss computation, treat labels as float (0.0 or 1.0)
    targets = targets_raw.float()
    targets[~valid_mask] = 0.0  # set ignore pixels to 0 (won't affect loss due to mask)
    fl = focal_loss(logits, targets, valid_mask)
    dl = dice_loss(logits, targets, valid_mask)
    return fl + dl


# -----------------------------------------
# METRICS (only on labeled pixels)
# -----------------------------------------
def compute_metrics(logits, targets_raw, threshold):
    valid_mask = (targets_raw != IGNORE_IDX)
    pred_prob  = torch.sigmoid(logits)
    pred_bin   = (pred_prob > threshold).float()
    targets    = targets_raw.float()
    # Only evaluate on valid (non-ignore) pixels
    pred_bin = pred_bin[valid_mask]
    targets  = targets[valid_mask]
    inter    = (pred_bin * targets).sum()
    union    = pred_bin.sum() + targets.sum() - inter
    iou      = (inter / (union + 1e-6)).item()
    tp = (pred_bin * targets).sum().item()
    fp = (pred_bin * (1 - targets)).sum().item()
    fn = ((1 - pred_bin) * targets).sum().item()
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    return iou, f1


# -----------------------------------------
# MAIN GUARD
# -----------------------------------------
if __name__ == "__main__":

    pin = (device.type == "cuda")

    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=args.workers, pin_memory=pin,
    )
    val_dl = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=args.workers, pin_memory=pin,
    )

    print(f"\nDataset   : {n:,} patches")
    print(f"   Train     : {len(train_ds):,}  (augmentation ON)")
    print(f"   Val       : {len(val_ds):,}  (augmentation OFF)")
    print(f"   Channels  : {IN_CH}")
    print(f"   pos_weight: {POS_WEIGHT:.1f}")
    print(f"   Ignore    : mask value {IGNORE_IDX} excluded from loss")
    log.info(f"Dataset: total={n}, train={len(train_ds)}, val={len(val_ds)}")

    model = UNetTransformer(in_channels=IN_CH).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,}")
    log.info(f"Model: UNetTransformer, params={total_params}, in_channels={IN_CH}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []

    print(f"\nTraining : {AOI} {YEAR}")
    print(f"   Epochs   : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}")
    print(f"   Patience : {PATIENCE}  |  Threshold : {THRESHOLD}\n")

    for epoch in range(1, EPOCHS + 1):

        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = combined_loss(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # ---- VALIDATION ----
        model.eval()
        val_loss  = 0.0
        total_iou = 0.0
        total_f1  = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y   = x.to(device), y.to(device)
                logits  = model(x)
                loss    = combined_loss(logits, y)
                val_loss += loss.item()
                iou, f1  = compute_metrics(logits, y, THRESHOLD)
                total_iou += iou
                total_f1  += f1

        val_loss  /= len(val_dl)
        avg_iou    = total_iou / len(val_dl)
        avg_f1     = total_f1  / len(val_dl)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"IoU: {avg_iou:.4f} | "
            f"F1: {avg_f1:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        log.info(
            f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}, "
            f"iou={avg_iou:.4f}, f1={avg_f1:.4f}, lr={current_lr:.2e}"
        )

        history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss,   6),
            "iou":        round(avg_iou,    6),
            "f1":         round(avg_f1,     6),
            "lr":         current_lr,
        })

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    val_loss,
                "iou":         avg_iou,
                "f1":          avg_f1,
                "config": {
                    "in_channels": IN_CH,
                    "threshold":   THRESHOLD,
                    "aoi":         AOI,
                    "year":        YEAR,
                    "pos_weight":  POS_WEIGHT,
                }
            }, BEST_CKPT)
            print(f"  Best model saved (val_loss={val_loss:.4f})")
            log.info(f"Best checkpoint: epoch={epoch}, val_loss={val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                log.info(f"Early stopping at epoch {epoch}")
                break

        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss":    val_loss,
        }, MODEL_PATH)

    history_path = MODEL_DIR / f"history_{YEAR}_{AOI}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved -> {history_path}")

    best = max(history, key=lambda x: x["iou"])
    print(f"\n{'='*55}")
    print(f"Training complete")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Best IoU      : {best['iou']:.4f}  (epoch {best['epoch']})")
    print(f"   Best F1       : {best['f1']:.4f}  (epoch {best['epoch']})")
    print(f"   Checkpoint    : {BEST_CKPT}")
    print(f"   History       : {history_path}")
    print(f"{'='*55}")
    log.info(f"Training complete. Best IoU={best['iou']:.4f} at epoch {best['epoch']}")
