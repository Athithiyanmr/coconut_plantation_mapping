# scripts/dl/train_unet.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import logging
import json
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from scripts.dl.dataset import BuiltupDataset
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
parser = argparse.ArgumentParser(description="Train UNet-Transformer for built-up area segmentation")
parser.add_argument("--year",        required=True)
parser.add_argument("--aoi",         required=True)
parser.add_argument("--epochs",      type=int,   default=40,   help="Max training epochs (default: 40)")
parser.add_argument("--batch",       type=int,   default=8,    help="Batch size (default: 8)")
parser.add_argument("--lr",          type=float, default=1e-4, help="Learning rate (default: 1e-4)")
parser.add_argument("--val_split",   type=float, default=0.2,  help="Validation split (default: 0.2)")
parser.add_argument("--patience",    type=int,   default=6,    help="Early stopping patience (default: 6)")
parser.add_argument("--threshold",   type=float, default=0.35, help="Binarization threshold (default: 0.35)")
parser.add_argument("--in_channels", type=int,   default=11,   help="Input bands (default: 11)")
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
IN_CH      = args.in_channels

MODEL_DIR  = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / f"unet_{YEAR}_{AOI}.pth"
BEST_CKPT  = MODEL_DIR / f"unet_{YEAR}_{AOI}_best.pth"

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

print(f"💻 Device : {device}")
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

ds = BuiltupDataset(img_dir, mask_dir)

if len(ds) == 0:
    raise RuntimeError("Dataset is empty — check make_patches.py output.")

torch.manual_seed(42)
val_size   = int(len(ds) * VAL_SPLIT)
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
    def __init__(self, alpha=0.8, gamma=2.0):
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
# METRICS
# -----------------------------------------
def compute_metrics(pred, target, threshold):
    pred_bin     = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union        = pred_bin.sum() + target.sum() - intersection
    iou          = (intersection / (union + 1e-6)).item()
    tp = (pred_bin * target).sum().item()
    fp = (pred_bin * (1 - target)).sum().item()
    fn = ((1 - pred_bin) * target).sum().item()
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    return iou, f1

# -----------------------------------------
# MAIN GUARD  ✅ required for macOS spawn multiprocessing
# -----------------------------------------
if __name__ == "__main__":

    # ✅ pin_memory only for CUDA — not supported on MPS
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

    print(f"\n📦 Dataset   : {len(ds):,} patches")
    print(f"   Train     : {len(train_ds):,}")
    print(f"   Val       : {len(val_ds):,}")
    log.info(f"Dataset: total={len(ds)}, train={len(train_ds)}, val={len(val_ds)}")

    # -----------------------------------------
    # MODEL
    # -----------------------------------------
    model = UNetTransformer(in_channels=IN_CH).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parameters: {total_params:,}")
    log.info(f"Model: UNetTransformer, params={total_params}, in_channels={IN_CH}")

    # -----------------------------------------
    # OPTIMIZER + SCHEDULER
    # -----------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # -----------------------------------------
    # TRAINING LOOP
    # -----------------------------------------
    best_val_loss    = float("inf")
    patience_counter = 0
    history          = []

    print(f"\n🚀 Training : {AOI} {YEAR}")
    print(f"   Epochs   : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}")
    print(f"   Patience : {PATIENCE}  |  Threshold : {THRESHOLD}\n")

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

        train_loss /= len(train_dl)

        # VALIDATION
        model.eval()
        val_loss  = 0.0
        total_iou = 0.0
        total_f1  = 0.0

        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred  = model(x)
                loss  = combined_loss(pred, y)
                val_loss += loss.item()
                iou, f1   = compute_metrics(pred, y, THRESHOLD)
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

        # BEST CHECKPOINT
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
                }
            }, BEST_CKPT)
            print(f"  ✅ Best model saved (val_loss={val_loss:.4f})")
            log.info(f"Best checkpoint: epoch={epoch}, val_loss={val_loss:.4f}")

        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print("⏹️  Early stopping triggered")
                log.info(f"Early stopping at epoch {epoch}")
                break

        # LATEST CHECKPOINT (every epoch)
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss":    val_loss,
        }, MODEL_PATH)

    # -----------------------------------------
    # SAVE HISTORY
    # -----------------------------------------
    history_path = MODEL_DIR / f"history_{YEAR}_{AOI}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n📊 History saved → {history_path}")

    # -----------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------
    best = max(history, key=lambda x: x["iou"])
    print(f"\n{'='*55}")
    print(f"✅ Training complete")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Best IoU      : {best['iou']:.4f}  (epoch {best['epoch']})")
    print(f"   Best F1       : {best['f1']:.4f}  (epoch {best['epoch']})")
    print(f"   Checkpoint    : {BEST_CKPT}")
    print(f"   History       : {history_path}")
    print(f"{'='*55}")
    log.info(f"Training complete. Best IoU={best['iou']:.4f} at epoch {best['epoch']}")
