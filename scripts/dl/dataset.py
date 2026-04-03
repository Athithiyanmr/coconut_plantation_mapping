# scripts/dl/dataset.py
#
# Improvement: Albumentations augmentation for training only.
# Val/test datasets should be instantiated with augment=False.
#
# Install: pip install albumentations

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

try:
    import albumentations as A
    AUG_AVAILABLE = True
except ImportError:
    AUG_AVAILABLE = False


def _build_aug():
    if not AUG_AVAILABLE:
        return None
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(0.0, 0.005), p=0.2),
    ])


class CoconutDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=True):
        self.imgs  = sorted(Path(img_dir).glob("*.npy"))
        self.masks = sorted(Path(mask_dir).glob("*.npy"))
        self.aug   = _build_aug() if augment else None

        if len(self.imgs) != len(self.masks):
            raise RuntimeError(
                f"Image/mask count mismatch: "
                f"{len(self.imgs)} images vs {len(self.masks)} masks"
            )

        if self.aug:
            print("[Dataset] Albumentations augmentation enabled")
        elif augment and not AUG_AVAILABLE:
            print(
                "[Dataset] WARNING: albumentations not installed -- "
                "no augmentation.  Install with: pip install albumentations"
            )
        else:
            print("[Dataset] No augmentation (augment=False)")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = np.load(self.imgs[idx],  allow_pickle=False).astype("float32")
        y = np.load(self.masks[idx], allow_pickle=False).astype("float32")

        if self.aug:
            # albumentations expects H x W x C for image, H x W for mask
            x_hwc = x.transpose(1, 2, 0)          # C,H,W -> H,W,C
            aug   = self.aug(image=x_hwc, mask=y)
            x     = aug["image"].transpose(2, 0, 1)  # back to C,H,W
            y     = aug["mask"]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0),
        )
