# scripts/dl/dataset.py
#
# Fix: filter macOS dot-underscore (._*) metadata files from .npy glob.
#      These are created automatically by macOS on external/network volumes
#      and are not valid numpy files.
#
# Fix: allow_pickle=True + _to_array() for legacy pickled patches.
# New: Albumentations augmentation pipeline for training.
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


def _glob_npy(directory):
    """
    Return sorted .npy files, skipping macOS dot-underscore metadata files
    (._filename.npy) written automatically on external / network volumes.
    """
    return sorted(
        p for p in Path(directory).glob("*.npy")
        if not p.name.startswith("._")
    )


def _to_array(loaded, dtype):
    """
    Safely extract a plain ndarray from a np.load result.
    Handles 0-d object arrays (pickled ndarrays or dicts).
    """
    arr = loaded
    if arr.dtype == object and arr.shape == ():
        inner = arr.item()
        if isinstance(inner, np.ndarray):
            arr = inner
        elif isinstance(inner, dict):
            for v in inner.values():
                if isinstance(v, np.ndarray):
                    arr = v
                    break
            else:
                raise ValueError(
                    "Pickled .npy file contains a dict with no ndarray values. "
                    "Re-run make_patches.py with --clean to regenerate patches."
                )
        else:
            raise ValueError(
                f"Pickled .npy file contains unsupported type: {type(inner)}. "
                "Re-run make_patches.py with --clean to regenerate patches."
            )
    return arr.astype(dtype)


class CoconutDataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=True):
        # _glob_npy skips macOS ._* ghost files automatically
        self.imgs  = _glob_npy(img_dir)
        self.masks = _glob_npy(mask_dir)
        self.aug   = _build_aug() if augment else None

        if len(self.imgs) != len(self.masks):
            raise RuntimeError(
                f"Image/mask count mismatch: "
                f"{len(self.imgs)} images vs {len(self.masks)} masks"
            )
        if len(self.imgs) == 0:
            raise RuntimeError(
                f"No .npy files found in {img_dir}. "
                "Run make_patches.py first."
            )

        if self.aug:
            print(f"[Dataset] Augmentation enabled  | {len(self.imgs):,} patches")
        elif augment and not AUG_AVAILABLE:
            print(
                "[Dataset] WARNING: albumentations not installed -- no augmentation.\n"
                "          Install with: pip install albumentations"
            )
        else:
            print(f"[Dataset] No augmentation (augment=False)  | {len(self.imgs):,} patches")

        # Probe first file; warn once if saved with pickle
        probe = np.load(self.imgs[0], allow_pickle=True)
        if probe.dtype == object:
            print(
                "[Dataset] WARNING: .npy files were saved with pickle (object dtype).\n"
                "          They will load correctly this run. Re-run make_patches.py\n"
                "          with --clean to write clean files and remove this warning."
            )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x_raw = np.load(self.imgs[idx],  allow_pickle=True)
        y_raw = np.load(self.masks[idx], allow_pickle=True)

        x = _to_array(x_raw, "float32")   # (C, H, W)
        y = _to_array(y_raw, "float32")   # (H, W)

        if x.ndim != 3:
            raise ValueError(
                f"Expected image ndim=3 (C,H,W), got {x.ndim} from {self.imgs[idx]}"
            )
        if y.ndim != 2:
            raise ValueError(
                f"Expected mask ndim=2 (H,W), got {y.ndim} from {self.masks[idx]}"
            )

        if self.aug:
            x_hwc = x.transpose(1, 2, 0)              # C,H,W -> H,W,C
            aug   = self.aug(image=x_hwc, mask=y)
            x     = aug["image"].transpose(2, 0, 1)   # back to C,H,W
            y     = aug["mask"]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0),
        )
