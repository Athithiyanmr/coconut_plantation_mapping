# scripts/dl/dataset.py
#
# Fix: allow_pickle=True so existing .npy files (saved as object arrays)
#      load correctly.  np.load returns an ndarray or 0-d object array;
#      _to_array() unwraps either case into a plain float32/uint8 ndarray.
#
# New: augment flag, Albumentations pipeline for training.
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


def _to_array(loaded, dtype):
    """
    Safely extract a plain ndarray from a np.load result.

    np.save() with object/dict data creates a 0-d object array:
        arr.shape == ()  and  arr.dtype == object
    Calling arr.item() returns the underlying Python object.
    If the object is already a ndarray we cast directly.
    If it is a dict (legacy format from some make_patches versions)
    we attempt to read a sensible value key.
    """
    arr = loaded
    # Unwrap 0-d object arrays
    if arr.dtype == object and arr.shape == ():
        inner = arr.item()
        if isinstance(inner, np.ndarray):
            arr = inner
        elif isinstance(inner, dict):
            # Heuristic: take the first value that is an ndarray
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
        self.imgs  = sorted(Path(img_dir).glob("*.npy"))
        self.masks = sorted(Path(mask_dir).glob("*.npy"))
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
            print("[Dataset] Albumentations augmentation enabled")
        elif augment and not AUG_AVAILABLE:
            print(
                "[Dataset] WARNING: albumentations not installed -- no augmentation.\n"
                "          Install with: pip install albumentations"
            )
        else:
            print("[Dataset] No augmentation (augment=False)")

        # Probe first file to report any pickle warning once at startup
        probe = np.load(self.imgs[0], allow_pickle=True)
        if probe.dtype == object:
            print(
                "[Dataset] WARNING: .npy files were saved with pickle (object dtype).\n"
                "          They will load correctly this run, but re-running\n"
                "          make_patches.py with --clean will write clean files\n"
                "          and remove this warning."
            )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # allow_pickle=True handles both clean and legacy pickled files
        x_raw = np.load(self.imgs[idx],  allow_pickle=True)
        y_raw = np.load(self.masks[idx], allow_pickle=True)

        x = _to_array(x_raw, "float32")   # shape: (C, H, W)
        y = _to_array(y_raw, "float32")   # shape: (H, W)

        # Sanity-check shapes
        if x.ndim != 3:
            raise ValueError(
                f"Expected image ndim=3 (C,H,W), got {x.ndim} "
                f"from {self.imgs[idx]}"
            )
        if y.ndim != 2:
            raise ValueError(
                f"Expected mask ndim=2 (H,W), got {y.ndim} "
                f"from {self.masks[idx]}"
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
