# scripts/dl/dataset.py
#
# Fix: filter macOS ._* ghost files, pickled .npy files, and mixed-size patches.
#      At init, all patches are scanned; any whose spatial dims differ from the
#      majority size are silently dropped with a warning.
# New: Albumentations augmentation pipeline for training.

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from collections import Counter

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
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(var_limit=(0.0, 0.005), p=0.2),
    ])


def _glob_npy(directory):
    """Sorted .npy files, skipping macOS ._* metadata ghost files."""
    return sorted(
        p for p in Path(directory).glob("*.npy")
        if not p.name.startswith("._")
    )


def _to_array(loaded, dtype):
    """Unwrap np.load result into a plain ndarray (handles pickled 0-d object arrays)."""
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
                raise ValueError("Pickled .npy dict has no ndarray values. Re-run make_patches.py --clean.")
        else:
            raise ValueError(f"Pickled .npy has unsupported type {type(inner)}. Re-run make_patches.py --clean.")
    return arr.astype(dtype)


def _scan_sizes(paths):
    """
    Read the header of every .npy file (cheap — only reads 128 bytes)
    and return (shape_counts, shape_per_path).
    Falls back to full load for pickled files.
    """
    shape_per_path = {}
    for p in paths:
        try:
            # np.lib.format.read_magic + read_array_header is cheapest,
            # but np.load with mmap is simpler and still O(1) for plain arrays.
            arr = np.load(p, allow_pickle=True, mmap_mode="r")
            arr = _to_array(arr, "float32")
            shape_per_path[p] = arr.shape
        except Exception:
            shape_per_path[p] = None   # mark as corrupt; will be filtered
    shape_counts = Counter(
        s for s in shape_per_path.values() if s is not None
    )
    return shape_counts, shape_per_path


class CoconutDataset(Dataset):
    """
    Loads (image, mask) patch pairs from .npy files.

    Safety filters applied at __init__:
      1. Skips macOS ._* ghost files
      2. Drops corrupt / unreadable files
      3. Drops patches whose spatial size (H, W) differs from the majority
         -- prevents RuntimeError: stack expects each tensor to be equal size

    Re-run make_patches.py with --clean to permanently fix a mixed-size dataset.
    """

    def __init__(self, img_dir, mask_dir, augment=True):
        raw_imgs  = _glob_npy(img_dir)
        raw_masks = _glob_npy(mask_dir)

        if len(raw_imgs) != len(raw_masks):
            raise RuntimeError(
                f"Image/mask count mismatch: {len(raw_imgs)} imgs vs {len(raw_masks)} masks"
            )
        if len(raw_imgs) == 0:
            raise RuntimeError(f"No .npy files found in {img_dir}. Run make_patches.py first.")

        # --- Scan spatial sizes of image patches ---
        shape_counts, shape_map = _scan_sizes(raw_imgs)

        if not shape_counts:
            raise RuntimeError("All patch files are corrupt or unreadable.")

        # Majority spatial size wins  (H, W from shape C,H,W -> index [1:])
        majority_shape = shape_counts.most_common(1)[0][0]   # e.g. (11, 256, 256)
        majority_hw    = majority_shape[1:]                   # (256, 256)

        # Filter to keep only patches matching the majority H x W
        valid_pairs = [
            (img, mask)
            for img, mask in zip(raw_imgs, raw_masks)
            if shape_map.get(img) is not None
            and shape_map[img][1:] == majority_hw
        ]

        dropped = len(raw_imgs) - len(valid_pairs)
        if dropped > 0:
            print(
                f"[Dataset] WARNING: dropped {dropped} patch(es) with size != "
                f"{majority_hw[0]}x{majority_hw[1]}px.\n"
                f"          Re-run make_patches.py with --clean to fix permanently."
            )

        self.imgs  = [p[0] for p in valid_pairs]
        self.masks = [p[1] for p in valid_pairs]
        self.patch_hw = majority_hw
        self.aug   = _build_aug() if augment else None

        if len(self.imgs) == 0:
            raise RuntimeError(
                "No valid same-size patches remain after filtering. "
                "Re-run make_patches.py with --clean."
            )

        # Log augmentation status
        if self.aug:
            print(f"[Dataset] Augmentation enabled  | {len(self.imgs):,} patches  "
                  f"| patch {majority_hw[0]}x{majority_hw[1]}")
        elif augment and not AUG_AVAILABLE:
            print(
                "[Dataset] WARNING: albumentations not installed -- no augmentation.\n"
                "          Install with: pip install albumentations"
            )
        else:
            print(f"[Dataset] No augmentation (augment=False)  | {len(self.imgs):,} patches  "
                  f"| patch {majority_hw[0]}x{majority_hw[1]}")

        # Warn once if files were saved with pickle
        probe = np.load(self.imgs[0], allow_pickle=True)
        if probe.dtype == object:
            print(
                "[Dataset] WARNING: .npy files were saved with pickle (object dtype).\n"
                "          Re-run make_patches.py with --clean to fix permanently."
            )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x_raw = np.load(self.imgs[idx],  allow_pickle=True)
        y_raw = np.load(self.masks[idx], allow_pickle=True)

        x = _to_array(x_raw, "float32")   # (C, H, W)
        y = _to_array(y_raw, "float32")   # (H, W)

        if x.ndim != 3:
            raise ValueError(f"Expected ndim=3 (C,H,W), got {x.ndim} from {self.imgs[idx]}")
        if y.ndim != 2:
            raise ValueError(f"Expected ndim=2 (H,W), got {y.ndim} from {self.masks[idx]}")

        if self.aug:
            x_hwc = x.transpose(1, 2, 0)
            aug   = self.aug(image=x_hwc, mask=y)
            x     = aug["image"].transpose(2, 0, 1)
            y     = aug["mask"]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0),
        )
