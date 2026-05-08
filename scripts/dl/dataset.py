import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class CoconutDataset(Dataset):
    """
    Loads pre-saved .npy patch pairs and applies on-the-fly augmentation
    during training.

    Augmentations (random, independent):
      - Horizontal flip
      - Vertical flip
      - Random 90deg rotation (0 / 90 / 180 / 270)
      - Spectral jitter: small Gaussian noise on each band
        (simulates sensor noise / slight atmospheric variability)
    """

    def __init__(self, img_dir, mask_dir, augment: bool = True):
        self.imgs    = sorted(Path(img_dir).glob("*.npy"))
        self.masks   = sorted(Path(mask_dir).glob("*.npy"))
        self.augment = augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        x = np.load(self.imgs[idx],  allow_pickle=False).astype("float32")  # (C, H, W)
        y = np.load(self.masks[idx], allow_pickle=False).astype("float32")  # (H, W)

        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2).copy()
                y = np.flip(y, axis=1).copy()

            # Vertical flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=0).copy()

            # Random 90deg rotation
            k = np.random.randint(0, 4)
            if k > 0:
                x = np.rot90(x, k, axes=(1, 2)).copy()
                y = np.rot90(y, k, axes=(0, 1)).copy()

            # Spectral jitter: sigma = 2% of unit std
            noise = np.random.normal(0, 0.02, x.shape).astype("float32")
            x = x + noise

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0),
        )
