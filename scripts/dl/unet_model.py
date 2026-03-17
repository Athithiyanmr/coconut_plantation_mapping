# scripts/dl/unet_model.py

import torch
import torch.nn as nn


def CBR(in_c, out_c, dropout=0.0):
    """Conv → BN → ReLU × 2 with optional spatial dropout."""
    layers = [
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),  # ✅ bias=False with BN
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels=11, dropout=0.1):
        super().__init__()

        # Encoder
        self.d1 = CBR(in_channels, 64)
        self.d2 = CBR(64,  128)
        self.d3 = CBR(128, 256)
        self.d4 = CBR(256, 512)              # ✅ extra encoder level

        self.pool = nn.MaxPool2d(2)

        # ✅ Learnable upsampling — ConvTranspose2d over bilinear Upsample
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)

        # Decoder
        self.u3 = CBR(512 + 256, 256, dropout=dropout)
        self.u2 = CBR(256 + 128, 128, dropout=dropout)
        self.u1 = CBR(128 + 64,  64,  dropout=dropout)

        # ✅ Raw logits — sigmoid applied externally for numerical stability
        self.out = nn.Conv2d(64, 1, kernel_size=1)

        # ✅ Kaiming weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        c1 = self.d1(x)                  # (B, 64,  H,   W)
        c2 = self.d2(self.pool(c1))      # (B, 128, H/2, W/2)
        c3 = self.d3(self.pool(c2))      # (B, 256, H/4, W/4)
        c4 = self.d4(self.pool(c3))      # (B, 512, H/8, W/8)  ✅ new

        # Decoder with skip connections
        u3 = self.u3(torch.cat([self.up3(c4), c3], dim=1))  # (B, 256, H/4, W/4)
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))  # (B, 128, H/2, W/2)
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))  # (B, 64,  H,   W)

        return self.out(u1)   # ✅ raw logits

    def predict(self, x):
        """Inference — returns sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))
