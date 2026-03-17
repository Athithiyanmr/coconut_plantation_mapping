# scripts/dl/unet_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),       # ✅ dropout inside MLP
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # ✅ Pre-norm style (more stable than post-norm)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


def CBR(in_c, out_c, dropout=0.0):
    """Conv → BN → ReLU × 2 with optional dropout."""
    layers = [
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),  # ✅ bias=False when using BN
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class UNetTransformer(nn.Module):
    def __init__(self, in_channels=11, dropout=0.1):
        super().__init__()

        # ✅ dropout exposed as argument — easy to tune
        self.d1 = CBR(in_channels, 64)
        self.d2 = CBR(64,  128)
        self.d3 = CBR(128, 256)
        self.d4 = CBR(256, 512)           # ✅ extra encoder depth

        self.pool = nn.MaxPool2d(2)

        # ✅ ConvTranspose2d instead of Upsample — learnable upsampling
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)

        # Transformer bottleneck at 512-dim
        self.transformer = TransformerBlock(dim=512, heads=8, mlp_dim=1024, dropout=dropout)

        # Decoder
        self.u3 = CBR(512 + 256, 256, dropout=dropout)
        self.u2 = CBR(256 + 128, 128, dropout=dropout)
        self.u1 = CBR(128 + 64,  64,  dropout=dropout)

        self.out = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            # ✅ No sigmoid here — moved out for numerical stability with BCEWithLogitsLoss
            # sigmoid applied externally during inference only
        )

        # ✅ Weight initialisation — kaiming for conv, ones/zeros for BN
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        c1 = self.d1(x)                    # (B, 64,  H,   W)
        c2 = self.d2(self.pool(c1))        # (B, 128, H/2, W/2)
        c3 = self.d3(self.pool(c2))        # (B, 256, H/4, W/4)
        c4 = self.d4(self.pool(c3))        # (B, 512, H/8, W/8)  ✅ new

        # Transformer bottleneck
        B, C, H, W = c4.shape
        t  = c4.flatten(2).transpose(1, 2)   # (B, H*W, C)
        t  = self.transformer(t)
        c4 = t.transpose(1, 2).reshape(B, C, H, W)

        # Decoder with skip connections
        u3 = self.u3(torch.cat([self.up3(c4), c3], dim=1))   # (B, 256, H/4, W/4)
        u2 = self.u2(torch.cat([self.up2(u3), c2], dim=1))   # (B, 128, H/2, W/2)
        u1 = self.u1(torch.cat([self.up1(u2), c1], dim=1))   # (B, 64,  H,   W)

        return self.out(u1)   # ✅ raw logits — sigmoid applied externally


    def predict(self, x):
        """Convenience method for inference — returns probabilities."""
        return torch.sigmoid(self.forward(x))
