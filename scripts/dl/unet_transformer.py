import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer block.
    mlp_dim = 4 * dim  (ViT convention).
    """
    def __init__(self, dim, heads=4):
        super().__init__()
        mlp_dim = dim * 4
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        # self-attention with residual
        h = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = h + attn_out
        # MLP with residual
        h = x
        x = self.norm2(x)
        x = h + self.mlp(x)
        return x


class UNetTransformer(nn.Module):
    """
    4-stage UNet with a Transformer bottleneck.

    Encoder : 64 -> 128 -> 256 -> 512
    Bottleneck : TransformerBlock(512)
    Decoder : 512+256 -> 256  ->  256+128 -> 128  ->  128+64 -> 64
    Output  : Conv1x1 -> raw logit (NO sigmoid here)

    Sigmoid is applied OUTSIDE the model (in loss and in predict_unet.py)
    so we avoid the numerically unstable sigmoid -> logit -> sigmoid path
    that was in the previous version.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        def CBR(in_c, out_c, dropout=0.0):
            layers = [
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        # ---- Encoder (4 stages) ----
        self.d1 = CBR(in_channels, 64)
        self.d2 = CBR(64,  128)
        self.d3 = CBR(128, 256)
        self.d4 = CBR(256, 512, dropout=0.15)   # deepest encoder stage

        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # ---- Transformer bottleneck at 512-dim ----
        self.transformer = TransformerBlock(dim=512, heads=8)

        # ---- Decoder (3 stages matching encoder skip connections) ----
        self.u3 = CBR(512 + 256, 256)
        self.u2 = CBR(256 + 128, 128)
        self.u1 = CBR(128 + 64,   64)

        # ---- Output: raw logit (no sigmoid) ----
        self.drop_out = nn.Dropout2d(0.15)
        self.out      = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        c1 = self.d1(x)            # (B,  64, H,   W)
        c2 = self.d2(self.pool(c1))  # (B, 128, H/2, W/2)
        c3 = self.d3(self.pool(c2))  # (B, 256, H/4, W/4)
        c4 = self.d4(self.pool(c3))  # (B, 512, H/8, W/8)

        # Transformer bottleneck
        B, C, H, W = c4.shape
        t  = c4.flatten(2).transpose(1, 2)   # (B, H*W, 512)
        t  = self.transformer(t)
        c4 = t.transpose(1, 2).reshape(B, C, H, W)

        # Decoder
        u3 = self.up(c4)
        u3 = self.u3(torch.cat([u3, c3], dim=1))  # 512+256

        u2 = self.up(u3)
        u2 = self.u2(torch.cat([u2, c2], dim=1))  # 256+128

        u1 = self.up(u2)
        u1 = self.u1(torch.cat([u1, c1], dim=1))  # 128+64

        # Raw logit — sigmoid applied externally
        return self.out(self.drop_out(u1))
