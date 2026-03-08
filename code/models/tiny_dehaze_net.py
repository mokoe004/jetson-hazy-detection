import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)


class TinyDehazeNet(nn.Module):
    """
    Lightweight dehazer designed for OD preprocessing.
    Residual output keeps optimization stable and preserves detector-relevant structure.
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(c1, c1),
        )
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc2 = nn.Sequential(
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(c2, c2),
        )
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(c3, c3),
        )

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = DepthwiseSeparableConv(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = DepthwiseSeparableConv(c1 + c1, c1)

        self.head = nn.Conv2d(c1, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        y2 = self.up2(x3)
        if y2.shape[-2:] != x2.shape[-2:]:
            y2 = F.interpolate(y2, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y2 = self.dec2(torch.cat([y2, x2], dim=1))

        y1 = self.up1(y2)
        if y1.shape[-2:] != x1.shape[-2:]:
            y1 = F.interpolate(y1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y1 = self.dec1(torch.cat([y1, x1], dim=1))

        residual = self.head(y1)
        return torch.clamp(inp + residual, 0.0, 1.0)
