import torch
import torch.nn as nn

class LCANet(nn.Module):
    def __init__(self, in_ch=3, feat_ch=50, latent_ch=10, out_act="sigmoid"):
        super().__init__()

        # =========================
        # Encoder
        # =========================
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),   # 512 -> 256

            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),   # 256 -> 128
        )

        # =========================
        # Latent representation
        # =========================
        self.bottleneck = nn.Sequential(
            nn.Conv2d(feat_ch, latent_ch, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(latent_ch, latent_ch, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(latent_ch, feat_ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        # =========================
        # Decoder
        # =========================
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(feat_ch, feat_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(feat_ch, in_ch, kernel_size=3, padding=1),
        )

        if out_act == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif out_act == "tanh":
            self.out_act = nn.Tanh()
        else:
            self.out_act = None

    def forward(self, x):
        z = self.encoder(x)      # (B, 50, 128, 128)
        z = self.bottleneck(z)   # (B, 50, 128, 128)
        y = self.decoder(z)      # (B, 3, 512, 512)

        if self.out_act is not None:
            y = self.out_act(y)

        return y


# Quick test
if __name__ == "__main__":
    x = torch.randn(1, 3, 512, 512)
    model = LCANet(out_act=None)
    y = model(x)
    print("out:", y.shape)