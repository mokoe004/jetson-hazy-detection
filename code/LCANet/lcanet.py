import torch
import torch.nn as nn

class LCAnet(nn.Module):
    def __init__(self):
        super(LCAnet, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(50, 50, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        # test global avg
        self.dense = nn.Sequential(
            nn.Linear(50*128*128, 10),
            nn.ReLU(),
            nn.Linear(10,10),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(50, 50, 3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(50, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        bottleneck = self.bottleneck(x)
        dense = self.dense(torch.flatten(bottleneck, 1))
        decoder = self.decoder(dense)
        return decoder

ex = torch.randn(1, 3, 512, 512)
model = LCAnet()
out = model(ex)

print(out.shape)