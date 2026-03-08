import torch
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attention

class AODnet(nn.Module):
    def __init__(self):
        super(AODnet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
        self.e_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)

        self.e_conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2, bias=True)
        self.e_conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3, bias=True)

        self.e_conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        source = []
        source.append(x)

        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        #concatenating the output of the two convolutions x1 and x2
        concat1 = torch.cat([x1, x2], 1)
        x3 = self.relu(self.e_conv3(concat1))

        #concatenating the output of the two convolutions x2 and x3
        concat2 = torch.cat([x2, x3], 1)
        x4 = self.relu(self.e_conv4(concat2))

        #concatenating the output of the two convolutions x1, x2, x3 and x4
        concat3 = torch.cat([x1, x2, x3, x4], 1)
        x5 = self.relu(self.e_conv5(concat3))

        #J(x) = T(x)*I(x)-T(x) + B
        clean_image = self.relu((x5*x)-x5+1)

        #return J(x)
        return clean_image


class AODnetDepthwiseSpatial(nn.Module):
    """
    AOD-Net variant with depthwise-separable convolutions and a spatial attention block.
    Spatial attention is applied on the multi-scale concatenated feature map before e_conv5.
    """

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
        self.e_conv2 = DepthwiseSeparableConv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.e_conv3 = DepthwiseSeparableConv2d(
            in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2, bias=True
        )
        self.e_conv4 = DepthwiseSeparableConv2d(
            in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3, bias=True
        )

        self.spatial_attention = SpatialAttention(kernel_size=7)
        self.e_conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat([x1, x2], 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat([x2, x3], 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat([x1, x2, x3, x4], 1)
        concat3 = self.spatial_attention(concat3)
        x5 = self.relu(self.e_conv5(concat3))

        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image
        
