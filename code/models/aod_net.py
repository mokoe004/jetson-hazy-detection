import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


def generate_gaussian_heatmap(image_size, center, sigma, device=None, dtype=torch.float32):
    """
    Build a normalized 2D Gaussian heatmap from a center point and sigma.
    """
    if len(image_size) != 2:
        raise ValueError("image_size must be a tuple (H, W).")

    h, w = int(image_size[0]), int(image_size[1])
    if h <= 0 or w <= 0:
        raise ValueError("image_size values must be positive.")
    if device is None:
        device = torch.device("cpu")
    y = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    x = torch.arange(w, device=device, dtype=dtype).view(1, w)
    cx, cy = center
    sigma = torch.clamp(torch.as_tensor(sigma, device=device, dtype=dtype), min=1e-6)
    heatmap = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma * sigma))

    max_val = heatmap.max()
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.unsqueeze(0)


def augment_heatmap(heatmap):
    """
    Optional robustness augmentation:
      - p=0.3: zero map
      - p=0.2: additive Gaussian noise
      - p=0.2: Gaussian blur
    """
    out = heatmap
    if torch.rand(1, device=heatmap.device).item() < 0.3:
        out = torch.zeros_like(out)
    if torch.rand(1, device=heatmap.device).item() < 0.2:
        noise = 0.05 * torch.randn_like(out)
        out = torch.clamp(out + noise, 0.0, 1.0)
    if torch.rand(1, device=heatmap.device).item() < 0.2:
        out = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
    return out


class GaussianAttention(nn.Module):
    def __init__(self, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, features, heatmap):
        if heatmap is None:
            return features
        target_size = features.shape[-2:]
        if heatmap.shape[-2:] != target_size:
            heatmap = F.interpolate(heatmap, size=target_size, mode="bilinear", align_corners=False)
        return features * (1.0 + self.alpha * heatmap)

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
    AOD-Net variant with depthwise-separable convolutions and feature-guided Gaussian attention.
    The Gaussian attention is applied on the multi-scale concatenated feature map before e_conv5.
    """

    def __init__(self, sigma_scale=0.3, heatmap_augmentation=True, alpha_init=0.5, use_spatial_attention=True):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.sigma_scale = float(sigma_scale)
        self.heatmap_augmentation = bool(heatmap_augmentation)
        self.use_spatial_attention = bool(use_spatial_attention)
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

        self.gaussian_attention = GaussianAttention(alpha_init=alpha_init)
        self.e_conv5 = nn.Conv2d(in_channels=12, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

    def _build_feature_guided_heatmap(self, features):
        b, _, h, w = features.shape
        spatial = torch.mean(features, dim=1, keepdim=True)
        flat = spatial.view(b, -1)
        peak_index = torch.argmax(flat, dim=1)
        center_x = (peak_index % w).to(features.dtype)
        center_y = torch.div(peak_index, w, rounding_mode="floor").to(features.dtype)

        sigma = self.sigma_scale * float(max(h, w))
        heatmaps = []
        for idx in range(b):
            heatmap = generate_gaussian_heatmap(
                image_size=(h, w),
                center=(center_x[idx], center_y[idx]),
                sigma=sigma,
                device=features.device,
                dtype=features.dtype,
            )
            heatmaps.append(heatmap)

        batch_heatmap = torch.stack(heatmaps, dim=0)
        if self.training and self.heatmap_augmentation:
            batch_heatmap = augment_heatmap(batch_heatmap)
        return batch_heatmap

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat([x1, x2], 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat([x2, x3], 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat([x1, x2, x3, x4], 1)
        if self.use_spatial_attention:
            batch_heatmap = self._build_feature_guided_heatmap(concat3)
            concat3 = self.gaussian_attention(concat3, batch_heatmap)
        x5 = self.relu(self.e_conv5(concat3))

        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image


class AODnetDepthwiseGaussian(nn.Module):
    """
    AOD-Net variant with depthwise-separable convolutions and feature-driven Gaussian attention.
    Attention is residual and soft: features are enhanced, not hard-masked.
    """

    def __init__(
        self,
        base_channels=3,
        sigma_scale=0.3,
        heatmap_augmentation=True,
        alpha_init=0.5,
        use_gaussian_attention=True,
        use_se_attention=True,
        se_reduction=8,
        **kwargs,
    ):
        super().__init__()

        _ = kwargs
        self.relu = nn.ReLU(inplace=True)
        self.base_channels = int(base_channels)
        self.sigma_scale = float(sigma_scale)
        self.heatmap_augmentation = bool(heatmap_augmentation)
        self.use_gaussian_attention = bool(use_gaussian_attention)
        self.use_se_attention = bool(use_se_attention)

        if self.base_channels < 3:
            raise ValueError("base_channels must be >= 3.")

        self.e_conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.base_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.e_conv2 = DepthwiseSeparableConv2d(
            in_channels=self.base_channels,
            out_channels=self.base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.e_conv3 = DepthwiseSeparableConv2d(
            in_channels=self.base_channels * 2,
            out_channels=self.base_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )
        self.e_conv4 = DepthwiseSeparableConv2d(
            in_channels=self.base_channels * 2,
            out_channels=self.base_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=True,
        )

        self.gaussian_attention = GaussianAttention(alpha_init=alpha_init)
        self.se_attention = SEBlock(channels=self.base_channels * 4, reduction=se_reduction)
        self.e_conv5 = nn.Conv2d(
            in_channels=self.base_channels * 4,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def _build_feature_guided_heatmap(self, features):
        b, _, h, w = features.shape
        spatial = torch.mean(features, dim=1, keepdim=True)
        flat = spatial.view(b, -1)
        peak_index = torch.argmax(flat, dim=1)
        center_x = (peak_index % w).to(features.dtype)
        center_y = torch.div(peak_index, w, rounding_mode="floor").to(features.dtype)

        sigma = self.sigma_scale * float(max(h, w))
        heatmaps = []
        for idx in range(b):
            heatmap = generate_gaussian_heatmap(
                image_size=(h, w),
                center=(center_x[idx], center_y[idx]),
                sigma=sigma,
                device=features.device,
                dtype=features.dtype,
            )
            heatmaps.append(heatmap)

        batch_heatmap = torch.stack(heatmaps, dim=0)
        if self.training and self.heatmap_augmentation:
            batch_heatmap = augment_heatmap(batch_heatmap)
        return batch_heatmap

    def forward(self, x, boxes=None):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat([x1, x2], 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat([x2, x3], 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat([x1, x2, x3, x4], 1)
        if self.use_gaussian_attention:
            batch_heatmap = self._build_feature_guided_heatmap(concat3)
            concat3 = self.gaussian_attention(concat3, batch_heatmap)
        if self.use_se_attention:
            concat3 = self.se_attention(concat3)
        x5 = self.relu(self.e_conv5(concat3))

        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image
