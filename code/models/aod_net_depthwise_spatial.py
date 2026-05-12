import torch
import torch.nn as nn
import torch.nn.functional as F

from models.aod_net import (
    DepthwiseSeparableConv2d,
    GaussianAttention,
    augment_heatmap,
    generate_gaussian_heatmap,
)


class LightweightChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced_channels = max(1, channels // max(1, int(reduction)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return scale


class ODGuidedAttention(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int,
        alpha_init: float = 0.35,
        use_channel_gate: bool = True,
        channel_gate_reduction: int = 8,
    ):
        super().__init__()
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.spatial_gain = nn.Parameter(torch.tensor(float(alpha_init)))
        self.channel_gate = (
            LightweightChannelGate(feature_channels, reduction=channel_gate_reduction)
            if use_channel_gate
            else None
        )
        self.channel_gain = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        features: torch.Tensor,
        gaussian_heatmap: torch.Tensor,
        edge_map: torch.Tensor,
    ) -> torch.Tensor:
        saliency = torch.mean(torch.abs(features), dim=1, keepdim=True)
        saliency = saliency / torch.clamp(saliency.amax(dim=(-2, -1), keepdim=True), min=1e-6)

        if gaussian_heatmap.shape[-2:] != features.shape[-2:]:
            gaussian_heatmap = F.interpolate(
                gaussian_heatmap,
                size=features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if edge_map.shape[-2:] != features.shape[-2:]:
            edge_map = F.interpolate(edge_map, size=features.shape[-2:], mode="bilinear", align_corners=False)

        spatial_mask = torch.sigmoid(self.spatial_gate(torch.cat([saliency, gaussian_heatmap, edge_map], dim=1)))
        out = features * (1.0 + self.spatial_gain * spatial_mask)
        if self.channel_gate is not None:
            channel_mask = self.channel_gate(features)
            out = out * (1.0 + self.channel_gain * channel_mask)
        return out


class AODnetDepthwiseSpatial(nn.Module):
    """
    Lightweight AOD-Net variant for OD preprocessing.
    Supports the legacy Gaussian attention path for backward-compatible ablations and
    a newer OD-guided attention path that fuses multi-peak saliency, input edges, and
    a small channel gate without adding much overhead.
    """

    def __init__(
        self,
        base_channels=3,
        sigma_scale=0.3,
        heatmap_augmentation=True,
        alpha_init=0.5,
        attention_variant="legacy_gaussian",
        num_attention_peaks=1,
        use_input_edge=True,
        use_channel_gate=False,
        channel_gate_reduction=8,
        spatial_hidden_channels=None,
        **kwargs,
    ):
        super().__init__()

        _ = kwargs
        self.relu = nn.ReLU(inplace=True)
        self.base_channels = int(base_channels)
        self.sigma_scale = float(sigma_scale)
        self.heatmap_augmentation = bool(heatmap_augmentation)
        self.attention_variant = str(attention_variant).strip().lower()
        self.num_attention_peaks = max(1, int(num_attention_peaks))
        self.use_input_edge = bool(use_input_edge)
        feature_channels = self.base_channels * 4
        hidden_channels = (
            max(4, feature_channels // 3)
            if spatial_hidden_channels is None
            else max(4, int(spatial_hidden_channels))
        )

        if self.base_channels < 3:
            raise ValueError("base_channels must be >= 3.")
        if self.attention_variant not in {"legacy_gaussian", "od_guided", "hybrid", "none"}:
            raise ValueError(
                "attention_variant must be one of: legacy_gaussian, od_guided, hybrid, none."
            )

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

        self.gaussian_attention = (
            GaussianAttention(alpha_init=alpha_init)
            if self.attention_variant in {"legacy_gaussian", "hybrid"}
            else None
        )
        self.guided_attention = (
            ODGuidedAttention(
                feature_channels=feature_channels,
                hidden_channels=hidden_channels,
                alpha_init=alpha_init,
                use_channel_gate=use_channel_gate,
                channel_gate_reduction=channel_gate_reduction,
            )
            if self.attention_variant in {"od_guided", "hybrid"}
            else None
        )
        self.e_conv5 = nn.Conv2d(
            in_channels=feature_channels,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def _build_input_edge_map(self, x: torch.Tensor, target_size) -> torch.Tensor:
        gray = torch.mean(x, dim=1, keepdim=True)
        dx = F.pad(torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1]), (0, 1, 0, 0))
        dy = F.pad(torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :]), (0, 0, 0, 1))
        edge = 0.5 * (dx + dy)
        edge = F.avg_pool2d(edge, kernel_size=3, stride=1, padding=1)
        edge = edge / torch.clamp(edge.amax(dim=(-2, -1), keepdim=True), min=1e-6)
        if edge.shape[-2:] != target_size:
            edge = F.interpolate(edge, size=target_size, mode="bilinear", align_corners=False)
        return edge

    def _build_feature_guided_heatmap(self, features):
        b, _, h, w = features.shape
        spatial = torch.mean(torch.abs(features), dim=1, keepdim=True)
        flat = spatial.view(b, -1)
        num_peaks = min(self.num_attention_peaks, flat.shape[1])
        peak_values, peak_indices = torch.topk(flat, k=num_peaks, dim=1)
        sigma = self.sigma_scale * float(max(h, w))
        heatmaps = []
        for idx in range(b):
            weighted_heatmap = None
            weights = peak_values[idx]
            weights = weights / torch.clamp(weights.sum(), min=1e-6)

            for peak_pos, weight in zip(peak_indices[idx], weights):
                center_x = (peak_pos % w).to(features.dtype)
                center_y = torch.div(peak_pos, w, rounding_mode="floor").to(features.dtype)
                peak_heatmap = generate_gaussian_heatmap(
                    image_size=(h, w),
                    center=(center_x, center_y),
                    sigma=sigma,
                    device=features.device,
                    dtype=features.dtype,
                )
                if weighted_heatmap is None:
                    weighted_heatmap = weight * peak_heatmap
                else:
                    weighted_heatmap = weighted_heatmap + weight * peak_heatmap

            heatmaps.append(weighted_heatmap if weighted_heatmap is not None else torch.zeros_like(spatial[idx]))

        batch_heatmap = torch.stack(heatmaps, dim=0)
        batch_heatmap = batch_heatmap / torch.clamp(
            batch_heatmap.amax(dim=(-2, -1), keepdim=True),
            min=1e-6,
        )
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
        batch_heatmap = self._build_feature_guided_heatmap(concat3)

        if self.attention_variant == "legacy_gaussian":
            concat3 = self.gaussian_attention(concat3, batch_heatmap)
        elif self.attention_variant == "od_guided":
            edge_map = self._build_input_edge_map(x, concat3.shape[-2:]) if self.use_input_edge else batch_heatmap
            concat3 = self.guided_attention(concat3, batch_heatmap, edge_map)
        elif self.attention_variant == "hybrid":
            edge_map = self._build_input_edge_map(x, concat3.shape[-2:]) if self.use_input_edge else batch_heatmap
            concat3 = self.gaussian_attention(concat3, batch_heatmap)
            concat3 = self.guided_attention(concat3, batch_heatmap, edge_map)
        x5 = self.relu(self.e_conv5(concat3))

        clean_image = self.relu((x5 * x) - x5 + 1)
        return clean_image
