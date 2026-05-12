import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from models.aod_net_depthwise_spatial import AODnetDepthwiseSpatial  # noqa: E402


class AODnetDepthwiseSpatialTests(unittest.TestCase):
    def test_legacy_gaussian_forward(self):
        model = AODnetDepthwiseSpatial(attention_variant="legacy_gaussian")
        x = torch.rand(2, 3, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all().item())

    def test_od_guided_forward_and_heatmap(self):
        model = AODnetDepthwiseSpatial(
            base_channels=6,
            attention_variant="od_guided",
            num_attention_peaks=4,
            use_input_edge=True,
            use_channel_gate=True,
        )
        x = torch.rand(2, 3, 64, 64)
        y = model(x)
        self.assertEqual(tuple(y.shape), tuple(x.shape))
        self.assertTrue(torch.isfinite(y).all().item())

        features = torch.rand(2, 24, 16, 16)
        heatmap = model._build_feature_guided_heatmap(features)
        self.assertEqual(tuple(heatmap.shape), (2, 1, 16, 16))
        self.assertGreaterEqual(float(heatmap.min().item()), 0.0)
        self.assertLessEqual(float(heatmap.max().item()), 1.0)

    def test_guided_variant_keeps_small_param_budget(self):
        legacy = AODnetDepthwiseSpatial(attention_variant="legacy_gaussian")
        guided = AODnetDepthwiseSpatial(
            base_channels=6,
            attention_variant="od_guided",
            num_attention_peaks=4,
            use_input_edge=True,
            use_channel_gate=True,
        )
        legacy_params = sum(p.numel() for p in legacy.parameters())
        guided_params = sum(p.numel() for p in guided.parameters())
        self.assertGreater(guided_params, legacy_params)
        self.assertLess(guided_params, legacy_params * 8)


if __name__ == "__main__":
    unittest.main()
