import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evaluation.evaluation import calculate_psnr_ssim  # noqa: E402
from evaluation.od_metrics import EvalSample, evaluate_detection, evaluate_detection_per_class  # noqa: E402
from evaluation.ssim_psnr_eval import psnr, ssim  # noqa: E402


class IdentityModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.conv1.weight.zero_()
            for channel_idx in range(3):
                self.conv1.weight[channel_idx, channel_idx, 0, 0] = 1.0

    def forward(self, x):
        return self.conv1(x)


class EvaluationMetricTests(unittest.TestCase):
    def test_calculate_psnr_ssim_averages_per_image(self):
        hazy = torch.stack(
            [
                torch.zeros(3, 8, 8),
                torch.full((3, 8, 8), 0.25),
                torch.full((3, 8, 8), 0.75),
            ]
        )
        clear = torch.stack(
            [
                torch.zeros(3, 8, 8),
                torch.full((3, 8, 8), 0.5),
                torch.full((3, 8, 8), 1.0),
            ]
        )

        loader = DataLoader(TensorDataset(hazy, clear), batch_size=2, shuffle=False)
        model = IdentityModel()

        avg_psnr, avg_ssim = calculate_psnr_ssim(model, loader, device=torch.device("cpu"))

        expected_psnr = float(np.mean(psnr(hazy, clear, reduction="none")))
        expected_ssim = float(ssim(hazy, clear, size_average=False).mean().item())

        self.assertAlmostEqual(avg_psnr, expected_psnr, places=6)
        self.assertAlmostEqual(avg_ssim, expected_ssim, places=6)

    def test_evaluate_detection_uses_ground_truth_taxonomy_for_summary(self):
        sample = EvalSample(
            pred_boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
            pred_scores=torch.tensor([0.95, 0.80]),
            pred_labels=torch.tensor([0, 7]),
            gt_boxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            gt_labels=torch.tensor([0]),
            image_id=0,
        )

        metrics = evaluate_detection([sample], iou_thresholds=[0.5])

        self.assertEqual(metrics["num_classes_observed"], 1.0)
        self.assertEqual(metrics["num_gt_classes_observed"], 1.0)
        self.assertEqual(metrics["num_pred_classes_observed"], 2.0)
        self.assertAlmostEqual(metrics["map50"], 1.0, places=6)

    def test_per_class_can_expose_pred_only_false_positives(self):
        sample = EvalSample(
            pred_boxes=torch.tensor([[20.0, 20.0, 30.0, 30.0]]),
            pred_scores=torch.tensor([0.80]),
            pred_labels=torch.tensor([7]),
            gt_boxes=torch.zeros((0, 4), dtype=torch.float32),
            gt_labels=torch.zeros((0,), dtype=torch.int64),
            image_id=0,
        )

        per_class = evaluate_detection_per_class([sample], iou_thr=0.5, include_pred_only=True)

        self.assertIn(7, per_class)
        self.assertEqual(per_class[7]["gt"], 0.0)
        self.assertEqual(per_class[7]["fp"], 1.0)
        self.assertEqual(per_class[7]["precision"], 0.0)


if __name__ == "__main__":
    unittest.main()
