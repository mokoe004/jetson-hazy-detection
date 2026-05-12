import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from detectors.yolov8_adapter import YOLOv8Adapter  # noqa: E402


class YOLOv8AdapterTests(unittest.TestCase):
    def test_backend_is_inferred_from_weight_suffix(self):
        self.assertEqual(YOLOv8Adapter._infer_backend_from_weights("model.pt"), "pytorch")
        self.assertEqual(YOLOv8Adapter._infer_backend_from_weights("model.engine"), "tensorrt")
        self.assertEqual(YOLOv8Adapter._infer_backend_from_weights("model.onnx"), "onnx")

    def test_letterbox_and_rescale_restore_original_boxes(self):
        image = torch.rand(3, 300, 500)
        original_boxes = torch.tensor([[50.0, 60.0, 400.0, 250.0]], dtype=torch.float32)

        _, meta = YOLOv8Adapter._letterbox_tensor(image, imgsz=512)
        ratio = float(meta["ratio"])
        pad_left = float(meta["pad_left"])
        pad_top = float(meta["pad_top"])

        letterboxed_boxes = original_boxes.clone()
        letterboxed_boxes[:, [0, 2]] = letterboxed_boxes[:, [0, 2]] * ratio + pad_left
        letterboxed_boxes[:, [1, 3]] = letterboxed_boxes[:, [1, 3]] * ratio + pad_top

        restored_boxes = YOLOv8Adapter._rescale_boxes_from_letterbox(letterboxed_boxes, meta)
        self.assertTrue(torch.allclose(restored_boxes, original_boxes, atol=1e-4))

    def test_letterbox_output_matches_requested_size(self):
        image = torch.rand(3, 287, 513)
        letterboxed, meta = YOLOv8Adapter._letterbox_tensor(image, imgsz=(512, 640))

        self.assertEqual(tuple(letterboxed.shape), (1, 3, 512, 640))
        self.assertEqual(int(meta["orig_h"]), 287)
        self.assertEqual(int(meta["orig_w"]), 513)


if __name__ == "__main__":
    unittest.main()
