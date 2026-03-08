from __future__ import annotations

from typing import List

import numpy as np
import torch

from detectors.base import DetectionBatch, DetectorAdapter


class YOLOv8Adapter(DetectorAdapter):
    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        max_det: int = 300,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for YOLOv8Adapter. Install with: pip install ultralytics"
            ) from exc

        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det

    @staticmethod
    def _to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
        if image.dim() != 3:
            raise ValueError(f"Expected CHW tensor, got shape={tuple(image.shape)}")

        img = image.detach().cpu().clamp(0.0, 1.0)
        img = (img * 255.0).round().to(torch.uint8)
        return img.permute(1, 2, 0).numpy()

    def predict(self, images: List[torch.Tensor]) -> List[DetectionBatch]:
        np_images = [self._to_uint8_hwc(img) for img in images]
        results = self.model.predict(
            source=np_images,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            verbose=False,
            device=self.device,
        )

        batches: List[DetectionBatch] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.xyxy is None:
                batches.append(
                    DetectionBatch(
                        boxes=torch.zeros((0, 4), dtype=torch.float32),
                        scores=torch.zeros((0,), dtype=torch.float32),
                        labels=torch.zeros((0,), dtype=torch.int64),
                    )
                )
                continue

            batches.append(
                DetectionBatch(
                    boxes=boxes.xyxy.detach().cpu().to(torch.float32),
                    scores=boxes.conf.detach().cpu().to(torch.float32),
                    labels=boxes.cls.detach().cpu().to(torch.int64),
                )
            )
        return batches

