from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

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
        half: bool = False,
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required for YOLOv8Adapter. Install with: pip install ultralytics"
            ) from exc

        self.model = YOLO(weights)
        self.weights = str(weights)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.half = bool(half)
        self.declared_backend = self._infer_backend_from_weights(self.weights)

    @staticmethod
    def _infer_backend_from_weights(weights: str) -> str:
        suffix = Path(weights).suffix.lower()
        return {
            ".pt": "pytorch",
            ".torchscript": "torchscript",
            ".onnx": "onnx",
            ".engine": "tensorrt",
        }.get(suffix, "unknown")

    @staticmethod
    def _serialize_runtime_value(value):
        if isinstance(value, torch.device):
            return str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return [YOLOv8Adapter._serialize_runtime_value(v) for v in value]
        if isinstance(value, list):
            return [YOLOv8Adapter._serialize_runtime_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): YOLOv8Adapter._serialize_runtime_value(v) for k, v in value.items()}
        return value

    @staticmethod
    def _infer_backend_from_autobackend(backend_model) -> str:
        for attr, name in (
            ("engine", "tensorrt"),
            ("pt", "pytorch"),
            ("jit", "torchscript"),
            ("onnx", "onnx"),
            ("xml", "openvino"),
            ("coreml", "coreml"),
            ("saved_model", "tf_saved_model"),
            ("pb", "tf_graphdef"),
            ("tflite", "tflite"),
            ("paddle", "paddle"),
            ("ncnn", "ncnn"),
            ("triton", "triton"),
        ):
            if bool(getattr(backend_model, attr, False)):
                return name
        return "unknown"

    def _get_backend_model(self):
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return None
        return getattr(predictor, "model", None)

    def _resolve_runtime_device(self) -> torch.device | None:
        backend_model = self._get_backend_model()
        runtime_device = getattr(backend_model, "device", None) if backend_model is not None else None
        if isinstance(runtime_device, torch.device):
            return runtime_device

        try:
            return torch.device(str(runtime_device if runtime_device is not None else self.device))
        except (TypeError, RuntimeError):
            return None

    @staticmethod
    def _resolve_imgsz(imgsz: int | tuple[int, int] | list[int]) -> tuple[int, int]:
        if isinstance(imgsz, int):
            return imgsz, imgsz
        if isinstance(imgsz, (tuple, list)) and len(imgsz) == 2:
            return int(imgsz[0]), int(imgsz[1])
        raise ValueError(f"Unsupported imgsz value: {imgsz!r}")

    @staticmethod
    def _ensure_chw(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 4:
            if image.size(0) != 1:
                raise ValueError(f"Expected a single image tensor, got shape={tuple(image.shape)}")
            image = image.squeeze(0)
        if image.dim() != 3:
            raise ValueError(f"Expected CHW tensor, got shape={tuple(image.shape)}")
        return image

    @staticmethod
    def _letterbox_tensor(
        image: torch.Tensor,
        imgsz: int | tuple[int, int] | list[int],
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        image = YOLOv8Adapter._ensure_chw(image).detach()
        if not image.is_floating_point():
            image = image.to(torch.float32)
        if image.max() > 1.0:
            image = image / 255.0
        image = image.clamp(0.0, 1.0)

        target_h, target_w = YOLOv8Adapter._resolve_imgsz(imgsz)
        _, orig_h, orig_w = image.shape

        ratio = min(target_h / orig_h, target_w / orig_w)
        resized_h = min(int(round(orig_h * ratio)), target_h)
        resized_w = min(int(round(orig_w * ratio)), target_w)

        resized = image.unsqueeze(0)
        if (resized_h, resized_w) != (orig_h, orig_w):
            resized = F.interpolate(
                resized,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )

        pad_h = target_h - resized_h
        pad_w = target_w - resized_w
        top = int(round(pad_h / 2 - 0.1))
        bottom = int(round(pad_h / 2 + 0.1))
        left = int(round(pad_w / 2 - 0.1))
        right = int(round(pad_w / 2 + 0.1))

        letterboxed = F.pad(resized, (left, right, top, bottom), value=114.0 / 255.0)
        meta = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "ratio": float(ratio),
            "pad_left": left,
            "pad_top": top,
        }
        return letterboxed, meta

    @staticmethod
    def _rescale_boxes_from_letterbox(
        boxes: torch.Tensor,
        meta: dict[str, float | int],
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes

        ratio = float(meta["ratio"])
        pad_left = float(meta["pad_left"])
        pad_top = float(meta["pad_top"])
        orig_w = float(meta["orig_w"])
        orig_h = float(meta["orig_h"])

        boxes = boxes.clone()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp_(0.0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp_(0.0, orig_h)
        return boxes

    @staticmethod
    def _to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
        img = YOLOv8Adapter._ensure_chw(image).detach().cpu().clamp(0.0, 1.0)
        img = (img * 255.0).round().to(torch.uint8)
        return img.permute(1, 2, 0).numpy()

    def _predict_tensor(self, images: List[torch.Tensor]) -> List[DetectionBatch]:
        letterboxed_images = []
        metas = []
        for image in images:
            letterboxed, meta = self._letterbox_tensor(image, self.imgsz)
            letterboxed_images.append(letterboxed)
            metas.append(meta)

        batch = torch.cat(letterboxed_images, dim=0)
        results = self.model.predict(
            source=batch,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            half=self.half,
            verbose=False,
            device=self.device,
        )

        batches: List[DetectionBatch] = []
        for result, meta in zip(results, metas):
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

            pred_boxes = boxes.xyxy.detach().cpu().to(torch.float32)
            pred_boxes = self._rescale_boxes_from_letterbox(pred_boxes, meta)
            batches.append(
                DetectionBatch(
                    boxes=pred_boxes,
                    scores=boxes.conf.detach().cpu().to(torch.float32),
                    labels=boxes.cls.detach().cpu().to(torch.int64),
                )
            )
        return batches

    def predict(self, images: List[torch.Tensor]) -> List[DetectionBatch]:
        return self._predict_tensor(images)

    def synchronize(self) -> None:
        runtime_device = self._resolve_runtime_device()
        if runtime_device is not None and runtime_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(runtime_device)

    def runtime_info(self) -> dict:
        info = {
            "detector_backend_declared": self.declared_backend,
            "detector_device_requested": str(self.device),
            "detector_half_requested": bool(self.half),
            "detector_imgsz_config": list(self._resolve_imgsz(self.imgsz)),
            "detector_weights_path": self.weights,
        }

        backend_model = self._get_backend_model()
        if backend_model is None:
            return info

        runtime_imgsz = getattr(backend_model, "imgsz", None)
        info.update(
            {
                "detector_backend_actual": self._infer_backend_from_autobackend(backend_model),
                "detector_device_runtime": self._serialize_runtime_value(getattr(backend_model, "device", None)),
                "detector_fp16_runtime": bool(getattr(backend_model, "fp16", False)),
                "detector_dynamic_runtime": bool(getattr(backend_model, "dynamic", False)),
                "detector_stride_runtime": int(getattr(backend_model, "stride", 0)),
                "detector_end2end_runtime": bool(getattr(backend_model, "end2end", False)),
                "detector_imgsz_runtime": self._serialize_runtime_value(runtime_imgsz),
            }
        )
        return info
