from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class DetectionBatch:
    boxes: torch.Tensor  # [N, 4] in xyxy pixel coordinates
    scores: torch.Tensor  # [N]
    labels: torch.Tensor  # [N] int class ids


class DetectorAdapter:
    def predict(self, images: List[torch.Tensor]) -> List[DetectionBatch]:
        raise NotImplementedError

