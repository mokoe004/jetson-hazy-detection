from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class EvalSample:
    pred_boxes: torch.Tensor
    pred_scores: torch.Tensor
    pred_labels: torch.Tensor
    gt_boxes: torch.Tensor
    gt_labels: torch.Tensor
    image_id: int


def _box_iou_single_to_many(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)

    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    union = area_box + area_boxes - inter
    return inter / union.clamp(min=1e-9)


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _class_ap_at_iou(samples: List[EvalSample], class_id: int, iou_thr: float) -> Tuple[float, int, int, int]:
    gt_by_image: Dict[int, torch.Tensor] = {}
    matched_by_image: Dict[int, torch.Tensor] = {}
    total_gt = 0

    preds = []
    for s in samples:
        gt_mask = s.gt_labels == class_id
        gt_cls_boxes = s.gt_boxes[gt_mask]
        gt_by_image[s.image_id] = gt_cls_boxes
        matched_by_image[s.image_id] = torch.zeros((gt_cls_boxes.shape[0],), dtype=torch.bool)
        total_gt += int(gt_cls_boxes.shape[0])

        pred_mask = s.pred_labels == class_id
        cls_pred_boxes = s.pred_boxes[pred_mask]
        cls_pred_scores = s.pred_scores[pred_mask]

        for box, score in zip(cls_pred_boxes, cls_pred_scores):
            preds.append((float(score.item()), s.image_id, box))

    if total_gt == 0:
        return float("nan"), 0, 0, 0

    preds.sort(key=lambda x: x[0], reverse=True)
    if len(preds) == 0:
        return 0.0, 0, 0, total_gt

    tp = np.zeros((len(preds),), dtype=np.float32)
    fp = np.zeros((len(preds),), dtype=np.float32)

    for i, (_, image_id, pred_box) in enumerate(preds):
        gt_boxes = gt_by_image[image_id]
        if gt_boxes.shape[0] == 0:
            fp[i] = 1.0
            continue

        ious = _box_iou_single_to_many(pred_box, gt_boxes)
        best_iou, best_idx = (float(torch.max(ious).item()), int(torch.argmax(ious).item()))

        if best_iou >= iou_thr and not matched_by_image[image_id][best_idx]:
            tp[i] = 1.0
            matched_by_image[image_id][best_idx] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    recall = tp_cum / max(total_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    ap = _compute_ap(recall, precision)

    final_tp = int(tp_cum[-1]) if tp_cum.size else 0
    final_fp = int(fp_cum[-1]) if fp_cum.size else 0
    return ap, final_tp, final_fp, total_gt


def evaluate_detection(
    samples: List[EvalSample],
    iou_thresholds: List[float] | None = None,
) -> Dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = [round(x, 2) for x in np.arange(0.5, 1.0, 0.05)]

    class_ids = set()
    for s in samples:
        class_ids.update(s.gt_labels.tolist())
        class_ids.update(s.pred_labels.tolist())
    class_ids = sorted(int(c) for c in class_ids)

    if len(class_ids) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "map50": 0.0,
            "map50_95": 0.0,
            "num_images": float(len(samples)),
            "num_classes_observed": 0.0,
        }

    map_per_thr = []
    p50_tp = 0
    p50_fp = 0
    p50_gt = 0
    ap50_values = []

    for thr in iou_thresholds:
        ap_values = []
        thr_tp = 0
        thr_fp = 0
        thr_gt = 0

        for class_id in class_ids:
            ap, tp, fp, n_gt = _class_ap_at_iou(samples, class_id=class_id, iou_thr=thr)
            if not np.isnan(ap):
                ap_values.append(ap)
            thr_tp += tp
            thr_fp += fp
            thr_gt += n_gt

        if len(ap_values) > 0:
            map_per_thr.append(float(np.mean(ap_values)))
        else:
            map_per_thr.append(0.0)

        if abs(thr - 0.5) < 1e-9:
            p50_tp, p50_fp, p50_gt = thr_tp, thr_fp, thr_gt
            ap50_values = ap_values

    precision = float(p50_tp / max(p50_tp + p50_fp, 1))
    recall = float(p50_tp / max(p50_gt, 1))
    map50 = float(np.mean(ap50_values)) if len(ap50_values) > 0 else 0.0
    map50_95 = float(np.mean(map_per_thr)) if len(map_per_thr) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "num_images": float(len(samples)),
        "num_classes_observed": float(len(class_ids)),
    }

