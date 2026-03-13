import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import pickle
import json
import random
import time
from typing import Optional, Union

import cv2
import numpy as np
from torchvision import transforms

from models import AODNet, AODnetDepthwiseSpatial, AODnetDepthwiseGaussian, FFANet, LCANet, LDNet, LFDNet, GCANet, TinyDehazeNet

def print_model_info(model):
    """
    Creates simple print statements with model information
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Model name              : {model.__class__.__name__}")
    print(f"Device                  : {device}")
    print(f"Total parameters        : {num_params:,}")
    print(f"Trainable parameters    : {num_trainable_params:,}")
    print("=" * 50)

def cfg_select_model(cfg, device: str) -> nn.Module:
    torch_device = torch.device(device)
    if cfg.model.name == "AODNet":
        model = AODNet().to(torch_device)
    elif cfg.model.name == "AODNetDepthwiseSpatial":
        sigma_scale = float(getattr(cfg.model, "sigma_scale", 0.3))
        heatmap_augmentation = bool(getattr(cfg.model, "heatmap_augmentation", True))
        alpha_init = float(getattr(cfg.model, "alpha_init", 0.5))
        model = AODnetDepthwiseSpatial(
            sigma_scale=sigma_scale,
            heatmap_augmentation=heatmap_augmentation,
            alpha_init=alpha_init,
        ).to(torch_device)
    elif cfg.model.name == "AODNetDepthwiseGaussian":
        base_channels = int(getattr(cfg.model, "base_channels", 3))
        sigma_scale = float(getattr(cfg.model, "sigma_scale", 0.3))
        heatmap_augmentation = bool(getattr(cfg.model, "heatmap_augmentation", True))
        alpha_init = float(getattr(cfg.model, "alpha_init", 0.5))
        use_gaussian_attention = bool(getattr(cfg.model, "use_gaussian_attention", True))
        use_se_attention = bool(getattr(cfg.model, "use_se_attention", True))
        se_reduction = int(getattr(cfg.model, "se_reduction", 8))
        model = AODnetDepthwiseGaussian(
            base_channels=base_channels,
            sigma_scale=sigma_scale,
            heatmap_augmentation=heatmap_augmentation,
            alpha_init=alpha_init,
            use_gaussian_attention=use_gaussian_attention,
            use_se_attention=use_se_attention,
            se_reduction=se_reduction,
        ).to(torch_device)
    elif cfg.model.name == "FFANet":
        model = FFANet().to(torch_device)
    elif cfg.model.name == "LCANet":
        model = LCANet().to(torch_device)
    elif cfg.model.name == "LDNet":
        model = LDNet().to(torch_device)
    elif cfg.model.name == "LFDNet":
        model = LFDNet().to(torch_device)
    elif cfg.model.name == "GCANet":
        model = GCANet().to(torch_device)
    elif cfg.model.name == "TinyDehazeNet":
        base_channels = int(getattr(cfg.model, "base_channels", 16))
        model = TinyDehazeNet(base_channels=base_channels).to(torch_device)
    else:
        print("Model from cfg file not known. Fallback to AODNet")
        model = AODNet().to(torch_device)

    return model


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def is_gcanet_model(model: nn.Module) -> bool:
    return _unwrap_model(model).__class__.__name__ == "GCANet"


def get_model_input_channels(model: nn.Module) -> int:
    unwrapped = _unwrap_model(model)
    conv1 = getattr(unwrapped, "conv1", None)
    if isinstance(conv1, nn.Conv2d):
        return int(conv1.in_channels)

    for module in unwrapped.modules():
        if isinstance(module, nn.Conv2d):
            return int(module.in_channels)

    raise ValueError(f"Could not infer input channels for model {unwrapped.__class__.__name__}.")


def compute_edge_channel(image: torch.Tensor) -> torch.Tensor:
    """Compute the official GCANet edge map from an RGB tensor in BCHW layout."""
    if image.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(image.shape)}.")
    if image.size(1) != 3:
        raise ValueError(f"Edge computation expects RGB input with 3 channels, got {image.size(1)}.")

    x_diffx = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    x_diffy = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    edge = image.new_zeros((image.size(0), image.size(1), image.size(2), image.size(3)))
    edge[:, :, :, 1:] += x_diffx
    edge[:, :, :, :-1] += x_diffx
    edge[:, :, 1:, :] += x_diffy
    edge[:, :, :-1, :] += x_diffy
    edge = torch.sum(edge, dim=1, keepdim=True) / 3.0
    edge = edge / 4.0
    return edge


def prepare_model_input(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Adapt a BCHW tensor to the channel count expected by the dehazing model."""
    if image.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(image.shape)}.")

    expected_channels = get_model_input_channels(model)
    actual_channels = int(image.size(1))
    if expected_channels == actual_channels:
        return image
    if expected_channels == 4 and actual_channels == 3:
        edge = compute_edge_channel(image)
        return torch.cat((image, edge), dim=1)

    raise ValueError(
        f"Model expects {expected_channels} input channels, but got tensor with {actual_channels} channels."
    )


def run_dehazer(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Run a dehazing model on normalized RGB input and return normalized RGB output."""
    if image.dim() != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {tuple(image.shape)}.")

    if is_gcanet_model(model):
        rgb_255 = image * 255.0
        edge = compute_edge_channel(rgb_255)
        model_input = torch.cat((rgb_255, edge), dim=1) - 128.0
        prediction = model(model_input)
        unwrapped = _unwrap_model(model)
        if bool(getattr(unwrapped, "only_residual", False)):
            prediction = prediction + rgb_255
        return prediction.clamp(0.0, 255.0) / 255.0

    model_input = prepare_model_input(model, image)
    return model(model_input)


def load_pretrained_dehazer(
    cfg,
    device: torch.device,
    checkpoint_path: str,
    project_root: Optional[Union[str, Path]] = None,
    strict: bool = True,
    print_info: bool = True,
):
    model = cfg_select_model(cfg, device.type)
    if print_info:
        print_model_info(model)

    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_absolute():
        if project_root is None:
            project_root = Path.cwd()
        ckpt_path = (Path(project_root) / ckpt_path).resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(ckpt_path, map_location=device)
    except pickle.UnpicklingError:
        # PyTorch >=2.6 defaults to weights_only=True; older checkpoints may need full unpickling.
        state = torch.load(ckpt_path, map_location=device, weights_only=False)

    if cfg.model.name == "FFANet":
        model=nn.DataParallel(model)
        model.load_state_dict(state['model'])
        model.eval()
        return model, ckpt_path
    
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=strict)
    model.eval()
    return model, ckpt_path


def visualize_random_od_predictions(
    cfg,
    num_samples: int = 12,
    score_thr: float = 0.25,
    seed: int = 42,
    save_dir: Optional[str] = None,
):
    """Visualize random RTTS samples with GT and YOLOv8 predictions.

    Saves images with overlays:
      - GT boxes in green: "GT:<class>"
      - Pred boxes in red: "P:<class> <conf>"
    """
    from datasets import RTTSDataset
    from detectors import YOLOv8Adapter

    def _remap_rtts_gt_labels_to_coco(gt_labels: torch.Tensor) -> torch.Tensor:
        rtts_to_coco = {1: 0, 2: 1, 3: 2, 4: 3, 5: 5}
        out = gt_labels.clone().to(torch.int64)
        unique_ids = set(int(v) for v in torch.unique(out).tolist())
        unknown = sorted(v for v in unique_ids if v not in rtts_to_coco)
        if unknown:
            raise ValueError(f"Found RTTS labels without COCO remap: {unknown}")
        for src_id, dst_id in rtts_to_coco.items():
            out[out == src_id] = dst_id
        return out

    def _to_bgr_uint8(image_chw: torch.Tensor) -> np.ndarray:
        img = image_chw.detach().cpu().clamp(0.0, 1.0)
        img = (img * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def _name(names_map, cls_id: int) -> str:
        if isinstance(names_map, dict):
            return str(names_map.get(cls_id, f"class_{cls_id}"))
        return f"class_{cls_id}"

    def _draw_box(img, box, color, text):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tx = max(0, x1)
        ty = max(12, y1 - 6)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    project_root = Path(__file__).resolve().parents[1]
    if save_dir is None:
        save_dir = str((project_root / "runs" / "od_viz" / time.strftime("run_%Y_%m_%d_%H_%M_%S")).resolve())
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    image_set = str(getattr(getattr(cfg, "evaluation_od", {}), "image_set", "test"))
    dataset = RTTSDataset(cfg, image_set=image_set, transforms=transform)

    subset = getattr(getattr(cfg, "dataset", {}), "subset", False)
    max_idx = len(dataset) if not subset else min(int(subset), len(dataset))
    if max_idx <= 0:
        raise ValueError("Dataset is empty for visualization.")

    rng = random.Random(seed)
    count = min(int(num_samples), max_idx)
    chosen_indices = sorted(rng.sample(range(max_idx), count))

    detector_device = str(getattr(getattr(cfg, "detector", {}), "device", "cuda:0"))
    if detector_device.startswith("cuda") and not torch.cuda.is_available():
        detector_device = "cpu"
    detector = YOLOv8Adapter(
        weights=str(getattr(getattr(cfg, "detector", {}), "weights")),
        device=detector_device,
        conf=float(getattr(getattr(cfg, "detector", {}), "conf", 0.25)),
        iou=float(getattr(getattr(cfg, "detector", {}), "iou", 0.7)),
        imgsz=int(getattr(getattr(cfg, "detector", {}), "imgsz", 640)),
        max_det=int(getattr(getattr(cfg, "detector", {}), "max_det", 300)),
    )

    dehaze_device_str = str(getattr(getattr(cfg, "evaluation_od", {}), "device", "cuda"))
    dehaze_device = torch.device(dehaze_device_str if torch.cuda.is_available() else "cpu")
    use_dehazer = bool(getattr(getattr(cfg, "evaluation_od", {}), "use_dehazer", True))
    dehazer = None
    if use_dehazer:
        ckpt = getattr(getattr(cfg, "evaluation_od", {}), "dehazer_checkpoint_path", None)
        if not ckpt:
            raise ValueError("Missing evaluation_od.dehazer_checkpoint_path in config.")
        dehazer, _ = load_pretrained_dehazer(
            cfg=cfg,
            device=dehaze_device,
            checkpoint_path=str(ckpt),
            project_root=project_root,
            strict=True,
            print_info=False,
        )

    dehazer_size = getattr(getattr(cfg, "evaluation_od", {}), "dehazer_input_size", False)
    names_map = getattr(detector.model, "names", {})

    meta = []
    with torch.no_grad():
        for rank, idx in enumerate(chosen_indices):
            image, target = dataset[idx]
            image_batched = image.to(dehaze_device).unsqueeze(0)
            original_h, original_w = image_batched.shape[-2:]

            dehaze_input = image_batched
            if use_dehazer and dehazer_size:
                size = int(dehazer_size)
                dehaze_input = F.interpolate(image_batched, size=(size, size), mode="bilinear", align_corners=False)

            if use_dehazer:
                dehazed = run_dehazer(dehazer, dehaze_input)
                if dehazer_size:
                    dehazed = F.interpolate(dehazed, size=(original_h, original_w), mode="bilinear", align_corners=False)
            else:
                dehazed = dehaze_input

            dehazed = dehazed.clamp(0.0, 1.0).squeeze(0).cpu()
            pred = detector.predict([dehazed])[0]

            gt_boxes = target["boxes"].detach().cpu().to(torch.float32)
            gt_labels = _remap_rtts_gt_labels_to_coco(target["labels"].detach().cpu().to(torch.int64))

            canvas = _to_bgr_uint8(dehazed)

            for box, label in zip(gt_boxes, gt_labels):
                cls_id = int(label.item())
                _draw_box(canvas, box.tolist(), (0, 220, 0), f"GT:{_name(names_map, cls_id)}")

            keep = pred.scores >= float(score_thr)
            pred_boxes = pred.boxes[keep]
            pred_labels = pred.labels[keep]
            pred_scores = pred.scores[keep]
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                cls_id = int(label.item())
                sc = float(score.item())
                _draw_box(canvas, box.tolist(), (0, 0, 255), f"P:{_name(names_map, cls_id)} {sc:.2f}")

            img_id = dataset.ids[idx] if hasattr(dataset, "ids") else f"idx_{idx}"
            out_name = f"{rank:03d}_{img_id}.jpg"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), canvas)

            meta.append(
                {
                    "rank": rank,
                    "dataset_index": int(idx),
                    "image_id": str(img_id),
                    "saved_image": str(out_path),
                    "num_gt": int(gt_boxes.shape[0]),
                    "num_pred_kept": int(pred_boxes.shape[0]),
                    "score_thr": float(score_thr),
                }
            )

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(meta)} visualizations to: {out_dir}")
    return out_dir
