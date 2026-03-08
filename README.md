# Jetson Hazy Detection

PyTorch-based dehazing project for low-visibility scenes, with training on paired hazy/clear datasets (RESIDE-OTS style) and inference benchmarking on NVIDIA Jetson devices.

## Features

- Multiple dehazing backbones: `AODnet`, `FFANet`, `LCANet`, `LDNet`, `LDFNet`
- Config-driven training with YAML (`OmegaConf`)
- Validation metrics: PSNR and SSIM
- Object detection evaluation after dehazing (`precision`, `recall`, `mAP@0.5`, `mAP@0.5:0.95`)
- Jetson-focused benchmark pipeline with optional `tegrastats` monitoring
- Docker setup for Jetson Xavier NX inference runs

## Repository Layout

```text
.
|- code/
|  |- train.py                  # training entry point
|  |- run_jetson.py             # benchmark entry point
|  |- dataloaders.py            # dataset loaders (RESIDE-OTS, RTTS, paired)
|  |- models/                   # model definitions
|  \- evaluation/               # PSNR/SSIM + benchmark utilities
|- configs/
|  |- train/
|  |  |- config.yaml            # default training config
|  |  \- train_*.yaml           # model-specific training configs
|  |- evaluate/
|  |  |- evaluate.yaml          # dehazing evaluation config
|  |  \- evaluate_od.yaml       # dehaze + OD evaluation config
|  \- inference/
|     \- inference_config.yaml  # benchmark config
|- pretrained_models/           # local checkpoints
|- inference.Dockerfile         # Jetson inference container
\- docker_run.sh               # helper script for Jetson Docker run
```

## Installation

### Local (Python)

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Jetson Docker (Xavier NX)

```bash
bash docker_run.sh
```

This script:
- sets max performance mode (`nvpmodel`, `jetson_clocks`)
- builds `inference-jetson` from `inference.Dockerfile`
- starts a container with NVIDIA runtime and project mounts

## Dataset Setup

Training expects a RESIDE-OTS-like structure (see `configs/train/train_*.yaml`):

```text
datasets/
\- RESIDE-OTS/
   |- hazy-part2/
   \- clear/
```

Set dataset paths in your selected config file:
- `dataset.root`
- `dataset.hazy_path`
- `dataset.clear_path`

## Training

Run training with a config:

```bash
python code/train.py --config configs/train/train_aodnet.yaml
```

Other ready-to-use configs:
- `configs/train/train_ffanet.yaml`
- `configs/train/train_lcanet.yaml`
- `configs/train/train_ldnet.yaml`
- `configs/train/train_lfdnet.yaml`

Outputs are written under the configured `model.save_path`:
- timestamped run folder
- `models/best_model.pth`
- `models/last_model.pth`
- `training_log.csv`
- validation sample images in `outputs/`

## Jetson Benchmark

Run latency/FPS benchmark:

```bash
python code/run_jetson.py --config configs/inference/inference_config.yaml
```

Main benchmark config options:
- `benchmark.device` (`cuda`/`cpu`)
- `benchmark.use_fp16`
- `benchmark.input_size`
- `benchmark.runs`, `benchmark.warmup`
- `benchmark.jetson.enable_tegrastats`

Results are saved to `runs/benchmark/run_YYYYMMDD_HHMMSS/`:
- `metrics.json`
- `metrics.csv`
- `environment.json`
- copied benchmark config

## Dehaze + Object Detection Evaluation

Run end-to-end evaluation of `dehazer -> detector`:

```bash
python code/evaluate_od.py --config configs/evaluate/evaluate_od.yaml
```

Default setup uses:
- `RTTSDataset` with bounding boxes (`dataset.return_bboxes: true`)
- dehazing checkpoint from `evaluation_od.dehazer_checkpoint_path`
- YOLOv8 detector from `detector.weights`

Results are saved to `runs/od_eval/.../run_YYYY_MM_DD_HH_MM_SS/`:
- `metrics.json`
- `metrics.csv`
- `run_config.yaml`

## Notes

- Use exact model names from `code/utils.py` in configs: `AODnet`, `FFANet`, `LCANet`, `LDNet`, `LDFNet`.
- `yolov5_annotations.py` and `yolov8_annotations.py` are helper templates and require path/model adaptation before use.
