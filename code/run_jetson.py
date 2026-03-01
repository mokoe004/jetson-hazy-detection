import argparse
import torch
from omegaconf import OmegaConf, DictConfig

from evaluation.evaluation import run_benchmark
from utils import cfg_select_model


def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    results = run_benchmark(cfg)
    print("Benchmark results:")
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmark on Jetson")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )

    args = parser.parse_args()

    # Config laden
    cfg = OmegaConf.load(args.config)

    main(cfg)
