import torch
import time
from tqdm import tqdm

from models.lca_net import LCANet
from models.aod_net import AODnet

def inference_benchmark(cfg, weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # model = LCANet(out_act="sigmoid").to(device)
    model = AODnet().to(device)
    model.eval()
    num_runs = 50

    # -------------------------
    # Model Info
    # -------------------------
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

    # Dummy Input
    x = torch.rand(1, 3, 512, 512).to(device)

    print("\nWarmup phase (10 runs)...")
    with torch.no_grad():
        for _ in tqdm(range(10), desc="Warmup", leave=False):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # -------------------------
    # Timing
    # -------------------------
    print(f"\nRunning benchmark ({num_runs} runs)...")

    start = time.time()

    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="Benchmark", leave=False):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()

    avg_time = (end - start) / num_runs
    fps = 1 / avg_time

    # -------------------------
    # Results
    # -------------------------
    print("\n" + "=" * 50)
    print("PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"Input size              : {tuple(x.shape)}")
    print(f"Average forward time    : {avg_time:.6f} seconds")
    print(f"Frames per second (FPS) : {fps:.2f}")
    print("=" * 50)
