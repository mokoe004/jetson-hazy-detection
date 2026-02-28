import time
import torch

def benchmark_single_image(
    model,
    device,
    input_size=(1, 3, 256, 256),
    warmup=10,
    runs=50
):
    """
    Misst reine Forward-Zeit für ein Bild
    """

    model.eval()
    model.to(device)

    dummy_input = torch.randn(input_size).to(device)

    # -------- Warmup --------
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # -------- Benchmark --------
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / runs
    avg_ms = avg_time * 1000
    fps = 1.0 / avg_time

    print(f"\nInference Benchmark ({device})")
    print(f"Avg time: {avg_ms:.2f} ms")
    print(f"FPS:      {fps:.2f}")

    return avg_ms, fps

def benchmark_dataloader(
    model,
    dataloader,
    device,
    warmup_batches=3
):
    """
    Misst Inference-Zeit über echten DataLoader.
    """

    model.eval()
    model.to(device)

    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for i, (hazy, _) in enumerate(dataloader):

            hazy = hazy.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(hazy)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            if i >= warmup_batches:
                total_time += (end - start)
                total_images += hazy.size(0)

    avg_time = total_time / total_images
    avg_ms = avg_time * 1000
    fps = 1.0 / avg_time

    print(f"\nDataloader Benchmark ({device})")
    print(f"Avg time per image: {avg_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")

    return avg_ms, fps
