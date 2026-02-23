import torch
import time

from aodnetX import DehazeNetAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DehazeNetAttention().to(device)
model.eval()

# Dummy Input
x = torch.rand(1, 3, 512, 512).to(device)

# Dummy bounding boxes
# Format: [class, x_center, y_center, w, h, conf]
dummy_boxes = [
    [
        [0, 0.5, 0.5, 0.4, 0.4, 1.0],
        [0, 0.2, 0.3, 0.2, 0.2, 1.0]
    ]
]

# -------------------------
# Warmup
# -------------------------
with torch.no_grad():
    for _ in range(10):
        _ = model(x, dummy_boxes)

if device.type == "cuda":
    torch.cuda.synchronize()

# -------------------------
# Timing
# -------------------------
num_runs = 50
start = time.time()

with torch.no_grad():
    for _ in range(num_runs):
        _ = model(x, dummy_boxes)

if device.type == "cuda":
    torch.cuda.synchronize()

end = time.time()

avg_time = (end - start) / num_runs

print(f"Average forward time: {avg_time:.6f} seconds")
print(f"FPS: {1/avg_time:.2f}")