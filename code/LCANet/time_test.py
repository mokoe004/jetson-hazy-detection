import torch
import time
from tqdm import tqdm

from lca_net import LCANet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LCANet(out_act="sigmoid").to(device)
model.eval()

# Dummy Input
x = torch.rand(1, 3, 512, 512).to(device)

# -------------------------
# Warmup (sehr wichtig!)
# -------------------------
with torch.no_grad():
    for _ in tqdm(range(10)):
        _ = model(x)

if device.type == "cuda":
    torch.cuda.synchronize()

# -------------------------
# Timing
# -------------------------
num_runs = 50
start = time.time()

with torch.no_grad():
    for _ in tqdm(range(num_runs)):
        _ = model(x)

if device.type == "cuda":
    torch.cuda.synchronize()

end = time.time()

avg_time = (end - start) / num_runs

print(f"Average forward time: {avg_time:.6f} seconds")
print(f"FPS: {1/avg_time:.2f}")
print(device)