# =========================================================
# Dockerfile: Inference Image for Jetson Xavier NX
# =========================================================

# Base image with CUDA, TensorRT, PyTorch etc.
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3-pip git libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
# RUN pip3 install ultralytics opencv-python
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY code /workspace/project/code
COPY pretrained_models /workspace/project/pretrained_models
COPY configs /workspace/project/configs

# Standardbefehl: YOLO Demo starten (z. B. Webcam)
CMD ["bash"]
