# =========================================================
# Dockerfile: Jetson Inference Dev Image
# =========================================================
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

WORKDIR /workspace

# System packages
RUN apt-get update && apt-get install -y \
    python3-pip git libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install --upgrade pip

COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# create non-root user
RUN groupadd -g 1000 devuser && \
    useradd -m -u 1000 -g 1000 devuser && \
    chown -R devuser:devuser /workspace

USER devuser

# container stays alive
CMD ["sleep", "infinity"]