#!/bin/bash

sudo nvpmodel -m 0
sudo jetson_clocks

docker build -t inference-jetson -f inference.Dockerfile .

docker run -d \
  --name inference-jetson \
  --runtime nvidia \
  --network host \
  --ipc host \
  -v $(pwd):/workspace \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats \
  inference-jetson
