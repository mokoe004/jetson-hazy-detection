#!/bin/bash

sudo nvpmodel -m 0
sudo jetson_clocks

docker build -t inference-jetson -f inference.Dockerfile

docker run -it \
  --runtime nvidia \
  --network host \
  --ipc=host \
  --privileged \
  -v $(pwd):/workspace \
  -v /usr/bin/tegrastats:/usr/bin/tegrastats \
  inference-jetson
