#!/bin/bash

sudo nvpmodel -m 0
sudo jetson_clocks

docker run -it \
  --runtime nvidia \
  --network host \
  --ipc=host \
  --privileged \
  -v $(pwd):/workspace \
  your_image
