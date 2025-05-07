#!/bin/bash

echo "MuTILs container is running..."

docker run \
    --name Mutils \
    --gpus '"device=0,1,2,3,4"' \
    --rm \
    -it \
    -v /path/to/the/slides:/home/input \
    -v /path/to/the/output:/home/output \
    -v /path/to/the/mutils/models:/home/models \
    --ulimit core=0 \
    szolgyen/mutils:v2 \
    bash