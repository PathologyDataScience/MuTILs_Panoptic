#!/bin/bash

echo "Test started. Container is running..."

slide_path=$1
models_path=$2

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SLIDE_BASE_DIR=$(echo $slide_path | cut -d'/' -f1,2)
MODELS_BASE_DIR=$(echo $models_path | cut -d'/' -f1,2)

docker run \
    --name MutilsTest \
    --rm \
    -it \
    -v $BASE_DIR/MuTILs_Panoptic:/home/MuTILs_Panoptic \
    -v $BASE_DIR/MuTILs_Panoptic/tests/input:/home/input \
    -v $BASE_DIR/MuTILs_Panoptic/tests/output:/home/output \
    -v $SLIDE_BASE_DIR:$SLIDE_BASE_DIR \
    -v $MODELS_BASE_DIR:$MODELS_BASE_DIR \
    szolgyen/mutils:v1 \
    bash -c "source /home/MuTILs_Panoptic/tests/.bashrc $slide_path $models_path && bash"

echo "Test finished. Container exited."