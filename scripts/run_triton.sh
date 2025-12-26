#!/bin/bash

REPO_PATH=$(pwd)/model_repository

echo "=== Starting Triton Inference Server (CPU Mode) ==="
echo "Model Repository: $REPO_PATH"

if [ ! -d "$REPO_PATH" ]; then
    echo "Ошибка: Папка model_repository не найдена."
    exit 1
fi

docker run --rm \
    --name triton_server \
    -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $REPO_PATH:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models
