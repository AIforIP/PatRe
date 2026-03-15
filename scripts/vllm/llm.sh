#!/bin/bash

MODEL_PATH=/data/shared/users/wangqiyao/models/Qwen3.5-9B
PORT=${1:-8000}

MODEL_NAME=$(basename $MODEL_PATH)

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1