#!/bin/bash

MODEL_PATH=$1
PORT=${2:-8000}

MODEL_NAME=/data/shared/users/wangqiyao/models/Qwen3.5-9B

CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --host 0.0.0.0 \
    --port $PORT \
    --tensor-parallel-size 1