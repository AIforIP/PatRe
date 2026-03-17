#!/bin/bash

MODEL_PATH=/data/shared/users/wangqiyao/models/Llama-3.1-8B-Instruct
PORT=${1:-8001}

MODEL_NAME=$(basename $MODEL_PATH)

CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --host 0.0.0.0 \
    --port $PORT \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1