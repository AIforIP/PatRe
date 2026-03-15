python run_exp.py \
    --model Qwen3.5-9B \
    --task oa \
    --ipc A \
    --num_samples 1 \
    --test_mode zero_shot \
    --retrieval no_ref \
    --backend vllm \
    --generate_only 

# python run_exp.py --task oa --generate_only