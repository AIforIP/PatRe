python run_exp.py \
    --model Llama-3.3-70B-Instruct \
    --task oa \
    --test_mode zero_shot \
    --retrieval bm25_retrieved \
    --backend vllm \
    --generate_only 