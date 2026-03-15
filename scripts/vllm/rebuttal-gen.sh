python run_exp.py \
    --model gemma-3-12b-it \
    --task rebuttal \
    --test_mode zero_shot \
    --retrieval no_ref \
    --backend vllm \
    --generate_only 