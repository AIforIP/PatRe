python run_exp.py \
    --model deepseek-v3.2 \
    --task oa \
    --test_mode zero_shot \
    --retrieval bm25_retrieved \
    --backend api \
    --generate_only 

# no_ref|oracle_summary|bm25_retrieved