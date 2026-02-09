import json
import os
from backends import PatentAgentEvaluator

# 这里的 DATA_FILE 是我之前建议你生成的那个对齐后的 jsonl 文件
DATA_FILE = "patre_final_dataset_for_nips.jsonl"
RESULT_DIR = "results"

def construct_prompt(sample, setting="oracle"):
    claims = sample['input']['claims']
    oa = sample['input']['office_action_text']
    refs = sample['input']['reference_summaries']
    
    if setting == "blind": # 不给任何引文内容，只给编号
        return f"Patent Claims:\n{claims}\n\nOffice Action:\n{oa}\n\nPlease write a rebuttal."
    
    if setting == "oracle": # 给引文摘要（学长要求的设置）
        return f"Patent Claims:\n{claims}\n\nOffice Action:\n{oa}\n\nReference Summaries:\n{refs}\n\nPlease write a rebuttal based on technical differences."

def run_experiment(setting="oracle", model_name="my-vllm-model"):
    evaluator = PatentAgentEvaluator() # 默认连本地 vLLM
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f][:20] # 先拿20个小样本测试

    output_results = []
    for sample in samples:
        print(f"🚀 Testing Patent: {sample['meta']['patent_id']}")
        prompt = construct_prompt(sample, setting=setting)
        
        # 模型生成的答案
        prediction = evaluator.generate_rebuttal(prompt, model_name=model_name)
        
        output_results.append({
            "patent_id": sample['meta']['patent_id'],
            "setting": setting,
            "prompt": prompt,
            "prediction": prediction,
            "ground_truth": sample['output']['author_remarks']
        })

    # 保存结果
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/{setting}_results.json", 'w', encoding='utf-8') as f:
        json.dump(output_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_experiment(setting="oracle")