# -*- coding: utf-8 -*-
import os
import json
import time
from backends import PatentAgentEvaluator

# ===================== 路径配置 =====================
MAIN_DATA_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\cleaned_json_output"
REF_POOL_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\ReferencePool"
RESULT_DIR = "./results"

# ===================== 辅助函数 =====================

def get_ref_summary(ref_num):
    """从本地引用池中获取摘要内容"""
    # 统一编号格式，去掉横杠
    clean_num = ref_num.replace("-", "").strip()
    if not clean_num.startswith("US"): 
        clean_num = "US" + clean_num
    
    path = os.path.join(REF_POOL_DIR, f"{clean_num}.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return f"Reference {ref_num} (Title: {data.get('title')}): {data.get('abstract')}"
    return f"Reference {ref_num}: Summary not found in local library."

def run_test(num_samples=5, setting="oracle"):
    """执行评测主函数"""
    # 初始化后端，切换 backend_type="vllm" 如果你想用本地模型
    evaluator = PatentAgentEvaluator(backend_type="api") 
    
    if not os.path.exists(MAIN_DATA_DIR):
        print(f"❌ 错误：找不到主数据目录 {MAIN_DATA_DIR}")
        return

    json_files = [f for f in os.listdir(MAIN_DATA_DIR) if f.endswith(".json")]
    test_files = json_files[:num_samples] 
    
    all_results = []

    for filename in test_files:
        path = os.path.join(MAIN_DATA_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            p_id = data.get("patent_number")
            initial_claims = data.get("initial_submission", {}).get("claims")
            
            # 遍历 OA，寻找有 Remark 的轮次进行反驳评测
            for oa in data.get("office_actions", []):
                remarks_gt = oa.get("remarks", "")
                if not remarks_gt or "Access denied" in remarks_gt:
                    continue
                
                # 构造引用上下文
                ref_context = ""
                if setting == "oracle":
                    ref_list = [ref.get("document_number") for ref in oa.get("references_cited", [])]
                    ref_context = "\n".join([get_ref_summary(r) for r in ref_list])

                # 构造指令
                prompt = (
                    f"You are a patent attorney. Write a detailed rebuttal (Remarks) to the following Office Action.\n\n"
                    f"### Current Patent Claims ###\n{initial_claims}\n\n"
                    f"### Examiner's Rejection ###\n{oa.get('office_action_text')}\n\n"
                    f"### Cited References Summaries ###\n{ref_context}\n\n"
                    f"### Task ###\nDistinguish the present invention from the references. Provide a legal and technical argument."
                )

                print(f"🚀 Processing Patent: {p_id} (OA #{oa.get('oa_number')})")
                prediction = evaluator.generate(prompt)
                
                all_results.append({
                    "patent_id": p_id,
                    "setting": setting,
                    "prediction": prediction,
                    "ground_truth": remarks_gt
                })
                break # 每个专利只测第一轮有效反驳

    # 保存结果文件
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    
    out_path = os.path.join(RESULT_DIR, f"{setting}_raw_outputs.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评测原始数据已保存至: {out_path}")

# ===================== 指标计算逻辑 =====================

def calculate_metrics(result_file):
    """从预测结果 JSON 中计算指标"""
    from rouge_score import rouge_scorer
    
    if not os.path.exists(result_file):
        print(f"❌ 找不到指标文件: {result_file}")
        return

    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("⚠️ 预测结果为空，无法计算指标。")
        return

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_rouge = 0
    
    print(f"📊 Calculating metrics for {len(data)} samples...")
    for entry in data:
        pred = entry['prediction']
        gt = entry['ground_truth']
        score = scorer.score(gt, pred)
        total_rouge += score['rougeL'].fmeasure
        
    avg_rouge = total_rouge / len(data)
    print(f"\n--- Metric Result ---")
    print(f"Average ROUGE-L Score: {avg_rouge:.4f}")

# ===================== 主入口 =====================

if __name__ == "__main__":
    SETTING = "oracle"
    NUM_SAMPLES = 5 # 先跑5个样本
    RESULT_FILE = f"./results/{SETTING}_raw_outputs.json"

    print("--- Step 1: Running Evaluation ---")
    run_test(num_samples=NUM_SAMPLES, setting=SETTING)

    print("\n--- Step 2: Calculating Metrics ---")
    calculate_metrics(RESULT_FILE)