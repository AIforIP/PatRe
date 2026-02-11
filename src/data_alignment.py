# -*- coding: utf-8 -*-
import json
import os

# ===================== 路径配置 =====================
RAW_DATA_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\cleaned_json_output"
REF_POOL_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\ReferencePool_Full"
SAVE_PATH = "dataset/processed/test_set.jsonl"

# ===================== 辅助函数 =====================

def get_ref_content(ref_list):
    """从引用池里抓取摘要，并进行简单的清洗"""
    if not ref_list:
        return "No prior art cited."
    
    summaries = []
    for ref in ref_list:
        doc_num = ref.get("document_number", "").replace("-", "").strip()
        # 兼容性处理：有些文件名带 US 前缀，有些不带
        possible_names = [f"{doc_num}.json", f"US{doc_num}.json"] if not doc_num.startswith("US") else [f"{doc_num}.json"]
        
        found = False
        for name in possible_names:
            path = os.path.join(REF_POOL_DIR, name)
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    ref_data = json.load(f)
                    summaries.append(f"Ref {doc_num}: {ref_data.get('abstract', 'No abstract available')}")
                    found = True
                    break
        if not found:
            summaries.append(f"Ref {doc_num}: [Summary not found in library]")
            
    return "\n".join(summaries)

def process_claims(claims):
    """
    鲁棒性处理：专利数据中 Claims 可能是字符串，也可能是列表
    """
    if isinstance(claims, list):
        # 提取 text 字段并合并，或者直接连接字符串
        processed = []
        for c in claims:
            if isinstance(c, dict):
                processed.append(c.get("text", ""))
            else:
                processed.append(str(c))
        return "\n".join(processed)
    return str(claims) if claims else ""

# ===================== 主逻辑 =====================

def align_data():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    count = 0
    
    with open(SAVE_PATH, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(RAW_DATA_DIR):
            if not filename.endswith(".json"): continue
            
            with open(os.path.join(RAW_DATA_DIR, filename), 'r', encoding='utf-8') as f:
                try:
                    item = json.load(f)
                except json.JSONDecodeError:
                    print(f"⚠️ 跳过损坏的 JSON 文件: {filename}")
                    continue
                
                # 预处理权利要求文本
                raw_claims = item.get("initial_submission", {}).get("claims", "")
                claims_text = process_claims(raw_claims)
                
                # 过滤掉没有权利要求的脏数据
                if not claims_text.strip():
                    continue

                office_actions = item.get("office_actions", [])
                for idx, oa in enumerate(office_actions):
                    # 过滤掉没有 OA 文本的轮次
                    oa_body = oa.get("office_action_text", "")
                    if not oa_body or len(oa_body) < 100: # 假设长度过短的是无效数据
                        continue
                    next_oa_body = ""
                    if idx + 1 < len(office_actions):
                        nxt = office_actions[idx + 1]
                        next_oa_body = nxt.get("office_action_text", "")

                    # 对于反驳任务，使用下一轮的OA作为输入（当前轮次的真实OA）
                    # 对于审查任务，使用当前轮次的OA作为输入
                    input_oa = oa_body  # 默认使用当前轮次OA
                    
                    formatted_item = {
                        "id": item.get("patent_number"),
                        "meta": {
                            "ipc": item.get("ipc_category", "Unknown"),
                            "claim_count": len(raw_claims) if isinstance(raw_claims, list) else 1,
                            "round_type": "rebuttal" if next_oa_body else "examination"  # 标记轮次类型
                        },
                        "input": {
                            "claims": claims_text,
                            "oracle_reference_summaries": get_ref_content(oa.get("references_cited", [])),
                            "rebuttal": oa.get("remarks", "N/A"),
                            "office_action": input_oa,  # 对于反驳：使用当前轮次OA；对于审查：使用当前轮次OA
                            "next_office_action": next_oa_body  # 下一轮OA，供反驳任务使用
                        },
                        "ground_truth": {
                            "oa_text": oa_body,
                            "rebuttal_text": oa.get("remarks", "N/A"),
                            "next_oa_text": next_oa_body,
                            "decision": oa.get("decision", "Non-Final Rejection")
                        }
                    }
                    outfile.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")
                    count += 1

    print(f"✅ 数据对齐完成！共生成 {count} 条测试样本。")
    print(f"📂 保存位置: {SAVE_PATH}")

if __name__ == "__main__":
    align_data()
