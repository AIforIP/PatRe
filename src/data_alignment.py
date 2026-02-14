# -*- coding: utf-8 -*-
import json
import os
import re

# ===================== 路径配置 =====================
# 使用相对路径以提高通用性
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_BASE = os.path.join(BASE_DIR, "dataset")
TARGET_CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H"]
REF_POOL_DIR = os.path.join(BASE_DIR, "dataset", "ReferencePool_Full")
SAVE_PATH = os.path.join(BASE_DIR, "dataset", "processed", "test_set.jsonl")
SAVE_PATH_OA = os.path.join(BASE_DIR, "dataset", "processed", "test_set_oa.jsonl")
SAVE_PATH_REBUTTAL = os.path.join(BASE_DIR, "dataset", "processed", "test_set_rebuttal.jsonl")

# ===================== 辅助函数 =====================

def get_ref_content(ref_list):
    """从引用池里抓取摘要，并进行简单的清洗"""
    if not ref_list:
        return "No prior art cited.", True

    summaries = []
    any_missing = False
    for ref in ref_list:
        doc_num = ref.get("document_number", "").replace("-", "").strip()
        # 兼容性处理：有些文件名带 US 前缀，有些不带
        possible_names = [f"{doc_num}.json", f"US{doc_num}.json"] if not doc_num.startswith("US") else [f"{doc_num}.json"]

        found = False
        for name in possible_names:
            path = os.path.join(REF_POOL_DIR, name)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    ref_data = json.load(f)
                    summaries.append(f"Ref {doc_num}: {ref_data.get('abstract', 'No abstract available')}")
                    found = True
                    break
        if not found:
            summaries.append(f"Ref {doc_num}: [Summary not found in library]")
            any_missing = True

    return "\n".join(summaries), any_missing


def _standardize_pid(pid):
    p = re.sub(r"[^a-zA-Z0-9]", "", str(pid)).upper()
    if not p:
        return ""
    if not p.startswith("US"):
        p = "US" + p
    return p


def extract_citations_gt(ref_list):
    """从 references_cited 抽取结构化引用并过滤未爬到的"""
    if not ref_list:
        return []
    citations = []
    seen = set()
    for ref in ref_list:
        doc_num = _standardize_pid(ref.get("document_number", ""))
        if not doc_num:
            continue
        path = os.path.join(REF_POOL_DIR, f"{doc_num}.json")
        if not os.path.exists(path):
            continue
        if doc_num in seen:
            continue
        citations.append(doc_num)
        seen.add(doc_num)
    return citations


def process_claims(claims):
    """
    鲁棒性处理：专利数据中 Claims 可能是字符串，也可能是列表
    """
    if isinstance(claims, list):
        # 鎻愬彇 text 瀛楁骞跺悎骞讹紝鎴栬€呯洿鎺ヨ繛鎺ュ瓧绗︿覆
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
    count_oa = 0
    count_rebuttal = 0

    with open(SAVE_PATH, "w", encoding="utf-8") as outfile, \
         open(SAVE_PATH_OA, "w", encoding="utf-8") as outfile_oa, \
         open(SAVE_PATH_REBUTTAL, "w", encoding="utf-8") as outfile_rebuttal:
        for cat in TARGET_CATEGORIES:
            raw_dir = os.path.join(RAW_DATA_BASE, f"PatTimeline_{cat}_Category", "cleaned_json_output")
            if not os.path.exists(raw_dir):
                continue
            for filename in os.listdir(raw_dir):
                if not filename.endswith(".json"):
                    continue

                with open(os.path.join(raw_dir, filename), "r", encoding="utf-8") as f:
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
                    if not oa_body or len(oa_body) < 100:  # 假设长度过短的是无效数据
                        continue

                    next_oa_body = ""
                    if idx + 1 < len(office_actions):
                        nxt = office_actions[idx + 1]
                        next_oa_body = nxt.get("office_action_text", "")

                    # 对于反驳任务，使用下一轮的 OA 作为输入（当前轮的真实 OA）
                    # 对于审查任务，使用当前轮 de OA 作为输入
                    input_oa = oa_body  # 默认使用当前轮次 OA

                    ref_text, ref_missing = get_ref_content(oa.get("references_cited", []))
                    citations_gt = extract_citations_gt(oa.get("references_cited", []))
                    decision_value = oa.get("decision", "")
                    decision_value = str(decision_value).strip()
                    decision_missing = not bool(decision_value)
                    if decision_missing:
                        decision_value = "Unknown"

                    ipc_value = item.get("ipc_category") or cat
                    formatted_item = {
                        "id": item.get("patent_number"),
                        "meta": {
                            "ipc": ipc_value,
                            "claim_count": len(raw_claims) if isinstance(raw_claims, list) else 1,
                            "round_type": "rebuttal" if next_oa_body else "examination",  # 标记轮次类型
                            "reference_missing": ref_missing,
                            "decision_missing": decision_missing,
                            "has_rebuttal": bool(oa.get("remarks") and str(oa.get("remarks")).strip() and str(oa.get("remarks")).strip().upper() != "N/A"),
                        },
                        "input": {
                            "claims": claims_text,
                            "oracle_reference_summaries": ref_text,
                            "rebuttal": oa.get("remarks", "N/A"),
                            "office_action": input_oa,  # 对于反驳：使用当前轮次 OA；对于审查：使用当前轮次 OA
                            "next_office_action": next_oa_body,  # 下一轮 OA，供反驳任务使用
                        },
                        "ground_truth": {
                            "oa_text": oa_body,
                            "rebuttal_text": oa.get("remarks", "N/A"),
                            "next_oa_text": next_oa_body,
                            "decision": decision_value,
                            "citations_gt": citations_gt,
                        },
                    }
                    line = json.dumps(formatted_item, ensure_ascii=False) + "\n"
                    outfile.write(line)
                    outfile_oa.write(line)
                    count_oa += 1

                    rebuttal_text = formatted_item["input"].get("rebuttal", "")
                    if isinstance(rebuttal_text, str):
                        rebuttal_text = rebuttal_text.strip()
                    if rebuttal_text and rebuttal_text.upper() != "N/A":
                        outfile_rebuttal.write(line)
                        count_rebuttal += 1
                    count += 1

    print(f"Alignment complete. Generated {count} samples.")
    print(f"Saved to: {SAVE_PATH}")
    print(f"OA-only saved to: {SAVE_PATH_OA} ({count_oa} samples)")
    print(f"Rebuttal-only saved to: {SAVE_PATH_REBUTTAL} ({count_rebuttal} samples)")


if __name__ == "__main__":
    align_data()

