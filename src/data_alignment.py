# -*- coding: utf-8 -*-
import json
import os

# ===================== 路径配置 =====================
# 使用相对路径以提高通用性
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_ROOT = os.path.join(BASE_DIR, "dataset")
RAW_DATA_DIRS = []
for name in sorted(os.listdir(RAW_DATA_ROOT)):
    if name.startswith("PatTimeline_") and name.endswith("_Category"):
        candidate = os.path.join(RAW_DATA_ROOT, name, "cleaned_json_output")
        if os.path.isdir(candidate):
            RAW_DATA_DIRS.append(candidate)
REF_POOL_DIR = os.path.join(BASE_DIR, "dataset", "ReferencePool_Full")
SAVE_PATH = os.path.join(BASE_DIR, "dataset", "processed", "test_set_new.jsonl")

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
                    abstract_text = ref_data.get("abstract", "No abstract available")
                    summary_text = ref_data.get("summary_of_invention", "")
                    if isinstance(summary_text, str):
                        summary_text = summary_text.strip()
                    else:
                        summary_text = ""
                    if summary_text and summary_text.lower() != "summary content not found":
                        summaries.append(
                            f"Ref {doc_num}: Abstract: {abstract_text}\nSummary of invention: {summary_text}"
                        )
                    else:
                        summaries.append(f"Ref {doc_num}: {abstract_text}")
                    found = True
                    break
        if not found:
            summaries.append(f"Ref {doc_num}: [Summary not found in library]")
            any_missing = True

    return "\n".join(summaries), any_missing


def infer_ipc_from_dir(raw_dir):
    parent = os.path.basename(os.path.dirname(raw_dir))
    if parent.startswith("PatTimeline_") and parent.endswith("_Category"):
        mid = parent[len("PatTimeline_"):-len("_Category")]
        if mid:
            return mid.upper()
    return "Unknown"


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

    if not RAW_DATA_DIRS:
        print("WARNING: No raw data directories found under dataset.")
        return

    with open(SAVE_PATH, "w", encoding="utf-8") as outfile:
        for raw_dir in RAW_DATA_DIRS:
            inferred_ipc = infer_ipc_from_dir(raw_dir)
            for filename in os.listdir(raw_dir):
                if not filename.endswith(".json"):
                    continue

                with open(os.path.join(raw_dir, filename), "r", encoding="utf-8") as f:
                    try:
                        item = json.load(f)
                    except json.JSONDecodeError:
                        print(f"?? ????? JSON ??: {filename}")
                        continue

                    raw_claims = item.get("initial_submission", {}).get("claims", "")
                    claims_text = process_claims(raw_claims)

                    if not claims_text.strip():
                        continue

                    office_actions = item.get("office_actions", [])
                    # Ensure deterministic round order by oa_number (fallback to original order).
                    if isinstance(office_actions, list) and office_actions:
                        def _oa_sort_key(oa_obj, idx):
                            if not isinstance(oa_obj, dict):
                                return (1, idx)
                            num = oa_obj.get("oa_number")
                            try:
                                return (0, int(num))
                            except Exception:
                                return (1, idx)
                        office_actions = [
                            oa for _, oa in sorted(
                                enumerate(office_actions),
                                key=lambda pair: _oa_sort_key(pair[1], pair[0])
                            )
                        ]
                    for idx, oa in enumerate(office_actions):
                        oa_body = oa.get("office_action_text", "")
                        if not oa_body or len(oa_body) < 100:
                            continue

                        next_oa_body = ""
                        if idx + 1 < len(office_actions):
                            nxt = office_actions[idx + 1]
                            next_oa_body = nxt.get("office_action_text", "")

                        input_oa = oa_body

                        # Use rebuttal from the *current* OA.
                        # This aligns with: OA#1 -> remarks (response) -> OA#2.
                        cur_remarks = ""
                        if isinstance(oa, dict):
                            cur_remarks = oa.get("remarks", "N/A")

                        ref_text, ref_missing = get_ref_content(oa.get("references_cited", []))
                        decision_value = oa.get("decision", "")
                        decision_value = str(decision_value).strip()
                        decision_missing = not bool(decision_value)
                        if decision_missing:
                            decision_value = "Unknown"

                        formatted_item = {
                            "id": item.get("patent_number"),
                            "meta": {
                                "ipc": item.get("ipc_category") or inferred_ipc,
                                "claim_count": len(raw_claims) if isinstance(raw_claims, list) else 1,
                                "round_type": "rebuttal" if next_oa_body else "examination",
                                "reference_missing": ref_missing,
                                "decision_missing": decision_missing,
                            },
                            "input": {
                                "claims": claims_text,
                                "oracle_reference_summaries": ref_text,
                                "rebuttal": cur_remarks if cur_remarks is not None else "",
                                "office_action": input_oa,
                                "next_office_action": next_oa_body,
                            },
                            "ground_truth": {
                                "oa_text": oa_body,
                                "rebuttal_text": cur_remarks if cur_remarks is not None else "",
                                "next_oa_text": next_oa_body,
                                "decision": decision_value,
                            },
                        }
                        outfile.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")
                        count += 1

    print(f"? ?????????? {count} ??????")
    print(f"?? ????: {SAVE_PATH}")


if __name__ == "__main__":
    align_data()

