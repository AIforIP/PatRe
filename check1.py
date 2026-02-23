import os
import json
import re

# ================= 配置区域 =================
DATA_FOLDER = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\cleaned_json_output"
# ===========================================

def local_audit(oa_item):
    errors = []
    rejection_types = str(oa_item.get("rejection_types", []))
    cited_refs = str(oa_item.get("references_cited", []))
    oa_text = oa_item.get("office_action_text", "")

    # 1. 检查条款漏标 (102, 103, 112)
    patterns = {
        "102": r"35\s+U\.S\.C\.?\s+102|anticipat",
        "103": r"35\s+U\.S\.C\.?\s+103|obvious",
        "112": r"35\s+U\.S\.C\.?\s+112|indefinite|written description|enablement"
    }
    
    for code, regex in patterns.items():
        if re.search(regex, oa_text, re.IGNORECASE):
            if code not in rejection_types:
                errors.append(f"【条款漏标】正文疑似提到 {code} 相关内容，但 JSON rejection_types 中缺失。")

    # 2. 检查文献漏标
    # 匹配专利格式如 US 2003/0080939 或 Patent No. 1234567
    ref_pattern = r"(?:US|No\.|Patent)\s*([\d\/\,]{7,15})"
    found_refs = re.findall(ref_pattern, oa_text)
    for ref in list(set(found_refs)):
        clean_ref = ref.replace("/", "").replace(",", "")
        # 如果正文提到的专利号不在引文列表里（简单模糊匹配）
        if clean_ref not in cited_refs.replace("/", "").replace(",", ""):
            if len(clean_ref) >= 7: # 排除掉短数字干扰
                errors.append(f"【引文漏标】正文提到专利号 {ref}，但未在 references_cited 列表中找到。")

    # 3. 检查状态逻辑
    if "Notice of Allowance" in oa_item.get("oa_type", ""):
        if "rejection" in oa_text.lower() and "rejection_types" in oa_item and len(oa_item["rejection_types"]) > 0:
            errors.append("【逻辑错误】标注为 Notice of Allowance，但正文中仍保留驳回条款。")

    return errors

def main():
    if not os.path.exists(DATA_FOLDER):
        print(f"找不到文件夹: {DATA_FOLDER}")
        return

    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
    print(f"开始本地审计 {len(files)} 个文件...\n")
    
    report_count = 0
    for file_name in files:
        file_path = os.path.join(DATA_FOLDER, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except: continue
                
            for oa in data.get("office_actions", []):
                oa_num = oa.get("oa_number", "?")
                discrepancies = local_audit(oa)
                
                if discrepancies:
                    report_count += 1
                    print(f"!!! [{file_name}] OA#{oa_num}")
                    for d in discrepancies:
                        print(f"    - {d}")
                    print("-" * 50)

    print(f"审计完成，共发现 {report_count} 处潜在数据问题。")

if __name__ == "__main__":
    main()