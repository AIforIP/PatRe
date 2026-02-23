import os
import json
import time
from openai import OpenAI

# ================= 配置区域 =================
# 配置本地 Ollama
client = OpenAI(
    api_key="ollama",  # Ollama 默认密钥
    base_url="http://localhost:11434/v1"  # Ollama 本地 API 地址
)

MODEL_NAME = "phi4-mini"  # 确保你已经通过 ollama pull phi4-mini 下载了模型

DATA_FOLDER = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\cleaned_json_output"
OUTPUT_LOG = "ollama_audit_report.txt"
# ===========================================

def deep_logic_check(oa_item, patent_id):
    """
    利用本地 Ollama 模型进行逻辑核校
    """
    rejection_types = oa_item.get("rejection_types", [])
    cited_refs = [ref.get("name", "") for ref in oa_item.get("references_cited", [])]
    # 截取前 3000 字，本地模型处理超长文本速度会变慢，3000字是平衡点
    oa_text = oa_item.get("office_action_text", "")[:3000]
    remarks = oa_item.get("remarks", "")[:1500]

    prompt = f"""
You are a patent data auditor. Analyze the consistency between JSON fields and the actual text for Patent: {patent_id}

[Input Data]
- Rejection Types in JSON: {rejection_types}
- Cited References in JSON: {cited_refs}
- Office Action Text: {oa_text}
- Applicant Remarks: {remarks}

[Audit Tasks]
1. Statutory Basis: If the text discusses rejections under 102, 103, or 112, are they all listed in the JSON "rejection_types"?
2. Reference Chain: Are the prior art references (e.g., Kobayashi) discussed in the text present in the "references_cited" list?
3. Claim Status: Do the "Remarks" mention any claim cancellations or amendments that aren't reflected in the data logic?

[Output Requirement]
- If everything is correct, reply ONLY with "Consistent".
- If there are errors, list them clearly as "Discrepancy: [Description]".
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Local Ollama Error: {str(e)}"

def main():
    if not os.path.exists(DATA_FOLDER):
        print(f"Path not found: {DATA_FOLDER}")
        return

    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
    print(f"Starting local audit with {MODEL_NAME} on {len(files)} files...\n")
    
    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        for file_name in files:
            file_path = os.path.join(DATA_FOLDER, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except: continue
                
                patent_id = data.get("patent_number", file_name)
                print(f"Auditing: {file_name}")

                for oa in data.get("office_actions", []):
                    # 跳过授权通知等非实质性检查
                    if not oa.get("rejection_types") and "Allowance" in oa.get("oa_type", ""):
                        continue
                    
                    result = deep_logic_check(oa, patent_id)
                    
                    if "Consistent" not in result:
                        report = f"!!! File: {file_name} | OA#{oa.get('oa_number')}\n{result}\n"
                        print(report)
                        log.write(report + "-"*30 + "\n")
            
            # 本地运行不需要长时间 sleep，只需给 CPU/GPU 一点缓冲
            time.sleep(0.1)

    print(f"\nAudit finished. Report saved to: {os.path.abspath(OUTPUT_LOG)}")

if __name__ == "__main__":
    main()