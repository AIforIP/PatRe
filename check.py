import os
import json
import time
from openai import OpenAI

# ================= 配置区域 =================
# 1. API 配置 (已按要求更新)
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)
# 指定使用 DeepSeek R1 模型
MODEL_NAME = "tngtech/deepseek-r1t2-chimera:free"

# 2. 路径配置
DATA_FOLDER = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\cleaned_json_output"
OUTPUT_LOG = "oa_check_report.txt"
# ===========================================

def check_oa_consistency(oa_type, oa_text):
    """
    调用 DeepSeek R1 核对文本内容与标签是否一致
    """
    # 专利OA的关键判定通常在前 2000 字
    content_sample = oa_text[:2000].replace('\n', ' ')
    
    prompt = f"""
你是一名资深的专利数据专家。请根据提供的专利 Office Action (OA) 文本片段，判断其对应的 OA 类型标注是否正确。

【标注类型】: {oa_type}
【文本片段】: {content_sample}

【核查逻辑参考】:
- Non-Final Rejection: 包含 "Detailed Action", "Claim Rejections"，但没有 "FINAL" 字样。
- Final Rejection: 必须包含 "This office action is made FINAL" 或 "Notice of Final Rejection"。
- Notice of Allowance: 包含 "NOTICE OF ALLOWANCE" 或 "Reasons for Allowance"。
- Restriction Requirement: 包含 "Requirement for Restriction/Election"。

【输出要求】:
1. 如果标注正确，请只回答：“一致”。
2. 如果标注有误，请回答：“不一致，建议修正为：[正确类型]”，并简述原因。
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # 设低温度以获得稳定的判定结果
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"接口调用失败: {str(e)}"

def main():
    if not os.path.exists(DATA_FOLDER):
        print(f"错误：找不到路径 {DATA_FOLDER}")
        return

    json_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.json')]
    print(f"共找到 {len(json_files)} 个 JSON 文件，开始执行 DeepSeek 逻辑核查...\n")

    with open(OUTPUT_LOG, "w", encoding="utf-8") as log:
        log.write(f"--- OA 类型一致性检查报告 ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n")

        for file_name in json_files:
            file_path = os.path.join(DATA_FOLDER, file_name)
            print(f"正在处理: {file_name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except Exception:
                    print(f"  [跳过] 文件解析失败: {file_name}")
                    continue

                oas = data.get("office_actions", [])
                for oa in oas:
                    oa_num = oa.get("oa_number", "N/A")
                    oa_type = oa.get("oa_type", "Unknown")
                    oa_text = oa.get("office_action_text", "")

                    if not oa_text:
                        continue

                    # 调用模型进行分析
                    result = check_oa_consistency(oa_type, oa_text)
                    
                    # 只有不一致时才在控制台打印亮点，减少刷屏
                    if "不一致" in result:
                        msg = f"【发现潜在错误】\n文件: {file_name}\nOA序号: {oa_num}\n原始标注: {oa_type}\n模型反馈: {result}\n"
                        print(msg)
                        log.write(msg + "-"*30 + "\n")
                    else:
                        log.write(f"OK - 文件: {file_name} | OA#{oa_num} 一致\n")
            
            # OpenRouter 免费模型可能有限速，建议保留小延时
            time.sleep(0.5)

    print(f"\n检查完毕！详细报告已生成至: {os.path.abspath(OUTPUT_LOG)}")

if __name__ == "__main__":
    main()