# -*- coding: utf-8 -*-
import json
import re
import os

# -------------------------- 配置路径 --------------------------
SOURCE_MD = r"D:\study\competition\LLM_based_IP_Review_Research\Project\2024-09-24_Applicant_Arguments_Remarks_Made_in_an_Amendment_17749934_XML_patentpull.md"
TARGET_JSON = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_G_Category\cleaned_json_output\Patent_12189863_App_17749934.json"

def lossless_clean_remarks(text: str) -> str:
    """
    无损清洗字符：仅移除Markdown转义、OCR噪声和排版装饰线，严禁改变文字内容和大小写。
    """
    # 1. 移除转换工具产生的图像/SVG占位符 (针对此文件可能存在的 SVG 信息)
    text = re.sub(r"SVG\s*\{EMBED\s*HERE\}[^ \n]*", "", text)
    
    # 2. 还原 Markdown 转义 (例如把 \. 还原为 . 方便阅读序号)
    text = text.replace("\\.", ".").replace("\\_", "_").replace("\\[", "[").replace("\\]", "]")
    
    # 3. 修正 OCR 可能产生的符号误匹配 (例如把 @ 还原为 §)
    text = text.replace(" @ ", " § ")
    
    lines = text.splitlines()
    clean_lines = []
    
    # 4. 逐行处理布局杂质
    for line in lines:
        stripped = line.strip()
        # 移除纯装饰性的分割线 (如 ======= 或 -------)
        if re.match(r"^[=\-\*_ ]{3,}$", stripped):
            continue
        # 压缩行内由于对齐产生的多余空格 (保留单个空格)
        line = re.sub(r"[ \t]{2,}", " ", line)
        clean_lines.append(line.rstrip())
    
    # 5. 段落重组：合并被硬换行切断的法律陈述
    final_text = ""
    for line in clean_lines:
        if not line:
            if not final_text.endswith("\n\n"):
                final_text += "\n\n"
            continue
        
        if final_text and not final_text.endswith("\n\n"):
            # 如果当前行是小写字母开头，或者是句子的延续
            if re.match(r"^[a-z0-9]", line):
                final_text += " " + line
            else:
                # 如果当前行以大写字母开头，可能是新起的一行，但也可能是上一句的延续
                # 这里采用较保守的逻辑：如果是常见的法律连接词开头则合并
                if re.match(r"^(is|of|the|and|by|which|to)\b", line, re.I):
                    final_text += " " + line
                else:
                    final_text += "\n" + line
        else:
            final_text += line
            
    # 6. 最终清理冗余换行
    res = re.sub(r'\n{3,}', '\n\n', final_text)
    return res.strip()

def update_json_gesture_recognition():
    if not os.path.exists(SOURCE_MD):
        print(f"❌ 找不到源 MD 文件: {SOURCE_MD}")
        return

    # 读取原文
    with open(SOURCE_MD, "r", encoding="utf-8", errors="ignore") as f:
        md_content = f.read()

    # 定位 REMARKS 章节 (通常在文件后半段)
    match = re.search(r"REMARKS\b(.*?)$", md_content, re.S | re.I)
    raw_remarks = match.group(1) if match else md_content

    # 执行清洗
    cleaned = lossless_clean_remarks(raw_remarks)

    # 载入并更新 JSON
    if not os.path.exists(TARGET_JSON):
        print(f"❌ 找不到目标 JSON 文件: {TARGET_JSON}")
        return

    with open(TARGET_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 精准定位 oa_number 为 1 的条目
    found = False
    for oa in data.get("office_actions", []):
        if str(oa.get("oa_number")) == "1":
            oa["remarks"] = cleaned
            found = True
            print(f"🎯 成功匹配 OA #1 (Gesture Recognition)，正在注入...")
            break
    
    if not found:
        print("⚠️ 未在 JSON 中找到 oa_number: 1，请确认 JSON 内容。")
        return

    # 写回 JSON 文件
    with open(TARGET_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ 处理完成！")
    print(f"📝 注入长度: {len(cleaned)} 字符。")
    print("-" * 30)
    print(f"清洗后的内容预览:\n{cleaned[:200]}...")

if __name__ == "__main__":
    update_json_gesture_recognition()