import os
import json

# 你的 JSON 目录路径
TARGET_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\2026-02-16\cleaned_json_output1"

def find_patents_with_refs_except_last():
    qualified_patents = []

    if not os.path.exists(TARGET_DIR):
        print(f"❌ 目录不存在: {TARGET_DIR}")
        return

    json_files = [f for f in os.listdir(TARGET_DIR) if f.endswith('.json')]
    print(f"📊 正在扫描 {len(json_files)} 个 JSON 文件...\n")

    for filename in json_files:
        filepath = os.path.join(TARGET_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            oa_list = data.get("office_actions", [])
            total_oa = len(oa_list)

            # 只有一轮的可能不符合“除了最后一轮都有”的逻辑，这里设定至少有2轮
            if total_oa < 2:
                continue

            # 检查前 N-1 轮
            all_previous_have_refs = True
            for i in range(total_oa - 1):  # 遍历 0 到 N-2 轮
                refs = oa_list[i].get("references_cited", [])
                if not refs: # 如果中间某一轮引文为空
                    all_previous_have_refs = False
                    break
            
            # 检查最后一轮是否为空（根据你的需求，最后一轮通常是空的）
            last_oa_empty = len(oa_list[-1].get("references_cited", [])) == 0

            if all_previous_have_refs:
                status = " (最后一轮也是满的)" if not last_oa_empty else " (最后一轮确实为空)"
                qualified_patents.append({
                    "filename": filename,
                    "patent_number": data.get("patent_number"),
                    "total_oa": total_oa,
                    "info": status
                })

        except Exception as e:
            print(f"⚠️ 读取失败 {filename}: {str(e)}")

    # 打印结果
    if qualified_patents:
        print(f"✅ 找到 {len(qualified_patents)} 个符合条件的专利:")
        print("-" * 60)
        for item in qualified_patents:
            print(f"文件: {item['filename']} | 专利号: {item['patent_number']} | 总OA轮数: {item['total_oa']}{item['info']}")
    else:
        print("❌ 未找到符合条件的专利。")

if __name__ == "__main__":
    find_patents_with_refs_except_last()