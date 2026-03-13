import json

# 指向你生成的 jsonl 路径
SAVE_PATH = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\processed\test_set_new.jsonl"

def check_category_a():
    total_count = 0
    a_category_count = 0
    a_patent_ids = set()
    other_categories = {}

    with open(SAVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            total_count += 1
            ipc = item.get("meta", {}).get("ipc", "Unknown")
            
            if ipc == "A":
                a_category_count += 1
                a_patent_ids.add(item["id"])
            else:
                other_categories[ipc] = other_categories.get(ipc, 0) + 1

    print("-" * 30)
    print(f"📊 总体统计结果:")
    print(f"总样本数: {total_count}")
    print(f"A 类别样本数: {a_category_count}")
    print(f"A 类别唯一专利数: {len(a_patent_ids)}")
    print("-" * 30)
    print(f"📁 其他类别分布: {other_categories}")
    print("-" * 30)
    
    if len(a_patent_ids) > 0:
        print(f"✅ 前 5 个 A 类别专利 ID 示例: {list(a_patent_ids)[:5]}")
    else:
        print("❌ 警告：未在文件中发现 A 类别专利！")

if __name__ == "__main__":
    check_category_a()