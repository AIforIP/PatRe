import os
import json
import sys  # 导入 sys 用于重定向
from collections import Counter

# 路径配置
CLEANED_DATA_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\cleaned_json_output"
# 报告输出路径
REPORT_PATH = os.path.join(os.path.dirname(CLEANED_DATA_DIR), "dataset_analysis_report.txt")

def analyze_dataset():
    total_patents = 0
    oa_counts = []
    rejection_reasons = Counter()
    outcome_stats = Counter()
    oa_type_distribution = Counter()
    ref_cited_counts = []

    json_files = [f for f in os.listdir(CLEANED_DATA_DIR) if f.endswith('.json')]
    
    for filename in json_files:
        total_patents += 1
        with open(os.path.join(CLEANED_DATA_DIR, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
            oas = data.get("office_actions", [])
            oa_counts.append(len(oas))
            
            if not oas:
                outcome_stats["No OA Data"] += 1
                continue
            
            last_oa = oas[-1]
            last_type = last_oa.get("oa_type", "Unknown")
            
            if last_type == "Notice of Allowance":
                outcome_stats["Allowed (Success)"] += 1
            elif "Rejection" in last_type:
                outcome_stats["Rejected/Abandoned"] += 1
            else:
                outcome_stats[f"Other ({last_type})"] += 1

            for oa in oas:
                oa_type_distribution[oa.get("oa_type")] += 1
                for r_type in oa.get("rejection_types", []):
                    rejection_reasons[r_type] += 1
                ref_cited_counts.append(len(oa.get("references_cited", [])))

    # --- 记录逻辑开始 ---
    # 使用 tee 类的逻辑：同时输出到控制台和文件
    with open(REPORT_PATH, 'w', encoding='utf-8') as report_file:
        def log(content):
            print(content)  # 打印到控制台
            report_file.write(content + '\n')  # 写入文件

        log("="*40)
        log(f"📊 专利数据集统计报告")
        log(f"生成时间: {os.path.basename(os.path.dirname(CLEANED_DATA_DIR))}")
        log("="*40)
        log(f"1. 规模统计:")
        log(f"   - 总专利数: {total_patents}")
        log(f"   - 总 Office Actions 数: {sum(oa_counts)}")
        avg_oa = sum(oa_counts)/total_patents if total_patents else 0
        log(f"   - 平均每案 OA 轮数: {avg_oa:.2f}")
        
        log(f"\n2. 最终结局 (Outcome):")
        for outcome, count in outcome_stats.items():
            percent = (count / total_patents) * 100 if total_patents else 0
            log(f"   - {outcome}: {count} ({percent:.1f}%)")

        log(f"\n3. 拒绝条款分布 (所有轮次累加):")
        for r_type, count in rejection_reasons.most_common():
            log(f"   - {r_type}: {count} 次")

        log(f"\n4. 引文情况 (References):")
        if ref_cited_counts:
            avg_ref = sum(ref_cited_counts)/len(ref_cited_counts)
            log(f"   - 平均引文数/每轮OA: {avg_ref:.2f}")
            log(f"   - 单轮最高引文数: {max(ref_cited_counts)}")

        log(f"\n5. OA 类型全分布:")
        for oa_t, count in oa_type_distribution.most_common():
            log(f"   - {oa_t}: {count}")
        log("="*40)
        log(f"🎉 报告已保存在: {REPORT_PATH}")

if __name__ == "__main__":
    analyze_dataset()