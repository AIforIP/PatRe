# -*- coding: utf-8 -*-
"""
脚本名称: evaluate_all_categories_quality.py
功能：自动识别并检测 PatRe 目录下所有分类文件夹（A, H, G 等）的 JSON 质量。
"""

import os
import json
import shutil
from pathlib import Path

# --- 核心配置 ---
# 指向包含所有 Category 的父目录
BASE_DATASET_DIR = Path(r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset")

# --- 过滤阈值配置 ---
MIN_OA_TEXT_LEN = 300      # OA 文本至少要有 300 字符
MIN_CLAIMS_LEN = 200       # 权利要求文本至少要有 200 字符

def ensure_dir(p):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def check_json_quality(data):
    """ 质量评估逻辑 """
    # 1. 检查初始提交
    init_sub = data.get("initial_submission", {})
    if not init_sub.get("claims") or len(init_sub["claims"]) < MIN_CLAIMS_LEN:
        return False, "Initial_Claims_Missing_or_Too_Short"

    # 2. 检查是否有 Office Actions
    oas = data.get("office_actions", [])
    if not oas:
        return False, "No_Office_Actions_Found"

    # 3. 检查每个 OA 的正文内容
    for i, oa in enumerate(oas):
        text = oa.get("office_action_text", "")
        # 如果不是授权通知书，但文本特别短，说明爬取失败了
        if len(text) < MIN_OA_TEXT_LEN:
            return False, f"OA_{i+1}_Text_Invalid_Short"
    
    return True, "Valid"

def process_single_category(cat_dir):
    """ 处理单个分类文件夹 """
    source_dir = cat_dir / "cleaned_json_output"
    if not source_dir.exists():
        return None # 如果该分类还没跑完清洗脚本，直接跳过

    wrong_dir = cat_dir / "wrong_data"
    ensure_dir(wrong_dir)
    
    all_jsons = list(source_dir.glob("*.json"))
    cat_stats = {"total": len(all_jsons), "valid": 0, "wrong": 0, "reasons": {}}

    print(f"📂 正在扫描分类: {cat_dir.name}")

    for fpath in all_jsons:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            is_valid, reason = check_json_quality(data)
            
            if is_valid:
                cat_stats["valid"] += 1
            else:
                cat_stats["wrong"] += 1
                cat_stats["reasons"][reason] = cat_stats["reasons"].get(reason, 0) + 1
                shutil.move(str(fpath), str(wrong_dir / fpath.name))

        except Exception as e:
            cat_stats["wrong"] += 1
            shutil.move(str(fpath), str(wrong_dir / fpath.name))

    return cat_stats

def main():
    # 查找所有以 PatTimeline 开头的文件夹
    categories = [d for d in BASE_DATASET_DIR.iterdir() if d.is_dir() and d.name.startswith("PatTimeline")]
    
    overall_report = {}

    print(f"🚀 启动全品类质量精筛...")
    print(f"目标父目录: {BASE_DATASET_DIR}\n")

    for cat_dir in categories:
        report = process_single_category(cat_dir)
        if report:
            overall_report[cat_dir.name] = report

    # --- 打印汇总报告 ---
    print("\n" + "项目数据清洗全汇总报告".center(50, "="))
    total_valid_all = 0
    
    for cat_name, stats in overall_report.items():
        print(f"\n【{cat_name}】")
        print(f"  - 处理总数: {stats['total']}")
        print(f"  - 符合要求: {stats['valid']}")
        print(f"  - 剔除数量: {stats['wrong']}")
        total_valid_all += stats['valid']
        if stats['reasons']:
            reasons_str = ", ".join([f"{k}: {v}" for k, v in stats['reasons'].items()])
            print(f"  - 剔除原因: {reasons_str}")
    
    print("\n" + "="*50)
    print(f"🎉 所有品类处理完成！")
    print(f"🔥 全库当前可用黄金样本总数: {total_valid_all}")
    print("="*50)

if __name__ == "__main__":
    main()