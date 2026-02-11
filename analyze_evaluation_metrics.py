#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PatRe 评估指标分析脚本
用于分析和可视化各种评估指标的结果
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import argparse


def load_results(result_file_path):
    """加载评估结果文件"""
    with open(result_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_point_wise_scores(results):
    """分析 point-wise 评分"""
    print("\n📊 Point-wise 评估分析:")
    print("-" * 50)
    
    scores = []
    for result in results.get('metrics', []):
        if 'point_wise_scores' in result:
            scores.append(result['point_wise_scores'])
    
    if not scores:
        print("未找到 point-wise 评分数据")
        return
    
    # 计算各项指标的平均分
    score_keys = set()
    for score_dict in scores:
        score_keys.update(score_dict.keys())
    
    print(f"评分维度: {list(score_keys)}")
    
    for key in score_keys:
        if key != 'error':
            values = [score_dict[key] for score_dict in scores if key in score_dict and isinstance(score_dict[key], (int, float))]
            if values:
                avg_score = sum(values) / len(values)
                print(f"{key}: 平均分 = {avg_score:.2f}")


def analyze_pairwise_comparison(results):
    """分析 pairwise 比较结果"""
    print("\n⚖️ Pairwise 比较分析:")
    print("-" * 50)
    
    comparisons = []
    for result in results.get('metrics', []):
        if 'pairwise_winner' in result:
            comparisons.append(result['pairwise_winner'])
    
    if not comparisons:
        print("未找到 pairwise 比较数据")
        return
    
    # 统计获胜情况
    from collections import Counter
    counter = Counter(comparisons)
    print(f"获胜统计: {dict(counter)}")
    
    # 计算胜率
    total = len(comparisons)
    for winner, count in counter.items():
        win_rate = count / total * 100
        print(f"{winner}: 胜率 = {win_rate:.1f}% ({count}/{total})")


def analyze_claim_level_scores(results):
    """分析 claim-level 评分"""
    print("\n🔍 Claim-level 评估分析:")
    print("-" * 50)
    
    scores = []
    for result in results.get('metrics', []):
        if 'claim_level_scores' in result:
            scores.append(result['claim_level_scores'])
    
    if not scores:
        print("未找到 claim-level 评分数据")
        return
    
    # 计算各项指标的平均分
    score_keys = set()
    for score_dict in scores:
        score_keys.update(score_dict.keys())
    
    print(f"评分维度: {list(score_keys)}")
    
    for key in score_keys:
        if key != 'error':
            values = [score_dict[key] for score_dict in scores if key in score_dict and isinstance(score_dict[key], (int, float))]
            if values:
                avg_score = sum(values) / len(values)
                print(f"{key}: 平均分 = {avg_score:.2f}")


def generate_detailed_report(results):
    """生成详细的评估报告"""
    print("\n📋 详细评估报告:")
    print("=" * 60)
    
    # 基础指标
    print(f"实验ID: {results.get('experiment_id', 'N/A')}")
    print(f"时间戳: {results.get('timestamp', 'N/A')}")
    print(f"总样本数: {len(results.get('results', []))}")
    
    # 汇总指标
    agg_metrics = results.get('aggregate_metrics', {})
    print(f"\n📈 汇总指标:")
    for key, value in agg_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 运行各专项分析
    analyze_point_wise_scores(results)
    analyze_pairwise_comparison(results)
    analyze_claim_level_scores(results)


def visualize_results(results, output_dir):
    """可视化评估结果"""
    print(f"\n🖼️ 生成可视化图表...")
    
    # 创建输出目录
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 汇总指标柱状图
    agg_metrics = results.get('aggregate_metrics', {})
    numeric_metrics = {k: v for k, v in agg_metrics.items() if isinstance(v, (int, float))}
    
    if numeric_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        keys = list(numeric_metrics.keys())
        values = list(numeric_metrics.values())
        
        bars = ax.bar(keys, values)
        ax.set_title('汇总评估指标', fontsize=14)
        ax.set_ylabel('分数')
        
        # 在柱子上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2f}',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'aggregate_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  - 汇总指标图已保存至: {os.path.join(viz_dir, 'aggregate_metrics.png')}")
    
    # 2. 分析 point-wise 评分
    point_scores = []
    for result in results.get('metrics', []):
        if 'point_wise_scores' in result:
            point_scores.append(result['point_wise_scores'])
    
    if point_scores:
        # 收集所有评分维度
        all_dims = set()
        for score_dict in point_scores:
            all_dims.update([k for k, v in score_dict.items() if isinstance(v, (int, float)) and k != 'error'])
        
        if all_dims:
            # 创建评分矩阵
            df_data = []
            for i, score_dict in enumerate(point_scores):
                row = {'sample_id': i}
                for dim in all_dims:
                    row[dim] = score_dict.get(dim, 0)
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            
            # 绘制各维度评分分布
            fig, axes = plt.subplots(1, min(3, len(all_dims)), figsize=(15, 5))
            if len(all_dims) == 1:
                axes = [axes]
            elif len(all_dims) == 2:
                axes = axes[:2]
            
            for idx, dim in enumerate(list(all_dims)[:3]):
                if idx < len(axes):
                    axes[idx].hist(df[dim], bins=20, alpha=0.7)
                    axes[idx].set_title(f'{dim} 分布')
                    axes[idx].set_xlabel('分数')
                    axes[idx].set_ylabel('频次')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'point_wise_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  - Point-wise 分布图已保存至: {os.path.join(viz_dir, 'point_wise_distribution.png')}")


def main():
    parser = argparse.ArgumentParser(description="PatRe 评估指标分析工具")
    parser.add_argument("--result-file", type=str, required=True, 
                       help="评估结果文件路径 (JSON格式)")
    parser.add_argument("--output-dir", type=str, default="./analysis_output",
                       help="输出目录路径")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载结果
    print(f"📂 正在加载评估结果: {args.result_file}")
    results = load_results(args.result_file)
    
    # 生成详细报告
    generate_detailed_report(results)
    
    # 可视化结果
    visualize_results(results, args.output_dir)
    
    print(f"\n✅ 分析完成！结果已保存至: {args.output_dir}")


if __name__ == "__main__":
    main()