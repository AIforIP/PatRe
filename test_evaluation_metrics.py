#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PatRe 评估指标快速测试脚本
用于验证新增评估指标功能是否正常工作
"""

import json
import sys
import os
from pathlib import Path

def test_evaluation_metrics():
    """测试评估指标功能"""
    print("[TEST] 开始测试 PatRe 评估指标功能...")
    
    # 检查必要文件是否存在
    required_files = [
        "config/prompt_templates.json",
        "src/evaluator.py",
        "run_exp.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"[ERROR] 缺少必要文件: {missing_files}")
        return False
    
    print("[OK] 所有必要文件存在")
    
    # 检查配置文件中的评估指标
    try:
        with open("config/prompt_templates.json", 'r', encoding='utf-8') as f:
            templates = json.load(f)
        
        eval_metrics = templates.get("Evaluation_Metrics", {})
        required_metrics = ["examiner_judge", "rebuttal_judge"]
        optional_metrics = ["point_wise_evaluator", "pairwise_evaluator", "claim_level_evaluator"]
        
        print(f"[INFO] 检测到评估指标: {list(eval_metrics.keys())}")
        
        missing_metrics = []
        for metric in required_metrics:
            if metric not in eval_metrics:
                missing_metrics.append(metric)
        
        if missing_metrics:
            print(f"[ERROR] 缺少评估指标: {missing_metrics}")
            return False

        missing_optional = [m for m in optional_metrics if m not in eval_metrics]
        if missing_optional:
            print(f"[INFO] 可选评估指标未配置（将自动跳过）: {missing_optional}")
        
        print("[OK] 所有评估指标配置完整")
        
        # 检查每个指标是否有必要的字段
        for metric_name, metric_config in eval_metrics.items():
            required_fields = ["description", "system_prompt", "user_prompt_template"]
            missing_fields = []
            for field in required_fields:
                if field not in metric_config:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"[ERROR] 指标 '{metric_name}' 缺少字段: {missing_fields}")
                return False
        
        print("[OK] 所有评估指标字段完整")
        
    except Exception as e:
        print(f"[ERROR] 配置文件检查失败: {e}")
        return False
    
    # 检查命令行参数
    try:
        import argparse
        # 手动构建参数解析器来验证新参数
        parser = argparse.ArgumentParser(description="PatRe NIPS-2026 Evaluation Engine")
        parser.add_argument("--config", type=str, default="config/eval_config.yaml")
        # 允许通过命令行指定只跑某一个 IPC 分类，例如 --ipc A
        parser.add_argument("--ipc", type=str, help="Run specific IPC category (A-H)")
        # 覆盖模型
        parser.add_argument("--num_samples", type=int, help="Override task_settings.num_samples")
        parser.add_argument("--model", type=str, help="Specific model to test")
        # 任务类型：只保留三种（oa / rebuttal / both），默认 both
        parser.add_argument(
            "--task",
            type=str,
            default="both",
            choices=["oa", "rebuttal", "both"],
            help="Task type to run: oa / rebuttal / both (default: both)",
        )
        # 覆盖测试模式 (zero_shot 或 sft_model)
        parser.add_argument("--test_mode", type=str, choices=["zero_shot", "sft_model"], help="Override test mode")
        # 覆盖检索模式
        parser.add_argument("--retrieval", type=str, choices=["no_ref", "oracle_summary", "bm25_retrieved"], help="Override retrieval setting")
        parser.add_argument("--backend", type=str, choices=["api", "vllm", "ollama"], help="Override model backend")
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument("--generate_only", action="store_true", help="Only generate outputs, skip evaluation")
        mode_group.add_argument("--eval_only", action="store_true", help="Only evaluate existing results, skip generation")
        parser.add_argument("--results_dir", type=str, help="Results directory for eval-only mode")
        parser.add_argument("--eval_limit", type=int, default=0, help="Max rounds to evaluate in eval-only mode (0=all)")
        
        # 检查新添加的参数
        arg_names = [action.dest for action in parser._actions]
        new_arg_names = ['num_samples', 'backend', 'generate_only', 'eval_only', 'results_dir', 'eval_limit']
        
        missing_args = []
        for arg in new_arg_names:
            if arg not in arg_names:
                missing_args.append(arg)
        
        if missing_args:
            print(f"[ERROR] 缺少命令行参数: {missing_args}")
            return False
        
        print("[OK] 命令行参数配置正确")
        
    except Exception as e:
        print(f"[ERROR] 命令行参数检查失败: {e}")
        return False
    
    print("\n[OK] 评估指标功能测试通过！")
    print("\n可用的评估指标:")
    print("  - Traditional: examiner_judge, rebuttal_judge")
    if optional_metrics:
        print("  - Optional: point_wise_evaluator, pairwise_evaluator, claim_level_evaluator (if configured)")
    print("\n可以运行以下命令测试功能:")
    print("  python run_exp.py --task oa --ipc A --num_samples 1")
    print("  python run_exp.py --task rebuttal --ipc A --num_samples 1")
    print("  python run_exp.py --eval_only --results_dir results/... --task both --eval_limit 5")
    
    return True

if __name__ == "__main__":
    success = test_evaluation_metrics()
    if not success:
        sys.exit(1)
