# -*- coding: utf-8 -*-
import os
import yaml
import json
import argparse
from datetime import datetime
# 注意这里：为了模块化，具体的逻辑被封装在 src.evaluator
from src.evaluator import PatentEvaluator
from src.utils import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="PatRe NIPS-2026 Evaluation Engine")
    parser.add_argument("--config", type=str, default="config/eval_config.yaml")
    # 允许通过命令行指定只跑某一个 IPC 分类，例如 --ipc A
    parser.add_argument("--ipc", type=str, help="Run specific IPC category (A-H)")
    # 覆盖模型
    parser.add_argument("--model", type=str, help="Specific model to test")
    # 运行指定任务 (例如: v1_initial_examination / v1_high_fidelity_rebuttal)
    parser.add_argument("--task", type=str, help="Specific task key to run")
    # 一次性顺序运行两类任务 (OA + Rebuttal)
    parser.add_argument("--run_both", action="store_true", help="Run Office Action and Rebuttal tasks sequentially")
    # 一次性顺序运行多轮审查流程
    parser.add_argument("--run_chain3", action="store_true", help="Run OA → Rebuttal → Examiner Response sequence")
    parser.add_argument("--run_multiround", action="store_true", help="Run multi-round OA-Rebuttal cycles based on data structure")
    # 覆盖测试模式 (zero_shot 或 sft_model)
    parser.add_argument("--test_mode", type=str, choices=["zero_shot", "sft_model"], help="Override test mode")
    # 覆盖检索模式
    parser.add_argument("--retrieval", type=str, choices=["no_ref", "oracle_summary", "bm25_retrieved"], help="Override retrieval setting")
    # 新增评估指标选项
    parser.add_argument("--run_point_wise", action="store_true", help="Run point-wise evaluation against ground truth")
    parser.add_argument("--run_pairwise", action="store_true", help="Run pairwise comparison between models")
    parser.add_argument("--run_claim_level", action="store_true", help="Run claim-level evaluation")
    parser.add_argument("--run_claim_rediscovery", action="store_true", help="Run claim-centric rediscovery evaluation")
    parser.add_argument("--run_extra_metrics", action="store_true", help="Run additional rule-based metrics")
    parser.add_argument("--backend", type=str, choices=["api", "vllm", "ollama"], help="Override model backend")
    return parser.parse_args()

def analyze_dataset_rounds(config):
    """分析数据集中的实际轮数分布"""
    test_set_path = config['paths']['test_set']
    if not os.path.exists(test_set_path):
        return 3  # 默认3轮
    
    patent_rounds = {}
    with open(test_set_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            patent_id = item['id']
            if patent_id not in patent_rounds:
                patent_rounds[patent_id] = 0
            patent_rounds[patent_id] += 1
    
    if not patent_rounds:
        return 3
    
    max_rounds = max(patent_rounds.values())
    print(f"📊 数据集分析: 最大轮数为 {max_rounds} 轮")
    print(f"   专利轮数分布: {sorted(set(patent_rounds.values()))}")
    return max_rounds

def generate_multiround_sequence_from_data(max_rounds):
    """根据实际数据轮数生成多轮审查序列"""
    sequence = ["v1_initial_examination"]  # 开始总是初始审查
    
    # 根据实际轮数生成序列
    # 轮数1: initial_examination
    # 轮数2: initial_examination -> rebuttal
    # 轮数3: initial_examination -> rebuttal -> response
    # 轮数4+: 继续交替添加 rebuttal 和 response
    
    for i in range(1, max_rounds):
        if i % 2 == 1:  # 奇数轮次（除了第1轮）是反驳
            sequence.append("v1_high_fidelity_rebuttal")
        else:  # 偶数轮次是审查员回复
            sequence.append("v1_response_to_rebuttal")
    
    return sequence

def _save_patent_organized_results(raw_results, eval_report, config, exp_dir):
    """按专利组织结果文件的保存方法，支持追加各轮次结果"""
    # 将原始结果和评估结果合并
    oa_schema = [
        "disposition_accuracy",
        "statute_precision",
        "citation_accuracy",
        "citation_list_match",
        "bleu",
        "rouge_l",
        "bert_score",
        "embed_cos",
        "formatting_rule",
        "accuracy",
        "clarity",
        "constructiveness",
        "completeness",
        "quality",
    ]
    rebuttal_schema = [
        "point_wise_coverage",
        "bleu",
        "rouge_l",
        "bert_score",
        "embed_cos",
        "formatting_rule",
        "accuracy",
        "clarity",
        "constructiveness",
        "completeness",
        "quality",
        "disposition_accuracy",
        "statute_precision",
        "citation_accuracy",
        "citation_list_match",
    ]
    for raw_result, eval_result in zip(raw_results, eval_report):
        patent_id = raw_result['patent_id']
        patent_dir = os.path.join(exp_dir, f"patent_{patent_id}")
        os.makedirs(patent_dir, exist_ok=True)
        
        round_info = raw_result['round_info']
        round_num = round_info['round_number']
        task_key = round_info['task']
        
        # 简化任务标签以用于文件夹命名
        if task_key == "v1_initial_examination":
            task_label = "initial_examination"
        elif task_key == "v1_response_to_rebuttal":
            task_label = "response"
        elif task_key == "v1_high_fidelity_rebuttal":
            task_label = "rebuttal"
        else:
            task_label = "other"
            
        round_dir = os.path.join(patent_dir, f"round_{round_num}_{task_label}")
        os.makedirs(round_dir, exist_ok=True)
        
        # 1. input.json (AI 实际输入)
        exam_input_path = os.path.join(round_dir, "input.json")
        with open(exam_input_path, 'w', encoding='utf-8') as f:
            ai_input_data = raw_result['ai_exam']['input'].copy()
            # Persist OA for traceability (not used by the model directly).
            oa_text = raw_result.get('evaluation_context', {}).get('full_context', {}).get('office_action', "")
            if oa_text:
                ai_input_data["office_action"] = oa_text
            json.dump(ai_input_data, f, indent=2, ensure_ascii=False)
        # 2. ai_output.txt (AI生成的答案)
        exam_output_path = os.path.join(round_dir, "ai_output.txt")
        with open(exam_output_path, 'w', encoding='utf-8') as f:
            f.write(raw_result['ai_exam']['output'])
        
        # 3. context.json (评估上下文)
        eval_context_path = os.path.join(round_dir, "context.json")
        with open(eval_context_path, 'w', encoding='utf-8') as f:
            context_data = {
                "ai_input": raw_result['ai_exam']['input'],
                "ai_output": raw_result['ai_exam']['output'],
                "ground_truth": raw_result['evaluation_context'].get("ground_truth", ""),
                "restricted_info": raw_result['ai_exam'].get('restricted_info', []),
                "office_action": raw_result.get('evaluation_context', {}).get('full_context', {}).get('office_action', "")
            }
            json.dump(context_data, f, indent=2, ensure_ascii=False)
        
        # 4. scores.json (评分结果)
        scores_path = os.path.join(round_dir, "scores.json")
        with open(scores_path, 'w', encoding='utf-8') as f:
            score_data = {
                "total_score": eval_result.get('total_score', 0),
                "scores": eval_result.get('scores', {}),
                "justification": eval_result.get('justification', ""),
                "justification_details": eval_result.get('justification_details', {}),
                "raw_judge_output": eval_result.get('raw_judge_output', ""),
                "rediscovery_details": eval_result.get('rediscovery_details', {}),
                "metric_details": eval_result.get('metric_details', {})
            }
            json.dump(score_data, f, indent=2, ensure_ascii=False)

        summary_scores = {}
        schema = rebuttal_schema if task_key == "v1_high_fidelity_rebuttal" else oa_schema
        name_map_oa = {
            "disposition_accuracy": "Dispositional Accuracy",
            "statute_precision": "Statute Precision",
            "citation_accuracy": "Citation Accuracy",
            "citation_list_match": "Citation List Match",
            "bleu": "BLEU",
            "rouge_l": "ROUGE-L",
            "bert_score": "BERTScore",
            "embed_cos": "EmbedCos",
            "formatting_rule": "formatting_",
            "accuracy": "Accuracy",
            "clarity": "Clarity",
            "constructiveness": "Constructiveness",
            "completeness": "Completeness",
            "quality": "Quality",
        }
        name_map_rebuttal = {
            "point_wise_coverage": "Point-wise Coverage",
            "bleu": "BLEU",
            "rouge_l": "ROUGE-L",
            "bert_score": "BERTScore",
            "embed_cos": "EmbedCos",
            "formatting_rule": "formatting_",
            "accuracy": "Accuracy",
            "clarity": "Clarity",
            "constructiveness": "Constructiveness",
            "completeness": "Completeness",
            "quality": "Quality",
            "disposition_accuracy": "Dispositional Accuracy",
            "statute_precision": "Statute Precision",
            "citation_accuracy": "Citation Accuracy",
            "citation_list_match": "Citation List Match",
        }
        name_map = name_map_rebuttal if task_key == "v1_high_fidelity_rebuttal" else name_map_oa
        src_scores = eval_result.get("scores", {})
        percent_metrics = {
            "disposition_accuracy": 100.0,
            "statute_precision": 100.0,
            "citation_accuracy": 10.0,
            "point_wise_coverage": 10.0,
        }
        for key in schema:
            if key == "bert_score":
                val = src_scores.get("bert_score", src_scores.get("bert_score_f1", None))
            elif key == "embed_cos":
                val = src_scores.get("embed_cos", src_scores.get("embed_cosine", None))
            elif key == "citation_accuracy":
                val = src_scores.get("citation_accuracy", src_scores.get("reference_citation_accuracy", None))
            else:
                val = src_scores.get(key, None)

            out_key = name_map.get(key, key)
            if key in percent_metrics and isinstance(val, (int, float)):
                summary_scores[out_key] = f"{val * percent_metrics[key]:.2f}%"
            else:
                summary_scores[out_key] = val
        summary_path = os.path.join(round_dir, "scores_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "task_type": task_key,
                    "total_score": eval_result.get('total_score', 0),
                    "scores": summary_scores
                },
                f,
                indent=2,
                ensure_ascii=False
            )
    # 专利级别的汇总将在 _generate_summaries 函数中处理
    pass

def _calculate_patent_average_scores(results):
    """计算单个专利的平均分数"""
    if not results: return {}
    score_sums = {}
    score_counts = {}
    for res in results:
        scores = res.get('scores', {})
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                score_sums[k] = score_sums.get(k, 0) + v
                score_counts[k] = score_counts.get(k, 0) + 1
    return {k: score_sums[k] / score_counts[k] for k in score_sums} if score_counts else {}

def _calculate_overall_metrics(reports):
    """计算整体评估指标"""
    if not reports: return {}
    score_sums = {}
    score_counts = {}
    for res in reports:
        scores = res.get('scores', {})
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                score_sums[k] = score_sums.get(k, 0) + v
                score_counts[k] = score_counts.get(k, 0) + 1
    return {k: score_sums[k] / score_counts[k] for k in score_sums} if score_counts else {}

def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def _calculate_disposition_micro_metrics(reports):
    """Compute micro-averaged disposition metrics across all samples."""
    tp = fp = fn = 0
    correct = 0
    total = 0
    for res in reports:
        md = res.get("metric_details", {}) if isinstance(res, dict) else {}
        pred = md.get("disposition_pred")
        gt = md.get("disposition_gt")
        if not pred or not gt or pred == "unknown" or gt == "unknown":
            continue
        total += 1
        if pred == gt:
            correct += 1
        pred_pos = pred in ("final_rejection", "non_final_rejection", "rejection")
        gt_pos = gt in ("final_rejection", "non_final_rejection", "rejection")
        if pred_pos and gt_pos:
            tp += 1
        elif pred_pos and not gt_pos:
            fp += 1
        elif (not pred_pos) and gt_pos:
            fn += 1

    if total == 0:
        return {}
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "disposition_accuracy_overall": round(correct / total, 4),
        "disposition_micro_precision": round(precision, 4),
        "disposition_micro_recall": round(recall, 4),
        "disposition_micro_f1": round(f1, 4),
        "disposition_support": total,
        "disposition_tp": tp,
        "disposition_fp": fp,
        "disposition_fn": fn,
    }

def _generate_summaries(exp_dir, grouped_data, all_results_map, config):
    """生成实验汇总文件"""
    # 1. 每个专利的 summary.json
    for pid, results in all_results_map.items():
        patent_dir = os.path.join(exp_dir, f"patent_{pid}")
        os.makedirs(patent_dir, exist_ok=True)
        
        patent_summary = {
            "total_rounds_executed": len(results),
            "average_scores": _calculate_patent_average_scores(results)
        }
        summary_path = os.path.join(patent_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(patent_summary, f, indent=2, ensure_ascii=False)
    
    # 2. 整体实验概览 experiment_overview.json
    all_reports = [item for sublist in all_results_map.values() for item in sublist]
    disposition_micro = _calculate_disposition_micro_metrics(all_reports)
    experiment_overview = {
        "config_snapshot": config,
        "timestamp": datetime.now().isoformat(),
        "total_patents": len(grouped_data),
        "total_rounds_run": sum(len(results) for results in all_results_map.values()),
        "overall_metrics": _calculate_overall_metrics(all_reports),
        "overall_disposition": disposition_micro
    }
    overview_path = os.path.join(exp_dir, "experiment_overview.json")
    with open(overview_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_overview, f, indent=4, ensure_ascii=False)

def main():
    args = parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    models_to_run = [args.model] if args.model else config['models']['backbone_list']
    test_set_path = config['paths']['test_set']
    ipc_filter = args.ipc.strip().upper() if args.ipc else None
    if ipc_filter:
        config.setdefault('task_settings', {})['ipc_filter'] = ipc_filter
    
    # 处理命令行参数覆盖
    if args.test_mode:
        config['active_settings']['test_mode'] = args.test_mode
    if args.retrieval:
        config['active_settings']['retrieval_setting'] = args.retrieval
    if args.backend:
        config['models']['backend'] = args.backend
    
    auto_extra = bool(config.get('task_settings', {}).get('auto_run_extra_metrics', False))
    auto_rediscovery = bool(config.get('task_settings', {}).get('auto_run_claim_rediscovery', False))

    # 获取命名配置
    naming_cfg = config.get('experiment_naming', {})
    use_academic_format = naming_cfg.get('use_academic_format', True)
    timestamp_format = naming_cfg.get('timestamp_format', "%Y%m%d_%H%M")
    include_model_name = naming_cfg.get('include_model_name', True)

    if args.run_multiround:
        # 加载并分组数据
        with open(test_set_path, 'r', encoding='utf-8') as f:
            all_data = []
            for line in f:
                if not line or not line.strip():
                    continue
                s = line.lstrip()
                if s.startswith("//"):
                    continue
                all_data.append(json.loads(line))
        if ipc_filter:
            all_data = [
                item for item in all_data
                if str(item.get("meta", {}).get("ipc", "")).upper() == ipc_filter
            ]
        
        # 遵循 config 中的样本限制
        limit = config['task_settings'].get('num_samples', 9999)
        all_data = all_data[:limit]
        
        grouped_data = {}
        for item in all_data:
            pid = item['id']
            if pid not in grouped_data: grouped_data[pid] = []
            grouped_data[pid].append(item)
            
        max_lines = max(len(v) for v in grouped_data.values()) if grouped_data else 0

        for model_idx, model_name in enumerate(models_to_run, start=1):
            config['models']['active_model'] = model_name
            timestamp = datetime.now().strftime(timestamp_format)
            if use_academic_format:
                # 学术规范命名：模型_多轮专利审查_时间戳
                multiround_exp_dir = os.path.join(config['paths']['results'], f"{model_name}_multiround_patent_review_{timestamp}")
            else:
                # 原始命名方式
                multiround_exp_dir = os.path.join(config['paths']['results'], f"multiround_{model_name}_{timestamp}")
            if not include_model_name:
                if use_academic_format:
                    multiround_exp_dir = os.path.join(config['paths']['results'], f"multiround_patent_review_{timestamp}")
                else:
                    multiround_exp_dir = os.path.join(config['paths']['results'], f"multiround_{timestamp}")
                if len(models_to_run) > 1:
                    multiround_exp_dir = f"{multiround_exp_dir}_run{model_idx}"
            os.makedirs(multiround_exp_dir, exist_ok=True)
            
            logger = setup_logger(multiround_exp_dir)
            logger.info(f"🚀 Starting multi-round for model: {model_name}")

            all_results_map = {} # pid -> [eval_result, ...]

            for line_idx in range(max_lines):
                # 1. Examiner Round
                task_oa = "v1_initial_examination" if line_idx == 0 else "v1_response_to_rebuttal"
                config['task_settings']['current_task'] = task_oa
                batch_oa = [lines[line_idx] for pid, lines in grouped_data.items() if line_idx < len(lines)]
                
                if batch_oa:
                    logger.info(f"📌 Round {line_idx*2+1}: {task_oa} for {len(batch_oa)} patents...")
                    engine = PatentEvaluator(config, logger)
                    engine.dataset = batch_oa
                    raw = engine.run_batch()
                    for r in raw: r['round_info']['round_number'] = line_idx*2+1
                    report = engine.run_scoring(raw)
                    if args.run_claim_rediscovery or auto_rediscovery:
                        report = engine.run_claim_rediscovery_evaluation(report)
                    if args.run_extra_metrics or auto_extra:
                        report = engine.run_additional_metrics(report)
                    _save_patent_organized_results(raw, report, config, multiround_exp_dir)
                    
                    for r, e in zip(raw, report):
                        pid = r['patent_id']
                        if pid not in all_results_map: all_results_map[pid] = []
                        all_results_map[pid].append(e)

                # 2. Applicant Round (Rebuttal)
                task_rebuttal = "v1_high_fidelity_rebuttal"
                config['task_settings']['current_task'] = task_rebuttal
                batch_rebuttal = []
                for pid, lines in grouped_data.items():
                    if line_idx < len(lines):
                        item = lines[line_idx]
                        rebuttal_text = (
                            item.get("input", {}).get("rebuttal")
                            or item.get("ground_truth", {}).get("rebuttal_text")
                            or ""
                        )
                        if isinstance(rebuttal_text, str):
                            rebuttal_text = rebuttal_text.strip()
                        if rebuttal_text and str(rebuttal_text).upper() != "N/A":
                            batch_rebuttal.append(item)
                
                if batch_rebuttal:
                    logger.info(f"📌 Round {line_idx*2+2}: {task_rebuttal} for {len(batch_rebuttal)} patents...")
                    engine = PatentEvaluator(config, logger)
                    engine.dataset = batch_rebuttal
                    raw = engine.run_batch()
                    for r in raw: r['round_info']['round_number'] = line_idx*2+2
                    report = engine.run_scoring(raw)
                    if args.run_claim_rediscovery or auto_rediscovery:
                        report = engine.run_claim_rediscovery_evaluation(report)
                    if args.run_extra_metrics or auto_extra:
                        report = engine.run_additional_metrics(report)
                    _save_patent_organized_results(raw, report, config, multiround_exp_dir)
                    
                    for r, e in zip(raw, report):
                        pid = r['patent_id']
                        if pid not in all_results_map: all_results_map[pid] = []
                        all_results_map[pid].append(e)

            # 生成汇总
            _generate_summaries(multiround_exp_dir, grouped_data, all_results_map, config)
            logger.info(f"✅ Multiround finished. Results: {multiround_exp_dir}")

    else:
        # 单任务模式
        tasks_to_run = [args.task] if args.task else [config['task_settings']['current_task']]
        for task_key in tasks_to_run:
            config['task_settings']['current_task'] = task_key
            for model_idx, model_name in enumerate(models_to_run, start=1):
                config['models']['active_model'] = model_name
                timestamp = datetime.now().strftime(timestamp_format)
                if use_academic_format:
                    # 学术规范命名：模型_任务缩写_检索方式_时间戳
                    retrieval_abbr = {
                        "no_ref": "noctx",
                        "oracle_summary": "oracle",
                        "bm25_retrieved": "bm25"
                    }.get(config['active_settings'].get('retrieval_setting', 'oracle_summary'), 'unknown')
                    
                    task_abbr = {
                        "v1_initial_examination": "oa_gen",
                        "v1_high_fidelity_rebuttal": "rebuttal_gen",
                        "v1_response_to_rebuttal": "oa_response"
                    }.get(task_key, task_key)
                    
                    exp_dir = os.path.join(config['paths']['results'], f"{model_name}_{task_abbr}_{retrieval_abbr}_{timestamp}")
                else:
                    # 原始命名方式
                    exp_dir = os.path.join(config['paths']['results'], f"{model_name}_{task_key}_{timestamp}")
                if not include_model_name:
                    if use_academic_format:
                        exp_dir = os.path.join(config['paths']['results'], f"{task_abbr}_{retrieval_abbr}_{timestamp}")
                    else:
                        exp_dir = os.path.join(config['paths']['results'], f"{task_key}_{timestamp}")
                    if len(models_to_run) > 1:
                        exp_dir = f"{exp_dir}_run{model_idx}"
                os.makedirs(exp_dir, exist_ok=True)
                logger = setup_logger(exp_dir)
                try:
                    engine = PatentEvaluator(config, logger)
                    raw = engine.run_batch()
                    report = engine.run_scoring(raw)
                    if args.run_claim_rediscovery or auto_rediscovery:
                        report = engine.run_claim_rediscovery_evaluation(report)
                    if args.run_extra_metrics or auto_extra:
                        report = engine.run_additional_metrics(report)
                    _save_patent_organized_results(raw, report, config, exp_dir)
                    # 单任务模式也生成一个简单的 overview
                    all_results_map = {r['patent_id']: [e] for r,e in zip(raw, report)}
                    _generate_summaries(exp_dir, {k:[None] for k in all_results_map}, all_results_map, config)
                    logger.info(f"✅ Finished. Results: {exp_dir}")
                except Exception as e:
                    logger.error(f"❌ Failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
