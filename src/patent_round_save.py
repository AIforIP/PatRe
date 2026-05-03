#Persist one patent-round to disk (same layout as run_exp multiround folders).

import json
import os

_OA_SCHEMA = [
    "disposition_accuracy",
    "statute_precision",
    "citation_accuracy",
    "bleu",
    "rouge_l",
    "bert_score",
    "formatting_rule",
    "accuracy",
    "clarity",
    "language_style",
    "constructiveness",
    "completeness",
]
_REBUTTAL_SCHEMA = [
    "point_wise_coverage",
    "bleu",
    "rouge_l",
    "bert_score",
    "formatting_rule",
    "accuracy",
    "clarity",
    "language_style",
    "constructiveness",
    "completeness",
]


def save_one_patent_round_artifacts(raw_result, exp_dir, write_scores=False, eval_result=None):
    #Write input/ai_input.json, output/ai_output.txt, eval/context.json, and optionally eval/scores.json + scores_summary.json for one sample.
    eval_result = eval_result if isinstance(eval_result, dict) else {}
    patent_id = raw_result["patent_id"]
    patent_dir = os.path.join(exp_dir, f"patent_{patent_id}")
    os.makedirs(patent_dir, exist_ok=True)

    round_info = raw_result["round_info"]
    round_num = round_info["round_number"]
    task_key = round_info["task"]

    if task_key == "oa":
        task_label = "oa"
    elif task_key == "rebuttal":
        task_label = "rebuttal"
    else:
        task_label = "other"

    round_dir = os.path.join(patent_dir, f"round_{round_num:02d}_{task_label}")
    input_dir = os.path.join(round_dir, "input")
    output_dir = os.path.join(round_dir, "output")
    eval_dir = os.path.join(round_dir, "eval")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    exam_input_path = os.path.join(input_dir, "ai_input.json")
    with open(exam_input_path, "w", encoding="utf-8") as f:
        ai_input_data = raw_result["ai_exam"]["input"].copy()
        json.dump(ai_input_data, f, indent=2, ensure_ascii=False)

    exam_output_path = os.path.join(output_dir, "ai_output.txt")
    with open(exam_output_path, "w", encoding="utf-8") as f:
        f.write(raw_result["ai_exam"]["output"])

    eval_context_path = os.path.join(eval_dir, "context.json")
    with open(eval_context_path, "w", encoding="utf-8") as f:
        context_data = {
            "ai_input": raw_result["ai_exam"]["input"],
            "ai_output": raw_result["ai_exam"]["output"],
            "ground_truth": raw_result["evaluation_context"].get("ground_truth", ""),
            "ground_truth_decision": raw_result.get("evaluation_context", {}).get(
                "ground_truth_decision", ""
            ),
            "restricted_info": raw_result["ai_exam"].get("restricted_info", []),
            "office_action": raw_result.get("evaluation_context", {})
            .get("full_context", {})
            .get("office_action", ""),
            "rebuttal_history": raw_result.get("evaluation_context", {})
            .get("full_context", {})
            .get("rebuttal_history", "N/A"),
            "citations_gt": raw_result.get("evaluation_context", {}).get("citations_gt", []),
        }
        json.dump(context_data, f, indent=2, ensure_ascii=False)

    if not write_scores:
        return

    scores_path = os.path.join(eval_dir, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        score_data = {
            "total_score": eval_result.get("total_score", 0),
            "scores": eval_result.get("scores", {}),
            "llm_total_score": eval_result.get("judge_total_score", None),
            "llm_scores": eval_result.get("judge_scores", {}),
            "llm_system_prompt": eval_result.get("judge_system_prompt", ""),
            "llm_user_prompt": eval_result.get("judge_user_prompt", ""),
            "llm_attempts": eval_result.get("judge_attempts", []),
            "llm_parsed_output": eval_result.get("judge_parsed_output", {}),
            "llm_justification": eval_result.get("judge_justification", ""),
            "llm_justification_details": eval_result.get("judge_justification_details", {}),
            "llm_evaluation_report": eval_result.get("judge_evaluation_report", ""),
            "justification": eval_result.get("justification", ""),
            "justification_details": eval_result.get("justification_details", {}),
            "raw_judge_output": eval_result.get("raw_judge_output", ""),
            "metric_details": eval_result.get("metric_details", {}),
            "rule_scores": eval_result.get("rule_scores", {}),
        }
        json.dump(score_data, f, indent=2, ensure_ascii=False)

    summary_scores = {}
    schema = _REBUTTAL_SCHEMA if task_key == "rebuttal" else _OA_SCHEMA
    name_map_oa = {
        "disposition_accuracy": "Dispositional Accuracy",
        "statute_precision": "Statute Precision",
        "citation_accuracy": "Citation Accuracy",
        "citation_list_match": "Citation List Match",
        "bleu": "BLEU",
        "rouge_l": "ROUGE-L",
        "bert_score": "BERTScore",
        "formatting_rule": "Formatting",
        "accuracy": "Accuracy",
        "clarity": "Clarity",
        "language_style": "Language Style",
        "constructiveness": "Constructiveness",
        "completeness": "Completeness",
    }
    name_map_rebuttal = {
        "point_wise_coverage": "Point-wise Coverage",
        "bleu": "BLEU",
        "rouge_l": "ROUGE-L",
        "bert_score": "BERTScore",
        "formatting_rule": "Formatting",
        "accuracy": "Accuracy",
        "clarity": "Clarity",
        "language_style": "Language Style",
        "constructiveness": "Constructiveness",
        "completeness": "Completeness",
        "disposition_accuracy": "Dispositional Accuracy",
        "statute_precision": "Statute Precision",
        "citation_accuracy": "Citation Accuracy",
        "citation_list_match": "Citation List Match",
    }
    name_map = name_map_rebuttal if task_key == "rebuttal" else name_map_oa
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
        elif key == "citation_accuracy":
            val = src_scores.get(
                "citation_accuracy", src_scores.get("reference_citation_accuracy", None)
            )
        else:
            val = src_scores.get(key, None)

        out_key = name_map.get(key, key)
        if key in percent_metrics and isinstance(val, (int, float)):
            summary_scores[out_key] = f"{val * percent_metrics[key]:.2f}%"
        else:
            summary_scores[out_key] = val
    ls_err = src_scores.get("language_style_error")
    if isinstance(ls_err, str) and ls_err.strip():
        summary_scores["Language Style Error"] = ls_err.strip()
    summary_path = os.path.join(eval_dir, "scores_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_type": task_key,
                "scores": summary_scores,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
