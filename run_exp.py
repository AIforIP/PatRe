import os
import yaml
import json
import copy
import argparse
import subprocess
import sys
from datetime import datetime
# src.evaluator
from src.evaluator import PatentEvaluator
from src.utils import setup_logger, load_dataset_json
from src.patent_round_save import save_one_patent_round_artifacts
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="PatRe")
    parser.add_argument("--config", type=str, default="config/eval_config.yaml")
    # Allow specifying a single IPC category via command line (e.g., --ipc A)
    parser.add_argument("--ipc", type=str, help="Run specific IPC category (A-H)")
    parser.add_argument("--num_samples", type=int, help="Override task_settings.num_samples")
    # model
    parser.add_argument("--model", type=str, help="Specific model to test")

    # (oa / rebuttal / both), default is both
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["oa", "rebuttal", "both"],
        help="Task type to run: oa / rebuttal / both (default: both)",
    )


    # Override test_mode: zero_shot vs sft_model
    parser.add_argument("--test_mode", type=str, choices=["zero_shot", "sft_model"], help="Override test mode")
    # Override retrieval_setting: no_ref, oracle_summary, bm25_retrieved
    parser.add_argument("--retrieval", type=str, choices=["no_ref", "oracle_summary", "bm25_retrieved"], help="Override retrieval setting")

    parser.add_argument("--backend", type=str, choices=["api", "vllm", "ollama"], help="Override model backend")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--generate_only", action="store_true", help="Only generate outputs, skip evaluation")
    mode_group.add_argument("--eval_only", action="store_true", help="Only evaluate existing results, skip generation")
    parser.add_argument("--results_dir", type=str, help="Results directory for eval-only mode")
    parser.add_argument("--eval_limit", type=int, default=0, help="Max rounds to evaluate in eval-only mode (0=all)")
    parser.add_argument(
        "--eval_force",
        action="store_true",
        help="In --eval_only mode, re-evaluate even if eval/scores.json already exists.",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Existing experiment directory to resume: skip patents that already have valid generation (and eval scores when not --generate_only).",
    )
    parser.add_argument(
        "--resume_retry_failed",
        action="store_true",
        help="With --resume_dir, regenerate when the saved output is empty or FAILED_TO_GENERATE.",
    )
    return parser.parse_args()

def analyze_dataset_rounds(config):
    """Analyze actual round distribution in the dataset."""
    test_set_path = config["paths"]["test_set"]
    if not os.path.exists(test_set_path):
        return 3

    patent_rounds = {}
    for item in load_dataset_json(test_set_path):
        patent_id = item["id"]
        patent_rounds[patent_id] = patent_rounds.get(patent_id, 0) + 1

    if not patent_rounds:
        return 3

    max_rounds = max(patent_rounds.values())
    print(f"Dataset max rounds: {max_rounds}")
    print(f"Round distribution: {sorted(set(patent_rounds.values()))}")
    return max_rounds

def generate_multiround_sequence_from_data(max_rounds):
    # Generate multi-round examination sequence based on actual dataset rounds.
    sequence = ["oa"]  # start as OA
    

    # round 1: oa
    # round 2: oa -> rebuttal
    # round 3: oa -> rebuttal -> oa
    # round 4+: oa -> rebuttal -> oa -> rebuttal ...
    
    for i in range(1, max_rounds):
        if i % 2 == 1:
            sequence.append("rebuttal")
        else:
            sequence.append("oa")
    
    return sequence



def _normalize_0_1(value):
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    v = float(value)
    if v < 0:
        return v
    if v <= 1:
        return round(v, 4)
    if v <= 10:
        return round(v / 10.0, 4)
    if v <= 100:
        return round(v / 100.0, 4)
    return round(v, 4)


def _pick_score(scores, *keys):
    for k in keys:
        v = scores.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def _build_scores_table(task_key, scores):
    if not isinstance(scores, dict):
        scores = {}

    table = {}
    if task_key == "rebuttal":
        pw = _normalize_0_1(_pick_score(scores, "point_wise_coverage"))
        if pw is not None:
            table["Point-wise Coverage"] = pw
    else:
        disp = _normalize_0_1(_pick_score(scores, "disposition_accuracy"))
        if disp is not None:
            table["Disposition Accuracy"] = disp
        stat = _normalize_0_1(_pick_score(scores, "statute_precision", "legal_statute_precision"))
        if stat is not None:
            table["Statute Precision"] = stat
        cit = _normalize_0_1(_pick_score(scores, "reference_citation_accuracy", "citation_accuracy"))
        if cit is not None:
            table["Citation Accuracy"] = cit

    bleu = _normalize_0_1(_pick_score(scores, "bleu"))
    if bleu is not None:
        table["BLEU"] = bleu
    rouge = _normalize_0_1(_pick_score(scores, "rouge_l"))
    if rouge is not None:
        table["ROUGE-L"] = rouge
    bert = _normalize_0_1(_pick_score(scores, "bert_score", "bert_score_f1"))
    if bert is not None:
        table["BERTScore"] = bert
    fmt = _pick_score(scores, "formatting_rule", "formatting_schema_score", "formatting_citation_style")
    if fmt is not None:
        f = float(fmt)
        # 新存盘为 0–10；旧结果为 0–1，按 0–10 展示
        table["Formatting"] = round(f * 10.0, 4) if f <= 1.0 else round(min(f, 10.0), 4)

    for key, label in (
        ("accuracy", "Accuracy"),
        ("clarity", "Clarity"),
        ("constructiveness", "Constructiveness"),
        ("completeness", "Completeness"),
        ("language_style", "Language Style"),
    ):
        val = _pick_score(scores, key)
        if val is not None:
            table[label] = val

    return table


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _round_task_from_name(name):
    # Expected like: round_1_oa or round_2_rebuttal
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[2].lower()
    return ""


def _build_result_entry(patent_id, round_dir, ctx):
    ai_input = ctx.get("ai_input", {})
    ai_output = ctx.get("ai_output", "")
    office_action = ctx.get("office_action", "")
    ground_truth = ctx.get("ground_truth", "")
    refs = ai_input.get("refs", "")

    
    if not ai_output:
        ai_txt = os.path.join(round_dir, "ai_output.txt")
        if os.path.exists(ai_txt):
            with open(ai_txt, "r", encoding="utf-8") as f:
                ai_output = f.read()

    task = ai_input.get("task_type") or _round_task_from_name(os.path.basename(round_dir))
    round_name = os.path.basename(round_dir)
    round_no = 0
    if round_name.startswith("round_"):
        try:
            round_no = int(round_name.split("_")[1])
        except Exception:
            round_no = 0

    input_payload = {
        "claims": ai_input.get("claims", ""),
        "refs": refs,
        "task_type": task,
        "prosecution_history": ai_input.get("prosecution_history", "N/A"),
        "granularity": ai_input.get("granularity", "full_text"),
    }
    # For rebuttal, we don't output retrieval_setting to keep the ai_input clean.
    if task != "rebuttal":
        input_payload["retrieval_setting"] = ai_input.get("retrieval_setting", "")

    return {
        "_round_dir": round_dir,
        "patent_id": patent_id,
        "round_info": {
            "task": task,
            "round_number": round_no,
            "exam_phase": "eval_only",
            "granularity": ai_input.get("granularity", "full_text"),
            "claim_number": "all",
        },
        "ai_exam": {
            "input": input_payload,
            "output": ai_output,
            "available_info": list(ai_input.keys()),
            "restricted_info": ctx.get("restricted_info", []),
        },
        "evaluation_context": {
            "full_context": {
                "claims": ai_input.get("claims", ""),
                "refs": refs,
                "office_action": office_action,
                "rebuttal_history": ctx.get("rebuttal_history", "N/A"),
            },
            "ground_truth": ground_truth,
            "citations_gt": ctx.get("citations_gt", []),
            "ground_truth_decision": ctx.get("ground_truth_decision", ""),
            "input_refs_preview": (refs[:200] + "...") if isinstance(refs, str) else "",
        },
        "meta": ctx.get("meta") or {},
    }


def _save_scores(round_dir, eval_result):
    out_dir = round_dir
    eval_subdir = os.path.join(round_dir, "eval")
    if os.path.isdir(eval_subdir):
        out_dir = eval_subdir

    scores_path = os.path.join(out_dir, "scores.json")
    payload = {
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
        "evaluation_report": eval_result.get("evaluation_report", ""),
        "justification_details": eval_result.get("justification_details", {}),
        "raw_judge_output": eval_result.get("raw_judge_output", ""),
        "metric_details": eval_result.get("metric_details", {}),
        "rule_scores": eval_result.get("rule_scores", {}),
    }
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    
    summary_path = os.path.join(out_dir, "scores_summary.json")
    task = eval_result.get("round_info", {}).get("task", "")
    percent_metrics = {
        "disposition_accuracy": 100.0,
        "statute_precision": 100.0,
        "citation_accuracy": 10.0,
        "point_wise_coverage": 10.0,
    }
    oa_schema = [
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
    rebuttal_schema = [
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
    name_map_oa = {
        "disposition_accuracy": "Dispositional Accuracy",
        "statute_precision": "Statute Precision",
        "citation_accuracy": "Citation Accuracy",
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
    }
    schema = rebuttal_schema if task == "rebuttal" else oa_schema
    name_map = name_map_rebuttal if task == "rebuttal" else name_map_oa
    src_scores = eval_result.get("scores", {}) or {}
    summary_scores = {}
    for key in schema:
        out_key = name_map.get(key, key)
        summary_scores[out_key] = None
        if key == "bert_score":
            val = src_scores.get("bert_score", src_scores.get("bert_score_f1", None))
        elif key == "citation_accuracy":
            val = src_scores.get("citation_accuracy", src_scores.get("reference_citation_accuracy", None))
        else:
            val = src_scores.get(key, None)

        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        if key in percent_metrics:
            summary_scores[out_key] = f"{val * percent_metrics[key]:.2f}%"
        else:
            summary_scores[out_key] = val
    ls_err = src_scores.get("language_style_error")
    if isinstance(ls_err, str) and ls_err.strip():
        summary_scores["Language Style Error"] = ls_err.strip()
    summary = {
        "task_type": task,
        "total_score": eval_result.get("total_score", 0),
        "scores": summary_scores,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
def _save_patent_organized_results(raw_results, eval_report, config, exp_dir, write_scores=True):
    """Save results grouped by patent, appending per-round outputs."""
    eval_report = eval_report or []
    for idx, raw_result in enumerate(raw_results or []):
        eval_result = (
            eval_report[idx] if idx < len(eval_report) and isinstance(eval_report[idx], dict) else {}
        )
        save_one_patent_round_artifacts(
            raw_result, exp_dir, write_scores=write_scores, eval_result=eval_result
        )

def _calculate_patent_average_scores(results):
    """Compute average scores for a single patent."""
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
    """Compute overall evaluation metrics."""
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
    """Generate experiment summary files."""

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

def _get_nested_text(item, *keys):
    cur = item
    for k in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(k)
    return cur if isinstance(cur, str) else ""

def _extract_round_texts(item):
    claims = _get_nested_text(item, "input", "claims") or _get_nested_text(item, "ground_truth", "claims")
    office_action = (
        _get_nested_text(item, "input", "office_action")
        or _get_nested_text(item, "ground_truth", "oa_text")
        or _get_nested_text(item, "ground_truth", "next_oa_text")
    )
    remarks = (
        _get_nested_text(item, "input", "rebuttal")
        or _get_nested_text(item, "ground_truth", "rebuttal_text")
    )
    return claims, office_action, remarks

def _dataset_item_has_oracle_refs(item):
    """Whether test-set row has oracle reference ids or non-empty oracle summaries."""
    inp = item.get("input", {}) if isinstance(item, dict) else {}
    ids = inp.get("oracle_reference_ids", [])
    summ = inp.get("oracle_reference_summaries", "")
    if isinstance(ids, list) and len(ids) > 0:
        return True
    if isinstance(summ, str) and summ.strip():
        return True
    return False


def _build_previous_round_context(items, current_round_idx, task_type):
    """Build context from only the previous same-type round.

    Args:
        items: List of rounds for a patent (same type only)
        current_round_idx: Current round index (0-based)
        task_type: Current task type ('oa' or 'rebuttal')

    Returns:
        String containing previous round context, or 'N/A' if no previous round
    """
    if current_round_idx <= 0:
        return "N/A"

    prev_idx = current_round_idx - 1
    if prev_idx >= len(items):
        return "N/A"

    prev_item = items[prev_idx]
    _, office_action, remarks = _extract_round_texts(prev_item)

    if task_type == "oa":
        if not office_action or office_action.strip() == "":
            return "N/A"
        return f"Previous Round Office Action:\n{office_action}"

    if task_type == "rebuttal":
        if not remarks or remarks.strip() == "" or remarks.upper() == "N/A":
            return "N/A"
        return f"Previous Round Rebuttal:\n{remarks}"

    return "N/A"


def _task_label(task_key):
    if task_key == "oa":
        return "oa"
    if task_key == "rebuttal":
        return "rebuttal"
    return "other"


def _round_dir_for_patent(exp_dir, patent_id, round_no, task_key):
    label = _task_label(task_key)
    return os.path.join(exp_dir, f"patent_{patent_id}", f"round_{round_no:02d}_{label}")


def _resolve_ai_output_path(round_dir):
    p_new = os.path.join(round_dir, "output", "ai_output.txt")
    p_legacy = os.path.join(round_dir, "ai_output.txt")
    if os.path.isfile(p_new):
        return p_new
    if os.path.isfile(p_legacy):
        return p_legacy
    return p_new


def _generation_is_complete(output_path, retry_failed):
    if not os.path.isfile(output_path):
        return False
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return False
    text = (text or "").strip()
    if not text:
        return False
    if text == "FAILED_TO_GENERATE" and not retry_failed:
        return False
    return True


def _resume_restricted_info(task_key):
    if task_key == "oa":
        return ["ground_truth_oa", "rebuttal_history", "human_decisions"]
    return ["ground_truth_final_outcome"]


def _load_resume_raw_result(patent_id, round_dir, round_no, task_key, item):
    """Rebuild a run_batch-style result dict from saved round files (for scoring after resume)."""
    ctx_path = os.path.join(round_dir, "eval", "context.json")
    in_path = os.path.join(round_dir, "input", "ai_input.json")
    out_path = _resolve_ai_output_path(round_dir)

    ai_output = ""
    if os.path.isfile(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            ai_output = f.read()
    if not str(ai_output).strip():
        return None

    office_action = ""
    ground_truth = "N/A"
    citations_gt = []
    rebuttal_history = "N/A"
    ai_input = {}

    ground_truth_decision = ""
    if isinstance(item, dict):
        ground_truth_decision = (item.get("ground_truth") or {}).get("decision", "") or ""

    if os.path.isfile(ctx_path):
        ctx = _load_json(ctx_path)
        ai_input = dict(ctx.get("ai_input") or {})
        if ctx.get("ai_output") is not None:
            ai_output = str(ctx.get("ai_output"))
        office_action = ctx.get("office_action") or ""
        ground_truth = ctx.get("ground_truth") if ctx.get("ground_truth") is not None else "N/A"
        cg = ctx.get("citations_gt", [])
        citations_gt = cg if isinstance(cg, list) else []
        rh = ctx.get("rebuttal_history")
        rebuttal_history = rh if isinstance(rh, str) else "N/A"
        gtd = ctx.get("ground_truth_decision")
        if isinstance(gtd, str) and gtd.strip():
            ground_truth_decision = gtd
    elif os.path.isfile(in_path):
        ai_input = _load_json(in_path)
        inp = item.get("input", {}) if isinstance(item, dict) else {}
        gt = item.get("ground_truth", {}) if isinstance(item, dict) else {}
        office_action = (
            inp.get("office_action", "")
            or gt.get("oa_text", "")
            or gt.get("next_oa_text", "")
        )
        if task_key == "rebuttal":
            ground_truth = gt.get("rebuttal_text", "N/A")
        else:
            ground_truth = gt.get("oa_text", "N/A")
        cg = gt.get("citations_gt", [])
        citations_gt = cg if isinstance(cg, list) else []
        rb = inp.get("rebuttal", "N/A")
        rebuttal_history = rb if isinstance(rb, str) else "N/A"
    else:
        return None

    if not str(ai_output).strip():
        return None

    if not ai_input.get("claims"):
        ic = item.get("input", {}).get("claims", "") if isinstance(item, dict) else ""
        if ic:
            ai_input = dict(ai_input)
            ai_input["claims"] = ic

    refs = ai_input.get("refs", "") or ""
    preview = (refs[:200] + "...") if isinstance(refs, str) and len(refs) > 200 else refs

    claims_ctx = item.get("input", {}).get("claims", "") if isinstance(item, dict) else ""
    if not office_action and isinstance(item, dict):
        office_action = (
            item.get("input", {}).get("office_action", "")
            or item.get("ground_truth", {}).get("oa_text", "")
            or item.get("ground_truth", {}).get("next_oa_text", "")
        )

    return {
        "patent_id": patent_id,
        "round_info": {
            "task": task_key,
            "round_number": round_no,
            "exam_phase": "context_aware",
            "granularity": ai_input.get("granularity", "full_text"),
            "claim_number": ai_input.get("claim_number", "all"),
        },
        "ai_exam": {
            "input": ai_input,
            "output": ai_output,
            "available_info": list(ai_input.keys()),
            "restricted_info": _resume_restricted_info(task_key),
        },
        "evaluation_context": {
            "full_context": {
                "claims": claims_ctx or ai_input.get("claims", ""),
                "refs": refs,
                "office_action": office_action,
                "rebuttal_history": rebuttal_history,
            },
            "ground_truth": ground_truth,
            "citations_gt": citations_gt,
            "ground_truth_decision": ground_truth_decision,
            "input_refs_preview": preview,
        },
        "meta": item.get("meta") or {} if isinstance(item, dict) else {},
    }


def _partition_resume_batch(items, exp_dir, round_no, task_key, args):
    """Split batch into items that still need API generation vs. disk-only scoring."""
    if not args.resume_dir:
        return list(items), []

    to_generate = []
    score_only = []
    for item in items:
        pid = item["id"]
        rd = _round_dir_for_patent(exp_dir, pid, round_no, task_key)
        out_path = _resolve_ai_output_path(rd)
        gen_ok = _generation_is_complete(out_path, args.resume_retry_failed)

        if not gen_ok:
            to_generate.append(item)
            continue

        if args.generate_only:
            continue

        scores_path = os.path.join(rd, "eval", "scores.json")
        if os.path.isfile(scores_path):
            continue

        score_only.append((item, rd))

    return to_generate, score_only


def _run_one_task_round(
    args,
    config,
    logger,
    multiround_exp_dir,
    batch_items,
    round_no,
    task_key,
    all_results_map,
    auto_extra,
):
    if not batch_items:
        return

    to_gen, score_pairs = _partition_resume_batch(batch_items, multiround_exp_dir, round_no, task_key, args)
    if args.resume_dir:
        if args.generate_only:
            logger.info(
                f"Resume round {round_no} ({task_key}): {len(to_gen)} to generate, "
                f"{len(batch_items) - len(to_gen)} already on disk (skipped)"
            )
        else:
            logger.info(
                f"Resume round {round_no} ({task_key}): {len(to_gen)} to generate, "
                f"{len(score_pairs)} pending eval from disk, "
                f"{len(batch_items) - len(to_gen) - len(score_pairs)} complete (skipped)"
            )

    if score_pairs and not args.generate_only:
        config["task_settings"]["current_task"] = task_key
        raws = []
        for item, rd in score_pairs:
            r = _load_resume_raw_result(item["id"], rd, round_no, task_key, item)
            if r:
                raws.append(r)
        if raws:
            engine = PatentEvaluator(config, logger)
            engine.dataset = []
            scoring_ok = False
            try:
                report = engine.run_scoring(raws)
                if auto_extra:
                    report = engine.run_additional_metrics(report)
                scoring_ok = True
            except Exception as e:
                logger.warning(f"Scoring failed (resume from disk); saving generation only. Error: {e}")
                report = [{} for _ in raws]
                scoring_ok = False
            _save_patent_organized_results(
                raws, report, config, multiround_exp_dir, write_scores=scoring_ok
            )
            if scoring_ok:
                for r, e in zip(raws, report):
                    pid = r["patent_id"]
                    if pid not in all_results_map:
                        all_results_map[pid] = []
                    all_results_map[pid].append(e)

    if not to_gen:
        if batch_items and args.resume_dir:
            logger.info(f"Round {round_no} ({task_key}): no new generations needed.")
        return

    config["task_settings"]["current_task"] = task_key
    config["task_settings"]["_run_round_number"] = round_no
    if config.get("task_settings", {}).get("incremental_disk_save", True):
        config["task_settings"]["_incremental_save_exp_dir"] = multiround_exp_dir
    else:
        config["task_settings"].pop("_incremental_save_exp_dir", None)

    if not args.resume_dir:
        logger.info(f"[Round {round_no}] {task_key}: {len(to_gen)} patents...")
    else:
        logger.info(f"[Round {round_no}] {task_key}: calling API for {len(to_gen)} patents...")
    engine = PatentEvaluator(config, logger)
    engine.dataset = to_gen
    raw = engine.run_batch()
    config["task_settings"].pop("_incremental_save_exp_dir", None)
    config["task_settings"].pop("_run_round_number", None)
    for r in raw:
        r["round_info"]["round_number"] = round_no
    scoring_ok = False
    if args.generate_only:
        report = [{} for _ in raw]
    else:
        try:
            report = engine.run_scoring(raw)
            if auto_extra:
                report = engine.run_additional_metrics(report)
            scoring_ok = True
        except Exception as e:
            logger.warning(f"Scoring failed; saving generation only. Error: {e}")
            report = [{} for _ in raw]
            scoring_ok = False
    _save_patent_organized_results(
        raw, report, config, multiround_exp_dir, write_scores=(not args.generate_only and scoring_ok)
    )

    if not args.generate_only and scoring_ok:
        for r, e in zip(raw, report):
            pid = r["patent_id"]
            if pid not in all_results_map:
                all_results_map[pid] = []
            all_results_map[pid].append(e)


def main():
    args = parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    models_to_run = [args.model] if args.model else config['models']['backbone_list']
    test_set_path = config['paths']['test_set']
    ipc_filter = args.ipc.strip().upper() if args.ipc else None
    if ipc_filter:
        config.setdefault('task_settings', {})['ipc_filter'] = ipc_filter
    if args.num_samples is not None:
        config.setdefault('task_settings', {})['num_samples'] = int(args.num_samples)
    

    if args.test_mode:
        config['active_settings']['test_mode'] = args.test_mode
    if args.retrieval:
        config['active_settings']['retrieval_setting'] = args.retrieval
    if args.backend:
        config['models']['backend'] = args.backend
    
    auto_extra = bool(config.get('task_settings', {}).get('auto_run_extra_metrics', True))

    if args.eval_only:
        if not args.results_dir:
            raise SystemExit("--results_dir is required when --eval_only is set")
        if not os.path.exists(args.results_dir):
            raise SystemExit(f"results_dir not found: {args.results_dir}")

        config.setdefault('task_settings', {})
        models_cfg = config.get("models", {})
        if "active_model" not in models_cfg:
            # Evaluation should use judge_model; backend still expects active_model.
            models_cfg["active_model"] = models_cfg.get("judge_model", "")
            config["models"] = models_cfg

        logger = setup_logger(args.results_dir)
        engine = PatentEvaluator(config, logger)

        dataset_by_pid = None
        def _ensure_dataset_loaded():
            nonlocal dataset_by_pid
            if dataset_by_pid is not None:
                return
            dataset_by_pid = {}
            if not os.path.exists(test_set_path):
                return
            for item in load_dataset_json(test_set_path):
                pid = str(item.get("id", "")).strip()
                if not pid:
                    continue
                dataset_by_pid.setdefault(pid, []).append(item)

        def _infer_round_from_dir_name(round_dir_name):
            parts = str(round_dir_name or "").split("_")
            if len(parts) >= 3 and parts[0] == "round":
                try:
                    return int(parts[1]), str(parts[2]).lower()
                except Exception:
                    return 0, ""
            return 0, ""

        def _extract_from_item(item, key):
            gt = item.get("ground_truth", {}) if isinstance(item, dict) else {}
            inp = item.get("input", {}) if isinstance(item, dict) else {}
            if key == "claims":
                v = inp.get("claims", "")
                return v if isinstance(v, str) else ""
            if key == "office_action":
                for k in ("office_action",):
                    v = inp.get(k, "")
                    if isinstance(v, str) and v.strip():
                        return v
                for k in ("oa_text", "next_oa_text"):
                    v = gt.get(k, "")
                    if isinstance(v, str) and v.strip():
                        return v
                return ""
            if key == "ground_truth_oa":
                v = gt.get("oa_text", "")
                return v if isinstance(v, str) else ""
            if key == "ground_truth_rebuttal":
                v = gt.get("rebuttal_text", "")
                return v if isinstance(v, str) else ""
            if key == "rebuttal_history":
                v = inp.get("rebuttal", "N/A")
                return v if isinstance(v, str) else "N/A"
            return ""

        def _build_ctx_from_dataset(patent_id, round_dir_name, has_both_tasks):
            _ensure_dataset_loaded()
            items = (dataset_by_pid or {}).get(patent_id) if dataset_by_pid is not None else None
            if not items:
                return None
            round_no, task = _infer_round_from_dir_name(round_dir_name)
            if round_no <= 0 or task not in ("oa", "rebuttal"):
                return None

            if has_both_tasks:
                if task == "oa":
                    idx = (round_no - 1) // 2
                else:
                    idx = (round_no - 2) // 2
            else:
                idx = round_no - 1

            if idx < 0 or idx >= len(items):
                return None

            item = items[idx]
            claims = _extract_from_item(item, "claims")
            office_action = _extract_from_item(item, "office_action")
            # Use only previous round context (not all history)
            prev_context = _build_previous_round_context(items, idx, task)
            if task == "rebuttal":
                ground_truth = _extract_from_item(item, "ground_truth_rebuttal") or "N/A"
            else:
                ground_truth = _extract_from_item(item, "ground_truth_oa") or "N/A"
            gt_citations = item.get("ground_truth", {}).get("citations_gt", [])
            if not isinstance(gt_citations, list):
                gt_citations = []

            retrieval_setting = config.get("active_settings", {}).get("retrieval_setting", "")
            ai_input = {
                "claims": claims,
                "refs": "",
                "task_type": task,
                "prosecution_history": prev_context,
                "granularity": "full_text",
            }
            # For rebuttal, we don't output retrieval_setting into ai_input.json.
            if task != "rebuttal":
                ai_input["retrieval_setting"] = retrieval_setting
            return {
                "ai_input": ai_input,
                "office_action": office_action,
                "ground_truth": ground_truth,
                "ground_truth_decision": (item.get("ground_truth") or {}).get("decision", ""),
                "meta": item.get("meta") or {},
                "rebuttal_history": _extract_from_item(item, "rebuttal_history"),
                "citations_gt": gt_citations,
                "restricted_info": [],
            }

        ipc_allowed = None
        if ipc_filter:
            ipc_allowed = set()
            if os.path.exists(test_set_path):
                try:
                    for item in load_dataset_json(test_set_path):
                        pid = str(item.get("id", "")).strip()
                        ipc = str(item.get("meta", {}).get("ipc", "")).strip().upper()
                        if pid and ipc == ipc_filter:
                            ipc_allowed.add(pid)
                except Exception:
                    ipc_allowed = None

        grouped = {}
        scanned_rounds = 0
        base_name = os.path.basename(os.path.normpath(args.results_dir))
        if base_name.startswith("patent_"):
            patent_dirs = [(base_name.replace("patent_", ""), args.results_dir)]
        else:
            patent_dirs = []
            for name in os.listdir(args.results_dir):
                if not name.startswith("patent_"):
                    continue
                patent_dir = os.path.join(args.results_dir, name)
                if not os.path.isdir(patent_dir):
                    continue
                patent_id = name.replace("patent_", "")
                patent_dirs.append((patent_id, patent_dir))

        for patent_id, patent_dir in patent_dirs:
            if ipc_allowed is not None and patent_id not in ipc_allowed:
                continue
            round_names = [n for n in os.listdir(patent_dir) if os.path.isdir(os.path.join(patent_dir, n))]
            has_both_tasks = any("_rebuttal" in n for n in round_names) and any("_oa" in n for n in round_names)
            for round_name in round_names:
                round_dir = os.path.join(patent_dir, round_name)
                ctx_path = os.path.join(round_dir, "eval", "context.json")
                if not os.path.exists(ctx_path):
                    ctx_path = os.path.join(round_dir, "context.json")
                ctx = None
                if os.path.exists(ctx_path):
                    ctx = _load_json(ctx_path)
                    _ensure_dataset_loaded()
                    ctx_ds = _build_ctx_from_dataset(patent_id, round_name, has_both_tasks)
                    if isinstance(ctx, dict) and isinstance(ctx_ds, dict):
                        gtd = ctx.get("ground_truth_decision")
                        if not (isinstance(gtd, str) and gtd.strip()):
                            ctx["ground_truth_decision"] = ctx_ds.get(
                                "ground_truth_decision", ""
                            )
                        if not ctx.get("meta"):
                            ctx["meta"] = ctx_ds.get("meta") or {}
                else:
                    ctx = _build_ctx_from_dataset(patent_id, round_name, has_both_tasks)
                    if isinstance(ctx, dict):
                        ai_txt = os.path.join(round_dir, "ai_output.txt")
                        if os.path.exists(ai_txt):
                            with open(ai_txt, "r", encoding="utf-8") as f:
                                ctx["ai_output"] = f.read()
                if not isinstance(ctx, dict):
                    continue
                res = _build_result_entry(patent_id, round_dir, ctx)
                task = res.get("round_info", {}).get("task", "")
                if args.task != "both" and task != args.task:
                    continue
                grouped.setdefault(task, []).append(res)
                scanned_rounds += 1
                if args.eval_limit and scanned_rounds >= int(args.eval_limit):
                    break
            if args.eval_limit and scanned_rounds >= int(args.eval_limit):
                break

        for task, results in grouped.items():
            if not results:
                continue
            config['task_settings']['current_task'] = task
            # Evaluate one-by-one so each result is written immediately
            pbar = tqdm(total=len(results), desc=f"Eval {task}", unit="sample")
            for res in results:
                if not args.eval_force:
                    rd = res.get("_round_dir") or ""
                    if isinstance(rd, str) and rd:
                        scores_path = os.path.join(rd, "eval", "scores.json")
                        if os.path.isfile(scores_path):
                            pbar.update(1)
                            continue
                scored = engine.run_scoring([res], show_progress=False)
                if auto_extra:
                    scored = engine.run_additional_metrics(scored)
                _save_scores(scored[0]["_round_dir"], scored[0])
                pbar.update(1)
            pbar.close()

        logger.info("Eval-only complete.")

        if config.get("task_settings", {}).get("auto_run_aggregate_scores", True):
            try:
                agg_script = os.path.relpath(
                    os.path.join(os.path.dirname(__file__), "aggregate_scores.py"),
                    start=os.getcwd(),
                )
                out_csv = os.path.join(args.results_dir, "metrics_mean_by_task.csv")
                subprocess.run(
                    [
                        sys.executable,
                        agg_script,
                        "--root",
                        args.results_dir,
                        "--out",
                        out_csv,
                    ],
                    check=False,
                )
            except Exception as e:
                logger.warning(f"Aggregate scores failed: {e}")
        return

    # name
    naming_cfg = config.get('experiment_naming', {})
    use_academic_format = naming_cfg.get('use_academic_format', True)
    timestamp_format = naming_cfg.get('timestamp_format', "%Y%m%d_%H%M")
    include_model_name = naming_cfg.get('include_model_name', True)


    all_data = load_dataset_json(test_set_path)
    if ipc_filter:
        all_data = [
            item for item in all_data
            if str(item.get("meta", {}).get("ipc", "")).upper() == ipc_filter
        ]
    

    limit = config['task_settings'].get('num_samples', 9999)
    all_data = all_data[:limit]
    
    grouped_data = {}
    for item in all_data:
        pid = item['id']
        if pid not in grouped_data:
            grouped_data[pid] = []
        grouped_data[pid].append(item)

    # Sort by oa_number/oa_date and split into same-type lists for correct sequencing
    def _sort_key(it):
        meta = it.get("meta", {}) if isinstance(it, dict) else {}
        num = meta.get("oa_number")
        try:
            num_val = int(num)
        except Exception:
            num_val = 10 ** 9
        date_val = meta.get("oa_date") or ""
        round_type = str(meta.get("round_type", "")).lower()
        round_order = 0 if round_type == "oa" else 1
        return (num_val, date_val, round_order)

    grouped_data = {pid: sorted(items, key=_sort_key) for pid, items in grouped_data.items()}
    grouped_oa = {
        pid: [it for it in items if str(it.get("meta", {}).get("round_type", "")).lower() == "oa"]
        for pid, items in grouped_data.items()
    }
    grouped_rebuttal = {
        pid: [it for it in items if str(it.get("meta", {}).get("round_type", "")).lower() == "rebuttal"]
        for pid, items in grouped_data.items()
    }

    max_lines = max(len(v) for v in grouped_data.values()) if grouped_data else 0
    max_oa_lines = max((len(v) for v in grouped_oa.values()), default=0)
    max_rebuttal_lines = max((len(v) for v in grouped_rebuttal.values()), default=0)

    if args.resume_dir and len(models_to_run) > 1:
        print("Warning: --resume_dir with multiple models may write to the same directory; use --model for a single backbone.")

    for model_idx, model_name in enumerate(models_to_run, start=1):
        config['models']['active_model'] = model_name
        timestamp = datetime.now().strftime(timestamp_format)
        tm = config.get('active_settings', {}).get('test_mode', 'zero_shot')
        retr = config.get('active_settings', {}).get('retrieval_setting', 'oracle_summary')
        task_tag = args.task
        ipc_tag = ipc_filter if ipc_filter else "ALL"
        def _safe(s):
            return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))
        if args.resume_dir:
            multiround_exp_dir = os.path.abspath(args.resume_dir)
        else:
            parts = ["PatRe"]
            if include_model_name:
                parts.append(_safe(model_name))
            parts.extend([_safe(task_tag), f"tm-{_safe(tm)}", f"ret-{_safe(retr)}", f"ipc-{_safe(ipc_tag)}", timestamp])
            folder_name = "_".join(parts)
            multiround_exp_dir = os.path.join(config['paths']['results'], folder_name)
            if len(models_to_run) > 1:
                multiround_exp_dir = f"{multiround_exp_dir}_run{model_idx}"
        os.makedirs(multiround_exp_dir, exist_ok=True)
        
        logger = setup_logger(multiround_exp_dir)
        logger.info(f"Starting multi-round for model: {model_name}")

        all_results_map = {} # pid -> [eval_result, ...]

        total_rounds = max(max_oa_lines, max_rebuttal_lines) if args.task == "both" else max_lines
        for line_idx in range(total_rounds):
            # 1. Examiner Round (OA)
            if args.task in ("oa", "both"):
                task_oa = "oa"
                config['task_settings']['current_task'] = task_oa
                batch_oa = []

                for pid, lines in grouped_oa.items():
                    if line_idx < len(lines):
                        item = lines[line_idx]
                        item_oa = copy.deepcopy(item)
                        item_oa.setdefault("input", {})
                        # Use only previous OA
                        prev_context = _build_previous_round_context(lines, line_idx, "oa")
                        item_oa["input"]["prosecution_history"] = prev_context
                        # For OA generation, use previous rebuttal if exists
                        prev_remarks = ""
                        if line_idx - 1 >= 0 and pid in grouped_rebuttal:
                            rb_lines = grouped_rebuttal.get(pid, [])
                            if line_idx - 1 < len(rb_lines):
                                _, _, prev_remarks = _extract_round_texts(rb_lines[line_idx - 1])
                        if isinstance(prev_remarks, str):
                            prev_remarks = prev_remarks.strip()
                        item_oa["input"]["rebuttal"] = prev_remarks if prev_remarks is not None else ""
                        if config.get("active_settings", {}).get("retrieval_setting") == "oracle_summary":
                            if not _dataset_item_has_oracle_refs(item_oa):
                                continue
                        batch_oa.append(item_oa)

                if batch_oa:
                    round_no = line_idx * 2 + 1 if args.task == "both" else line_idx + 1
                    _run_one_task_round(
                        args,
                        config,
                        logger,
                        multiround_exp_dir,
                        batch_oa,
                        round_no,
                        task_oa,
                        all_results_map,
                        auto_extra,
                    )

            # 2. Applicant Round (Rebuttal)
            if args.task in ("rebuttal", "both"):
                task_rebuttal = "rebuttal"
                config['task_settings']['current_task'] = task_rebuttal
                batch_rebuttal = []

                for pid, lines in grouped_rebuttal.items():
                    if line_idx < len(lines):
                        item = lines[line_idx]
                        item_rb = copy.deepcopy(item)
                        item_rb.setdefault("input", {})
                        # Use only previous rebuttal
                        prev_context = _build_previous_round_context(lines, line_idx, "rebuttal")
                        item_rb["input"]["prosecution_history"] = prev_context
                        # For rebuttal generation, use current round's remarks.
                        _, _, cur_remarks = _extract_round_texts(item_rb)
                        if isinstance(cur_remarks, str):
                            cur_remarks = cur_remarks.strip()
                        item_rb["input"]["rebuttal"] = cur_remarks if cur_remarks is not None else ""
                        rebuttal_text = (
                            item_rb.get("input", {}).get("rebuttal")
                            or item_rb.get("ground_truth", {}).get("rebuttal_text")
                            or ""
                        )
                        if isinstance(rebuttal_text, str):
                            rebuttal_text = rebuttal_text.strip()
                        if rebuttal_text and str(rebuttal_text).upper() != "N/A":
                            if config.get("active_settings", {}).get("retrieval_setting") == "oracle_summary":
                                if not _dataset_item_has_oracle_refs(item_rb):
                                    continue
                            batch_rebuttal.append(item_rb)

                if batch_rebuttal:
                    round_no = line_idx * 2 + 2 if args.task == "both" else line_idx + 1
                    _run_one_task_round(
                        args,
                        config,
                        logger,
                        multiround_exp_dir,
                        batch_rebuttal,
                        round_no,
                        task_rebuttal,
                        all_results_map,
                        auto_extra,
                    )

        logger.info(f"Multiround finished. Results: {multiround_exp_dir}")

        if (
            not args.generate_only
            and config.get("task_settings", {}).get("auto_run_aggregate_scores", True)
        ):
            try:
                agg_script = os.path.relpath(
                    os.path.join(os.path.dirname(__file__), "aggregate_scores.py"),
                    start=os.getcwd(),
                )
                out_csv = os.path.join(multiround_exp_dir, "metrics_mean_by_task.csv")
                subprocess.run(
                    [
                        sys.executable,
                        agg_script,
                        "--root",
                        multiround_exp_dir,
                        "--out",
                        out_csv,
                    ],
                    check=False,
                )
            except Exception as e:
                logger.warning(f"Aggregate scores failed: {e}")


if __name__ == "__main__":
    main()
