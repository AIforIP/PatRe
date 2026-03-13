import argparse
import json
import os
from collections import defaultdict

import yaml

from src.evaluator import PatentEvaluator
from src.utils import setup_logger


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

    # Fall back to files if not present in context.json
    if not ai_output:
        ai_txt_new = os.path.join(round_dir, "output", "ai_output.txt")
        ai_txt_old = os.path.join(round_dir, "ai_output.txt")
        if os.path.exists(ai_txt_new):
            with open(ai_txt_new, "r", encoding="utf-8") as f:
                ai_output = f.read()
        elif os.path.exists(ai_txt_old):
            with open(ai_txt_old, "r", encoding="utf-8") as f:
                ai_output = f.read()

    task = ai_input.get("task_type") or _round_task_from_name(os.path.basename(round_dir))
    round_name = os.path.basename(round_dir)
    round_no = 0
    if round_name.startswith("round_"):
        try:
            round_no = int(round_name.split("_")[1])
        except Exception:
            round_no = 0

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
            "input": {
                "claims": ai_input.get("claims", ""),
                "refs": refs,
                "task_type": task,
                "retrieval_setting": ai_input.get("retrieval_setting", ""),
                "prosecution_history": ai_input.get("prosecution_history", "N/A"),
                "granularity": ai_input.get("granularity", "full_text"),
            },
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
            "input_refs_preview": (refs[:200] + "...") if isinstance(refs, str) else "",
        },
    }


def _save_scores(round_dir, eval_result):
    eval_dir = os.path.join(round_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    scores_path = os.path.join(eval_dir, "scores.json")
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

    # Lightweight summary for quick browsing
    summary_path = os.path.join(eval_dir, "scores_summary.json")
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
    summary = {
        "task_type": task,
        "total_score": eval_result.get("total_score", 0),
        "scores": summary_scores,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)




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
    fmt = _normalize_0_1(_pick_score(scores, "formatting_rule", "formatting_schema_score", "formatting_citation_style"))
    if fmt is not None:
        table["Formatting"] = fmt

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

def parse_args():
    parser = argparse.ArgumentParser(description="PatRe Evaluation Only Runner")
    parser.add_argument("--config", type=str, default="config/eval_config.yaml")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["oa", "rebuttal", "both"], default="both")
    parser.add_argument("--run_extra_metrics", action="store_true")
    parser.add_argument("--eval_limit", type=int, default=0, help="Max rounds to evaluate (0=all)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.results_dir):
        raise SystemExit(f"results_dir not found: {args.results_dir}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models_cfg = config.get("models", {})
    if "active_model" not in models_cfg:
        # Evaluation should use judge_model; backend still expects active_model.
        models_cfg["active_model"] = models_cfg.get("judge_model", "")
        config["models"] = models_cfg

    logger = setup_logger(args.results_dir)
    engine = PatentEvaluator(config, logger)

    grouped = defaultdict(list)
    scanned_rounds = 0

    for name in os.listdir(args.results_dir):
        if not name.startswith("patent_"):
            continue
        patent_dir = os.path.join(args.results_dir, name)
        if not os.path.isdir(patent_dir):
            continue
        patent_id = name.replace("patent_", "")
        for round_name in os.listdir(patent_dir):
            round_dir = os.path.join(patent_dir, round_name)
            if not os.path.isdir(round_dir):
                continue
            # Support new layout eval/context.json; fallback to legacy root
            ctx_path = os.path.join(round_dir, "eval", "context.json")
            if not os.path.exists(ctx_path):
                ctx_path = os.path.join(round_dir, "context.json")
                if not os.path.exists(ctx_path):
                    continue
            ctx = _load_json(ctx_path)
            res = _build_result_entry(patent_id, round_dir, ctx)
            task = res.get("round_info", {}).get("task", "")
            if args.task != "both" and task != args.task:
                continue
            grouped[task].append(res)
            scanned_rounds += 1
            if args.eval_limit and scanned_rounds >= int(args.eval_limit):
                break
        if args.eval_limit and scanned_rounds >= int(args.eval_limit):
            break

    for task, results in grouped.items():
        if not results:
            continue
        config["task_settings"]["current_task"] = task
        # Evaluate one-by-one so each result is written immediately
        for res in results:
            scored = engine.run_scoring([res])
            if args.run_extra_metrics:
                scored = engine.run_additional_metrics(scored)
            _save_scores(scored[0]["_round_dir"], scored[0])

    logger.info("Eval-only complete.")


if __name__ == "__main__":
    main()
