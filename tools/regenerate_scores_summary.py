import argparse
import json
import os


def _infer_task_from_round_dir(round_dir_name):
    n = str(round_dir_name or "").lower()
    if "_rebuttal" in n or n.endswith("rebuttal"):
        return "rebuttal"
    if "_oa" in n or n.endswith("oa"):
        return "oa"
    return ""


def _build_summary_scores(task, scores):
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

    src_scores = scores if isinstance(scores, dict) else {}
    out = {}
    for key in schema:
        label = name_map.get(key, key)
        out[label] = None
        if key == "bert_score":
            val = src_scores.get("bert_score", src_scores.get("bert_score_f1", None))
        elif key == "citation_accuracy":
            val = src_scores.get("citation_accuracy", src_scores.get("reference_citation_accuracy", None))
        else:
            val = src_scores.get(key, None)

        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue

        if key in percent_metrics:
            out[label] = f"{val * percent_metrics[key]:.2f}%"
        else:
            out[label] = val
    return out


def main():
    parser = argparse.ArgumentParser(description="Regenerate scores_summary.json from scores.json without calling LLMs")
    parser.add_argument("--root", required=True, help="Results directory to scan")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scores_summary.json")
    args = parser.parse_args()

    rewritten = 0
    skipped = 0
    scanned = 0

    for dirpath, _, filenames in os.walk(args.root):
        if "scores.json" not in filenames:
            continue
        scanned += 1
        scores_path = os.path.join(dirpath, "scores.json")
        summary_path = os.path.join(dirpath, "scores_summary.json")
        if os.path.exists(summary_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            with open(scores_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped += 1
            continue

        task = _infer_task_from_round_dir(os.path.basename(dirpath)) or data.get("task_type") or ""
        combined_scores = {}
        base_scores = data.get("scores", {})
        if isinstance(base_scores, dict):
            combined_scores.update(base_scores)
        lp = data.get("llm_parsed_output", None)
        if isinstance(lp, dict):
            lp_scores = lp.get("scores") if isinstance(lp.get("scores"), dict) else lp
            if isinstance(lp_scores, dict):
                for k, v in lp_scores.items():
                    if k not in combined_scores and isinstance(v, (int, float)) and not isinstance(v, bool):
                        combined_scores[k] = v

        summary_scores = _build_summary_scores(task, combined_scores)
        summary = {
            "task_type": task or "unknown",
            "total_score": data.get("total_score", 0),
            "scores": summary_scores,
        }
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            rewritten += 1
        except Exception:
            skipped += 1

    print(f"Scanned {scanned} round dirs. Rewrote {rewritten}. Skipped {skipped}.")


if __name__ == "__main__":
    main()
