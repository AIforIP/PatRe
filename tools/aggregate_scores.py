import argparse
import csv
import json
import os
from collections import defaultdict


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


PERCENT_METRICS = {
    "disposition_accuracy": 100.0,
    "statute_precision": 100.0,
    "citation_accuracy": 10.0,
    "point_wise_coverage": 10.0,
}

OA_METRICS = [
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

REBUTTAL_METRICS = [
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

TASK_METRICS = {
    "oa": OA_METRICS,
    "rebuttal": REBUTTAL_METRICS,
    # Backward compatibility for older results
    "initial_examination": OA_METRICS,
    "response_to_rebuttal": OA_METRICS,
}


def parse_value(metric, value):
    if is_number(value):
        return float(value)
    if isinstance(value, str) and value.endswith("%"):
        try:
            return float(value.strip("%"))
        except Exception:
            return None
    return None


CANONICAL_KEY = {
    "Disposition Accuracy": "disposition_accuracy",
    "Dispositional Accuracy": "disposition_accuracy",
    "Statute Precision": "statute_precision",
    "Citation Accuracy": "citation_accuracy",
    "Point-wise Coverage": "point_wise_coverage",
    "Point Wise Coverage": "point_wise_coverage",
    "BLEU": "bleu",
    "ROUGE-L": "rouge_l",
    "BERTScore": "bert_score",
    "Formatting": "formatting_rule",
    "Accuracy": "accuracy",
    "Clarity": "clarity",
    "Language Style": "language_style",
    "Constructiveness": "constructiveness",
    "Completeness": "completeness",
}


def canonicalize_metric_key(key):
    if not isinstance(key, str):
        return None
    k = key.strip()
    if not k:
        return None
    mapped = CANONICAL_KEY.get(k)
    if mapped:
        return mapped
    return k


def main():
    parser = argparse.ArgumentParser(description="Aggregate mean scores by task_type from scores*.json")
    parser.add_argument("--root", default="results/nips_exp", help="Root directory to scan")
    parser.add_argument("--filename", default="scores_summary.json", help="Filename to aggregate (scores_summary.json or scores.json)")
    parser.add_argument("--out", default="results/nips_exp/metrics_mean_by_task.csv", help="Output CSV path")
    parser.add_argument("--out_json", default="", help="Optional output JSON path")
    parser.add_argument("--include_all_metrics", action="store_true", help="Include all observed metrics, not only the default schema")
    args = parser.parse_args()

    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    file_count = defaultdict(int)
    total_files = 0

    for dirpath, _, filenames in os.walk(args.root):
        for name in filenames:
            if name != args.filename:
                continue
            total_files += 1
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            task = data.get("task_type") or "unknown"
            file_count[task] += 1

            # total_score
            total_score = data.get("total_score", None)
            if is_number(total_score):
                sums[task]["total_score"] += float(total_score)
                counts[task]["total_score"] += 1

            scores = data.get("scores", {}) or {}
            for k, v in scores.items():
                if v is None:
                    continue
                canon_k = canonicalize_metric_key(k)
                if canon_k is None:
                    continue
                parsed = parse_value(canon_k, v)
                if parsed is not None:
                    sums[task][canon_k] += float(parsed)
                    counts[task][canon_k] += 1

    rows = []
    for task in file_count.keys():
        metric_sums = sums.get(task, {})
        present_metrics = set(metric_sums.keys())
        metric_list = TASK_METRICS.get(task, OA_METRICS)
        if args.include_all_metrics:
            metric_list = sorted(set(metric_list) | present_metrics)

        for metric in metric_list:
            s = metric_sums.get(metric, 0.0)
            c = counts[task].get(metric, 0)
            if c == 0:
                mean_str = "NaN"
            else:
                mean = s / c
                if metric in PERCENT_METRICS:
                    mean_str = f"{round(mean, 2)}%"
                else:
                    mean_str = round(mean, 6)
            rows.append({
                "task_type": task,
                "metric": metric,
                "mean": mean_str,
                "count": counts[task].get(metric, 0),
                "files_with_task": file_count.get(task, 0)
            })

    rows.sort(key=lambda r: (r["task_type"], r["metric"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "metric", "mean", "count", "files_with_task"])
        writer.writeheader()
        writer.writerows(rows)

    if args.out_json:
        payload = {
            "scanned_files": total_files,
            "by_task": rows,
        }
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Scanned {total_files} files. Wrote {len(rows)} rows to {args.out}")
    if args.out_json:
        print(f"Wrote JSON to {args.out_json}")


if __name__ == "__main__":
    main()
