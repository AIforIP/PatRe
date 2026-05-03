import argparse
import csv
import json
import os
from collections import defaultdict


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# These metrics are stored as strings like "12.34%" inside scores_summary.json.
# We parse them to 12.34 and format mean back with a trailing "%".
PERCENT_METRICS = {
    "disposition_accuracy",
    "statute_precision",
    "citation_accuracy",
    "point_wise_coverage",
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
}


CANONICAL_KEY = {
    "Dispositional Accuracy": "disposition_accuracy",
    "Statute Precision": "statute_precision",
    "Citation Accuracy": "citation_accuracy",
    "Point-wise Coverage": "point_wise_coverage",
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


def parse_value(value):
    if value is None:
        return None
    if is_number(value):
        return float(value)

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.lower() in ("none", "null", "nan", "n/a"):
            return None
        if s.endswith("%"):
            try:
                return float(s[:-1].strip())
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None

    return None


def canonicalize_metric_key(key):
    if not isinstance(key, str):
        return None
    k = key.strip()
    if not k:
        return None
    return CANONICAL_KEY.get(k, k)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate mean scores by task_type from scores_summary.json files."
    )
    parser.add_argument("--root", default="results", help="Root directory to scan")
    parser.add_argument("--filename", default="scores_summary.json", help="File to aggregate")
    parser.add_argument(
        "--out",
        default="metrics_mean_by_task.csv",
        help="Output CSV path (default: metrics_mean_by_task.csv in CWD)",
    )
    parser.add_argument("--out_json", default="", help="Optional output JSON path")
    parser.add_argument("--include_all_metrics", action="store_true", help="Include all observed metrics")
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

            total_score = data.get("total_score", None)
            parsed_total = parse_value(total_score)
            if parsed_total is not None:
                sums[task]["total_score"] += parsed_total
                counts[task]["total_score"] += 1

            scores = data.get("scores", {}) or {}
            if not isinstance(scores, dict):
                continue

            for raw_k, raw_v in scores.items():
                canon_k = canonicalize_metric_key(raw_k)
                if canon_k is None:
                    continue
                parsed = parse_value(raw_v)
                if parsed is None:
                    continue
                sums[task][canon_k] += float(parsed)
                counts[task][canon_k] += 1

    rows = []
    tasks = sorted(file_count.keys())
    for task in tasks:
        metric_sums = sums.get(task, {})
        present_metrics = set(metric_sums.keys())
        default_metric_list = TASK_METRICS.get(task, OA_METRICS)

        metric_list = default_metric_list
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

            rows.append(
                {
                    "task_type": task,
                    "metric": metric,
                    "mean": mean_str,
                    "count": c,
                    "files_with_task": file_count.get(task, 0),
                }
            )

    rows.sort(key=lambda r: (r["task_type"], r["metric"]))

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task_type", "metric", "mean", "count", "files_with_task"])
        writer.writeheader()
        writer.writerows(rows)

    if args.out_json:
        payload = {
            "scanned_files": total_files,
            "out_csv": args.out,
            "by_task": rows,
        }
        out_json_dir = os.path.dirname(args.out_json)
        if out_json_dir:
            os.makedirs(out_json_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Scanned {total_files} files. Wrote {len(rows)} rows to {args.out}")
    if args.out_json:
        print(f"Wrote JSON to {args.out_json}")


if __name__ == "__main__":
    main()

