import json
import os
import sys

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
from src.utils import load_dataset_json


def _has_oracle_summary(text):
    if not isinstance(text, str):
        return False
    t = text.strip()
    if not t:
        return False
    bad_markers = [
        "no prior art cited.",
        "warning: oracle summary data is missing",
        "[summary not found",
        "references corpus not available.",
    ]
    tl = t.lower()
    return not any(m in tl for m in bad_markers)


def filter_dataset(in_path, out_path):
    records = load_dataset_json(in_path)
    total = len(records)
    kept_records = []
    for obj in records:
        refs = obj.get("input", {}).get("oracle_reference_summaries", "")
        if _has_oracle_summary(refs):
            kept_records.append(obj)
    kept = len(kept_records)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if out_path.lower().endswith(".json"):
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(kept_records, f_out, ensure_ascii=False, indent=2)
            f_out.write("\n")
    else:
        with open(out_path, "w", encoding="utf-8") as f_out:
            for obj in kept_records:
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Filtered {kept}/{total} samples with oracle_reference_summaries.")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/filter_dataset_oracle.py <in.json|.jsonl> <out.json|.jsonl>")
        sys.exit(1)
    filter_dataset(sys.argv[1], sys.argv[2])
