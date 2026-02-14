# -*- coding: utf-8 -*-
import json
import os
import sys


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
    total = 0
    kept = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(in_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line or not line.strip():
                continue
            s = line.lstrip()
            if s.startswith("//"):
                continue
            obj = json.loads(line)
            refs = obj.get("input", {}).get("oracle_reference_summaries", "")
            if _has_oracle_summary(refs):
                f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
            total += 1
    print(f"Filtered {kept}/{total} samples with oracle_reference_summaries.")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/filter_dataset_oracle.py <in_jsonl> <out_jsonl>")
        sys.exit(1)
    filter_dataset(sys.argv[1], sys.argv[2])
