# -*- coding: utf-8 -*-
import os
import re
import json
from collections import OrderedDict

BASE_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\ReferencePool_Full"

FIELD_ORDER = [
    "patent_number",
    "application_number",
    "std_pid",
    "title",
    "publication_date",
    "abstract",
    "summary_of_invention",
    "full_description",
    "claims",
    "category",
    "source_file",
    "oa_index",
    "oa_number",
    "oa_date",
    "oa_type",
    "rejection_types",
    "references_cited",
    "oa_text_len",
    "match_kind",
    "match_raw",
    "match_norm",
    "match_count",
    "context",
    "provenance",
    "data_quality",
    "crawl_time",
]

DEFAULTS = {
    "patent_number": "",
    "application_number": "",
    "std_pid": "",
    "title": "",
    "publication_date": "",
    "abstract": "",
    "summary_of_invention": "",
    "full_description": "",
    "claims": [],
    "category": "",
    "source_file": "",
    "oa_index": None,
    "oa_number": None,
    "oa_date": "",
    "oa_type": "",
    "rejection_types": [],
    "references_cited": [],
    "oa_text_len": None,
    "match_kind": "",
    "match_raw": "",
    "match_norm": "",
    "match_count": 0,
    "context": "",
    "provenance": [],
    "data_quality": "",
    "crawl_time": "",
}

def standardize_pid(pid):
    if not pid:
        return ""
    p = re.sub(r"[^a-zA-Z0-9]", "", str(pid)).upper()
    if not p:
        return ""
    if p[0].isdigit():
        p = "US" + p
    return p

def extract_summary_from_full_description(text, max_len=20000):
    if not text:
        return ""
    m = re.search(r"SUMMARY(?: OF THE INVENTION)?", text, re.I)
    if not m:
        return ""
    stop_re = re.compile(r"(?:\r?\n|\r|\b)(BRIEF DESCRIPTION|DETAILED DESCRIPTION|DESCRIPTION OF THE DRAWINGS|DESCRIPTION OF EMBODIMENTS|DESCRIPTION OF THE INVENTION|BACKGROUND|FIELD|CLAIMS)\b", re.I)    
    stop = stop_re.search(rest)
    snippet = rest[:stop.start()] if stop else rest[:max_len]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet

def compute_data_quality(abstract, summary, claims):
    if abstract and summary and claims:
        return "OK"
    if abstract or summary or claims:
        return "Partial"
    return "Missing"

def normalize_record(data, filename):
    # std_pid
    std_pid = data.get("std_pid")
    if not std_pid:
        std_pid = standardize_pid(data.get("match_norm") or data.get("patent_number") or os.path.splitext(filename)[0])
    data["std_pid"] = std_pid

    # patent_number fallback
    if not data.get("patent_number"):
        data["patent_number"] = std_pid

    # summary
    summary = data.get("summary_of_invention") or ""
    if summary == "Summary Content Not Found" or not summary.strip():
        extracted = extract_summary_from_full_description(data.get("full_description", ""))
        if extracted:
            data["summary_of_invention"] = extracted

    # defaults
    for k, v in DEFAULTS.items():
        if k not in data:
            data[k] = v

    # normalize empty lists
    if data.get("claims") is None:
        data["claims"] = []
    if data.get("rejection_types") is None:
        data["rejection_types"] = []
    if data.get("references_cited") is None:
        data["references_cited"] = []
    if data.get("provenance") is None:
        data["provenance"] = []

    # data_quality
    data["data_quality"] = compute_data_quality(
        data.get("abstract", ""),
        data.get("summary_of_invention", ""),
        data.get("claims", [])
    )

    # order keys
    ordered = OrderedDict()
    for key in FIELD_ORDER:
        ordered[key] = data.get(key, DEFAULTS.get(key))

    for key in sorted(k for k in data.keys() if k not in ordered):
        ordered[key] = data[key]

    return ordered

def main():
    if not os.path.exists(BASE_DIR):
        raise SystemExit(f"ReferencePool_Full not found: {BASE_DIR}")

    files = [fn for fn in os.listdir(BASE_DIR) if fn.endswith('.json') and fn != 'crawl_status.json']
    updated = 0
    for fn in files:
        path = os.path.join(BASE_DIR, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue

        normalized = normalize_record(data, fn)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(normalized, f, ensure_ascii=False, indent=2)
        updated += 1

    print(f"Normalized {updated} files.")

if __name__ == "__main__":
    main()
