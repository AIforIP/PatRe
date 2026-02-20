import argparse
import json
import re
from pathlib import Path
from typing import Optional


def iter_json_files(cleaned_dir: Path):
    for p in sorted(cleaned_dir.glob("*.json")):
        if p.is_file():
            yield p


def _context(text: str, start: int, end: int, window: int = 80) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)
    snippet = text[left:right]
    return re.sub(r"\s+", " ", snippet).strip()


def _norm_us_pub(year: str, num: str, kind: Optional[str]):
    base = f"US-{year}{num}"
    if kind:
        return f"{base}-{kind.upper()}"
    return base


def _norm_us_pat(num: str, kind: Optional[str]):
    num = num.replace(",", "")
    base = f"US-{num}"
    if kind:
        return f"{base}-{kind.upper()}"
    return base


def _norm_generic(prefix: str, main: str, kind: Optional[str]):
    base = f"{prefix.upper()}-{main}"
    if kind:
        return f"{base}-{kind.upper()}"
    return base


US_PREFIX = r"U\.?\s*S\.?[-\s]*"
US_PAT_NUM = r"(?:\d{7,9}|\d{1,3}(?:,\d{3}){1,2})"

PATTERNS = [
    # U.S. publication number: US 2003/0080939 A1
    (
        "us_pub_slash",
        re.compile(
            rf"\b{US_PREFIX}(\d{{4}})\s*[-/]\s*(\d{{6,7}})\s*([A-Z]\d?)?\b",
            re.IGNORECASE,
        ),
        lambda m: _norm_us_pub(m.group(1), m.group(2), m.group(3)),
    ),
    # U.S. publication number without slash: US 20030080939 A1
    (
        "us_pub_noslash",
        re.compile(
            rf"\b{US_PREFIX}(\d{{10,11}})\s*([A-Z]\d?)?\b",
            re.IGNORECASE,
        ),
        lambda m: _norm_generic("US", m.group(1), m.group(2)),
    ),
    # U.S. patent number: US 7,123,456 B2 or US 7123456
    (
        "us_pat",
        re.compile(
            rf"\b{US_PREFIX}({US_PAT_NUM})\s*([A-Z]\d?)?\b",
            re.IGNORECASE,
        ),
        lambda m: _norm_us_pat(m.group(1), m.group(2)),
    ),
    # U.S. Patent No. 7,123,456
    (
        "us_pat_no",
        re.compile(
            rf"\b{US_PREFIX}Patent\s*(?:No\.|Number)?\s*({US_PAT_NUM})\b",
            re.IGNORECASE,
        ),
        lambda m: _norm_us_pat(m.group(1), None),
    ),
    # U.S. Patent Application Publication No. 2003/0080939
    (
        "us_pub_app",
        re.compile(
            rf"\b{US_PREFIX}Patent\s*Application\s*Publication\s*(?:No\.|Number)?\s*(\d{{4}})\s*[-/]\s*(\d{{6,7}})\b",
            re.IGNORECASE,
        ),
        lambda m: _norm_us_pub(m.group(1), m.group(2), None),
    ),
    # WO / EP / JP / CN / KR basic patterns
    (
        "wo_pub",
        re.compile(r"\bWO\s*(\d{4})\s*[-/]\s*(\d{6,7})\s*([A-Z]\d?)?\b", re.IGNORECASE),
        lambda m: _norm_generic("WO", f"{m.group(1)}{m.group(2)}", m.group(3)),
    ),
    (
        "ep_pat",
        re.compile(r"\bEP\s*(\d{6,8})\s*([A-Z]\d?)?\b", re.IGNORECASE),
        lambda m: _norm_generic("EP", m.group(1), m.group(2)),
    ),
    (
        "jp_pub",
        re.compile(r"\bJP\s*(\d{4})\s*[-/]\s*(\d{6,7})\b", re.IGNORECASE),
        lambda m: _norm_generic("JP", f"{m.group(1)}{m.group(2)}", None),
    ),
    (
        "cn_pat",
        re.compile(r"\bCN\s*(\d{7,9})\s*([A-Z]\d?)?\b", re.IGNORECASE),
        lambda m: _norm_generic("CN", m.group(1), m.group(2)),
    ),
    (
        "kr_pub",
        re.compile(r"\bKR\s*(\d{4})\s*[-/]\s*(\d{6,7})\b", re.IGNORECASE),
        lambda m: _norm_generic("KR", f"{m.group(1)}{m.group(2)}", None),
    ),
]


def extract_refs_from_text(text: str):
    matches = []
    for kind, pattern, normalizer in PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(0)
            norm = normalizer(m)
            matches.append(
                {
                    "match_kind": kind,
                    "match_raw": raw,
                    "match_norm": norm,
                    "start": m.start(),
                    "end": m.end(),
                    "context": _context(text, m.start(), m.end()),
                }
            )

    # De-duplicate by normalized id + raw text, keep first occurrence and count hits
    dedup = {}
    for m in matches:
        key = (m["match_norm"], m["match_raw"])
        if key not in dedup:
            m["match_count"] = 1
            dedup[key] = m
        else:
            dedup[key]["match_count"] += 1

    # Sort by first appearance in text
    return sorted(dedup.values(), key=lambda x: x["start"])


def process_dirs(input_dirs, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_oas = 0
    total_hits = 0

    with output_path.open("w", encoding="utf-8") as out_f:
        for category, cleaned_dir in input_dirs:
            for json_path in iter_json_files(cleaned_dir):
                total_files += 1
                try:
                    data = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                patent_number = data.get("patent_number")
                application_number = data.get("application_number")
                oas = data.get("office_actions") or []

                for oa_index, oa in enumerate(oas):
                    total_oas += 1
                    oa_text = oa.get("office_action_text") or ""
                    if not oa_text.strip():
                        continue

                    refs = extract_refs_from_text(oa_text)
                    if not refs:
                        continue

                    cited_list = [
                        r.get("document_number")
                        for r in (oa.get("references_cited") or [])
                        if r.get("document_number")
                    ]

                    for ref in refs:
                        total_hits += 1
                        record = {
                            "category": category,
                            "source_file": str(json_path),
                            "patent_number": patent_number,
                            "application_number": application_number,
                            "oa_index": oa_index,
                            "oa_number": oa.get("oa_number"),
                            "oa_date": oa.get("oa_date"),
                            "oa_type": oa.get("oa_type"),
                            "rejection_types": oa.get("rejection_types"),
                            "references_cited": cited_list,
                            "oa_text_len": len(oa_text),
                            "match_kind": ref["match_kind"],
                            "match_raw": ref["match_raw"],
                            "match_norm": ref["match_norm"],
                            "match_count": ref["match_count"],
                            "context": ref["context"],
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Scanned files: {total_files}")
    print(f"Scanned OA entries: {total_oas}")
    print(f"Extracted references: {total_hits}")
    print(f"Output: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract patent references from office_action_text in A/B/C datasets."
    )
    parser.add_argument(
        "--root",
        default="dataset",
        help="Dataset root folder (default: dataset).",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["A", "B", "C"],
        help="Categories to scan (default: A B C).",
    )
    parser.add_argument(
        "--output",
        default="dataset/oa_text_references.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)

    input_dirs = []
    for cat in args.categories:
        cleaned = root / f"PatTimeline_{cat}_Category" / "cleaned_json_output"
        if cleaned.exists():
            input_dirs.append((cat, cleaned))
        else:
            print(f"[WARN] Missing directory: {cleaned}")

    if not input_dirs:
        print("No input directories found. Nothing to do.")
        return

    process_dirs(input_dirs, Path(args.output))


if __name__ == "__main__":
    main()
