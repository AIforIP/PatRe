# fix_missing_oa_text.py
import json
from pathlib import Path
import re

BASE = Path(r"dataset\PatTimeline_A_Category\2026-02-16")
OUT_DIR = BASE / "cleaned_json_output1"

def is_oa_text_filename(filename: str) -> bool:
    name = filename.lower()
    if not name.endswith(".txt"):
        return False
    if "summary" in name:
        return False
    keywords = [
        "office_action", "office action",
        "final_rejection", "final rejection",
        "non_final", "non-final", "nonfinal",
        "rejection",
        "notice_of_allowance", "notice of allowance", "allowance",
        "advisory", "quayle"
    ]
    return any(k in name for k in keywords)

def clean_office_action(text: str) -> str:
    # 简化版清洗
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines)

def infer_oa_type_from_folder_name(folder_name: str) -> str:
    name = folder_name.lower()
    if "notice_of_allowance" in name or "notice of allowance" in name or "allowance" in name:
        return "Notice of Allowance"
    if "final_rejection" in name or "final rejection" in name or "final office action" in name:
        return "Final Rejection"
    if "non-final" in name or "non_final" in name or "nonfinal" in name:
        return "Non-Final Rejection"
    if "restriction" in name:
        return "Restriction Requirement"
    if "advisory" in name:
        return "Advisory Action"
    if "quayle" in name:
        return "Ex Parte Quayle Action"
    if "office_action" in name or "office action" in name:
        return "Office Action"
    return ""

for f in OUT_DIR.glob("Patent_*.json"):
    data = json.loads(f.read_text(encoding="utf-8"))
    patent_dir = BASE / f.stem

    oa_folders = sorted([p for p in patent_dir.iterdir() if p.is_dir() and p.name.startswith("OA")])

    changed = False
    for oa in data.get("office_actions", []):
        if oa.get("office_action_text"):
            continue

        idx = oa.get("oa_number", 0) - 1
        if idx < 0 or idx >= len(oa_folders):
            continue

        oa_folder = oa_folders[idx]
        txt_files = [p for p in oa_folder.iterdir() if is_oa_text_filename(p.name)]

        if not txt_files:
            continue

        txt_path = txt_files[0]
        text = txt_path.read_text(encoding="utf-8", errors="ignore")
        oa["office_action_text"] = clean_office_action(text)

        if not oa.get("oa_type"):
            oa["oa_type"] = infer_oa_type_from_folder_name(oa_folder.name)

        changed = True

    if changed:
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"fixed: {f.name}")
