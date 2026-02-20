# fix_oa_type.py
import json
import re
import sys
from pathlib import Path

DEFAULT_BASE = Path(r"dataset\PatTimeline_A_Category\2026-02-16\cleaned_json_output1")


def infer_oa_type_from_text(oa_text: str) -> str:
    if not oa_text:
        return ""
    text_lower = oa_text.lower()

    # Match clean_data.py logic: allowance wins only when no active rejection exists.
    def _has_active_rejection(txt: str) -> bool:
        for seg in re.split(r"[.\n]", txt):
            if not seg:
                continue
            if "withdrawn" in seg:
                continue
            if re.search(r"\b(are|is|be|been)\s+rejected\b", seg):
                return True
            if re.search(r"\brejected\s+under\b|\brejected\s+on\s+the\s+ground\b", seg):
                return True
            if "provisionally rejected" in seg:
                return True
        return False

    active_rejection = _has_active_rejection(text_lower)

    if re.search(r"\bnotice of allowance\b|\breasons for allowance\b|\ballowable\b|\ballowed\b", text_lower) and not active_rejection:
        return "Notice of Allowance"
    if re.search(r"\bnon[-\s]?final (rejection|office action)\b", text_lower) and active_rejection:
        return "Non-Final Rejection"
    if re.search(r"\bthis action is made final\b|\bfinal office action\b|\bfinal rejection\b", text_lower) and active_rejection:
        return "Final Rejection"
    if active_rejection:
        return "Office Action"
    if "restriction requirement" in text_lower or "requirement for restriction" in text_lower or "election requirement" in text_lower:
        return "Restriction Requirement"
    if "advisory action" in text_lower:
        return "Advisory Action"
    if "ex parte quayle" in text_lower:
        return "Ex Parte Quayle Action"
    if "office action" in text_lower:
        return "Office Action"
    return "Office Action"


def main():
    bases = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else [DEFAULT_BASE]
    for base in bases:
        if not base.exists():
            print(f"Base dir not found: {base}")
            continue

        changed_files = 0
        for f in base.glob("Patent_*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            changed = False
            for oa in data.get("office_actions", []):
                oa_text = oa.get("office_action_text", "")
                new_type = infer_oa_type_from_text(oa_text)
                if new_type and new_type != oa.get("oa_type"):
                    oa["oa_type"] = new_type
                    changed = True
            if changed:
                f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                changed_files += 1
                print(f"updated: {f.name}")
        print(f"done. base={base} updated_files={changed_files}")


if __name__ == "__main__":
    main()
