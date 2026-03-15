# -*- coding: utf-8 -*-
import json
import os
import sys
import io
import re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_ROOT = os.path.join(BASE_DIR, "dataset")
RAW_DATA_DIRS = []
for name in sorted(os.listdir(RAW_DATA_ROOT)):
    if name.startswith("PatTimeline_") and name.endswith("_Category"):
        candidate = os.path.join(RAW_DATA_ROOT, name, "cleaned_json_output")
        if os.path.isdir(candidate):
            RAW_DATA_DIRS.append(candidate)
REF_POOL_DIR = os.path.join(BASE_DIR, "dataset", "ReferencePool_Full")
SAVE_PATH = os.path.join(BASE_DIR, "dataset", "processed", "test_set_new.jsonl")
REF_TEXT_LIMIT = 20000  # Keep aligned with config/eval_config.yaml: reference_summaries



def _normalize_ref_list(ref_list):
    if not ref_list:
        return []
    normalized = []
    for ref in ref_list:
        if not isinstance(ref, dict):
            continue
        if "document_number" in ref:
            normalized.append(ref)
        elif "doc_number" in ref:
            normalized.append({"document_number": ref.get("doc_number", "")})
    return normalized


def _is_independent_claim(claim_text):
    if not claim_text:
        return False
    m = re.match(r"^\s*(\d+)\s*[\.\)]\s*", claim_text)
    claim_num = m.group(1) if m else None
    for ref_num in re.findall(r"\bclaim\s+(\d+)\b", claim_text, flags=re.IGNORECASE):
        if claim_num is None or ref_num != claim_num:
            return False
    return True


def _format_claims(claims_list, independent_only=False):
    if not claims_list:
        return ""
    lines = []
    for c in claims_list:
        if isinstance(c, dict):
            text = c.get("text", "")
        else:
            text = str(c)
        text = text.strip()
        if not text:
            continue
        if independent_only and not _is_independent_claim(text):
            continue
        lines.append(text)
    return "\n".join(lines)


def _build_ref_entry(doc_num, ref_data, independent_only=False):
    claims_text = _format_claims(ref_data.get("claims", []), independent_only=independent_only)
    if claims_text:
        return f"Ref {doc_num}: Claims:\n{claims_text}"
    abstract_text = ref_data.get("abstract", "No abstract available")
    return f"Ref {doc_num}: {abstract_text}"


def _normalize_doc_id(doc_num):
    if not doc_num:
        return ""
    s = str(doc_num).upper().strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def _ref_possible_filenames(clean_doc_id):
    if not clean_doc_id:
        return []
    if clean_doc_id.startswith(("US", "WO", "EP", "AU")):
        return [f"{clean_doc_id}.json"]
    return [f"{clean_doc_id}.json", f"US{clean_doc_id}.json"]


def get_ref_ids(ref_list):
    ref_list = _normalize_ref_list(ref_list)
    if not ref_list:
        return [], True

    ids = []
    any_missing = False
    for ref in ref_list:
        raw = ref.get("document_number", "")
        clean_id = _normalize_doc_id(raw)
        if not clean_id:
            any_missing = True
            continue
        ids.append(clean_id)

        found = False
        for name in _ref_possible_filenames(clean_id):
            path = os.path.join(REF_POOL_DIR, name)
            if os.path.exists(path):
                found = True
                break
        if not found:
            any_missing = True

    return ids, any_missing


def infer_ipc_from_dir(raw_dir):
    parent = os.path.basename(os.path.dirname(raw_dir))
    if parent.startswith("PatTimeline_") and parent.endswith("_Category"):
        mid = parent[len("PatTimeline_"):-len("_Category")]
        if mid:
            return mid.upper()
    return "Unknown"


def process_claims(claims):

    if isinstance(claims, list):
        processed = []
        for c in claims:
            if isinstance(c, dict):
                processed.append(c.get("text", ""))
            else:
                processed.append(str(c))
        return "\n".join(processed)
    return str(claims) if claims else ""


def _count_claims_text(claims_text):
    if not claims_text:
        return 0
    import re
    # Match claim numbers followed by optional backslash + dot, or parenthesis
    # Handles both "1." and "1\." formats
    hits = re.findall(r"(?m)^\s*(\d+)\s*(?:\\?\.|\))", claims_text)
    return len(hits) if hits else 1




def align_data():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    count = 0

    if not RAW_DATA_DIRS:
        print("WARNING: No raw data directories found under dataset.")
        return

    with open(SAVE_PATH, "w", encoding="utf-8") as outfile:
        for raw_dir in RAW_DATA_DIRS:
            inferred_ipc = infer_ipc_from_dir(raw_dir)
            for filename in os.listdir(raw_dir):
                if not filename.endswith(".json"):
                    continue

                with open(os.path.join(raw_dir, filename), "r", encoding="utf-8") as f:
                    try:
                        item = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON file: {filename}")
                        continue

                    office_actions = item.get("office_actions", [])
                    if not isinstance(office_actions, list):
                        office_actions = []
                    # Ensure deterministic round order by oa_number (fallback to original order).
                    if isinstance(office_actions, list) and office_actions:
                        def _oa_sort_key(oa_obj, idx):
                            if not isinstance(oa_obj, dict):
                                return (1, idx)
                            num = oa_obj.get("oa_number")
                            try:
                                return (0, int(num))
                            except Exception:
                                return (1, idx)
                        office_actions = [
                            oa for _, oa in sorted(
                                enumerate(office_actions),
                                key=lambda pair: _oa_sort_key(pair[1], pair[0])
                            )
                        ]
                    for idx, oa in enumerate(office_actions):
                        oa_body = oa.get("office_action_text", "")
                        if not oa_body or len(oa_body) < 100:
                            continue

                        next_oa_body = ""
                        if idx + 1 < len(office_actions):
                            nxt = office_actions[idx + 1]
                            next_oa_body = nxt.get("office_action_text", "")

                        input_oa = oa_body

                        oa_type = ""
                        if isinstance(oa, dict):
                            oa_type = str(oa.get("oa_type", "")).strip()
                        is_notice_of_allowance = "notice of allowance" in oa_type.lower()

                        # Use rebuttal from the *current* OA unless this is Notice of Allowance.
                        # This aligns with: OA#1 -> remarks (response) -> OA#2.
                        cur_remarks = ""
                        if isinstance(oa, dict) and not is_notice_of_allowance:
                            cur_remarks = oa.get("remarks", "N/A")

                        # Claims: prefer OA current_claims; fallback to initial_submission.claims.
                        raw_claims = oa.get("current_claims", "")
                        if not raw_claims:
                            raw_claims = item.get("initial_submission", {}).get("claims", "")
                        claims_text = process_claims(raw_claims)
                        if not claims_text.strip():
                            continue

                        # References: prefer references_cited; fallback to extracted_reference_contexts.
                        refs = oa.get("references_cited", [])
                        if not refs:
                            refs = oa.get("extracted_reference_contexts", [])
                        ref_ids, ref_missing = get_ref_ids(refs)
                        decision_value = oa.get("decision", "")
                        decision_value = str(decision_value).strip()
                        decision_missing = not bool(decision_value)
                        if decision_missing:
                            decision_value = "Unknown"

                        if is_notice_of_allowance:
                            # Terminate at Notice of Allowance: no remarks, no next OA.
                            next_oa_body = ""

                        formatted_item = {
                            "id": item.get("patent_number"),
                            "meta": {
                                "ipc": item.get("ipc_category") or inferred_ipc,
                                "claim_count": _count_claims_text(claims_text),
                                "round_type": "rebuttal" if next_oa_body else "examination",
                                "reference_missing": ref_missing,
                                "decision_missing": decision_missing,
                                "oa_type": oa_type,
                            },
                            "input": {
                                "claims": claims_text,
                                "oracle_reference_summaries": "",
                                "oracle_reference_ids": ref_ids,
                                "rebuttal": cur_remarks if cur_remarks is not None else "",
                                "office_action": input_oa,
                                "next_office_action": next_oa_body,
                            },
                            "ground_truth": {
                                "oa_text": oa_body,
                                "rebuttal_text": cur_remarks if cur_remarks is not None else "",
                                "next_oa_text": next_oa_body,
                                "decision": decision_value,
                            },
                        }
                        outfile.write(json.dumps(formatted_item, ensure_ascii=False) + "\n")
                        count += 1

                        if is_notice_of_allowance:
                            break

    print(f"Successfully aligned {count} data samples.")
    print(f"Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    align_data()

