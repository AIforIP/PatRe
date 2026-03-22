# -*- coding: utf-8 -*-
import json
import os
import sys
import io
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.ref_pool_dates import (
    oa_cutoff_date,
    ref_doc_effective_date,
    reference_pool_date_policy_log_line,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

RAW_DATA_ROOT = os.path.join(BASE_DIR, "dataset1")
RAW_DATA_DIRS = []
for name in sorted(os.listdir(RAW_DATA_ROOT)):
    if name.startswith("PatTimeline_") and name.endswith("_Category"):
        candidate = os.path.join(RAW_DATA_ROOT, name, "cleaned_json_output")
        if os.path.isdir(candidate):
            RAW_DATA_DIRS.append(candidate)
REF_POOL_DIR = os.path.join(BASE_DIR, "dataset", "ReferencePool_Full")
SAVE_PATH = os.path.join(BASE_DIR, "dataset", "processed", "test_set_new.json")
REF_TEXT_LIMIT = 20000  # Keep aligned with config/eval_config.yaml: reference_summaries
BM25_TOP_K = 5

# Whether to compute and store BM25 retrieved reference summaries into the
# generated test set (`test_set_new.json`).
# Default is OFF, so existing experiments that don't rely on BM25 won't pay
# the extra cost and won't persist BM25-derived references.
BM25_ENABLE = str(os.getenv("PATRE_ENABLE_BM25_ALIGNMENT", "")).strip().lower() in (
    "1",
    "true",
    "yes",
)


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


def _ref_stems_for_pool_lookup(clean_doc_id):
    """Ordered unique stems (no .json) to try under ReferencePool_Full."""
    if not clean_doc_id:
        return []
    stems = []
    seen = set()

    def add(s):
        if s and s not in seen:
            seen.add(s)
            stems.append(s)

    add(clean_doc_id)
    # Raw data e.g. APP-18_375137 -> normalize -> APP18375137; pool files stay APP-* only (no US* fallback).
    if clean_doc_id.startswith("APP") and len(clean_doc_id) > 3:
        rest = clean_doc_id[3:]
        if rest.isdigit() and len(rest) >= 8:
            add(f"APP-{rest[:2]}_{rest[2:]}")
    return stems


def _ref_possible_filenames(clean_doc_id):
    if not clean_doc_id:
        return []
    names = []
    nseen = set()

    def add_name(fn):
        if fn not in nseen:
            nseen.add(fn)
            names.append(fn)

    if clean_doc_id.startswith(("US", "WO", "EP", "AU")):
        add_name(f"{clean_doc_id}.json")
        return names

    for stem in _ref_stems_for_pool_lookup(clean_doc_id):
        add_name(f"{stem}.json")
        if stem.startswith("APP"):
            continue
        if not stem.startswith(("US", "WO", "EP", "AU")):
            add_name(f"US{stem}.json")
    return names


def _load_reference_record(doc_num):
    clean_id = _normalize_doc_id(doc_num)
    if not clean_id:
        return "", None
    for name in _ref_possible_filenames(clean_id):
        path = os.path.join(REF_POOL_DIR, name)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                return os.path.splitext(name)[0], json.load(f)
        except Exception:
            continue
    return clean_id, None


def _build_oracle_reference_summaries(ref_ids):
    if not isinstance(ref_ids, list) or not ref_ids:
        return ""
    entries = []
    for rid in ref_ids:
        doc_id, ref_data = _load_reference_record(rid)
        if ref_data:
            entries.append(_build_ref_entry(doc_id, ref_data))
        else:
            entries.append(f"Ref {doc_id}: [content unavailable]")
    text = "\n".join(entries).strip()
    if REF_TEXT_LIMIT and isinstance(text, str) and len(text) > REF_TEXT_LIMIT:
        return text[:REF_TEXT_LIMIT] + "\n[Truncated to input limit]"
    return text


def _tokenize(text):
    return [w for w in re.findall(r"[A-Za-z0-9]+", str(text).lower()) if w]


def _load_bm25_corpus():
    docs = []
    ids = []
    doc_dates = []
    ref_dir = REF_POOL_DIR
    if not ref_dir or not os.path.exists(ref_dir):
        return docs, ids, doc_dates
    for name in os.listdir(ref_dir):
        if not name.endswith(".json"):
            continue
        path = os.path.join(ref_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            title = data.get("title", "")
            abstract = data.get("abstract", "")
            summary = data.get("summary_of_invention", "")
            content = " ".join([title, abstract, summary]).strip()
            tokens = _tokenize(content)
            docs.append(tokens)
            ids.append(os.path.splitext(name)[0])
            d_doc, _ = ref_doc_effective_date(data)
            doc_dates.append(d_doc)
        except Exception:
            continue
    return docs, ids, doc_dates


def _bm25_topk(query_text, docs, ids, top_k, doc_dates=None, cutoff_date=None):
    if not docs or not ids:
        return []
    q = _tokenize(query_text)
    if not q:
        return []
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(docs)
        scores = bm25.get_scores(q)
        scored = [(float(scores[i]), i) for i in range(len(scores))]
        scored.sort(reverse=True, key=lambda x: x[0])
        if cutoff_date is not None and doc_dates:
            scored = [
                (s, i)
                for s, i in scored
                if i < len(doc_dates)
                and doc_dates[i] is not None
                and doc_dates[i] < cutoff_date
            ]
    except Exception:
        N = len(docs)
        df = {}
        for term in set(q):
            c = 0
            seen = set()
            for doc in docs:
                if term in set(doc):
                    c += 1
            df[term] = c
        k1 = 1.5
        b = 0.75
        scored = []
        avgdl = sum(len(d) for d in docs) / len(docs) if docs else 0.0
        for i, doc in enumerate(docs):
            doc_tf = {}
            for t in doc:
                doc_tf[t] = doc_tf.get(t, 0) + 1
            dl = len(doc) if doc else 1
            s = 0.0
            for term in q:
                if term not in doc_tf:
                    continue
                dfi = df.get(term, 0)
                idf = max(0.0, ((N - dfi + 0.5) / (dfi + 0.5))) if dfi > 0 else 0.0
                tf = doc_tf[term]
                denom = tf + k1 * (1 - b + b * (dl / (avgdl if avgdl else dl)))
                s += (idf) * ((tf * (k1 + 1)) / denom)
            scored.append((s, i))
        scored.sort(reverse=True, key=lambda x: x[0])
        if cutoff_date is not None and doc_dates:
            scored = [
                (s, i)
                for s, i in scored
                if i < len(doc_dates)
                and doc_dates[i] is not None
                and doc_dates[i] < cutoff_date
            ]

    out = []
    for _, i in scored[:top_k]:
        doc_id = ids[i]
        path = os.path.join(REF_POOL_DIR, f"{doc_id}.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            title = data.get("title", "")
            abstract = data.get("abstract", "")
            out.append(f"Ref {doc_id}: {title}. {abstract}")
        except Exception:
            out.append(f"Ref {doc_id}: [content unavailable]")
    return out


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

    if BM25_ENABLE:
        print(reference_pool_date_policy_log_line())
    docs, doc_ids, doc_dates = (
        _load_bm25_corpus() if BM25_ENABLE else ([], [], [])
    )

    records = []
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

                        cur_remarks = ""
                        if isinstance(oa, dict) and not is_notice_of_allowance:
                            cur_remarks = oa.get("remarks", "N/A")

                        prev_remarks = ""
                        if idx - 1 >= 0:
                            prev = office_actions[idx - 1]
                            prev_remarks = prev.get("remarks", "N/A") if isinstance(prev, dict) else ""

                        # Claims: use previous round claims as input (fallback to initial_submission.claims).
                        prev_claims = ""
                        if idx - 1 >= 0:
                            prev_oa = office_actions[idx - 1]
                            if isinstance(prev_oa, dict):
                                prev_claims = prev_oa.get("current_claims", "")
                        if not prev_claims:
                            prev_claims = item.get("initial_submission", {}).get("claims", "")
                        claims_text = process_claims(prev_claims)
                        if not claims_text.strip():
                            continue

                        # References:
                        # - For oracle/bm25 "input refs", we merge references_cited + extracted_reference_contexts.
                        # - For contrast "citations_gt" (citation-level evaluation), we only use extracted_reference_contexts.
                        refs_cited = oa.get("references_cited", []) or []
                        refs_extracted = oa.get("extracted_reference_contexts", []) or []

                        ref_ids_cited, _ = get_ref_ids(refs_cited)
                        ref_ids_extracted, _ = get_ref_ids(refs_extracted)

                        # Merge (dedupe) for oracle input.
                        oracle_ref_ids = []
                        seen = set()
                        for rid in (ref_ids_cited + ref_ids_extracted):
                            if rid in seen:
                                continue
                            seen.add(rid)
                            oracle_ref_ids.append(rid)

                        oracle_ref_summaries = _build_oracle_reference_summaries(oracle_ref_ids)
                        bm25_ref_summaries = ""
                        if BM25_ENABLE and docs and doc_ids:
                            oa_cut = (
                                oa_cutoff_date(oa.get("oa_date"))
                                if isinstance(oa, dict)
                                else None
                            )
                            if oa_cut is None and isinstance(oa, dict) and oa.get("oa_date"):
                                print(
                                    "WARNING: oa_date not parseable; BM25 without temporal pool filter.",
                                    flush=True,
                                )
                            bm25_entries = _bm25_topk(
                                claims_text,
                                docs,
                                doc_ids,
                                BM25_TOP_K,
                                doc_dates=doc_dates,
                                cutoff_date=oa_cut,
                            )
                            bm25_ref_summaries = "\n".join(bm25_entries).strip()
                        # Prefer explicit decision; else use USPTO OA kind label (oa_type).
                        decision_value = oa.get("decision") or oa.get("oa_type") or ""
                        decision_value = str(decision_value).strip()
                        if not decision_value:
                            decision_value = "Unknown"

                        if is_notice_of_allowance:
                            # Terminate at Notice of Allowance: no rebuttal round and no next OA.
                            next_oa_body = ""

                        oa_item = {
                            "id": item.get("patent_number"),
                            "meta": {
                                "ipc": item.get("ipc_category") or inferred_ipc,
                                "claim_count": _count_claims_text(claims_text),
                                "round_type": "oa",
                                "oa_type": oa_type,
                                "oa_number": oa.get("oa_number") if isinstance(oa, dict) else None,
                                "oa_date": oa.get("oa_date") if isinstance(oa, dict) else "",
                                "rejection_types": oa.get("rejection_types", []) if isinstance(oa, dict) else [],
                            },
                            "input": {
                                "claims": claims_text,
                                "oracle_reference_summaries": oracle_ref_summaries,
                                "oracle_reference_ids": oracle_ref_ids,
                                "bm25_reference_summaries": bm25_ref_summaries,
                                "rebuttal": prev_remarks if prev_remarks is not None else "",
                                "office_action": input_oa,
                                "next_office_action": next_oa_body,
                            },
                            "ground_truth": {
                                "oa_text": oa_body,
                                "rebuttal_text": prev_remarks if prev_remarks is not None else "",
                                "next_oa_text": next_oa_body,
                                # Ground-truth citations used by citation-level evaluation.
                                # Here we only keep extracted_reference_contexts references.
                                # `ref_ids_extracted` are normalized reference pool document ids (e.g. "USxxxx...").
                                "citations_gt": ref_ids_extracted,
                                "decision": decision_value,
                            },
                        }
                        records.append(oa_item)
                        count += 1

                        if not is_notice_of_allowance and cur_remarks and str(cur_remarks).strip().upper() != "N/A":
                            rebuttal_item = {
                                "id": item.get("patent_number"),
                                "meta": {
                                    "ipc": item.get("ipc_category") or inferred_ipc,
                                    "claim_count": _count_claims_text(claims_text),
                                    "round_type": "rebuttal",
                                    "oa_type": oa_type,
                                    "oa_number": oa.get("oa_number") if isinstance(oa, dict) else None,
                                    "oa_date": oa.get("oa_date") if isinstance(oa, dict) else "",
                                    "rejection_types": oa.get("rejection_types", []) if isinstance(oa, dict) else [],
                                },
                                "input": {
                                    "claims": claims_text,
                                    "oracle_reference_summaries": oracle_ref_summaries,
                                "oracle_reference_ids": oracle_ref_ids,
                                    "bm25_reference_summaries": bm25_ref_summaries,
                                    "rebuttal": cur_remarks,
                                    "office_action": input_oa,
                                    "next_office_action": next_oa_body,
                                },
                                "ground_truth": {
                                    "oa_text": oa_body,
                                    "rebuttal_text": cur_remarks,
                                    "next_oa_text": next_oa_body,
                                # Same extracted-only reference set as the aligned office action.
                                "citations_gt": ref_ids_extracted,
                                    "decision": decision_value,
                                },
                            }
                            records.append(rebuttal_item)
                            count += 1

                        if is_notice_of_allowance:
                            break

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Successfully aligned {count} data samples.")
    print(f"Saved to: {SAVE_PATH}")


if __name__ == "__main__":
    align_data()

