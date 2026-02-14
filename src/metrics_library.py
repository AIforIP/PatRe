import json
import re

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None


_STATUTE_RE = re.compile(r"(?:35\\s*U\\.S\\.C\\.?\\s*§?\\s*)?(101|102|103|112)(?:\\([a-z]\\))?", re.IGNORECASE)


def _extract_json_block(text):
    if not text:
        return None
    m = re.search(r"```json\\s*(.*?)\\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\\{.*\\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return None


def safe_parse_json(text):
    block = _extract_json_block(text)
    if not block:
        return None
    try:
        return json.loads(block)
    except Exception:
        return None


def extract_statutes(text):
    if not text:
        return set()
    return set(m.group(1) for m in _STATUTE_RE.finditer(text))


def extract_disposition(text, json_obj=None):
    if json_obj:
        status = json_obj.get("claim_status") or json_obj.get("decision")
        if isinstance(status, str):
            s = status.strip().lower()
            if "final" in s and "rejection" in s:
                return "final_rejection"
            if "non" in s and "rejection" in s:
                return "non_final_rejection"
            if "allow" in s:
                return "allowance"
            if "reject" in s:
                return "rejection"
    if not text:
        return "unknown"
    t = text.lower()
    if "final rejection" in t:
        return "final_rejection"
    if "non-final rejection" in t or "nonfinal rejection" in t:
        return "non_final_rejection"
    if "rejected under" in t or "rejection" in t:
        return "rejection"
    if "allowance" in t or "allowed" in t or "allowable" in t:
        return "allowance"
    return "unknown"


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def disposition_metrics(pred_label, gt_label):
    if pred_label == "unknown" or gt_label == "unknown":
        return {}
    accuracy = 1.0 if pred_label == gt_label else 0.0
    pred_pos = pred_label in ("final_rejection", "non_final_rejection", "rejection")
    gt_pos = gt_label in ("final_rejection", "non_final_rejection", "rejection")
    tp = 1 if pred_pos and gt_pos else 0
    fp = 1 if pred_pos and not gt_pos else 0
    fn = 1 if (not pred_pos) and gt_pos else 0
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "disposition_accuracy": round(accuracy, 4),
        "disposition_precision": round(precision, 4),
        "disposition_recall": round(recall, 4),
        "disposition_f1": round(f1, 4),
    }


def statute_metrics(pred_set, gt_set):
    if not pred_set and not gt_set:
        return {}
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "statute_precision": round(precision, 4),
        "statute_recall": round(recall, 4),
        "statute_f1": round(f1, 4),
    }


_REF_RE = re.compile(
    r"(US\\s*\\d{4}/?\\d{6,8}\\s*[A-Z]\\d?|US\\s*\\d{7,}\\s*[A-Z]\\d?|US\\s*\\d{4}/?\\d{6,8}|US\\s*\\d{7,}|WO\\s*\\d{4}\\d+\\s*[A-Z]\\d?|WO\\s*\\d{4}\\d+|EP\\s*\\d{7,}\\s*[A-Z]\\d?|EP\\s*\\d{7,})",
    re.IGNORECASE
)


def normalize_ref_id(text):
    """
    Normalize reference IDs to a stable form like 'US20210267678'.
    Keeps prefix (US/WO/EP) + digits only, strips kind codes (A1/B2).
    """
    if not text:
        return None
    raw = re.sub(r"\\s+", "", str(text).upper())
    m = re.match(r"^(US|WO|EP)(.+)$", raw)
    if not m:
        return raw
    prefix, rest = m.group(1), m.group(2)
    # Remove kind codes like A1/B2 (letters + optional digit at end)
    rest = re.sub(r"[A-Z]\\d?$", "", rest)
    # Remove any non-digits (slashes, commas)
    digits = re.sub(r"\\D", "", rest)
    return f"{prefix}{digits}" if digits else raw


def extract_reference_ids(text):
    if not text:
        return set()
    ids = set()
    for m in _REF_RE.finditer(text):
        norm = normalize_ref_id(m.group(1))
        if norm:
            ids.add(norm)
    return ids


def citation_metrics(pred_text, refs_text, gt_ids=None):
    if gt_ids is not None:
        ref_ids = set(filter(None, (normalize_ref_id(x) for x in gt_ids)))
    else:
        ref_ids = extract_reference_ids(refs_text)
    pred_ids = extract_reference_ids(pred_text)
    if not ref_ids:
        return {}, ref_ids, pred_ids
    if not pred_ids:
        return {
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "citation_f1": 0.0,
            "citation_accuracy": 0.0,
        }, ref_ids, pred_ids
    tp = len(pred_ids & ref_ids)
    fp = len(pred_ids - ref_ids)
    fn = len(ref_ids - pred_ids)
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "citation_precision": round(precision, 4),
        "citation_recall": round(recall, 4),
        "citation_f1": round(f1, 4),
        "citation_accuracy": round(precision * 10, 4),
    }, ref_ids, pred_ids


def citation_list_match(pred_text, refs_text):
    """
    Check whether predicted citation IDs are in the provided reference list.
    Returns precision in [0,1].
    """
    pred_ids = extract_reference_ids(pred_text)
    ref_ids = extract_reference_ids(refs_text)
    if not pred_ids:
        return 1.0
    if not ref_ids:
        return 0.0
    precision = len(pred_ids & ref_ids) / len(pred_ids)
    return round(precision, 4)


def rouge_l_f1(pred_text, gt_text):
    if not rouge_scorer or not pred_text or not gt_text:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(gt_text, pred_text)["rougeL"].fmeasure
    return round(score, 4)


def bleu_score(pred_text, gt_text):
    if not pred_text or not gt_text:
        return None
    try:
        import sacrebleu
        return round(sacrebleu.sentence_bleu(pred_text, [gt_text]).score / 100.0, 4)
    except Exception:
        return None


def bert_score_f1(pred_text, gt_text):
    if not pred_text or not gt_text:
        return None
    try:
        from bert_score import score as bert_score
        _, _, f1 = bert_score([pred_text], [gt_text], lang="en", verbose=False)
        return round(float(f1[0]), 4)
    except Exception:
        return None


def embed_cosine(pred_text, gt_text):
    if not pred_text or not gt_text:
        return None
    try:
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_a = model.encode(pred_text, convert_to_tensor=True)
        emb_b = model.encode(gt_text, convert_to_tensor=True)
        return round(float(util.cos_sim(emb_a, emb_b).item()), 4)
    except Exception:
        return None


def formatting_score(text, task_type):
    if not text:
        return None
    t = text.lower()
    required = []
    if task_type in ("v1_initial_examination", "v1_response_to_rebuttal"):
        required = ["detailed action", "conclusion", "claim rejections"]
    elif "rebuttal" in task_type:
        required = ["response to office action", "amendment and remarks", "conclusion", "refined claims"]
    if not required:
        return None
    hits = sum(1 for h in required if h in t)
    return round(hits / len(required), 4)
