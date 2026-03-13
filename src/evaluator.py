# -*- coding: utf-8 -*-
import json
import os
import re
from tqdm import tqdm
from src.model_backend import ModelBackend
from src.metrics_library import (
    safe_parse_json,
    extract_statutes,
    extract_disposition,
    disposition_metrics,
    statute_metrics,
    citation_metrics,
    citation_list_match,
    rouge_l_f1,
    bleu_score,
    bert_score_f1,
    formatting_score,
    language_style_score,
)

class PatentEvaluator:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.backend = ModelBackend(config)
        self.judge_backend = ModelBackend(config, is_judge=True)
        self.logger.info(f"judge_base_url={getattr(self.judge_backend, '_base_url', '')}")
        self._corpus_loaded = False
        self._corpus_docs = []
        self._avgdl = 0.0
        self._doclens = []
        self._docids = []
        self._bm25_index = None
        
        with open(config['paths']['prompt_path'], 'r', encoding='utf-8') as f:
            self.templates = json.load(f)
        self._patch_eval_templates()
            
        self.dataset = self._load_dataset(config['paths']['test_set'])

    def _load_dataset(self, path):
        data = []
        ipc_filter = self.config.get('task_settings', {}).get('ipc_filter')
        if ipc_filter:
            ipc_filter = str(ipc_filter).upper()
        if not os.path.exists(path):
            self.logger.error(f"Test set path not found: {path}")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line or not line.strip():
                    continue
                s = line.lstrip()
                if s.startswith("//"):
                    continue
                item = json.loads(line)
                if ipc_filter:
                    ipc_value = str(item.get("meta", {}).get("ipc", "")).upper()
                    if ipc_value != ipc_filter:
                        continue
                data.append(item)
        # Attach prosecution history for single-task runs (use dataset order per patent)
        history_blocks = {}
        for item in data:
            pid = item.get("id")
            if not pid:
                continue
            item.setdefault("input", {})
            if not item["input"].get("prosecution_history"):
                blocks = history_blocks.get(pid, [])
                item["input"]["prosecution_history"] = "\n\n".join(blocks) if blocks else "N/A"

            claims, office_action, remarks = self._extract_round_texts(item)
            if claims or office_action or remarks:
                block = (
                    "Round {idx} (historical):\n"
                    "Claims:\n{claims}\n"
                    "Office Action:\n{oa}\n"
                    "Remarks/Rebuttal:\n{remarks}"
                ).format(
                    idx=len(history_blocks.get(pid, [])) + 1,
                    claims=claims or "[MISSING]",
                    oa=office_action or "[MISSING]",
                    remarks=remarks or "[MISSING]"
                )
                history_blocks.setdefault(pid, []).append(block)
        # Task-aware filtering (single dataset, avoid empty rebuttals)
        task_key = self.config.get('task_settings', {}).get('current_task', '')
        if "rebuttal" in task_key:
            filtered = []
            for item in data:
                rebut = item.get("input", {}).get("rebuttal", "")
                if isinstance(rebut, str):
                    rebut = rebut.strip()
                if rebut and rebut.upper() != "N/A":
                    filtered.append(item)
            data = filtered
        #  config 
        limit = self.config['task_settings'].get('num_samples', 9999)
        return data[:limit]

    def _patch_eval_templates(self):
        eval_metrics = self.templates.get("Evaluation_Metrics", {})
        if not isinstance(eval_metrics, dict):
            return

        def patch_scores_block(tmpl, keys):
            if not isinstance(tmpl, str):
                return tmpl
            m = re.search(r'("scores":\s*\{\n)([\s\S]*?)(\n\s*\}\s*,\s*\n\s*"justification")', tmpl)
            if not m:
                return tmpl
            head, body, tail = m.group(1), m.group(2), m.group(3)
            lines = body.splitlines(True)

            existing = set()
            for ln in lines:
                mm = re.search(r'"\s*([A-Za-z0-9_]+)\s*"\s*:', ln)
                if mm:
                    existing.add(mm.group(1))

            missing = [k for k in keys if k not in existing]
            if not missing:
                return tmpl

            last_idx = None
            for i in range(len(lines) - 1, -1, -1):
                if ":" in lines[i]:
                    last_idx = i
                    break
            if last_idx is None:
                return tmpl

            if not lines[last_idx].rstrip().endswith(","):
                lines[last_idx] = lines[last_idx].rstrip("\n") + ",\n"

            for i, k in enumerate(missing):
                comma = "," if i < len(missing) - 1 else ""
                lines.append(f'    "{k}": 0{comma}\n')

            new_body = "".join(lines)
            return tmpl[:m.start()] + head + new_body + tail + tmpl[m.end():]

        cfg = eval_metrics.get("rebuttal_judge")
        if isinstance(cfg, dict) and isinstance(cfg.get("user_prompt_template"), str):
            cfg["user_prompt_template"] = patch_scores_block(
                cfg["user_prompt_template"],
                ["constructiveness", "language_style", "point_wise_coverage"],
            )

        cfg = eval_metrics.get("examiner_judge")
        if isinstance(cfg, dict) and isinstance(cfg.get("user_prompt_template"), str):
            cfg["user_prompt_template"] = patch_scores_block(
                cfg["user_prompt_template"],
                ["language_style"],
            )

    def _get_nested_text(self, item, *keys):
        cur = item
        for k in keys:
            if not isinstance(cur, dict):
                return ""
            cur = cur.get(k)
        return cur if isinstance(cur, str) else ""

    def _extract_round_texts(self, item):
        claims = self._get_nested_text(item, "input", "claims") or self._get_nested_text(item, "ground_truth", "claims")
        office_action = (
            self._get_nested_text(item, "input", "office_action")
            or self._get_nested_text(item, "ground_truth", "oa_text")
            or self._get_nested_text(item, "ground_truth", "next_oa_text")
        )
        remarks = (
            self._get_nested_text(item, "input", "rebuttal")
            or self._get_nested_text(item, "ground_truth", "rebuttal_text")
        )
        return claims, office_action, remarks

    def _split_claims(self, claims_text):
        if not isinstance(claims_text, str):
            return []
        text = claims_text.strip()
        if not text or text.upper() == "N/A":
            return []
        # Prefer numbered claim boundaries (e.g., "1. ...", "2) ...").
        pattern = re.compile("(?m)^([0-9]+)\\s*[\\.|\\)]\\s+(.*?)(?=^[0-9]+\\s*[\\.|\\)]\\s+|\\Z)", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            return [(num.strip(), body.strip()) for num, body in matches if body.strip()]
        # Fallback: treat entire block as a single claim.
        return [("1", text)]

    def _resolve_template_task(self, task_key, item):
        if task_key != "oa":
            return task_key
        # For OA generation, decide prompt type based ONLY on current input rebuttal.
        # Ground-truth rebuttal is for evaluation, and should not force a response-to-rebuttal prompt.
        rebuttal_text = item.get("input", {}).get("rebuttal") or ""
        if isinstance(rebuttal_text, str):
            rebuttal_text = rebuttal_text.strip()

        prompt_variant = (
            self.config.get("active_settings", {}).get("oa_prompt_variant")
            or self.config.get("task_settings", {}).get("oa_prompt_variant")
            or "legacy"
        )

        if not rebuttal_text or rebuttal_text.upper() == "N/A":
            if prompt_variant == "retrieval_aligned":
                retrieval_setting = self.config.get("active_settings", {}).get("retrieval_setting", "oracle_summary")
                mapping = {
                    "no_ref": "initial_examination_zero_shot",
                    "oracle_summary": "initial_examination_oracle",
                    "bm25_retrieved": "initial_examination_bm25",
                }
                return mapping.get(retrieval_setting, "initial_examination")
            return "initial_examination"

        if prompt_variant == "retrieval_aligned":
            retrieval_setting = self.config.get("active_settings", {}).get("retrieval_setting", "oracle_summary")
            mapping = {
                "no_ref": "response_to_rebuttal_zero_shot",
                "oracle_summary": "response_to_rebuttal_oracle",
                "bm25_retrieved": "response_to_rebuttal_bm25",
            }
            return mapping.get(retrieval_setting, "response_to_rebuttal")
        return "response_to_rebuttal"
    def run_batch(self):
        results = []
        task_key = self.config['task_settings']['current_task']
        granularity = self.config.get('active_settings', {}).get('granularity', "full_text")
        

        if task_key in ("oa",) or task_key in self.templates.get("Office_Action_Generation", {}):
            root_key = "Office_Action_Generation"
        elif task_key in self.templates.get("Rebuttal_Generation", {}):
            root_key = "Rebuttal_Generation"
        else:
            self.logger.error(f"Unknown task key: {task_key}")
            return []
        
        self.logger.info(f" Starting batch inference for {len(self.dataset)} samples...")

        for item in tqdm(self.dataset, desc="Generating"):
            template_task = self._resolve_template_task(task_key, item)
            if template_task not in self.templates.get(root_key, {}):
                self.logger.error(f"Template not found in {root_key}: {template_task}")
                return []
            template_config = self.templates[root_key][template_task]

            # 1.
            ref_summary = self._process_retrieval(item)
            
            # 2. User Prompt input
            # Always use real dataset inputs (OA and rebuttal), never generated outputs.
            oa_content = item['input'].get('office_action', "No Office Action provided in dataset.")
            rebuttal_text = item['input'].get('rebuttal', item.get('ground_truth', {}).get('rebuttal_text', "N/A"))
            
            enforce_instruction = "\n\nIMPORTANT: Use the provided Reference Summaries to cite specific documents. DO NOT use placeholders like '[insert references]'."
            
            claim_blocks = [("all", item['input'].get('claims', ""))]
            if granularity == "atomic_claim":
                split_claims = self._split_claims(item['input'].get('claims', ""))
                if split_claims:
                    claim_blocks = split_claims
            
            for claim_no, claim_text in claim_blocks:
                # AI
                ai_input = {
                    "claims": f"Claim {claim_no}: {claim_text}" if granularity == "atomic_claim" else item['input']['claims'],
                    "refs": ref_summary,
                    "task_type": task_key,
                    "retrieval_setting": self.config['active_settings'].get('retrieval_setting', "oracle_summary"),
                    "prosecution_history": item['input'].get('prosecution_history', "N/A"),
                    "granularity": granularity
                }
                
                # AI
                if task_key == "oa":
                    # Avoid leaking ground-truth OA into OA generation.
                    ai_input.pop("office_action", None)
                
                try:
                    # AIprompt
                    template_text = template_config["user_prompt_template"]
                    user_prompt = template_text.format(
                        current_claims=ai_input['claims'],
                        reference_summaries=ai_input['refs'],
                        office_action=oa_content if template_task != "initial_examination" else "N/A",
                        rebuttal=rebuttal_text if template_task != "initial_examination" else "N/A",
                        claim_count=1 if granularity == "atomic_claim" else item['meta'].get('claim_count', 1),
                        prosecution_history=ai_input.get("prosecution_history", "N/A")
                    )
                    if "{reference_summaries}" in template_text:
                        user_prompt += enforce_instruction
                except KeyError as e:
                    self.logger.error(f"Prompt  {e}")
                    raise e

                # 3.
                prediction = self.backend.generate(
                    system_prompt=template_config["system_prompt"],
                    user_prompt=user_prompt
                )

                #  Ground Truth
                response_templates = {
                    "response_to_rebuttal",
                    "response_to_rebuttal_zero_shot",
                    "response_to_rebuttal_oracle",
                    "response_to_rebuttal_bm25",
                }
                if task_key == "oa" and template_task in response_templates:
                    ground_truth_text = item['ground_truth'].get('next_oa_text', item['ground_truth'].get('oa_text', "N/A"))
                elif task_key == "rebuttal":
                    ground_truth_text = item['ground_truth'].get('rebuttal_text', "N/A")
                else:
                    ground_truth_text = item['ground_truth'].get('oa_text', "N/A")

                gt_citations = item.get("ground_truth", {}).get("citations_gt", [])

                result_entry = {
                    "patent_id": item['id'],
                    "round_info": {
                        "task": task_key,
                        "round_number": self._get_round_number(task_key),
                        "exam_phase": "pure_prediction" if template_task == "initial_examination" else "context_aware",
                        "granularity": granularity,
                        "claim_number": claim_no if granularity == "atomic_claim" else "all"
                    },
                    "ai_exam": {
                        "input": ai_input,
                        "output": prediction,
                        "available_info": list(ai_input.keys()),
                        "restricted_info": self._get_restricted_info(task_key)
                    },
                    "evaluation_context": {
                        "full_context": {
                            "claims": item['input']['claims'],
                            "refs": ref_summary,
                            "office_action": oa_content,
                            "rebuttal_history": item['input'].get('rebuttal', "N/A")
                        },
                        "ground_truth": ground_truth_text,
                        "citations_gt": gt_citations,
                        "input_refs_preview": ref_summary[:200] + "..."  # 
                    }
                }
                results.append(result_entry)
        return results

    def _get_round_number(self, task_key):

        if task_key == "oa":
            return 1
        elif task_key == "rebuttal":
            return 2
        else:
            return 0

    def _get_restricted_info(self, task_key):
        """AI"""
        if task_key == "oa":
            return ["ground_truth_oa", "rebuttal_history", "human_decisions"]
        else:
            return ["ground_truth_final_outcome"]

    def _truncate_text(self, text, max_chars, head_ratio=0.6):
        if not text or not max_chars or max_chars <= 0:
            return text
        if len(text) <= max_chars:
            return text
        marker = "\n...[TRUNCATED]...\n"
        head = int(max_chars * head_ratio)
        tail = max_chars - head - len(marker)
        if tail <= 0:
            return text[:max_chars]
        return text[:head] + marker + text[-tail:]

    def _safe_template_format(self, template, fields):
        if not isinstance(template, str):
            return ""
        if not isinstance(fields, dict):
            fields = {}

        def repl(m):
            key = m.group(1)
            val = fields.get(key)
            if val is None:
                return m.group(0)
            return str(val)

        return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, template)

    def _shrink_prompt(self, template, fields, max_total):
        prompt = self._safe_template_format(template, fields)
        if len(prompt) <= max_total:
            return prompt
        shrink_keys = [
            "generated_text",
            "ground_truth",
            "office_action",
            "claims",
            "reference_summaries",
            "text_a",
            "text_b",
            "text",
            "claim_text",
        ]
        for key in shrink_keys:
            if len(prompt) <= max_total:
                break
            value = fields.get(key)
            if isinstance(value, str) and len(value) > 500:
                overflow = len(prompt) - max_total
                new_limit = max(500, len(value) - overflow - 50)
                fields[key] = self._truncate_text(value, new_limit)
                prompt = self._safe_template_format(template, fields)
        return prompt

    def _build_judge_prompt(self, template, fields):
        cfg = self.config.get("task_settings", {})
        max_total = cfg.get("judge_max_chars", 12000)
        default_limit = cfg.get("judge_default_field_max_chars", 4000)
        per_field = cfg.get("judge_field_max_chars", {}) or {}

        truncated = {}
        for k, v in fields.items():
            if isinstance(v, str):
                limit = per_field.get(k, default_limit)
                truncated[k] = self._truncate_text(v, limit)
            else:
                truncated[k] = v

        prompt = self._safe_template_format(template, truncated)
        if max_total and len(prompt) > max_total:
            prompt = self._shrink_prompt(template, truncated, max_total)
        return prompt

    def _split_reference_summaries(self, text):
        if not isinstance(text, str):
            return []
        lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
        if lines:
            return lines
        return [text.strip()] if text.strip() else []

    def _split_ref_blocks(self, text):
        if not isinstance(text, str):
            return []
        lines = text.splitlines()
        blocks = []
        buf = []
        for ln in lines:
            if not ln or not ln.strip():
                continue
            if re.match(r"^\s*Ref\s+[A-Za-z0-9]+:", ln):
                if buf:
                    blocks.append("\n".join(buf).strip())
                    buf = []
            buf.append(ln.rstrip())
        if buf:
            blocks.append("\n".join(buf).strip())
        return [b for b in blocks if b]

    def _extract_ref_id(self, text):
        if not isinstance(text, str):
            return ""
        m = re.search(r"Ref\\s+([A-Za-z0-9]+):", text)
        if m:
            return m.group(1).strip()
        # Try common publication/patent ID patterns.
        m = re.search(r"\\b[A-Z]{2}\\s?\\d{4}/\\d{7}\\s?[A-Z]\\d?\\b", text)
        if m:
            return re.sub(r"\\s+", "", m.group(0))
        m = re.search(r"\\b[A-Z]{2}\\d{7,}[A-Z]?\\d?\\b", text)
        if m:
            return m.group(0)
        return ""

    def _dedupe_refs(self, refs):
        seen = set()
        out = []
        for r in refs:
            if not isinstance(r, str):
                continue
            rid = self._extract_ref_id(r)
            key = rid if rid else r.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def _get_ref_text_limit(self):
        cfg = self.config.get("task_settings", {}) or {}
        per_field = cfg.get("judge_field_max_chars", {}) or {}
        limit = per_field.get("reference_summaries")
        try:
            limit = int(limit)
        except Exception:
            limit = 20000
        return max(2000, limit)

    def _normalize_doc_id(self, doc_num):
        if not doc_num:
            return ""
        s = str(doc_num).upper().strip()
        s = re.sub(r"[^A-Z0-9]", "", s)
        return s

    def _ref_id_variants(self, clean_id):
        if not clean_id:
            return []
        base = clean_id
        base2 = re.sub(r"[A-Z]\d?$", "", base)
        candidates = {base, base2}
        for kind in ("A1", "A2", "B1", "B2"):
            candidates.add(f"{base2}{kind}")
        return [c for c in candidates if c]

    def _ref_possible_names(self, clean_id):
        names = []
        for cid in self._ref_id_variants(clean_id):
            if cid.startswith(("US", "WO", "EP", "AU")):
                names.append(f"{cid}.json")
            else:
                names.append(f"{cid}.json")
                names.append(f"US{cid}.json")
        seen = set()
        out = []
        for n in names:
            if n in seen:
                continue
            seen.add(n)
            out.append(n)
        return out

    def _load_reference_record(self, doc_num):
        ref_dir = self.config.get("paths", {}).get("reference_pool")
        if not ref_dir or not os.path.exists(ref_dir):
            return "", None
        clean_id = self._normalize_doc_id(doc_num)
        if not clean_id:
            return "", None
        for name in self._ref_possible_names(clean_id):
            path = os.path.join(ref_dir, name)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return os.path.splitext(name)[0], json.load(f)
            except Exception:
                continue
        return clean_id, None

    def _is_independent_claim(self, claim_text):
        if not claim_text:
            return False
        m = re.match(r"^\s*(\d+)\s*[\.\)]\s*", claim_text)
        claim_num = m.group(1) if m else None
        for ref_num in re.findall(r"\bclaim\s+(\d+)\b", claim_text, flags=re.IGNORECASE):
            if claim_num is None or ref_num != claim_num:
                return False
        return True

    def _format_ref_claims(self, claims_list, independent_only=False):
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
            if independent_only and not self._is_independent_claim(text):
                continue
            lines.append(text)
        return "\n".join(lines)

    def _build_oracle_ref_entry(self, doc_id, ref_data, independent_only=False):
        if not isinstance(ref_data, dict):
            return f"Ref {doc_id}: [content unavailable]"
        claims_text = self._format_ref_claims(ref_data.get("claims", []), independent_only=independent_only)
        if claims_text:
            return f"Ref {doc_id}: Claims:\n{claims_text}"
        title = (ref_data.get("title") or "").strip()
        abstract = (ref_data.get("abstract") or "No abstract available").strip()
        if title:
            return f"Ref {doc_id}: {title}. {abstract}"
        return f"Ref {doc_id}: {abstract}"

    def _build_oracle_ref_abstract_line(self, doc_id, ref_data):
        if not isinstance(ref_data, dict):
            return f"Ref {doc_id}: [content unavailable]"
        title = (ref_data.get("title") or "").strip()
        abstract = (ref_data.get("abstract") or "No abstract available").strip()
        if title:
            return f"Ref {doc_id}: {title}. {abstract}"
        return f"Ref {doc_id}: {abstract}"

    def _process_retrieval(self, item):
        setting = self.config['active_settings'].get('retrieval_setting', "oracle_summary")
        
        if setting == "no_ref":
            return "No prior art references provided for this examination."
        
        elif setting == "oracle_summary":
            limit = self._get_ref_text_limit()
            content = item.get("input", {}).get('oracle_reference_summaries', "")
            if isinstance(content, str) and content.strip():
                content = content.strip()
                if len(content) > limit:
                    return content[:limit] + "\n[Truncated to input limit]"
                return content

            ids = item.get("input", {}).get("oracle_reference_ids", [])
            if not isinstance(ids, list) or not ids:
                return "Warning: Oracle reference IDs are missing in test set."
            entries = []
            for rid in ids:
                doc_id, ref_data = self._load_reference_record(rid)
                entries.append(self._build_oracle_ref_entry(doc_id, ref_data, independent_only=False))
            full_text = "\n".join(entries).strip()
            if len(full_text) <= limit:
                return full_text if full_text else "Warning: Oracle references not available."

            entries = []
            for rid in ids:
                doc_id, ref_data = self._load_reference_record(rid)
                entries.append(self._build_oracle_ref_entry(doc_id, ref_data, independent_only=True))
            short_text = "\n".join(entries).strip()
            if len(short_text) > limit:
                return short_text[:limit] + "\n[Truncated to input limit]"
            return short_text if short_text else "Warning: Oracle references not available."
        elif setting == "bm25_retrieved":
            limit = self._get_ref_text_limit()
            query_text = item.get("input", {}).get("claims", "")
            bm25_list = self._bm25_retrieve_list(query_text, top_k=5)

            ids = item.get("input", {}).get("oracle_reference_ids", [])
            oracle_text = item.get("input", {}).get("oracle_reference_summaries", "")

            def _build_pool(oracle_entries):
                # Prefer oracle entries (often richer / claim-based) when duplicates exist.
                pool = self._dedupe_refs(oracle_entries[:5] + bm25_list[:5])
                if len(pool) < 10:
                    for ref in oracle_entries[5:]:
                        pool = self._dedupe_refs(pool + [ref])
                        if len(pool) >= 10:
                            break
                if len(pool) < 10:
                    for ref in bm25_list[5:]:
                        pool = self._dedupe_refs(pool + [ref])
                        if len(pool) >= 10:
                            break
                return pool

            oracle_entries_full = []
            if isinstance(ids, list) and ids:
                for rid in ids:
                    doc_id, ref_data = self._load_reference_record(rid)
                    oracle_entries_full.append(self._build_oracle_ref_entry(doc_id, ref_data, independent_only=False))
            elif isinstance(oracle_text, str) and oracle_text.strip():
                oracle_entries_full = self._split_ref_blocks(oracle_text.strip())
                if not oracle_entries_full:
                    oracle_entries_full = self._split_reference_summaries(oracle_text.strip())

            pool = _build_pool(oracle_entries_full)
            if not pool:
                return "References corpus not available."
            joined = "\n".join(pool[:10]).strip()
            if len(joined) <= limit:
                return joined

            if isinstance(ids, list) and ids:
                oracle_entries_ind = []
                for rid in ids:
                    doc_id, ref_data = self._load_reference_record(rid)
                    oracle_entries_ind.append(self._build_oracle_ref_entry(doc_id, ref_data, independent_only=True))
                pool = _build_pool(oracle_entries_ind)
                joined = "\n".join(pool[:10]).strip()
                if len(joined) <= limit:
                    return joined

            if len(joined) > limit:
                return joined[:limit] + "\n[Truncated to input limit]"
            return joined
        return "Unknown retrieval setting"

    def _tokenize(self, text):
        return [w for w in re.findall(r"[A-Za-z0-9]+", text.lower()) if w]

    def _load_reference_corpus(self):
        ref_dir = self.config['paths'].get('reference_pool')
        if not ref_dir or not os.path.exists(ref_dir):
            return
        docs = []
        ids = []
        lens = []
        for name in os.listdir(ref_dir):
            if not name.endswith(".json"):
                continue
            path = os.path.join(ref_dir, name)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                title = data.get('title', '')
                abstract = data.get('abstract', '')
                summary = data.get('summary_of_invention', '')
                content = " ".join([title, abstract, summary]).strip()
                tokens = self._tokenize(content)
                docs.append(tokens)
                ids.append(os.path.splitext(name)[0])
                lens.append(len(tokens))
            except:
                continue
        self._corpus_docs = docs
        self._docids = ids
        self._doclens = lens
        self._avgdl = sum(lens) / len(lens) if lens else 0.0
        self._corpus_loaded = True

    def _bm25_retrieve_list(self, query_text, top_k=5):
        if not self._corpus_loaded:
            self._load_reference_corpus()
        if not self._corpus_docs:
            return []
        q = self._tokenize(query_text)
        if not q:
            return []
        scores = []
        try:
            from rank_bm25 import BM25Okapi
            if self._bm25_index is None:
                self._bm25_index = BM25Okapi(self._corpus_docs)
            s_arr = self._bm25_index.get_scores(q)
            scores = [(float(s_arr[i]), i) for i in range(len(s_arr))]
            scores.sort(reverse=True, key=lambda x: x[0])
        except Exception:
            N = len(self._corpus_docs)
            df = {}
            for term in set(q):
                c = 0
                seen = set()
                for doc in self._corpus_docs:
                    if term in set(doc):
                        c += 1
                df[term] = c
            k1 = 1.5
            b = 0.75
            tmp = []
            for i, doc in enumerate(self._corpus_docs):
                doc_tf = {}
                for t in doc:
                    doc_tf[t] = doc_tf.get(t, 0) + 1
                dl = self._doclens[i] if self._doclens else 1
                s = 0.0
                for term in q:
                    if term not in doc_tf:
                        continue
                    idf = 0.0
                    dfi = df.get(term, 0)
                    if dfi > 0:
                        idf = max(0.0, ((N - dfi + 0.5) / (dfi + 0.5)))
                    tf = doc_tf[term]
                    denom = tf + k1 * (1 - b + b * (dl / (self._avgdl if self._avgdl else dl)))
                    s += (idf) * ((tf * (k1 + 1)) / denom)
                tmp.append((s, i))
            tmp.sort(reverse=True, key=lambda x: x[0])
            scores = tmp
        top_k = int(top_k) if top_k else self.config.get('retrieval_params', {}).get('bm25_top_k', 5)
        ref_dir = self.config['paths'].get('reference_pool')
        out = []
        for s, i in scores[:top_k]:
            doc_id = self._docids[i]
            path = os.path.join(ref_dir, f"{doc_id}.json")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                title = data.get('title', '')
                abstract = data.get('abstract', '')
                out.append(f"Ref {doc_id}: {title}. {abstract}")
            except:
                out.append(f"Ref {doc_id}: [content unavailable]")
        return out

    def _bm25_retrieve(self, query_text):
        return "\n".join(self._bm25_retrieve_list(query_text))

    def run_scoring(self, results):
        """
        Task  examiner_judge rebuttal_judge
        """
        # 1.  ( "oa" "rebuttal")
        current_task = self.config['task_settings']['current_task']
        # 
        if current_task == "oa" or current_task in self.templates.get("Office_Action_Generation", {}):
            judge_key = "examiner_judge"
        else:
            judge_key = "rebuttal_judge"
        
        self.logger.info(f" Routing Evaluation to: {judge_key}")
        
        # 3. 
        judge_config = self.templates["Evaluation_Metrics"][judge_key]
        scored_results = []

        for res in tqdm(results, desc="Scoring"):
            #Prompt
            #Judge
            try:
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "generated_text": res['ai_exam']['output'],
                        "claims": res['ai_exam']['input']['claims'],
                        "reference_summaries": res['ai_exam']['input']['refs'],
                        "office_action": res['evaluation_context']['full_context']['office_action'],
                        "ground_truth": res['evaluation_context']['ground_truth'],
                    },
                )
            except KeyError as e:
                # Missing placeholders in the template; fall back with safe defaults.
                self.logger.warning(f" Prompt format warning: {e}")
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "generated_text": res['ai_exam'].get('output', ''),
                        "claims": res.get('ai_exam', {}).get('input', {}).get('claims', ''),
                        "reference_summaries": res.get('ai_exam', {}).get('input', {}).get('refs', ''),
                        "office_action": res.get('evaluation_context', {}).get('full_context', {}).get('office_action', ''),
                        "ground_truth": res.get('evaluation_context', {}).get('ground_truth', ''),
                    },
                )

            judge_attempts = []
            judge_system_prompt = judge_config["system_prompt"]
            # GPT-4o 
            raw_judge_res = self.judge_backend.generate(
                system_prompt=judge_config["system_prompt"],
                user_prompt=judge_prompt
            )
            judge_attempts.append({
                "attempt": 1,
                "user_prompt": judge_prompt,
                "raw_output": raw_judge_res
            })
            # Always store raw judge output for debugging
            res["raw_judge_output"] = raw_judge_res
            res["judge_system_prompt"] = judge_system_prompt
            res["judge_user_prompt"] = judge_prompt
            res["judge_attempts"] = judge_attempts
            res["judge_prompt_length"] = len(judge_prompt) if isinstance(judge_prompt, str) else 0

            try:
                # JSON 
                try:
                    score_data = self._extract_json(raw_judge_res)
                except Exception:
                    retry_prompt = judge_prompt + "\n\nReturn JSON only. Do not include any other text."
                    raw_judge_res = self.judge_backend.generate(
                        system_prompt=judge_config["system_prompt"],
                        user_prompt=retry_prompt
                    )
                    judge_attempts.append({
                        "attempt": len(judge_attempts) + 1,
                        "user_prompt": retry_prompt,
                        "raw_output": raw_judge_res
                    })
                    res["raw_judge_output"] = raw_judge_res
                    res["judge_attempts"] = judge_attempts
                    score_data = self._extract_json(raw_judge_res)
                judge_scores = {}
                if isinstance(score_data, dict):
                    raw_scores = score_data.get("scores")
                    if isinstance(raw_scores, dict):
                        judge_scores = raw_scores
                    else:
                        for k, v in score_data.items():
                            if k == "total_score":
                                continue
                            if isinstance(v, (int, float)) and not isinstance(v, bool):
                                judge_scores[k] = v
                # Keep PatentScore on its original 1-10 scale (no auto-scaling).
                res["judge_scores"] = judge_scores
                res["judge_parsed_output"] = score_data
                res["scores"] = dict(judge_scores)
                if (
                    "language_style" not in res["scores"]
                    and isinstance(res["scores"].get("clarity"), (int, float))
                ):
                    res["scores"]["language_style"] = res["scores"]["clarity"]
                res["judge_total_score"] = score_data.get("total_score", None)
                justification_summary = score_data.get("justification_summary", score_data.get("justification", ""))
                res["justification"] = justification_summary
                res["evaluation_report"] = score_data.get("evaluation_report", "")
                res["justification_details"] = score_data.get("justification_details", {})
                res["judge_justification"] = score_data.get("justification", "")
                res["judge_justification_details"] = score_data.get("justification_details", {})
                res["judge_evaluation_report"] = score_data.get("evaluation_report", "")

                # FORCE total_score to be the weighted average of sub-metrics
                combined_total = self._calculate_numeric_total(res.get("scores", {}), task_type=current_task)
                res["combined_total_score"] = combined_total
                res["total_score"] = combined_total
                # Keep the judge's subjective total score separate for reference
                res["judge_total_score"] = score_data.get("total_score", None)
                scored_results.append(res)
            except Exception as e:
                self.logger.warning(f" Scoring failed for {res['patent_id']}: {e} (prompt_len={res.get('judge_prompt_length', 0)})")
                res.setdefault("scores", {})
                res["scores"]["error"] = "PARSING_FAILED"
                res.setdefault("justification", "")
                res.setdefault("justification_details", {})
                scored_results.append(res)

        return scored_results

    def run_point_wise_evaluation(self, results):
        """
        Point-wise 
        """
        eval_templates = self.templates.get("Evaluation_Metrics", {})
        judge_config = eval_templates.get("point_wise_evaluator")
        if not judge_config:
            self.logger.info(" Point-wise evaluator template not found. Skipping.")
            return results
        self.logger.info(" Running point-wise evaluation...")
        evaluated_results = []

        for res in tqdm(results, desc="Point-wise Evaluation"):
            try:
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "generated_text": res['ai_exam']['output'],
                        "ground_truth": res['evaluation_context']['ground_truth'],
                        "claims": res['ai_exam']['input']['claims'],
                        "office_action": res['evaluation_context']['full_context']['office_action'],
                    },
                )
            except KeyError as e:
                self.logger.warning(f" Point-wise prompt format warning: {e}")
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "generated_text": res['ai_exam']['output'],
                        "ground_truth": res['evaluation_context']['ground_truth'],
                    },
                )

            raw_judge_res = self.judge_backend.generate(
                system_prompt=judge_config["system_prompt"],
                user_prompt=judge_prompt
            )

            try:
                try:
                    score_data = self._extract_json(raw_judge_res)
                except Exception:
                    retry_prompt = judge_prompt + "\n\nReturn JSON only. Do not include any other text."
                    raw_judge_res = self.judge_backend.generate(
                        system_prompt=judge_config["system_prompt"],
                        user_prompt=retry_prompt
                    )
                    score_data = self._extract_json(raw_judge_res)
                res["point_wise_scores"] = score_data.get("scores", {})
                res["point_wise_total"] = score_data.get("total_score", 0)
                res["point_wise_justification"] = score_data.get("justification", "")
                evaluated_results.append(res)
            except Exception as e:
                self.logger.warning(f" Point-wise evaluation failed for {res['patent_id']}: {e}")
                res["point_wise_scores"] = {"error": "PARSING_FAILED"}
                evaluated_results.append(res)

        return evaluated_results

    def run_pairwise_comparison(self, results):
        """
        Pair-wise 
        """
        eval_templates = self.templates.get("Evaluation_Metrics", {})
        judge_config = eval_templates.get("pairwise_evaluator")
        if not judge_config:
            self.logger.info(" Pairwise evaluator template not found. Skipping.")
            return results
        self.logger.info(" Running pairwise comparison...")
        compared_results = []

        #  results 

        for i in range(0, len(results), 2):
            if i + 1 >= len(results):
  
                res = results[i]
                res["pairwise_winner"] = None
                res["pairwise_comparison"] = "Insufficient pairs for comparison"
                compared_results.append(res)
                break
            
            res_a = results[i]
            res_b = results[i + 1]
            
            try:
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "text_a": res_a['ai_exam']['output'],
                        "text_b": res_b['ai_exam']['output'],
                        "claims": res_a['ai_exam']['input']['claims'],
                        "office_action": res_a['evaluation_context']['full_context']['office_action'],
                    },
                )
            except KeyError as e:
                self.logger.warning(f" Pairwise prompt format warning: {e}")
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "text_a": res_a['ai_exam']['output'],
                        "text_b": res_b['ai_exam']['output'],
                    },
                )

            raw_judge_res = self.judge_backend.generate(
                system_prompt=judge_config["system_prompt"],
                user_prompt=judge_prompt
            )

            try:
                try:
                    score_data = self._extract_json(raw_judge_res)
                except Exception:
                    retry_prompt = judge_prompt + "\n\nReturn JSON only. Do not include any other text."
                    raw_judge_res = self.judge_backend.generate(
                        system_prompt=judge_config["system_prompt"],
                        user_prompt=retry_prompt
                    )
                    score_data = self._extract_json(raw_judge_res)
                winner = score_data.get("winner", "unknown")
                
                # 
                res_a["pairwise_winner"] = winner
                res_a["pairwise_scores"] = score_data.get("scores", {})
                res_a["pairwise_justification"] = score_data.get("justification", "")
                
                # winner
                res_b["pairwise_winner"] = "A" if winner == "B" else "B" if winner == "A" else "tie"
                res_b["pairwise_scores"] = score_data.get("scores", {})
                res_b["pairwise_justification"] = score_data.get("justification", "")
                
                compared_results.extend([res_a, res_b])
            except Exception as e:
                self.logger.warning(f" Pairwise comparison failed for {res_a['patent_id']} vs {res_b['patent_id']}: {e}")
                res_a["pairwise_winner"] = "error"
                res_b["pairwise_winner"] = "error"
                compared_results.extend([res_a, res_b])

        return compared_results

    def run_claim_level_evaluation(self, results):
        """
        Claim-level 
        """
        eval_templates = self.templates.get("Evaluation_Metrics", {})
        judge_config = eval_templates.get("claim_level_evaluator")
        if not judge_config:
            self.logger.info(" Claim-level evaluator template not found. Skipping.")
            return results
        self.logger.info(" Running claim-level evaluation...")
        evaluated_results = []

        for res in tqdm(results, desc="Claim-level Evaluation"):
            try:
                # 
                claim_text = res.get('ai_exam', {}).get('input', {}).get('claims', '')
                claim_number = res.get('round_info', {}).get('round_number', '1')
                
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "generated_text": res['ai_exam']['output'],
                        "ground_truth": res['evaluation_context']['ground_truth'],
                        "claim_number": claim_number,
                        "claim_text": claim_text,
                        "office_action": res['evaluation_context']['full_context']['office_action'],
                    },
                )
            except KeyError as e:
                self.logger.warning(f" Claim-level prompt format warning: {e}")
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "generated_text": res['ai_exam']['output'],
                        "ground_truth": res['evaluation_context']['ground_truth'],
                        "claim_number": '1',
                        "claim_text": res['ai_exam']['input']['claims'],
                    },
                )

            raw_judge_res = self.judge_backend.generate(
                system_prompt=judge_config["system_prompt"],
                user_prompt=judge_prompt
            )

            try:
                try:
                    score_data = self._extract_json(raw_judge_res)
                except Exception:
                    retry_prompt = judge_prompt + "\n\nReturn JSON only. Do not include any other text."
                    raw_judge_res = self.judge_backend.generate(
                        system_prompt=judge_config["system_prompt"],
                        user_prompt=retry_prompt
                    )
                    score_data = self._extract_json(raw_judge_res)
                res["claim_level_scores"] = score_data.get("scores", {})
                res["claim_level_total"] = score_data.get("total_score", 0)
                res["claim_level_justification"] = score_data.get("justification", "")
                evaluated_results.append(res)
            except Exception as e:
                self.logger.warning(f" Claim-level evaluation failed for {res['patent_id']}: {e}")
                res["claim_level_scores"] = {"error": "PARSING_FAILED"}
                evaluated_results.append(res)

        return evaluated_results

    def run_additional_metrics(self, results):
        """
        Compute rule-based and text similarity metrics without changing prompts.
        """
        self.logger.info(" Running additional metrics...")
        sim_cfg = self.config.get("task_settings", {}).get("similarity_metrics", {})
        if not isinstance(sim_cfg, dict):
            sim_cfg = {}
        enable_bleu = sim_cfg.get("bleu", True)
        enable_rouge_l = sim_cfg.get("rouge_l", True)
        enable_bert_score = sim_cfg.get("bert_score", True)
        for res in tqdm(results, desc="Additional Metrics"):
            try:
                pred_text = res.get("ai_exam", {}).get("output", "")
                gt_text = res.get("evaluation_context", {}).get("ground_truth", "")
                refs_text = res.get("ai_exam", {}).get("input", {}).get("refs", "")
                task_type = res.get("round_info", {}).get("task", "")

                pred_json = safe_parse_json(pred_text)
                gt_json = safe_parse_json(gt_text)

                rule_scores = {}
                is_rebuttal = task_type == "rebuttal"
                if not is_rebuttal:
                    pred_disp = extract_disposition(pred_text, pred_json)
                    gt_disp = extract_disposition(gt_text, gt_json)

                    pred_stat = extract_statutes(pred_text)
                    gt_stat = extract_statutes(gt_text)

                    # Per-sample: only keep accuracy to avoid misleading precision/recall/F1
                    disp_scores = {}
                    if pred_disp != "unknown" and gt_disp not in ("unknown", "procedural_action"):
                        disp_scores["disposition_accuracy"] = round(1.0 if pred_disp == gt_disp else 0.0, 4)
                    stat_scores = statute_metrics(pred_stat, gt_stat)
                    gt_ids = res.get("evaluation_context", {}).get("citations_gt", None)
                    cit_scores, ref_ids, pred_ids = citation_metrics(pred_text, refs_text, gt_ids=gt_ids)
                    list_match = citation_list_match(pred_text, refs_text)

                    rule_scores.update(disp_scores)
                    rule_scores.update(stat_scores)
                    rule_scores.update(cit_scores)
                    rule_scores["citation_list_match"] = list_match
                    if "disposition_accuracy" in rule_scores:
                        rule_scores["disposition_accuracy_pct"] = round(rule_scores["disposition_accuracy"] * 100, 2)
                    if "statute_precision" in rule_scores:
                        rule_scores["statute_precision_pct"] = round(rule_scores["statute_precision"] * 100, 2)
                    rule_scores["citation_list_match_pct"] = round(list_match * 100, 2)

                if enable_rouge_l:
                    r_l = rouge_l_f1(pred_text, gt_text)
                    if r_l is not None:
                        rule_scores["rouge_l"] = r_l
                if enable_bleu:
                    b = bleu_score(pred_text, gt_text)
                    if b is not None:
                        rule_scores["bleu"] = b
                if enable_bert_score:
                    bs = bert_score_f1(pred_text, gt_text)
                    if bs is not None:
                        rule_scores["bert_score"] = bs

                fmt = formatting_score(pred_text, task_type)
                if fmt is not None:
                    rule_scores["formatting_rule"] = fmt
                if not isinstance(res.get("scores", {}).get("language_style"), (int, float)):
                    ls = language_style_score(pred_text)
                    if ls is not None:
                        rule_scores["language_style"] = ls

                res.setdefault("scores", {})
                res["rule_scores"] = rule_scores
                res["scores"].update(rule_scores)
                if not is_rebuttal:
                    res["metric_details"] = {
                        "disposition_pred": pred_disp,
                        "disposition_gt": gt_disp,
                        "statutes_pred": sorted(list(pred_stat)),
                        "statutes_gt": sorted(list(gt_stat)),
                        "citations_pred": sorted(list(pred_ids)),
                        "citations_gt": sorted(list(ref_ids)),
                    }
                self._merge_accuracy_from_core_metrics(res.get("scores", {}), task_type)
                # Recalculate total_score now that we have more accurate detail metrics (disposition, statute, citation)
                combined_total = self._calculate_numeric_total(res.get("scores", {}), task_type=task_type)
                res["combined_total_score"] = combined_total
                res["total_score"] = combined_total
            except Exception as e:
                self.logger.warning(f" Additional metrics failed for {res.get('patent_id', 'unknown')}: {e}")

        return results

    def _merge_accuracy_from_core_metrics(self, scores, task_type):
        if not isinstance(scores, dict):
            return
        if "accuracy" not in scores or not isinstance(scores.get("accuracy"), (int, float)):
            return

        is_rebuttal = "rebuttal" in str(task_type).lower()
        base_acc = float(scores.get("accuracy"))

        if is_rebuttal:
            p = scores.get("point_wise_coverage")
            if isinstance(p, (int, float)):
                p_score = p * 10.0 if p <= 1.0 else p
                merged = 0.7 * base_acc + 0.3 * p_score
                scores["accuracy"] = round(max(0.0, min(10.0, merged)), 4)
            return

        d = scores.get("disposition_accuracy")
        s = scores.get("statute_precision")
        c = scores.get("citation_accuracy")

        d_score = d * 10.0 if isinstance(d, (int, float)) else None
        s_score = s * 10.0 if isinstance(s, (int, float)) else None
        c_score = None
        if isinstance(c, (int, float)):
            c_score = c if c > 1.0 else c * 10.0

        weighted_sum = 0.0
        weight_sum = 0.0
        for val, w in ((base_acc, 0.55), (d_score, 0.20), (s_score, 0.15), (c_score, 0.10)):
            if isinstance(val, (int, float)):
                weighted_sum += val * w
                weight_sum += w

        if weight_sum > 0:
            merged = weighted_sum / weight_sum
            if isinstance(d_score, (int, float)) and d_score == 0.0:
                merged = min(merged, 3.0)
            scores["accuracy"] = round(max(0.0, min(10.0, merged)), 4)

    def _calculate_numeric_total(self, scores, task_type=None):
        """
        Calculate a weighted total score (0-10) based on specific sub-metrics.
        If a metric is missing, its weight is distributed among others or ignored.
        """
        if not task_type:
            task_type = self.config.get('task_settings', {}).get('current_task', 'oa')
        
        is_rebuttal = "rebuttal" in str(task_type).lower()
        
        # Define weights and whether they need normalization (0-1 -> 0-10)
        if is_rebuttal:
            weights = {
                "point_wise_coverage": (0.30, True),
                "accuracy": (0.20, False),
                "completeness": (0.12, False),
                "clarity": (0.12, False),
                "language_style": (0.08, False),
                "constructiveness": (0.08, False),
                "formatting_rule": (0.10, True)
            }
        else:
            weights = {
                "disposition_accuracy": (0.25, True),
                "statute_precision": (0.15, True),
                "citation_accuracy": (0.15, True),
                "accuracy": (0.15, False),
                "completeness": (0.08, False),
                "clarity": (0.08, False),
                "language_style": (0.08, False),
                "formatting_rule": (0.06, True)
            }

        total_weighted_score = 0.0
        sum_of_weights = 0.0

        for key, (weight, needs_norm) in weights.items():
            val = scores.get(key)
            
            # Special case for citation_accuracy which is often 0-10 in this project's scores dict
            if key == "citation_accuracy" and isinstance(val, (int, float)) and val > 1.0:
                # If it's already 0-10, don't normalize it again
                needs_norm = False
                
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                score_value = val * 10.0 if needs_norm else val
                total_weighted_score += score_value * weight
                sum_of_weights += weight
        
        if sum_of_weights == 0:
            return 0.0
        
        # Re-scale to 10 if some weights were missing
        final_score = total_weighted_score / sum_of_weights
        return round(final_score, 4)

    def _run_gpt5_coherence_judge(self, res):
        eval_templates = self.templates.get("Evaluation_Metrics", {})
        coherence_cfg = eval_templates.get("coherence_judge")
        if not coherence_cfg:
            return None

        try:
            judge_prompt = self._build_judge_prompt(
                coherence_cfg["user_prompt_template"],
                {
                    "generated_text": res['ai_exam']['output'],
                    "claims": res['ai_exam']['input']['claims'],
                    "reference_summaries": res['ai_exam']['input']['refs'],
                    "ground_truth": res['evaluation_context']['ground_truth'],
                },
            )
        except KeyError as e:
            self.logger.warning(f" Coherence prompt format warning: {e}")
            judge_prompt = self._build_judge_prompt(
                coherence_cfg["user_prompt_template"],
                {"generated_text": res['ai_exam']['output']},
            )

        coherence_model = self.config.get("models", {}).get("coherence_judge_model", "gpt-5-mini")
        raw = self.judge_backend.generate(
            system_prompt=coherence_cfg["system_prompt"],
            user_prompt=judge_prompt,
            is_judge=True,
            model_override=coherence_model
        )
        try:
            parsed = self._extract_json(raw)
        except Exception:
            retry_prompt = judge_prompt + "\n\nReturn JSON only. Do not include any other text."
            raw = self.judge_backend.generate(
                system_prompt=coherence_cfg["system_prompt"],
                user_prompt=retry_prompt,
                is_judge=True,
                model_override=coherence_model
            )
            parsed = self._extract_json(raw)

        # :
        # 1) {"coherence_score": 8, "justification": "..."}
        # 2) {"scores": {"coherence_score": 8}, "justification": "..."}
        if "coherence_score" in parsed:
            return {
                "coherence_score": parsed.get("coherence_score", 0),
                "justification": parsed.get("justification", "")
            }
        return {
            "coherence_score": parsed.get("scores", {}).get("coherence_score", 0),
            "justification": parsed.get("justification", "")
        }

    def _extract_json(self, text):
        """Lightweight JSON extractor."""
        if text is None:
            raise ValueError("No valid JSON found in response")
        if isinstance(text, (dict, list)):
            return text
        raw = str(text).strip()
        if not raw:
            raise ValueError("No valid JSON found in response")
        # 1) Prefer fenced JSON blocks
        for m in re.finditer(r'```json\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE):
            block = m.group(1).strip()
            try:
                return json.loads(block)
            except Exception:
                continue
        # 2) Try direct parse
        try:
            return json.loads(raw)
        except Exception:
            pass
        # 3) Find first valid JSON object/array using JSONDecoder
        decoder = json.JSONDecoder()
        for i, ch in enumerate(raw):
            if ch in "{[":
                try:
                    obj, _ = decoder.raw_decode(raw[i:])
                    return obj
                except Exception:
                    continue
        # 4) Fallback: greedy braces (may still fail)
        text_clean = re.search(r'\{.*\}', raw, re.DOTALL)
        if text_clean:
            try:
                return json.loads(text_clean.group(0))
            except Exception:
                pass
        # 5) Fallback: Python-literal parsing for near-JSON outputs
        try:
            import ast
            candidates = []
            obj_block = re.search(r'\{.*\}', raw, re.DOTALL)
            arr_block = re.search(r'\[.*\]', raw, re.DOTALL)
            if obj_block:
                candidates.append(obj_block.group(0))
            if arr_block:
                candidates.append(arr_block.group(0))
            if not candidates:
                candidates.append(raw)
            for cand in candidates:
                cleaned = cand
                cleaned = re.sub(r'\btrue\b', 'True', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'\bfalse\b', 'False', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'\bnull\b', 'None', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
                return ast.literal_eval(cleaned)
        except Exception:
            pass
        raise ValueError("No valid JSON found in response")


