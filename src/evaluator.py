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
    embed_cosine,
    formatting_score,
)

class PatentEvaluator:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.backend = ModelBackend(config)
        self.judge_backend = ModelBackend(config, is_judge=True)
        self._corpus_loaded = False
        self._corpus_docs = []
        self._avgdl = 0.0
        self._doclens = []
        self._docids = []
        
        with open(config['paths']['prompt_path'], 'r', encoding='utf-8') as f:
            self.templates = json.load(f)
            
        self.dataset = self._load_dataset(config['paths']['test_set'])

    def _load_dataset(self, path):
        data = []
        ipc_filter = self.config.get('task_settings', {}).get('ipc_filter')
        if ipc_filter:
            ipc_filter = str(ipc_filter).upper()
        if not os.path.exists(path):
            self.logger.error(f"?? {path}")
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
        rebuttal_text = (
            item.get("input", {}).get("rebuttal")
            or item.get("ground_truth", {}).get("rebuttal_text")
            or ""
        )
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
        """?- """
        results = []
        task_key = self.config['task_settings']['current_task']
        granularity = self.config.get('active_settings', {}).get('granularity', "full_text")
        
        # --- ?---
        if task_key in ("oa",) or task_key in self.templates.get("Office_Action_Generation", {}):
            root_key = "Office_Action_Generation"
        elif task_key in self.templates.get("Rebuttal_Generation", {}):
            root_key = "Rebuttal_Generation"
        else:
            self.logger.error(f"?: {task_key}")
            return []
        
        self.logger.info(f" Starting batch inference for {len(self.dataset)} samples...")

        for item in tqdm(self.dataset, desc="Generating"):
            template_task = self._resolve_template_task(task_key, item)
            if template_task not in self.templates.get(root_key, {}):
                self.logger.error(f"??{root_key} : {template_task}")
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
                    self.logger.error(f"?Prompt ? {e}")
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

                # ?
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
        """"""
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

    def _shrink_prompt(self, template, fields, max_total):
        prompt = template.format(**fields)
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
                prompt = template.format(**fields)
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

        prompt = template.format(**truncated)
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

    def _process_retrieval(self, item):
        setting = self.config['active_settings'].get('retrieval_setting', "oracle_summary")
        
        if setting == "no_ref":
            return "No prior art references provided for this examination."
        
        elif setting == "oracle_summary":
            content = item['input'].get('oracle_reference_summaries', "").strip()
            return content if content else "Warning: Oracle summary data is missing in test set."
        elif setting == "bm25_retrieved":
            query_text = item['input'].get('claims', '')
            bm25_list = self._bm25_retrieve_list(query_text, top_k=5)
            oracle_text = item['input'].get('oracle_reference_summaries', "")
            oracle_list = self._split_reference_summaries(oracle_text)
            # Mix: BM25 5 + Oracle 5, dedupe, then fill to 10 with oracle.
            pool = self._dedupe_refs(bm25_list[:5] + oracle_list[:5])
            if len(pool) < 10:
                for ref in oracle_list[5:]:
                    pool = self._dedupe_refs(pool + [ref])
                    if len(pool) >= 10:
                        break
            if len(pool) < 10:
                for ref in bm25_list[5:]:
                    pool = self._dedupe_refs(pool + [ref])
                    if len(pool) >= 10:
                        break
            if not pool:
                return "References corpus not available."
            return "\n".join(pool[:10])
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
        N = len(self._corpus_docs)
        df = {}
        for term in set(q):
            c = 0
            for doc in self._corpus_docs:
                if term in set(doc):
                    c += 1
            df[term] = c
        k1 = 1.5
        b = 0.75
        scores = []
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
                    idf = max(0.0, ( (N - dfi + 0.5) / (dfi + 0.5) ))
                tf = doc_tf[term]
                denom = tf + k1 * (1 - b + b * (dl / (self._avgdl if self._avgdl else dl)))
                s += (idf) * ((tf * (k1 + 1)) / denom)
            scores.append((s, i))
        scores.sort(reverse=True, key=lambda x: x[0])
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
        ?
         Task  examiner_judge ?rebuttal_judge
        """
        # 1.  ( "oa" ?"rebuttal")
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
            #  Prompt
            # ?Judge ?
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
                # ?{claims} ormat 
                # Python ?format  key ?key  unpack dict?
                #  {claims} ?
                # ?context ?
                self.logger.warning(f" Prompt format warning: {e}")
                # Fallback to simple format if needed, or just let it fail/warn
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {"generated_text": res['ai_exam']['output']},
                )

            # ?GPT-4o 
            raw_judge_res = self.judge_backend.generate(
                system_prompt=judge_config["system_prompt"],
                user_prompt=judge_prompt
            )
            # Always store raw judge output for debugging
            res["raw_judge_output"] = raw_judge_res
            res["judge_prompt_length"] = len(judge_prompt) if isinstance(judge_prompt, str) else 0

            try:
                # ?JSON 
                try:
                    score_data = self._extract_json(raw_judge_res)
                except Exception:
                    retry_prompt = judge_prompt + "\n\nReturn JSON only. Do not include any other text."
                    raw_judge_res = self.judge_backend.generate(
                        system_prompt=judge_config["system_prompt"],
                        user_prompt=retry_prompt
                    )
                    res["raw_judge_output"] = raw_judge_res
                    score_data = self._extract_json(raw_judge_res)
                judge_scores = score_data.get("scores", {})
                for k in ("patentscore_structure_stability", "patentscore_legal_clarity", "patentscore_semantic_logic"):
                    if isinstance(judge_scores.get(k), (int, float)):
                        judge_scores[k] = min(10, round(judge_scores[k] * 2, 4))
                res["judge_scores"] = judge_scores
                res["scores"] = dict(judge_scores)
                res["judge_total_score"] = score_data.get("total_score", None)
                justification_summary = score_data.get("justification_summary", score_data.get("justification", ""))
                res["justification"] = justification_summary
                res["justification_details"] = score_data.get("justification_details", {})

                # ?total_score 
                combined_total = self._calculate_numeric_total(res.get("scores", {}))
                res["combined_total_score"] = combined_total
                res["total_score"] = res.get("judge_total_score", combined_total)
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
        self.logger.info(" Running point-wise evaluation...")
        judge_config = self.templates["Evaluation_Metrics"]["point_wise_evaluator"]
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
        Pair-wise ?
        """
        self.logger.info(" Running pairwise comparison...")
        judge_config = self.templates["Evaluation_Metrics"]["pairwise_evaluator"]
        compared_results = []

        #  results 
        # ?
        for i in range(0, len(results), 2):
            if i + 1 >= len(results):
                # ?
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
        self.logger.info(" Running claim-level evaluation...")
        judge_config = self.templates["Evaluation_Metrics"]["claim_level_evaluator"]
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
                    if pred_disp != "unknown" and gt_disp != "unknown":
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

                r_l = rouge_l_f1(pred_text, gt_text)
                if r_l is not None:
                    rule_scores["rouge_l"] = r_l
                b = bleu_score(pred_text, gt_text)
                if b is not None:
                    rule_scores["bleu"] = b
                # Skip heavy metrics by default to avoid long stalls

                fmt = formatting_score(pred_text, task_type)
                if fmt is not None:
                    rule_scores["formatting_rule"] = fmt

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
                # If judge parsing failed (or total_score missing), still compute a numeric total
                # from available metrics so the run isn't forced to 0.
                if res.get("total_score") is None or res.get("scores", {}).get("error") == "PARSING_FAILED":
                    combined_total = self._calculate_numeric_total(res.get("scores", {}))
                    res["combined_total_score"] = combined_total
                    res["total_score"] = res.get("judge_total_score", combined_total)
            except Exception as e:
                self.logger.warning(f" Additional metrics failed for {res.get('patent_id', 'unknown')}: {e}")

        return results

    def run_claim_rediscovery_evaluation(self, results):
        """
        Claim-centric rediscovery evaluation:
        1) Extract atomic claims from ground truth OA and AI output
        2) Semantically match claims and compute recall/precision/F1
        """
        self.logger.info(" Running claim-centric rediscovery evaluation...")
        eval_templates = self.templates.get("Evaluation_Metrics", {})
        extractor_cfg = eval_templates.get("claim_rediscovery_extractor")
        matcher_cfg = eval_templates.get("claim_rediscovery_matcher")
        if not extractor_cfg or not matcher_cfg:
            self.logger.warning("?Claim rediscovery templates not found. Skipping.")
            return results

        match_cache = {}

        for res in tqdm(results, desc="Claim Rediscovery"):
            gt_text = res.get('evaluation_context', {}).get('ground_truth', '')
            ai_text = res.get('ai_exam', {}).get('output', '')

            gt_claims = self._rediscovery_extract_atomic_claims(gt_text, extractor_cfg)
            ai_claims = self._rediscovery_extract_atomic_claims(ai_text, extractor_cfg)

            gt_claims = [c for c in gt_claims if isinstance(c, str) and c.strip()]
            ai_claims = [c for c in ai_claims if isinstance(c, str) and c.strip()]

            matched_gt = 0
            matched_ai = 0

            for gt in gt_claims:
                if self._rediscovery_any_match(gt, ai_claims, matcher_cfg, match_cache):
                    matched_gt += 1

            for ai in ai_claims:
                if self._rediscovery_any_match(ai, gt_claims, matcher_cfg, match_cache):
                    matched_ai += 1

            recall = (matched_gt / len(gt_claims)) if gt_claims else 0.0
            precision = (matched_ai / len(ai_claims)) if ai_claims else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            res.setdefault("scores", {})
            res["scores"]["rediscovery_recall"] = round(recall, 4)
            res["scores"]["rediscovery_precision"] = round(precision, 4)
            res["scores"]["rediscovery_f1"] = round(f1, 4)
            res["rediscovery_details"] = {
                "ground_truth_claims": gt_claims,
                "ai_claims": ai_claims,
                "matched_ground_truth": matched_gt,
                "matched_ai": matched_ai
            }

        return results

    def _rediscovery_extract_atomic_claims(self, text, extractor_cfg):
        try:
            prompt = self._build_judge_prompt(
                extractor_cfg["user_prompt_template"],
                {"text": text},
            )
        except KeyError as e:
            self.logger.warning(f"?Rediscovery extractor prompt warning: {e}")
            prompt = self._build_judge_prompt(
                extractor_cfg["user_prompt_template"],
                {"text": text},
            )

        raw = self.judge_backend.generate(
            system_prompt=extractor_cfg["system_prompt"],
            user_prompt=prompt
        )
        try:
            parsed = self._extract_json(raw)
            return parsed.get("claims", [])
        except Exception as e:
            self.logger.warning(f"?Rediscovery extractor parsing failed: {e}")
            retry_prompt = prompt + "\n\nReturn JSON only. Do not include any other text."
            raw = self.judge_backend.generate(
                system_prompt=extractor_cfg["system_prompt"],
                user_prompt=retry_prompt
            )
            try:
                parsed = self._extract_json(raw)
                return parsed.get("claims", [])
            except Exception as e2:
                self.logger.warning(f"?Rediscovery extractor retry failed: {e2}")
                return []

    def _rediscovery_any_match(self, claim_a, claim_list, matcher_cfg, match_cache):
        for claim_b in claim_list:
            key = (claim_a, claim_b)
            if key in match_cache:
                if match_cache[key]:
                    return True
                continue
            try:
                prompt = self._build_judge_prompt(
                    matcher_cfg["user_prompt_template"],
                    {"claim_a": claim_a, "claim_b": claim_b},
                )
            except KeyError as e:
                self.logger.warning(f"?Rediscovery matcher prompt warning: {e}")
                prompt = self._build_judge_prompt(
                    matcher_cfg["user_prompt_template"],
                    {"claim_a": claim_a, "claim_b": claim_b},
                )
            raw = self.judge_backend.generate(
                system_prompt=matcher_cfg["system_prompt"],
                user_prompt=prompt
            )
            try:
                parsed = self._extract_json(raw)
                match = bool(parsed.get("match", False))
            except Exception as e:
                self.logger.warning(f"?Rediscovery matcher parsing failed: {e}")
                retry_prompt = prompt + "\n\nReturn JSON only. Do not include any other text."
                raw = self.judge_backend.generate(
                    system_prompt=matcher_cfg["system_prompt"],
                    user_prompt=retry_prompt
                )
                try:
                    parsed = self._extract_json(raw)
                    match = bool(parsed.get("match", False))
                except Exception as e2:
                    self.logger.warning(f"?Rediscovery matcher retry failed: {e2}")
                    match = False
            match_cache[key] = match
            if match:
                return True
        return False

    def _calculate_numeric_total(self, scores):
        total = 0.0
        for k, v in scores.items():
            if k == "quality":
                continue
            if isinstance(v, (int, float)):
                total += v
        return total

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


