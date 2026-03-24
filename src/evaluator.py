# -*- coding: utf-8 -*-
import json
import os
import re
import time
from tqdm import tqdm
from src.model_backend import ModelBackend
from src.utils import load_dataset_json
from src.ref_pool_dates import (
    oa_cutoff_date,
    ref_doc_effective_date,
    reference_pool_date_policy_log_line,
)
from src.patent_round_save import save_one_patent_round_artifacts
from src.metrics_library import (
    safe_parse_json,
    extract_statutes,
    extract_rejection_types,
    disposition_from_oa_type,
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
    normalize_rejection_types,
    rejection_type_metrics,
    resolve_predicted_disposition,
    MODEL_OA_TYPE_TRAILER_USER_SUFFIX,
    split_model_oa_type_trailer,
)

# Hard-disable prompt/reference truncation to avoid cutting long citations.
DISABLE_TRUNCATION = True


class PatentEvaluator:
    @staticmethod
    def _output_body_without_oa_type_trailer(text, task_type):
        if task_type != "oa" or not text:
            return text
        _, body = split_model_oa_type_trailer(text)
        return body

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
        self._doc_dates = []
        self._bm25_index = None
        self._bm25_temporal_policy_logged = False
        
        with open(config['paths']['prompt_path'], 'r', encoding='utf-8') as f:
            self.templates = json.load(f)
        self._patch_eval_templates()
            
        self.dataset = self._load_dataset(config['paths']['test_set'])

    def _load_dataset(self, path):
        ipc_filter = self.config.get('task_settings', {}).get('ipc_filter')
        if ipc_filter:
            ipc_filter = str(ipc_filter).upper()
        if not os.path.exists(path):
            self.logger.error(f"Test set path not found: {path}")
            return []
        data = load_dataset_json(path)
        if ipc_filter:
            data = [
                item
                for item in data
                if str(item.get("meta", {}).get("ipc", "")).upper() == ipc_filter
            ]
        # Attach prosecution history for single-task runs (use only the previous same-type round)
        patent_items = {}
        for item in data:
            pid = item.get("id")
            if pid:
                patent_items.setdefault(pid, []).append(item)
        
        for idx, item in enumerate(data):
            pid = item.get("id")
            if not pid:
                continue
            item.setdefault("input", {})
            
            # Build context from relevant previous rounds (ordered by OA number/date)
            items_for_pid = list(patent_items.get(pid, []))
            if items_for_pid:
                def _sort_key(it):
                    meta = it.get("meta", {}) if isinstance(it, dict) else {}
                    num = meta.get("oa_number")
                    try:
                        num_val = int(num)
                    except Exception:
                        num_val = 10 ** 9
                    date_val = meta.get("oa_date") or ""
                    round_type = str(meta.get("round_type", "")).lower()
                    round_order = 0 if round_type == "oa" else 1
                    return (num_val, date_val, round_order)
                items_for_pid = sorted(items_for_pid, key=_sort_key)

            current_idx_in_patent = None
            for i, it in enumerate(items_for_pid):
                if it is item or it.get("id") == pid and it == item:
                    current_idx_in_patent = i
                    break
            
            if current_idx_in_patent is None:
                current_idx_in_patent = idx
            
            task_key = self.config.get('task_settings', {}).get('current_task', '')
            prev_context = self._build_previous_round_context(items_for_pid, current_idx_in_patent, task_key)
            item["input"]["prosecution_history"] = prev_context

        
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
        # Apply num_samples cap from config
        limit = self.config['task_settings'].get('num_samples', 9999)
        return data[:limit]

    def _patch_eval_templates(self):
        eval_metrics = self.templates.get("Evaluation_Metrics", {})
        if not isinstance(eval_metrics, dict):
            return

        # Avoid 0 for language_style in the example JSON (models often copy it literally).
        # Also avoid 0 for constructiveness for the same reason: the model may copy example values.
        _patch_num_defaults = {"language_style": 5, "constructiveness": 5}

        def patch_scores_block(tmpl, keys, num_defaults=None):
            if not isinstance(tmpl, str):
                return tmpl
            num_defaults = num_defaults or {}
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
                val = num_defaults.get(k, 0)
                lines.append(f'    "{k}": {val}{comma}\n')

            new_body = "".join(lines)
            return tmpl[:m.start()] + head + new_body + tail + tmpl[m.end():]

        cfg = eval_metrics.get("rebuttal_judge")
        if isinstance(cfg, dict) and isinstance(cfg.get("user_prompt_template"), str):
            t = patch_scores_block(
                cfg["user_prompt_template"],
                ["constructiveness", "language_style", "point_wise_coverage"],
                num_defaults=_patch_num_defaults,
            )
            # Some templates (or patched JSON examples) may still contain "constructiveness": 0.
            # If the model copies that example literally, we would end up with constant zeros.
            t = t.replace('"constructiveness": 0', '"constructiveness": 5')
            t = t.replace('"constructiveness": 0.0', '"constructiveness": 5')
            t = t.replace(
                "## Output Format:",
                "## constructiveness (required): Integer 1-10 only. Rate how actionable and point-specific the rebuttal guidance is "
                "(i.e., whether it proposes concrete legal paths/amendments tied to the Office Action). Never use 0; if uncertain, match "
                "**clarity**.\n## language_style (required): Integer 1-10 only. Rate professional applicant-side legal prose (aligned with "
                "**clarity**). Never use 0; if uncertain, match **clarity**.\n\n## Output Format:",
                1,
            )
            cfg["user_prompt_template"] = t

        cfg = eval_metrics.get("examiner_judge")
        if isinstance(cfg, dict) and isinstance(cfg.get("user_prompt_template"), str):
            t = patch_scores_block(
                cfg["user_prompt_template"],
                ["language_style"],
                num_defaults=_patch_num_defaults,
            )
            t = t.replace(
                "## Output Format:",
                "## language_style (required): Integer 1-10 only. Rate professional USPTO-style prose (similar intent to **clarity**). "
                "Never use 0; if uncertain, match **clarity**.\n\n## Output Format:",
                1,
            )
            cfg["user_prompt_template"] = t

    @staticmethod
    def _score_in_one_to_ten(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and 1.0 <= float(x) <= 10.0

    def _resolve_language_style_after_judge(self, res, generated_body: str):
        """
        language_style must stay on 1-10. Never persist 0 or out-of-range values;
        record an English explanation in scores['language_style_error'] when a fallback is used.
        """
        scores = res.setdefault("scores", {})
        scores.pop("language_style_error", None)
        raw = scores.get("language_style")
        if self._score_in_one_to_ten(raw):
            scores["language_style"] = round(float(raw), 4)
            return

        reasons = []
        if raw == 0 or raw == 0.0:
            reasons.append("Judge returned language_style=0, which is invalid on the 1-10 rubric.")
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            reasons.append(
                f"Judge returned language_style={raw}, which is outside the required 1-10 range."
            )
        elif raw is not None:
            reasons.append("Judge returned a non-numeric language_style.")
        else:
            reasons.append("Judge omitted language_style.")

        heur = language_style_score(generated_body) if generated_body else None
        clar = scores.get("clarity")
        new_val = None
        if isinstance(heur, (int, float)):
            new_val = float(heur)
            reasons.append(f"Fallback: heuristic language_style_score={new_val:.4f}.")
        elif self._score_in_one_to_ten(clar):
            new_val = float(clar)
            reasons.append("Fallback: copied clarity score.")
        else:
            new_val = 1.0
            reasons.append("Fallback: default 1.0 (heuristic unavailable and clarity missing or invalid).")

        scores["language_style"] = round(new_val, 4)
        scores["language_style_error"] = " ".join(reasons)

    def _resolve_constructiveness_after_judge(self, res, generated_body: str):
        """
        constructiveness must stay on 1-10. If the judge omits it or returns 0, fall back to a safer value.
        """
        scores = res.setdefault("scores", {})
        raw = scores.get("constructiveness")
        if self._score_in_one_to_ten(raw):
            scores["constructiveness"] = round(float(raw), 4)
            scores.pop("constructiveness_error", None)
            return

        reasons = []
        if raw == 0 or raw == 0.0:
            reasons.append("Judge returned constructiveness=0, which is invalid on the 1-10 rubric.")
        elif raw is None:
            reasons.append("Judge omitted constructiveness.")
        elif isinstance(raw, (int, float)):
            reasons.append(f"Judge returned constructiveness={raw}, which is outside the required 1-10 range.")
        else:
            reasons.append("Judge returned a non-numeric constructiveness.")

        clar = scores.get("clarity")
        if self._score_in_one_to_ten(clar):
            new_val = float(clar)
            reasons.append(f"Fallback: copied clarity={new_val:.4f}.")
        else:
            new_val = 1.0
            reasons.append("Fallback: default 1.0 (clarity unavailable/invalid).")

        scores["constructiveness"] = round(new_val, 4)
        scores["constructiveness_error"] = " ".join(reasons)

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

            # Step 1: retrieval / reference summaries for the prompt
            ref_summary = self._process_retrieval(item, task_key=task_key)
            
            # Step 2: user prompt inputs
            # Always use real dataset inputs (OA and rebuttal), never generated outputs.
            oa_content = item['input'].get('office_action', "No Office Action provided in dataset.")
            rebuttal_text = item['input'].get('rebuttal', item.get('ground_truth', {}).get('rebuttal_text', "N/A"))
            
            enforce_instruction = "\n\nIMPORTANT: Use the provided Reference Summaries to cite specific documents. DO NOT use placeholders like '[insert references]'."
            
            claim_blocks = [("all", item['input'].get('claims', ""))]
            if granularity == "atomic_claim":
                split_claims = self._split_claims(item['input'].get('claims', ""))
                if split_claims:
                    claim_blocks = split_claims

            _rrn = self.config.get("task_settings", {}).get("_run_round_number")
            _round_num = int(_rrn) if _rrn is not None else self._get_round_number(task_key)

            for claim_no, claim_text in claim_blocks:
                # Model input payload
                ai_input = {
                    "claims": f"Claim {claim_no}: {claim_text}" if granularity == "atomic_claim" else item['input'].get('claims_for_generation', item['input'].get('claims', '')),
                    "refs": ref_summary,
                    "task_type": task_key,
                    "prosecution_history": item['input'].get('prosecution_history', "N/A"),
                    "granularity": granularity
                }
                # For rebuttal, we don't include retrieval_setting into the saved ai_input.json.
                if task_key != "rebuttal":
                    ai_input["retrieval_setting"] = self.config['active_settings'].get('retrieval_setting', "oracle_summary")

                # Include true prompt inputs for traceability
                if task_key == "rebuttal":
                    ai_input["office_action"] = oa_content
                elif task_key == "oa":
                    # OA response-to-rebuttal should include the rebuttal being responded to
                    if template_task != "initial_examination":
                        ai_input["rebuttal"] = rebuttal_text
                
                try:
                    # Build user prompt from template
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
                    if task_key == "oa":
                        user_prompt += MODEL_OA_TYPE_TRAILER_USER_SUFFIX
                except KeyError as e:
                    self.logger.error(f"Prompt template error: {e}")
                    raise e

                # Step 3: generation
                pid = item.get("id", "?")
                self.logger.info(
                    "API generate start: patent_id=%s template=%s claim_block=%s",
                    pid,
                    template_task,
                    claim_no,
                )
                t0 = time.perf_counter()
                prediction = self.backend.generate(
                    system_prompt=template_config["system_prompt"],
                    user_prompt=user_prompt
                )
                elapsed = time.perf_counter() - t0
                out_len = len(prediction) if isinstance(prediction, str) else 0
                self.logger.info(
                    "API generate done: patent_id=%s claim_block=%s elapsed_sec=%.2f out_chars=%s",
                    pid,
                    claim_no,
                    elapsed,
                    out_len,
                )

                # Ground truth label for metrics
                response_templates = {
                    "response_to_rebuttal",
                    "response_to_rebuttal_zero_shot",
                    "response_to_rebuttal_oracle",
                    "response_to_rebuttal_bm25",
                }
                if task_key == "oa" and template_task in response_templates:
                    # response_to_rebuttal: the model is expected to generate the CURRENT round OA in response
                    # to the applicant's rebuttal. Therefore, use current oa_text as ground truth.
                    ground_truth_text = item['ground_truth'].get('oa_text', "N/A")
                elif task_key == "rebuttal":
                    ground_truth_text = item['ground_truth'].get('rebuttal_text', "N/A")
                else:
                    ground_truth_text = item['ground_truth'].get('oa_text', "N/A")

                gt_citations = item.get("ground_truth", {}).get("citations_gt", [])

                result_entry = {
                    "patent_id": item['id'],
                    "meta": item.get("meta") or {},
                    "round_info": {
                        "task": task_key,
                        "round_number": _round_num,
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
                            "office_action": oa_content,
                            "rebuttal_history": item['input'].get('rebuttal', "N/A")
                        },
                        "ground_truth": ground_truth_text,
                        "citations_gt": gt_citations,
                        "ground_truth_decision": item.get("ground_truth", {}).get("decision", ""),
                    }
                }
                results.append(result_entry)

                inc_dir = self.config.get("task_settings", {}).get("_incremental_save_exp_dir")
                if inc_dir and self.config.get("task_settings", {}).get(
                    "incremental_disk_save", True
                ):
                    try:
                        save_one_patent_round_artifacts(
                            result_entry,
                            inc_dir,
                            write_scores=False,
                            eval_result=None,
                        )
                        if task_key == "oa":
                            _tl = "oa"
                        elif task_key == "rebuttal":
                            _tl = "rebuttal"
                        else:
                            _tl = "other"
                        self.logger.info(
                            "Disk checkpoint: patent_id=%s wrote %s",
                            item.get("id"),
                            os.path.join(
                                inc_dir,
                                f"patent_{item['id']}",
                                f"round_{_round_num:02d}_{_tl}",
                                "output",
                                "ai_output.txt",
                            ),
                        )
                    except Exception as e:
                        self.logger.warning("Incremental disk save failed: %s", e)
        return results

    def _get_round_number(self, task_key):

        if task_key == "oa":
            return 1
        elif task_key == "rebuttal":
            return 2
        else:
            return 0

    def _get_restricted_info(self, task_key):
        """Names of restricted evaluation fields for logging / UI."""
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

    def _safe_int(self, value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    def _count_tokens(self, text):
        if not isinstance(text, str) or not text:
            return 0
        # Optional dependency: use tiktoken when available for accurate token counting.
        try:
            import tiktoken  # noqa: F401
        except Exception:
            # Fallback heuristic when tiktoken is unavailable.
            return max(1, len(text) // 4)

        model_name = (
            self.config.get("models", {}).get("judge_model")
            or self.config.get("models", {}).get("backbone_list", [""])[0]
        )
        try:
            enc = tiktoken.encoding_for_model(model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text, disallowed_special=()))

    def _truncate_text_tokens(self, text, max_tokens, head_ratio=0.6):
        if not isinstance(text, str) or not text:
            return text
        max_tokens = self._safe_int(max_tokens, 0)
        if max_tokens <= 0:
            return text
        try:
            import tiktoken
            model_name = (
                self.config.get("models", {}).get("judge_model")
                or self.config.get("models", {}).get("backbone_list", [""])[0]
            )
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            toks = enc.encode(text, disallowed_special=())
            if len(toks) <= max_tokens:
                return text
            marker = "\n...[TRUNCATED]...\n"
            marker_toks = enc.encode(marker, disallowed_special=())
            head = int(max_tokens * head_ratio)
            tail = max_tokens - head - len(marker_toks)
            if tail <= 0:
                return enc.decode(toks[:max_tokens])
            return enc.decode(toks[:head]) + marker + enc.decode(toks[-tail:])
        except Exception:
            # Fallback to char-based truncation if tokenizer is unavailable.
            return self._truncate_text(text, max_tokens * 4, head_ratio=head_ratio)

    def _truncate_by_limits(self, text, max_chars=0, max_tokens=0):
        if DISABLE_TRUNCATION:
            return text
        out = text
        max_tokens = self._safe_int(max_tokens, 0)
        max_chars = self._safe_int(max_chars, 0)
        if max_tokens > 0:
            out = self._truncate_text_tokens(out, max_tokens)
        if max_chars > 0:
            out = self._truncate_text(out, max_chars)
        return out

    def _within_limits(self, text, max_chars=0, max_tokens=0):
        if DISABLE_TRUNCATION:
            return True
        max_tokens = self._safe_int(max_tokens, 0)
        max_chars = self._safe_int(max_chars, 0)
        if max_chars > 0 and len(text) > max_chars:
            return False
        if max_tokens > 0 and self._count_tokens(text) > max_tokens:
            return False
        return True

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

    def _shrink_prompt(self, template, fields, max_total_chars=0, max_total_tokens=0):
        if DISABLE_TRUNCATION:
            return self._safe_template_format(template, fields)
        prompt = self._safe_template_format(template, fields)
        if self._within_limits(prompt, max_total_chars, max_total_tokens):
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
            if self._within_limits(prompt, max_total_chars, max_total_tokens):
                break
            value = fields.get(key)
            if isinstance(value, str) and len(value) > 500:
                new_char_limit = 0
                new_token_limit = 0
                if max_total_chars and max_total_chars > 0:
                    overflow_chars = max(0, len(prompt) - max_total_chars)
                    new_char_limit = max(500, len(value) - overflow_chars - 50)
                if max_total_tokens and max_total_tokens > 0:
                    value_tokens = self._count_tokens(value)
                    prompt_tokens = self._count_tokens(prompt)
                    overflow_tokens = max(0, prompt_tokens - max_total_tokens)
                    new_token_limit = max(200, value_tokens - overflow_tokens - 20)
                fields[key] = self._truncate_by_limits(value, new_char_limit, new_token_limit)
                prompt = self._safe_template_format(template, fields)
        return prompt

    def _build_judge_prompt(self, template, fields):
        if DISABLE_TRUNCATION:
            processed_fields = fields.copy() if isinstance(fields, dict) else {}
            if "current_response_target" in processed_fields:
                response_target = processed_fields["current_response_target"]
                if response_target and isinstance(response_target, str) and response_target.strip():
                    processed_fields["response_target_section"] = (
                        f"- Content Being Responded To:\n{response_target}\n"
                    )
                else:
                    processed_fields["response_target_section"] = ""
            return self._safe_template_format(template, processed_fields)

        cfg = self.config.get("task_settings", {})
        max_total_chars = self._safe_int(cfg.get("judge_max_chars", 0), 0)
        max_total_tokens = self._safe_int(cfg.get("judge_max_tokens", 0), 0)
        default_chars = self._safe_int(cfg.get("judge_default_field_max_chars", 0), 0)
        default_tokens = self._safe_int(cfg.get("judge_default_field_max_tokens", 0), 0)
        per_field_chars = cfg.get("judge_field_max_chars", {}) or {}
        per_field_tokens = cfg.get("judge_field_max_tokens", {}) or {}

        # Build conditional section for response target (what this round is responding to)
        processed_fields = fields.copy()
        
        # Handle response_target_section - the content being responded to in this round
        if "current_response_target" in processed_fields:
            response_target = processed_fields["current_response_target"]
            if response_target and isinstance(response_target, str) and response_target.strip():
                processed_fields["response_target_section"] = f"- Content Being Responded To:\n{response_target}\n"
            else:
                processed_fields["response_target_section"] = ""

        truncated = {}
        for k, v in processed_fields.items():
            if isinstance(v, str):
                limit_chars = self._safe_int(per_field_chars.get(k, default_chars), 0)
                limit_tokens = self._safe_int(per_field_tokens.get(k, default_tokens), 0)
                if (not limit_chars or limit_chars <= 0) and (not limit_tokens or limit_tokens <= 0):
                    truncated[k] = v
                else:
                    truncated[k] = self._truncate_by_limits(v, limit_chars, limit_tokens)
            else:
                truncated[k] = v

        prompt = self._safe_template_format(template, truncated)
        if not self._within_limits(prompt, max_total_chars, max_total_tokens):
            prompt = self._shrink_prompt(
                template,
                truncated,
                max_total_chars=max_total_chars,
                max_total_tokens=max_total_tokens,
            )
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
        # oracle_reference_summaries typically uses lines like:
        # "Ref US20080038979A1: <content>"
        # Keep the regex permissive to handle small formatting variations.
        m = re.search(r"Ref\s+([A-Za-z0-9]+):", text)
        if m:
            return m.group(1).strip()
        # Try common publication/patent ID patterns.
        m = re.search(r"\b[A-Z]{2}\s?\d{4}/\d{7}\s?[A-Z]\d?\b", text)
        if m:
            return re.sub(r"\s+", "", m.group(0))
        m = re.search(r"\b[A-Z]{2}\d{7,}[A-Z]?\d?\b", text)
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
        # Unlimited reference text budget (explicitly disable all limits).
        return 10**12

    def _count_text_tokens(self, text):
        s = text if isinstance(text, str) else ""
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(s))
        except Exception:
            return len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", s))

    def _truncate_text_to_token_budget(self, text, budget, add_marker=True):
        """
        Truncate text by token budget (not character budget).
        Uses binary search on character prefix length to find a near-max prefix.
        """
        if DISABLE_TRUNCATION:
            return text
        s = text if isinstance(text, str) else ""
        if budget is None:
            budget = 0
        try:
            budget = int(budget)
        except Exception:
            budget = 0
        if budget <= 0:
            return ""
        if self._count_text_tokens(s) <= budget:
            return s

        marker = "\n[Truncated to input limit]"
        marker_toks = self._count_text_tokens(marker)
        use_marker = bool(add_marker) and budget > (marker_toks + 1)
        content_budget = budget - marker_toks if use_marker else budget

        lo, hi = 0, len(s)
        best = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = s[:mid]
            if self._count_text_tokens(cand) <= content_budget:
                best = cand
                lo = mid + 1
            else:
                hi = mid - 1

        out = best.rstrip()
        if use_marker and out:
            out = out + marker

        # Safety clamp (for tokenizer edge behavior around boundaries).
        while out and self._count_text_tokens(out) > budget:
            out = out[:-1]
        return out

    def _pack_ref_entries_with_limit(self, entries, limit):
        """
        Keep as many whole `Ref ...` entries as possible under `limit`.
        If the next entry would exceed `limit`, truncate only that entry's tail.
        """
        if not isinstance(entries, list):
            entries = []
        cleaned = [e.strip() for e in entries if isinstance(e, str) and e.strip()]
        if not cleaned:
            return ""

        marker = "\n[Truncated to input limit]"
        joined = "\n".join(cleaned).strip()
        if len(joined) <= limit:
            return joined

        out = []
        cur_len = 0
        for e in cleaned:
            sep_len = 1 if out else 0
            if cur_len + sep_len + len(e) <= limit:
                out.append(e)
                cur_len += sep_len + len(e)
                continue

            # Need to truncate this entry only.
            remain = limit - cur_len - sep_len
            if remain <= 0:
                break
            if remain <= len(marker):
                # No room for meaningful content; stop with previous full entries.
                break
            tail_budget = remain - len(marker)
            out.append(e[:tail_budget] + marker)
            cur_len = limit
            break

        return "\n".join(out).strip()

    def _pack_ref_entries_even_budget(self, entries, limit):
        """
        Even-budget packing:
        - Allocate token budget evenly across refs.
        - Used only after switching to independent-only refs and still over limit.
        """
        if not isinstance(entries, list):
            entries = []
        cleaned = [e.strip() for e in entries if isinstance(e, str) and e.strip()]
        n = len(cleaned)
        if n == 0:
            return ""

        try:
            limit = int(limit)
        except Exception:
            limit = 0
        if limit <= 0:
            return ""

        sep_tok = self._count_text_tokens("\n")
        sep_total_toks = max(0, n - 1) * sep_tok
        content_budget = max(0, limit - sep_total_toks)
        if content_budget <= 0:
            return ""

        entry_toks = [self._count_text_tokens(e) for e in cleaned]
        base = max(1, content_budget // n)
        alloc = [min(t, base) for t in entry_toks]
        used = sum(alloc)
        remain = content_budget - used

        # Round-robin distribute remaining tokens to refs not yet fully allocated.
        if remain > 0:
            i = 0
            while remain > 0 and any(alloc[j] < entry_toks[j] for j in range(n)):
                if alloc[i] < entry_toks[i]:
                    alloc[i] += 1
                    remain -= 1
                i = (i + 1) % n

        out = []
        for e, k, t in zip(cleaned, alloc, entry_toks):
            if k >= t:
                out.append(e)
            else:
                out.append(self._truncate_text_to_token_budget(e, k, add_marker=True))

        joined = "\n".join(out).strip()
        if self._count_text_tokens(joined) > limit:
            joined = self._truncate_text_to_token_budget(joined, limit, add_marker=True)
        return joined

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

    def _ref_stems_for_app_style_pool(self, clean_id):
        """Alternate APP-style filenames (no US* / bare-digit pool names for application ids)."""
        if not clean_id or not clean_id.startswith("APP") or len(clean_id) <= 3:
            return []
        rest = clean_id[3:]
        if not rest.isdigit() or len(rest) < 8:
            return []
        return [f"APP-{rest[:2]}_{rest[2:]}"]

    def _ref_possible_names(self, clean_id):
        names = []
        # First expand APP-style ids to multiple pool naming conventions, then kind-code variants.
        stem_seeds = [clean_id]
        if clean_id.startswith("APP"):
            stem_seeds.extend(self._ref_stems_for_app_style_pool(clean_id))
        seen_stems = set()
        ordered_stems = []
        for s in stem_seeds:
            if s in seen_stems:
                continue
            seen_stems.add(s)
            ordered_stems.append(s)

        for seed in ordered_stems:
            for cid in self._ref_id_variants(seed):
                if cid.startswith("APP"):
                    names.append(f"{cid}.json")
                elif cid.startswith(("US", "WO", "EP", "AU")):
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

    def _process_retrieval(self, item, task_key=None):
        setting = self.config['active_settings'].get('retrieval_setting', "oracle_summary")
        
        # Rebuttal generation does not perform inference-time retrieval.
        # Always use the per-sample aligned prosecution references:
        # 1) If extracted_reference_contexts exists => only keep those blocks.
        #    (mapped to ground_truth.citations_gt)
        # 2) Else => keep the remaining references cited in prosecution.
        # 3) If both empty => return empty string.
        if str(task_key).lower() == "rebuttal":
            oracle_text = item.get("input", {}).get("oracle_reference_summaries", "") or ""
            oracle_text = oracle_text.strip()
            if not oracle_text:
                return ""

            extracted_ids = item.get("ground_truth", {}).get("citations_gt", []) or []
            extracted_set = set(
                self._normalize_doc_id(x)
                for x in extracted_ids
                if isinstance(x, str) and x.strip()
            )

            limit = self._get_ref_text_limit()
            blocks = self._split_ref_blocks(oracle_text)
            if not blocks:
                blocks = [oracle_text]

            if extracted_set:
                kept = []
                for b in blocks:
                    rid = self._extract_ref_id(b)
                    if self._normalize_doc_id(rid) in extracted_set:
                        kept.append(b)
                blocks = kept

            packed = self._pack_ref_entries_with_limit(blocks, limit)
            return packed if packed else ""

        if setting == "no_ref":
            return "No prior art references provided for this examination."
        
        elif setting == "oracle_summary":
            limit = self._get_ref_text_limit()
            content = item.get("input", {}).get('oracle_reference_summaries', "")
            ids = item.get("input", {}).get("oracle_reference_ids", [])
            if isinstance(ids, list) and ids:
                entries_full = []
                for rid in ids:
                    doc_id, ref_data = self._load_reference_record(rid)
                    entries_full.append(self._build_oracle_ref_entry(doc_id, ref_data, independent_only=False))
                full_text = "\n".join(entries_full).strip()
                if full_text and self._count_text_tokens(full_text) <= limit:
                    return full_text

                # Over limit: prefer independent-claim-only per reference.
                entries_ind = []
                for rid in ids:
                    doc_id, ref_data = self._load_reference_record(rid)
                    entries_ind.append(self._build_oracle_ref_entry(doc_id, ref_data, independent_only=True))
                ind_text = "\n".join(entries_ind).strip()
                if ind_text and self._count_text_tokens(ind_text) <= limit:
                    return ind_text

                # Still over limit: use even per-reference budget on independent-only refs.
                packed = self._pack_ref_entries_even_budget(
                    entries_ind if entries_ind else entries_full, limit
                )
                return packed if packed else "Warning: Oracle references not available."

            # Fallback when ids are unavailable: use prebuilt summaries.
            if isinstance(content, str) and content.strip():
                content = content.strip()
                blocks = self._split_ref_blocks(content)
                if not blocks:
                    blocks = self._split_reference_summaries(content)
                packed = self._pack_ref_entries_with_limit(blocks if blocks else [content], limit)
                return packed if packed else "Warning: Oracle references not available."

            return "Warning: Oracle reference IDs are missing in test set."
        elif setting == "bm25_retrieved":
            limit = self._get_ref_text_limit()
            cached_bm25 = item.get("input", {}).get("bm25_reference_summaries", "")
            cached_bm25 = cached_bm25.strip() if isinstance(cached_bm25, str) else ""

            query_text = item.get("input", {}).get("claims", "")
            oa_d = oa_cutoff_date((item.get("meta") or {}).get("oa_date"))
            if oa_d is None and (item.get("meta") or {}).get("oa_date"):
                self.logger.warning(
                    "bm25_retrieved: meta.oa_date not parseable; BM25 runs without temporal pool filter."
                )

            # BM25 side: prefer precomputed summaries from the test set; otherwise run BM25 on claims.
            # Always merge with Oracle below (never return cached BM25 alone).
            bm25_list = []
            if cached_bm25:
                bm25_list = self._split_ref_blocks(cached_bm25)
                if not bm25_list:
                    bm25_list = self._split_reference_summaries(cached_bm25)
                if not bm25_list:
                    bm25_list = [cached_bm25]
            if not bm25_list:
                bm25_list = self._bm25_retrieve_list(query_text, top_k=5, cutoff_date=oa_d)

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
            if self._count_text_tokens(joined) <= limit:
                return joined

            # Over limit: prefer independent-claim-only oracle entries.
            pool_for_pack = pool
            if isinstance(ids, list) and ids:
                oracle_entries_ind = []
                for rid in ids:
                    doc_id, ref_data = self._load_reference_record(rid)
                    oracle_entries_ind.append(self._build_oracle_ref_entry(doc_id, ref_data, independent_only=True))
                pool_ind = _build_pool(oracle_entries_ind)
                joined_ind = "\n".join(pool_ind[:10]).strip()
                if self._count_text_tokens(joined_ind) <= limit:
                    return joined_ind
                pool_for_pack = pool_ind if pool_ind else pool

            # Still over limit: even budget across refs (after independent-only fallback).
            packed = self._pack_ref_entries_even_budget(pool_for_pack[:10], limit)
            return packed if packed else "References corpus not available."
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
        doc_dates = []
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
                d_doc, _ = ref_doc_effective_date(data)
                doc_dates.append(d_doc)
            except:
                continue
        self._corpus_docs = docs
        self._docids = ids
        self._doclens = lens
        self._doc_dates = doc_dates
        self._avgdl = sum(lens) / len(lens) if lens else 0.0
        self._corpus_loaded = True
        if docs and not self._bm25_temporal_policy_logged:
            self.logger.info(reference_pool_date_policy_log_line())
            self._bm25_temporal_policy_logged = True

    def _bm25_retrieve_list(self, query_text, top_k=5, cutoff_date=None):
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
        if cutoff_date is not None and self._doc_dates:
            filtered = []
            for s, i in scores:
                if i >= len(self._doc_dates):
                    continue
                dd = self._doc_dates[i]
                if dd is None or dd >= cutoff_date:
                    continue
                filtered.append((s, i))
            scores = filtered
        ref_dir = self.config['paths'].get('reference_pool')
        out = []
        seen_canonical = set()
        for s, i in scores:
            if len(out) >= top_k:
                break
            doc_id = self._docids[i]
            # Deduplicate publication variants by numeric core:
            # e.g., US1234567A / US1234567A1 / US1234567 -> one entry.
            canon = re.sub(r"[A-Z]\d?$", "", self._normalize_doc_id(doc_id))
            if not canon:
                canon = self._normalize_doc_id(doc_id)
            if canon in seen_canonical:
                continue
            seen_canonical.add(canon)
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

    def run_scoring(self, results, show_progress=True):
        """
        Run judge scoring (examiner_judge vs rebuttal_judge) per configured task.
        """
        # Route to the judge template that matches the current task (oa vs rebuttal)
        current_task = self.config['task_settings']['current_task']
        if current_task == "oa" or current_task in self.templates.get("Office_Action_Generation", {}):
            judge_key = "examiner_judge"
        else:
            judge_key = "rebuttal_judge"
        
        self.logger.info(f" Routing Evaluation to: {judge_key}")
        
        judge_config = self.templates["Evaluation_Metrics"][judge_key]
        scored_results = []

        for res in tqdm(results, desc="Scoring", disable=not bool(show_progress)):
            # Prompt
            task_type = res.get('round_info', {}).get('task', '')
            round_num = res.get('round_info', {}).get('round_number', 1)
            
            # Build evaluation input: claims + ground_truth + generated + current context to respond to
            # Note: Evaluation does NOT need refs or previous history!
            # Only needs the content being responded to in this round
            current_response_target = ""  # Either current OA (for rebuttal) or current rebuttal (for OA)
            
            if task_type == "oa":
                # For OA evaluation:
                # Round 1: Only claims + ground_truth + generated
                # Round 2+: Also need the rebuttal being responded to
                if round_num > 1:
                    # Get the rebuttal that this OA is responding to
                    prev_context = res['ai_exam']['input'].get('prosecution_history', '')
                    if prev_context and isinstance(prev_context, str):
                        if 'Current Round Rebuttal:' in prev_context:
                            current_response_target = prev_context.replace('Current Round Rebuttal:', '').strip()
                        elif 'Previous Round Rebuttal:' in prev_context:
                            current_response_target = prev_context.replace('Previous Round Rebuttal:', '').strip()
                        elif 'Current Round Rebuttal:' in prev_context or 'Previous Round Rebuttal:' in prev_context:
                            # Multi-section format - extract only remarks
                            lines = prev_context.split('\n')
                            current_section = None
                            for line in lines:
                                if 'Current Round Rebuttal:' in line or 'Previous Round Rebuttal:' in line:
                                    current_section = 'remarks'
                                elif current_section == 'remarks' and line.strip():
                                    current_response_target += line + '\n'
                
            elif task_type == "rebuttal":
                # For rebuttal evaluation:
                # Must have current OA (the one being responded to)
                current_response_target = res['evaluation_context']['full_context'].get('office_action', '')
            
            # Defaults used only when judge prompt template placeholders mismatch.
            prev_remarks = ""
            prev_oa = ""

            gen_for_judge = self._output_body_without_oa_type_trailer(
                res["ai_exam"].get("output", ""), task_type
            )
            try:
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "generated_text": gen_for_judge,
                        "ground_truth": res['evaluation_context']['ground_truth'],
                        "current_response_target": current_response_target.strip(),
                    },
                )
            except KeyError as e:
                # Missing placeholders in the template; fall back with safe defaults.
                self.logger.warning(f" Prompt format warning: {e}")
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "generated_text": gen_for_judge,
                        "ground_truth": res.get('evaluation_context', {}).get('ground_truth', ''),
                        "previous_remarks": prev_remarks or "",
                        "previous_office_action": prev_oa or "",
                    },
                )

            judge_attempts = []
            judge_system_prompt = judge_config["system_prompt"]
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
                # Parse judge JSON output
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
                self._resolve_language_style_after_judge(res, gen_for_judge)
                self._resolve_constructiveness_after_judge(res, gen_for_judge)
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
            _tt = res.get("round_info", {}).get("task", "")
            try:
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "generated_text": self._output_body_without_oa_type_trailer(
                            res["ai_exam"]["output"], _tt
                        ),
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
                        "generated_text": self._output_body_without_oa_type_trailer(
                            res["ai_exam"]["output"], _tt
                        ),
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
        Pairwise comparison of consecutive result entries using the pairwise judge template.
        """
        eval_templates = self.templates.get("Evaluation_Metrics", {})
        judge_config = eval_templates.get("pairwise_evaluator")
        if not judge_config:
            self.logger.info(" Pairwise evaluator template not found. Skipping.")
            return results
        self.logger.info(" Running pairwise comparison...")
        compared_results = []

        for i in range(0, len(results), 2):
            if i + 1 >= len(results):
  
                res = results[i]
                res["pairwise_winner"] = None
                res["pairwise_comparison"] = "Insufficient pairs for comparison"
                compared_results.append(res)
                break
            
            res_a = results[i]
            res_b = results[i + 1]
            _ta = res_a.get("round_info", {}).get("task", "")
            _tb = res_b.get("round_info", {}).get("task", "")

            try:
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "text_a": self._output_body_without_oa_type_trailer(
                            res_a["ai_exam"]["output"], _ta
                        ),
                        "text_b": self._output_body_without_oa_type_trailer(
                            res_b["ai_exam"]["output"], _tb
                        ),
                        "claims": res_a['ai_exam']['input']['claims'],
                        "office_action": res_a['evaluation_context']['full_context']['office_action'],
                    },
                )
            except KeyError as e:
                self.logger.warning(f" Pairwise prompt format warning: {e}")
                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "text_a": self._output_body_without_oa_type_trailer(
                            res_a["ai_exam"]["output"], _ta
                        ),
                        "text_b": self._output_body_without_oa_type_trailer(
                            res_b["ai_exam"]["output"], _tb
                        ),
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
                
                res_a["pairwise_winner"] = winner
                res_a["pairwise_scores"] = score_data.get("scores", {})
                res_a["pairwise_justification"] = score_data.get("justification", "")
                
                # Invert winner label for the paired sample
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
        Optional claim-level judge pass when a claim_level_evaluator template exists.
        """
        eval_templates = self.templates.get("Evaluation_Metrics", {})
        judge_config = eval_templates.get("claim_level_evaluator")
        if not judge_config:
            self.logger.info(" Claim-level evaluator template not found. Skipping.")
            return results
        self.logger.info(" Running claim-level evaluation...")
        evaluated_results = []

        for res in tqdm(results, desc="Claim-level Evaluation"):
            _ct = res.get("round_info", {}).get("task", "")
            try:
                claim_text = res.get('ai_exam', {}).get('input', {}).get('claims', '')
                claim_number = res.get('round_info', {}).get('round_number', '1')

                judge_prompt = self._build_judge_prompt(
                    judge_config["user_prompt_template"],
                    {
                        "generated_text": self._output_body_without_oa_type_trailer(
                            res["ai_exam"]["output"], _ct
                        ),
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
                        "generated_text": self._output_body_without_oa_type_trailer(
                            res["ai_exam"]["output"], _ct
                        ),
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
        Rule-based + similarity metrics. OA disposition: prefer <<<MODEL_OA_TYPE>>> JSON
        trailer in model output, else extract_disposition on body text (see metrics_library).
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
                task_type = res.get("round_info", {}).get("task", "")

                pred_json = safe_parse_json(pred_text)
                gt_json = safe_parse_json(gt_text)

                rule_scores = {}
                is_rebuttal = task_type == "rebuttal"
                pred_body = pred_text
                pred_disp_source = ""
                if not is_rebuttal:
                    pred_disp, pred_body, pred_disp_source = resolve_predicted_disposition(
                        pred_text
                    )
                    # GT disposition: (1) ground_truth.decision (2) meta.oa_type (3) OA body / JSON heuristics
                    gt_disp = "unknown"
                    gt_decision = (res.get("evaluation_context") or {}).get(
                        "ground_truth_decision", ""
                    )
                    if isinstance(gt_decision, str) and gt_decision.strip():
                        if gt_decision.strip().upper() != "UNKNOWN":
                            gt_disp = disposition_from_oa_type(gt_decision)
                    if gt_disp == "unknown":
                        gt_disp = disposition_from_oa_type(
                            (res.get("meta") or {}).get("oa_type", "")
                        )
                    if gt_disp == "unknown":
                        gt_disp = extract_disposition(gt_text, gt_json)

                    pred_stat = extract_statutes(pred_body)
                    gt_stat = extract_statutes(gt_text)

                    pred_rejections = extract_rejection_types(pred_body)

                    # Per-sample: only keep accuracy to avoid misleading precision/recall/F1
                    disp_scores = {}
                    if pred_disp != "unknown" and gt_disp not in ("unknown", "procedural_action"):
                        disp_scores["disposition_accuracy"] = round(1.0 if pred_disp == gt_disp else 0.0, 4)
                    stat_scores = statute_metrics(pred_stat, gt_stat)
                    gt_ids = res.get("evaluation_context", {}).get("citations_gt", None)
                    cit_scores = {}
                    ref_ids = set()
                    pred_ids = set()
                    list_match = None
                    refs_text = ""  # refs are intentionally removed from stored inputs

                    # Citation metrics: rely only on dataset ground-truth citations (citations_gt),
                    # not on retrieval-time reference text (which we intentionally removed from stored inputs).
                    if gt_ids is not None:
                        cit_scores, ref_ids, pred_ids = citation_metrics(
                            pred_body, "", gt_ids=gt_ids
                        )
                        # Citation List Match: compare predicted citations against the same GT set.
                        if not pred_ids:
                            list_match = 1.0
                        elif not ref_ids:
                            list_match = 0.0
                        else:
                            list_match = round(len(pred_ids & ref_ids) / len(pred_ids), 4)
                    else:
                        list_match = citation_list_match(pred_body, refs_text)

                    # Rejection type metrics (use dataset rejection_types as GT)
                    gt_rejections = normalize_rejection_types(
                        res.get("meta", {}).get("rejection_types", [])
                    )
                    rej_scores = rejection_type_metrics(pred_rejections, gt_rejections)

                    rule_scores.update(disp_scores)
                    rule_scores.update(stat_scores)
                    rule_scores.update(cit_scores)
                    rule_scores.update(rej_scores)
                    rule_scores["citation_list_match"] = list_match
                    if "disposition_accuracy" in rule_scores:
                        rule_scores["disposition_accuracy_pct"] = round(rule_scores["disposition_accuracy"] * 100, 2)
                    if "statute_precision" in rule_scores:
                        rule_scores["statute_precision_pct"] = round(rule_scores["statute_precision"] * 100, 2)
                    rule_scores["citation_list_match_pct"] = round(list_match * 100, 2)

                sim_pred = pred_body if not is_rebuttal else pred_text
                if enable_rouge_l:
                    r_l = rouge_l_f1(sim_pred, gt_text)
                    if r_l is not None:
                        rule_scores["rouge_l"] = r_l
                if enable_bleu:
                    b = bleu_score(sim_pred, gt_text)
                    if b is not None:
                        rule_scores["bleu"] = b
                if enable_bert_score:
                    bs = bert_score_f1(sim_pred, gt_text)
                    if bs is not None:
                        rule_scores["bert_score"] = bs

                fmt = formatting_score(sim_pred, task_type)
                if fmt is not None:
                    rule_scores["formatting_rule"] = fmt
                if not isinstance(res.get("scores", {}).get("language_style"), (int, float)):
                    ls = language_style_score(sim_pred)
                    if ls is not None:
                        rule_scores["language_style"] = ls

                res.setdefault("scores", {})
                res["rule_scores"] = rule_scores
                res["scores"].update(rule_scores)
                if not is_rebuttal:
                    res["metric_details"] = {
                        "disposition_pred": pred_disp,
                        "disposition_pred_source": pred_disp_source,
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
            # Rebuttal accuracy: recompute using judge sub-dimensions.
            # Keep 1-10 scale and renormalize weights when any sub-dimension is missing.
            effectiveness = scores.get("effectiveness")
            rationality = scores.get("rationality")  # kept for backward compatibility; not used directly here
            reliability = scores.get("reliability")
            feature_distinction = scores.get("feature_distinction")
            specification_support = scores.get("specification_support")

            # Default weighting scheme:
            # - accuracy dominates (0.70)
            # - the remaining 0.30 is split equally across the other four dimensions
            w_base = 0.70
            w_rest = 0.30 / 4.0

            weighted_sum = base_acc * w_base
            weight_sum = w_base

            def add_if_present(metric_val, w):
                nonlocal weighted_sum, weight_sum
                if isinstance(metric_val, (int, float)) and not isinstance(metric_val, bool):
                    weighted_sum += float(metric_val) * w
                    weight_sum += w

            add_if_present(effectiveness, w_rest)
            add_if_present(reliability, w_rest)
            add_if_present(feature_distinction, w_rest)
            add_if_present(specification_support, w_rest)

            if weight_sum > 0:
                merged = weighted_sum / weight_sum
                merged = max(0.0, min(10.0, merged))
                scores["accuracy"] = round(merged, 4)
            return

        # OA: Recompute "accuracy" using the same weighted scheme as tools/recompute_accuracy.py.
        # This uses LLM judge sub-scores (accuracy/rationality/reliability) and ignores
        # extracted core-metric fields (disposition/statute/citation precision) to avoid mixing scales.
        rationality = scores.get("rationality")
        reliability = scores.get("reliability")

        weighted_sum = base_acc * 0.70
        weight_sum = 0.70
        if isinstance(rationality, (int, float)) and not isinstance(rationality, bool):
            weighted_sum += float(rationality) * 0.15
            weight_sum += 0.15
        if isinstance(reliability, (int, float)) and not isinstance(reliability, bool):
            weighted_sum += float(reliability) * 0.15
            weight_sum += 0.15

        if weight_sum > 0:
            merged = weighted_sum / weight_sum
            # Clamp to 0-10 to match recompute_accuracy.py.
            merged = max(0.0, min(10.0, merged))
            scores["accuracy"] = round(merged, 4)

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
                "formatting_rule": (0.10, False)
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
                "formatting_rule": (0.06, False)
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

    def _build_previous_round_context(self, items, current_round_idx, task_type):
        """Build context from the previous same-type round.
        
        Rules:
        - OA: history = previous OA (skip rebuttal rounds)
        - Rebuttal: history = previous Rebuttal (skip OA rounds)
        """
        if current_round_idx <= 0 or current_round_idx >= len(items):
            return "N/A"

        if task_type == "oa":
            for j in range(current_round_idx - 1, -1, -1):
                prev_item = items[j]
                if str(prev_item.get("meta", {}).get("round_type", "")).lower() == "oa":
                    _, prev_oa, _ = self._extract_round_texts(prev_item)
                    if prev_oa and prev_oa.strip():
                        return f"Previous Round Office Action:\n{prev_oa}"
            return "N/A"

        if task_type == "rebuttal":
            for j in range(current_round_idx - 1, -1, -1):
                prev_item = items[j]
                if str(prev_item.get("meta", {}).get("round_type", "")).lower() == "rebuttal":
                    _, _, prev_rebuttal = self._extract_round_texts(prev_item)
                    if prev_rebuttal and prev_rebuttal.strip() and prev_rebuttal.upper() != "N/A":
                        return f"Previous Round Rebuttal:\n{prev_rebuttal}"
            return "N/A"

        return "N/A"


