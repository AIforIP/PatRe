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
        if not os.path.exists(path):
            self.logger.error(f"?? {path}")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        #  config 
        limit = self.config['task_settings'].get('num_samples', 9999)
        return data[:limit]

    def run_batch(self):
        """?- """
        results = []
        task_key = self.config['task_settings']['current_task']
        
        # --- ?---
        if task_key in self.templates.get("Office_Action_Generation", {}):
            root_key = "Office_Action_Generation"
        elif task_key in self.templates.get("Rebuttal_Generation", {}):
            root_key = "Rebuttal_Generation"
        else:
            self.logger.error(f"?: {task_key}")
            return []
            
        if task_key not in self.templates[root_key]:
            self.logger.error(f"??{root_key} : {task_key}")
            return []
            
        template_config = self.templates[root_key][task_key]
        
        self.logger.info(f" Starting batch inference for {len(self.dataset)} samples...")

        for item in tqdm(self.dataset, desc="Generating"):
            # 1. 
            ref_summary = self._process_retrieval(item)
            
            # 2. ?User Prompt
            # OAA?
            # OA
            if task_key == "v1_high_fidelity_rebuttal" and item['input'].get('next_office_action'):
                oa_content = item['input']['next_office_action']  # A
            else:
                oa_content = item['input'].get('office_action', "No Office Action provided in dataset.")
            
            enforce_instruction = "\n\nIMPORTANT: Use the provided Reference Summaries to cite specific documents. DO NOT use placeholders like '[insert references]'."
            
            # AI
            ai_input = {
                "claims": item['input']['claims'],
                "refs": ref_summary,
                "task_type": task_key,
                "retrieval_setting": self.config['active_settings'].get('retrieval_setting', "oracle_summary")
            }
            
            # AI
            if task_key == "v1_initial_examination":
                # IA?
                ai_input.pop("office_action", None)
                ai_input.pop("rebuttal_history", None)
            
            try:
                # AIprompt
                user_prompt = template_config["user_prompt_template"].format(
                    current_claims=ai_input['claims'],
                    reference_summaries=ai_input['refs'],
                    office_action=oa_content if task_key != "v1_initial_examination" else "N/A",
                    rebuttal=item['input'].get('rebuttal', "N/A") if task_key != "v1_initial_examination" else "N/A",
                    claim_count=item['meta'].get('claim_count', 1)
                ) + enforce_instruction
            except KeyError as e:
                self.logger.error(f"?Prompt ? {e}")
                raise e

            # 3. 
            prediction = self.backend.generate(
                system_prompt=template_config["system_prompt"],
                user_prompt=user_prompt
            )

            #  Ground Truth
            if task_key == "v1_response_to_rebuttal":
                ground_truth_text = item['ground_truth'].get('next_oa_text', item['ground_truth'].get('oa_text', "N/A"))
            elif "rebuttal" in task_key.lower():
                ground_truth_text = item['ground_truth'].get('rebuttal_text', "N/A")
            else:
                ground_truth_text = item['ground_truth']['oa_text']

            # ?
            result_entry = {
                "patent_id": item['id'],
                "round_info": {
                    "task": task_key,
                    "round_number": self._get_round_number(task_key),
                    "exam_phase": "pure_prediction" if task_key == "v1_initial_examination" else "context_aware"
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
                    "input_refs_preview": ref_summary[:200] + "..."  # 
                }
            }
            results.append(result_entry)
        return results

    def _get_round_number(self, task_key):
        """"""
        if task_key == "v1_initial_examination":
            return 1
        elif task_key == "v1_high_fidelity_rebuttal":
            return 2
        elif task_key == "v1_response_to_rebuttal":
            return 3
        else:
            return 0

    def _get_restricted_info(self, task_key):
        """AI"""
        if task_key == "v1_initial_examination":
            return ["ground_truth_oa", "rebuttal_history", "human_decisions"]
        else:
            return ["ground_truth_final_outcome"]

    def _process_retrieval(self, item):
        setting = self.config['active_settings'].get('retrieval_setting', "oracle_summary")
        
        if setting == "no_ref":
            return "No prior art references provided for this examination."
        
        elif setting == "oracle_summary":
            content = item['input'].get('oracle_reference_summaries', "").strip()
            return content if content else "Warning: Oracle summary data is missing in test set."
        elif setting == "bm25_retrieved":
            query_text = item['input'].get('claims', '')
            return self._bm25_retrieve(query_text)
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

    def _bm25_retrieve(self, query_text):
        if not self._corpus_loaded:
            self._load_reference_corpus()
        if not self._corpus_docs:
            return "References corpus not available."
        q = self._tokenize(query_text)
        if not q:
            return "No query terms found in claims."
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
        top_k = self.config.get('retrieval_params', {}).get('bm25_top_k', 5)
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
        return "\n".join(out)

    def run_scoring(self, results):
        """
        ?
         Task  examiner_judge ?rebuttal_judge
        """
        # 1.  ( "v1_initial_examination" ?"v1_high_fidelity_rebuttal")
        current_task = self.config['task_settings']['current_task']
        # 
        if current_task in self.templates.get("Office_Action_Generation", {}):
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
                judge_prompt = judge_config["user_prompt_template"].format(
                    generated_text=res['ai_exam']['output'],
                    claims=res['ai_exam']['input']['claims'],
                    reference_summaries=res['ai_exam']['input']['refs'],
                    office_action=res['evaluation_context']['full_context']['office_action'],
                    ground_truth=res['evaluation_context']['ground_truth']
                )
            except KeyError as e:
                # ?{claims} ormat 
                # Python ?format  key ?key  unpack dict?
                #  {claims} ?
                # ?context ?
                self.logger.warning(f" Prompt format warning: {e}")
                # Fallback to simple format if needed, or just let it fail/warn
                judge_prompt = judge_config["user_prompt_template"].format(
                    generated_text=res['ai_exam']['output']
                )

            # ?GPT-4o 
            raw_judge_res = self.judge_backend.generate(
                system_prompt=judge_config["system_prompt"],
                user_prompt=judge_prompt
            )

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
                    score_data = self._extract_json(raw_judge_res)
                res["scores"] = score_data.get("scores", {})
                res["justification"] = score_data.get("justification", "")

                # OA GPT-5 mini 
                if judge_key == "examiner_judge":
                    coherence_data = self._run_gpt5_coherence_judge(res)
                    if coherence_data:
                        res["scores"]["coherence_score"] = coherence_data.get("coherence_score", 0)
                        if coherence_data.get("justification"):
                            res["justification"] = (
                                (res["justification"] + "\n\n") if res["justification"] else ""
                            ) + f"[GPT-5 mini coherence] {coherence_data['justification']}"

                # ?total_score 
                res["total_score"] = self._calculate_numeric_total(res.get("scores", {}))
                scored_results.append(res)
            except Exception as e:
                self.logger.warning(f" Scoring failed for {res['patent_id']}: {e}")
                res["scores"] = {"error": "PARSING_FAILED"}
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
                judge_prompt = judge_config["user_prompt_template"].format(
                    generated_text=res['ai_exam']['output'],
                    ground_truth=res['evaluation_context']['ground_truth'],
                    claims=res['ai_exam']['input']['claims'],
                    office_action=res['evaluation_context']['full_context']['office_action']
                )
            except KeyError as e:
                self.logger.warning(f" Point-wise prompt format warning: {e}")
                judge_prompt = judge_config["user_prompt_template"].format(
                    generated_text=res['ai_exam']['output'],
                    ground_truth=res['evaluation_context']['ground_truth']
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
                judge_prompt = judge_config["user_prompt_template"].format(
                    text_a=res_a['ai_exam']['output'],
                    text_b=res_b['ai_exam']['output'],
                    claims=res_a['ai_exam']['input']['claims'],
                    office_action=res_a['evaluation_context']['full_context']['office_action']
                )
            except KeyError as e:
                self.logger.warning(f" Pairwise prompt format warning: {e}")
                judge_prompt = judge_config["user_prompt_template"].format(
                    text_a=res_a['ai_exam']['output'],
                    text_b=res_b['ai_exam']['output']
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
                
                judge_prompt = judge_config["user_prompt_template"].format(
                    generated_text=res['ai_exam']['output'],
                    ground_truth=res['evaluation_context']['ground_truth'],
                    claim_number=claim_number,
                    claim_text=claim_text,
                    office_action=res['evaluation_context']['full_context']['office_action']
                )
            except KeyError as e:
                self.logger.warning(f" Claim-level prompt format warning: {e}")
                judge_prompt = judge_config["user_prompt_template"].format(
                    generated_text=res['ai_exam']['output'],
                    ground_truth=res['evaluation_context']['ground_truth'],
                    claim_number='1',
                    claim_text=res['ai_exam']['input']['claims']
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

                pred_disp = extract_disposition(pred_text, pred_json)
                gt_disp = extract_disposition(gt_text, gt_json)

                pred_stat = extract_statutes(pred_text)
                gt_stat = extract_statutes(gt_text)

                disp_scores = disposition_metrics(pred_disp, gt_disp)
                stat_scores = statute_metrics(pred_stat, gt_stat)
                cit_scores, ref_ids, pred_ids = citation_metrics(pred_text, refs_text)

                scores = {}
                scores.update(disp_scores)
                scores.update(stat_scores)
                scores.update(cit_scores)

                r_l = rouge_l_f1(pred_text, gt_text)
                if r_l is not None:
                    scores["rouge_l"] = r_l
                b = bleu_score(pred_text, gt_text)
                if b is not None:
                    scores["bleu"] = b
                # Skip heavy metrics by default to avoid long stalls

                fmt = formatting_score(pred_text, task_type)
                if fmt is not None:
                    scores["formatting_rule"] = fmt

                res.setdefault("scores", {})
                res["scores"].update(scores)
                res["metric_details"] = {
                    "disposition_pred": pred_disp,
                    "disposition_gt": gt_disp,
                    "statutes_pred": sorted(list(pred_stat)),
                    "statutes_gt": sorted(list(gt_stat)),
                    "citations_pred": sorted(list(pred_ids)),
                    "citations_gt": sorted(list(ref_ids)),
                }
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
            prompt = extractor_cfg["user_prompt_template"].format(text=text)
        except KeyError as e:
            self.logger.warning(f"?Rediscovery extractor prompt warning: {e}")
            prompt = extractor_cfg["user_prompt_template"].format(text=text)

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
                prompt = matcher_cfg["user_prompt_template"].format(claim_a=claim_a, claim_b=claim_b)
            except KeyError as e:
                self.logger.warning(f"?Rediscovery matcher prompt warning: {e}")
                prompt = matcher_cfg["user_prompt_template"].format(claim_a=claim_a, claim_b=claim_b)
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
        for v in scores.values():
            if isinstance(v, (int, float)):
                total += v
        return total

    def _run_gpt5_coherence_judge(self, res):
        eval_templates = self.templates.get("Evaluation_Metrics", {})
        coherence_cfg = eval_templates.get("coherence_judge")
        if not coherence_cfg:
            return None

        try:
            judge_prompt = coherence_cfg["user_prompt_template"].format(
                generated_text=res['ai_exam']['output'],
                claims=res['ai_exam']['input']['claims'],
                reference_summaries=res['ai_exam']['input']['refs'],
                ground_truth=res['evaluation_context']['ground_truth']
            )
        except KeyError as e:
            self.logger.warning(f" Coherence prompt format warning: {e}")
            judge_prompt = coherence_cfg["user_prompt_template"].format(
                generated_text=res['ai_exam']['output']
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
        try:
            #  Markdown 
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # ?
            return json.loads(text.strip())
        except:
            #  JSON 
            text_clean = re.search(r'\{.*\}', text, re.DOTALL)
            if text_clean:
                return json.loads(text_clean.group(0))
            raise ValueError("No valid JSON found in response")


