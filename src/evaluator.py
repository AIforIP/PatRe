# -*- coding: utf-8 -*-
import json
import os
import re
from tqdm import tqdm
from src.model_backend import ModelBackend

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
            self.logger.error(f"❌ 数据文件不存在: {path}")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        # 根据 config 设置样本数量
        limit = self.config['task_settings'].get('num_samples', 9999)
        return data[:limit]

    def run_batch(self):
        """执行批处理生成任务 - 按专利分组，每轮独立输出"""
        results = []
        task_key = self.config['task_settings']['current_task']
        
        # --- 核心修复：优先根据模板成员关系决定任务类别，避免字符串匹配误判 ---
        if task_key in self.templates.get("Office_Action_Generation", {}):
            root_key = "Office_Action_Generation"
        elif task_key in self.templates.get("Rebuttal_Generation", {}):
            root_key = "Rebuttal_Generation"
        else:
            self.logger.error(f"❌ 未能在模板中找到任务: {task_key}")
            return []
            
        if task_key not in self.templates[root_key]:
            self.logger.error(f"❌ 在 {root_key} 下找不到任务: {task_key}")
            return []
            
        template_config = self.templates[root_key][task_key]
        
        self.logger.info(f"📊 Starting batch inference for {len(self.dataset)} samples...")

        for item in tqdm(self.dataset, desc="Generating"):
            # 1. 处理引文摘要内容
            ref_summary = self._process_retrieval(item)
            
            # 2. 组装最终 User Prompt
            # 对于反驳任务，使用下一轮的OA（即当前轮次的真实OA）
            # 对于审查任务，使用当前轮次的OA
            if task_key == "v1_high_fidelity_rebuttal" and item['input'].get('next_office_action'):
                oa_content = item['input']['next_office_action']  # 使用下一轮OA作为输入
            else:
                oa_content = item['input'].get('office_action', "No Office Action provided in dataset.")
            
            enforce_instruction = "\n\nIMPORTANT: Use the provided Reference Summaries to cite specific documents. DO NOT use placeholders like '[insert references]'."
            
            # 构建AI实际输入（考试时能看到的内容）
            ai_input = {
                "claims": item['input']['claims'],
                "refs": ref_summary,
                "task_type": task_key,
                "retrieval_setting": self.config['active_settings'].get('retrieval_setting', "oracle_summary")
            }
            
            # 移除AI不应该看到的信息
            if task_key == "v1_initial_examination":
                # 初始审查阶段，AI不应该看到OA和反驳历史
                ai_input.pop("office_action", None)
                ai_input.pop("rebuttal_history", None)
            
            try:
                # 使用AI实际能看到的输入构建prompt
                user_prompt = template_config["user_prompt_template"].format(
                    current_claims=ai_input['claims'],
                    reference_summaries=ai_input['refs'],
                    office_action=oa_content if task_key != "v1_initial_examination" else "N/A",
                    rebuttal=item['input'].get('rebuttal', "N/A") if task_key != "v1_initial_examination" else "N/A",
                    claim_count=item['meta'].get('claim_count', 1)
                ) + enforce_instruction
            except KeyError as e:
                self.logger.error(f"❌ Prompt 填充失败，缺少变量: {e}")
                raise e

            # 3. 推理
            prediction = self.backend.generate(
                system_prompt=template_config["system_prompt"],
                user_prompt=user_prompt
            )

            # 动态选择 Ground Truth
            if task_key == "v1_response_to_rebuttal":
                ground_truth_text = item['ground_truth'].get('next_oa_text', item['ground_truth'].get('oa_text', "N/A"))
            elif "rebuttal" in task_key.lower():
                ground_truth_text = item['ground_truth'].get('rebuttal_text', "N/A")
            else:
                ground_truth_text = item['ground_truth']['oa_text']

            # 构建清晰的结果结构
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
                    "input_refs_preview": ref_summary[:200] + "..."  # 方便溯源
                }
            }
            results.append(result_entry)
        return results

    def _get_round_number(self, task_key):
        """根据任务类型确定轮次编号"""
        if task_key == "v1_initial_examination":
            return 1
        elif task_key == "v1_high_fidelity_rebuttal":
            return 2
        elif task_key == "v1_response_to_rebuttal":
            return 3
        else:
            return 0

    def _get_restricted_info(self, task_key):
        """获取AI不应该看到的信息列表"""
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
        自动化裁判打分逻辑：
        根据当前 Task 自动选择 examiner_judge 或 rebuttal_judge
        """
        # 1. 获取当前任务名称 (例如 "v1_initial_examination" 或 "v1_high_fidelity_rebuttal")
        current_task = self.config['task_settings']['current_task']
        # 根据模板归属而非字符串，选择裁判类型
        if current_task in self.templates.get("Office_Action_Generation", {}):
            judge_key = "examiner_judge"
        else:
            judge_key = "rebuttal_judge"
        
        self.logger.info(f"⚖️ Routing Evaluation to: {judge_key}")
        
        # 3. 从模板中读取配置
        judge_config = self.templates["Evaluation_Metrics"][judge_key]
        scored_results = []

        for res in tqdm(results, desc="Scoring"):
            # 填充裁判 Prompt
            # 传入丰富的上下文供 Judge 判断准确性
            try:
                judge_prompt = judge_config["user_prompt_template"].format(
                    generated_text=res['ai_exam']['output'],
                    claims=res['ai_exam']['input']['claims'],
                    reference_summaries=res['ai_exam']['input']['refs'],
                    office_action=res['evaluation_context']['full_context']['office_action'],
                    ground_truth=res['evaluation_context']['ground_truth']
                )
            except KeyError as e:
                # 兼容旧模板：如果模板里没写 {claims} 等占位符，format 应该忽略多余参数吗？
                # Python 的 format 只要 key 存在于模板中就需要提供，多余的 key 传入通常没问题（除非 unpack dict）
                # 但是如果模板里有 {claims} 而这里没传，就会报错。
                # 这里的代码假设 context 都有。
                self.logger.warning(f"⚠️ Prompt format warning: {e}")
                # Fallback to simple format if needed, or just let it fail/warn
                judge_prompt = judge_config["user_prompt_template"].format(
                    generated_text=res['ai_exam']['output']
                )

            # 调用裁判后端（通常裁判用 GPT-4o 保证公平性）
            raw_judge_res = self.judge_backend.generate(
                system_prompt=judge_config["system_prompt"],
                user_prompt=judge_prompt
            )

            try:
                # 解析裁判返回的 JSON 分数
                score_data = self._extract_json(raw_judge_res)
                res["scores"] = score_data.get("scores", {})
                res["justification"] = score_data.get("justification", "")

                # OA评分增加基于 GPT-5 mini 的逻辑连贯性分
                if judge_key == "examiner_judge":
                    coherence_data = self._run_gpt5_coherence_judge(res)
                    if coherence_data:
                        res["scores"]["coherence_score"] = coherence_data.get("coherence_score", 0)
                        if coherence_data.get("justification"):
                            res["justification"] = (
                                (res["justification"] + "\n\n") if res["justification"] else ""
                            ) + f"[GPT-5 mini coherence] {coherence_data['justification']}"

                # 总分按数值型子项汇总，避免模板升级后 total_score 漏算
                res["total_score"] = self._calculate_numeric_total(res.get("scores", {}))
                scored_results.append(res)
            except Exception as e:
                self.logger.warning(f"⚠️ Scoring failed for {res['patent_id']}: {e}")
                res["scores"] = {"error": "PARSING_FAILED"}
                scored_results.append(res)

        return scored_results

    def run_point_wise_evaluation(self, results):
        """
        Point-wise 评估：将生成结果与真实值进行逐项对比
        """
        self.logger.info("📊 Running point-wise evaluation...")
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
                self.logger.warning(f"⚠️ Point-wise prompt format warning: {e}")
                judge_prompt = judge_config["user_prompt_template"].format(
                    generated_text=res['ai_exam']['output'],
                    ground_truth=res['evaluation_context']['ground_truth']
                )

            raw_judge_res = self.judge_backend.generate(
                system_prompt=judge_config["system_prompt"],
                user_prompt=judge_prompt
            )

            try:
                score_data = self._extract_json(raw_judge_res)
                res["point_wise_scores"] = score_data.get("scores", {})
                res["point_wise_total"] = score_data.get("total_score", 0)
                res["point_wise_justification"] = score_data.get("justification", "")
                evaluated_results.append(res)
            except Exception as e:
                self.logger.warning(f"⚠️ Point-wise evaluation failed for {res['patent_id']}: {e}")
                res["point_wise_scores"] = {"error": "PARSING_FAILED"}
                evaluated_results.append(res)

        return evaluated_results

    def run_pairwise_comparison(self, results):
        """
        Pair-wise 比较：比较两组结果哪一组更好
        """
        self.logger.info("⚖️ Running pairwise comparison...")
        judge_config = self.templates["Evaluation_Metrics"]["pairwise_evaluator"]
        compared_results = []

        # 假设 results 包含两组数据用于比较
        # 这里我们比较相同输入的不同模型输出
        for i in range(0, len(results), 2):
            if i + 1 >= len(results):
                # 如果是奇数个结果，最后一个单独处理
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
                self.logger.warning(f"⚠️ Pairwise prompt format warning: {e}")
                judge_prompt = judge_config["user_prompt_template"].format(
                    text_a=res_a['ai_exam']['output'],
                    text_b=res_b['ai_exam']['output']
                )

            raw_judge_res = self.judge_backend.generate(
                system_prompt=judge_config["system_prompt"],
                user_prompt=judge_prompt
            )

            try:
                score_data = self._extract_json(raw_judge_res)
                winner = score_data.get("winner", "unknown")
                
                # 记录比较结果
                res_a["pairwise_winner"] = winner
                res_a["pairwise_scores"] = score_data.get("scores", {})
                res_a["pairwise_justification"] = score_data.get("justification", "")
                
                # 为结果B也记录相同的比较结果，但调整winner
                res_b["pairwise_winner"] = "A" if winner == "B" else "B" if winner == "A" else "tie"
                res_b["pairwise_scores"] = score_data.get("scores", {})
                res_b["pairwise_justification"] = score_data.get("justification", "")
                
                compared_results.extend([res_a, res_b])
            except Exception as e:
                self.logger.warning(f"⚠️ Pairwise comparison failed for {res_a['patent_id']} vs {res_b['patent_id']}: {e}")
                res_a["pairwise_winner"] = "error"
                res_b["pairwise_winner"] = "error"
                compared_results.extend([res_a, res_b])

        return compared_results

    def run_claim_level_evaluation(self, results):
        """
        Claim-level 评估：对每个权利要求进行单独评估
        """
        self.logger.info("🔍 Running claim-level evaluation...")
        judge_config = self.templates["Evaluation_Metrics"]["claim_level_evaluator"]
        evaluated_results = []

        for res in tqdm(results, desc="Claim-level Evaluation"):
            try:
                # 提取权利要求相关信息
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
                self.logger.warning(f"⚠️ Claim-level prompt format warning: {e}")
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
                score_data = self._extract_json(raw_judge_res)
                res["claim_level_scores"] = score_data.get("scores", {})
                res["claim_level_total"] = score_data.get("total_score", 0)
                res["claim_level_justification"] = score_data.get("justification", "")
                evaluated_results.append(res)
            except Exception as e:
                self.logger.warning(f"⚠️ Claim-level evaluation failed for {res['patent_id']}: {e}")
                res["claim_level_scores"] = {"error": "PARSING_FAILED"}
                evaluated_results.append(res)

        return evaluated_results

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
            self.logger.warning(f"⚠️ Coherence prompt format warning: {e}")
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
        parsed = self._extract_json(raw)

        # 兼容两种返回格式:
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
        """鲁棒的 JSON 提取器"""
        try:
            # 尝试匹配 Markdown 格式
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # 尝试直接解析全文本
            return json.loads(text.strip())
        except:
            # 如果还是不行，尝试清洗掉前后的非 JSON 字符
            text_clean = re.search(r'\{.*\}', text, re.DOTALL)
            if text_clean:
                return json.loads(text_clean.group(0))
            raise ValueError("No valid JSON found in response")
