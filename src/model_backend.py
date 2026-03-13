# -*- coding: utf-8 -*-
import time
import os
import random
import json
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class ModelBackend:
    def __init__(self, config, is_judge=False):
        self.config = config
        self.model_name = config['models']['active_model']
        self.params = config['models']['generation_params']
        self._dry_run = str(os.getenv("PATRE_DRY_RUN", "")).strip().lower() in ("1", "true", "yes")
        

        if is_judge:
            self.backend = config['models'].get('judge_backend', 'api')
            self.model_name = config['models']['judge_model']
        else:
            self.backend = config['models'].get('backend', 'api')
        
        api_key = os.getenv('OPENAI_API_KEY') or config.get('api_key') or config['models'].get('api_key')
        base_url = os.getenv('OPENAI_BASE_URL') or config.get('base_url') or config['models'].get('base_url')

        # Allow separate judge credentials/endpoint
        if is_judge:
            api_key = config['models'].get('judge_api_key') or api_key
            base_url = config['models'].get('judge_base_url') or base_url
        
        # Ollama 
        if self.backend == 'ollama':
            api_key = config['models'].get('ollama_api_key') or "ollama"
            base_url = config['models'].get('ollama_base_url') or "http://localhost:11434/v1"
        elif self.backend == 'vllm':
            api_key = config['models'].get('vllm_api_key') or api_key or "EMPTY"
            base_url = config['models'].get('vllm_base_url') or base_url
        
        if not api_key and not self._dry_run:
            raise ValueError("API key is missing. Please provide it via OPENAI_API_KEY or config.")

        clean_base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().strip("`").strip()
        self._base_url = clean_base_url.rstrip("/")
        self._api_key = str(api_key or "").strip()

        print(f"[ModelBackend] backend={self.backend} is_judge={is_judge} base_url={self._base_url}")

        self._client = None
        if not self._dry_run:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def generate(self, system_prompt, user_prompt, is_judge=False, model_override=None):
        if self._dry_run:
            return self._dry_generate(system_prompt, user_prompt)
        if is_judge:
            model = self.config['models']['judge_model']
        else:
            if self.config.get('active_settings', {}).get('test_mode') == 'sft_model':
                model = self.config['models'].get('sft_model_id') or self.model_name
            else:
                model = self.model_name
        if model_override:
            model = model_override
        
        params = dict(self.params) if isinstance(self.params, dict) else {}
        if "max_tokens" in params and "max_completion_tokens" not in params:
            if isinstance(model, str) and model.startswith(("gpt-5", "gpt-4.1", "o1", "o3")):
                params["max_completion_tokens"] = params.pop("max_tokens")
        
        max_retries = int(self.config.get("models", {}).get("max_retries", 3))
        base_backoff = float(self.config.get("models", {}).get("retry_backoff_sec", 2))
        jitter = float(self.config.get("models", {}).get("retry_jitter_sec", 0))
        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    **params,
                )
                choice0 = resp.choices[0] if getattr(resp, "choices", None) else None
                content = None
                if choice0 and getattr(choice0, "message", None):
                    content = getattr(choice0.message, "content", None)
                if content is None and choice0:
                    content = getattr(choice0, "text", None)

                if content is None or not str(content).strip():
                    raise ValueError("Empty response content")
                return str(content)
            except Exception as e:
                msg = str(e)
                if "max_tokens" in msg and "max_completion_tokens" in msg:
                    if "max_tokens" in params and "max_completion_tokens" not in params:
                        params["max_completion_tokens"] = params.pop("max_tokens")
                        continue
                m_max_ctx = re.search(r"maximum context length is\s+(\d+)\s+tokens", msg)
                m_input = re.search(r"request has\s+(\d+)\s+input tokens", msg)
                if m_max_ctx and m_input:
                    max_ctx = int(m_max_ctx.group(1))
                    input_tokens = int(m_input.group(1))
                    allowed = max(1, (max_ctx - input_tokens) - 16)
                    if "max_tokens" in params:
                        params["max_tokens"] = min(int(params["max_tokens"]), allowed)
                        continue
                    if "max_completion_tokens" in params:
                        params["max_completion_tokens"] = min(int(params["max_completion_tokens"]), allowed)
                        continue
                print(f"⚠️ Error, retrying... {e}")
                if attempt < max_retries - 1:
                    sleep_s = base_backoff * (2 ** attempt)
                    if jitter:
                        sleep_s += random.uniform(0, jitter)
                    time.sleep(sleep_s)
        return "FAILED_TO_GENERATE"

    def _dry_generate(self, system_prompt, user_prompt):
        up = user_prompt or ""
        sp = system_prompt or ""
        up_l = up.lower()
        sp_l = sp.lower()

        def as_json(total, scores):
            payload = {
                "total_score": float(total),
                "scores": scores,
                "justification": "dry_run",
                "justification_details": {},
                "evaluation_report": "dry_run",
            }
            return json.dumps(payload, ensure_ascii=False)

        if "evaluate the generated" in up_l or "output format" in up_l or "audit report" in up_l:
            if "rebuttal" in up_l:
                scores = {
                    "effectiveness": 8.0,
                    "accuracy": 8.0,
                    "reliability": 8.0,
                    "statute_compliance": 8.0,
                    "feature_distinction": 8.0,
                    "specification_support": 8.0,
                    "clarity": 8.0,
                    "completeness": 8.0,
                    "constructiveness": 8.0,
                    "language_style": 8.0,
                    "point_wise_coverage": 8.0,
                }
                return as_json(8.0, scores)

            scores = {
                "accuracy": 8.0,
                "rationality": 8.0,
                "reliability": 8.0,
                "legal_statute_precision": 8.0,
                "feature_mapping_accuracy": 8.0,
                "reference_citation_accuracy": 8.0,
                "clarity": 8.0,
                "completeness": 8.0,
                "constructiveness": 8.0,
                "language_style": 8.0,
            }
            return as_json(8.0, scores)

        if "point-wise" in up_l or "point wise" in up_l:
            return as_json(8.0, {"point_wise_coverage": 8.0})

        if "pairwise" in up_l and ("text_a" in up_l or "text b" in up_l):
            return as_json(8.0, {"winner": 1.0})

        if "claim-level" in up_l or "claim level" in up_l:
            return as_json(8.0, {"accuracy": 8.0, "clarity": 8.0})

        if "detailed action" in up or "office action" in sp_l:
            return "DETAILED ACTION\nOPENING STATEMENT\nClaims are examined in dry_run mode.\nCONCLUSION\nThe application is in condition for allowance."

        if "rebuttal" in sp_l:
            return "REMARKS\nApplicant respectfully traverses the rejections in dry_run mode.\nCONCLUSION\nWithdrawal of the rejections is requested."

        return "DRY_RUN_OUTPUT"
