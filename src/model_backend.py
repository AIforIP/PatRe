# -*- coding: utf-8 -*-
from openai import OpenAI
import time
import os
import random

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 如果没有安装 python-dotenv，跳过加载

class ModelBackend:
    def __init__(self, config, is_judge=False):
        self.config = config
        self.model_name = config['models']['active_model']
        self.params = config['models']['generation_params']
        
        # 根据是否为裁判选择不同的后端
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
        
        # Ollama 配置
        if self.backend == 'ollama':
            api_key = config['models'].get('ollama_api_key') or "ollama"
            base_url = config['models'].get('ollama_base_url') or "http://localhost:11434/v1"
        elif self.backend == 'vllm':
            api_key = config['models'].get('vllm_api_key') or api_key or "EMPTY"
            base_url = config['models'].get('vllm_base_url') or base_url
        
        if not api_key:
            raise ValueError("API key is missing. Please provide it in config file or OPENAI_API_KEY environment variable.")

        # Debug: show which endpoint is used
        print(f"[ModelBackend] backend={self.backend} is_judge={is_judge} base_url={base_url}")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate(self, system_prompt, user_prompt, is_judge=False, model_override=None):
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
        for attempt in range(max_retries): # 自动重试
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    **params
                )
                content = response.choices[0].message.content
                if content is None or not str(content).strip():
                    raise ValueError("Empty response content")
                return content
            except Exception as e:
                msg = str(e)
                if "max_tokens" in msg and "max_completion_tokens" in msg:
                    if "max_tokens" in params and "max_completion_tokens" not in params:
                        params["max_completion_tokens"] = params.pop("max_tokens")
                        continue
                print(f"⚠️ Error, retrying... {e}")
                if attempt < max_retries - 1:
                    sleep_s = base_backoff * (2 ** attempt)
                    if jitter:
                        sleep_s += random.uniform(0, jitter)
                    time.sleep(sleep_s)
        return "FAILED_TO_GENERATE"
