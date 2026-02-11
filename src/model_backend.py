# -*- coding: utf-8 -*-
from openai import OpenAI
import time
import os

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
        
        # Ollama 配置
        if self.backend == 'ollama':
            api_key = config['models'].get('ollama_api_key') or "ollama"
            base_url = config['models'].get('ollama_base_url') or "http://localhost:11434/v1"
        elif self.backend == 'vllm':
            api_key = config['models'].get('vllm_api_key') or api_key or "EMPTY"
            base_url = config['models'].get('vllm_base_url') or base_url
        
        if not api_key:
            raise ValueError("API key is missing. Please provide it in config file or OPENAI_API_KEY environment variable.")
        
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
        
        for _ in range(3): # 自动重试 3 次，防止网络抖动
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    **self.params
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Error, retrying... {e}")
                time.sleep(2)
        return "FAILED_TO_GENERATE"
