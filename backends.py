# -*- coding: utf-8 -*-
import openai

class PatentAgentEvaluator:
    def __init__(self, backend_type="api"):
        self.backend_type = backend_type
        
        if backend_type == "vllm":
            self.api_key = "EMPTY"
            self.base_url = "http://localhost:8000/v1"
            self.model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" # 需与你vllm模型名一致
        else:
            # 使用你提供的 api.lqshop.cc 接口
            self.api_key = "sk-N0KfygnoEoUCERdR5f5cEd72C5374028A46d7f3f2cD6C308"
            self.base_url = "https://api.lqshop.cc/v1"
            self.model = "gpt-4o" # 或中转站支持的其他模型

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional patent attorney skilled in drafting rebuttals (Remarks) to USPTO Office Actions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.01, # 论文评测必须保持低随机性
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"