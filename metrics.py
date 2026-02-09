# -*- coding: utf-8 -*-
import json
from rouge_score import rouge_scorer

def calculate_metrics(result_file):
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_rouge = 0
    
    print(f"📊 Calculating metrics for {len(data)} samples...")
    for entry in data:
        pred = entry['prediction']
        gt = entry['ground_truth']
        
        # 计算 ROUGE-L
        score = scorer.score(gt, pred)
        total_rouge += score['rougeL'].fmeasure
        
    avg_rouge = total_rouge / len(data)
    print(f"\n--- Result Report ---")
    print(f"File: {result_file}")
    print(f"Average ROUGE-L Score: {avg_rouge:.4f}")

if __name__ == "__main__":
    calculate_metrics("results/oracle_raw_outputs.json")