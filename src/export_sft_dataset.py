import os
import sys
import json
import argparse
import random

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
from src.utils import load_dataset_json
def load_templates(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def _resolve_template_task(task_key, item):
    if task_key != "oa":
        return task_key
    rebuttal_text = (
        item.get("input", {}).get("rebuttal")
        or item.get("ground_truth", {}).get("rebuttal_text")
        or ""
    )
    if isinstance(rebuttal_text, str):
        rebuttal_text = rebuttal_text.strip()
    return "response_to_rebuttal" if rebuttal_text and rebuttal_text.upper() != "N/A" else "initial_examination"

def build_messages(task_key, templates, item, ref_summary, oa_content):
    template_task = _resolve_template_task(task_key, item)
    if template_task in templates.get("Office_Action_Generation", {}):
        template = templates["Office_Action_Generation"][template_task]
    else:
        template = templates["Rebuttal_Generation"][template_task]
    system_prompt = template["system_prompt"]
    user_prompt = template["user_prompt_template"].format(
        current_claims=item["input"]["claims"],
        reference_summaries=ref_summary,
        office_action=oa_content if template_task != "initial_examination" else "N/A",
        rebuttal=item["input"].get("rebuttal", "N/A") if template_task != "initial_examination" else "N/A",
        claim_count=item["meta"].get("claim_count", 1),
    )
    if task_key == "rebuttal":
        assistant_text = item["ground_truth"].get("rebuttal_text", "")
    else:
        assistant_text = item["ground_truth"].get("oa_text", "")
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_text},
        ]
    }
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset/processed/test_set_new.json")
    parser.add_argument("--templates", type=str, default="config/prompt_templates.json")
    parser.add_argument("--task", type=str, choices=["oa", "rebuttal"], required=True)
    parser.add_argument("--outdir", type=str, default="sft")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    templates = load_templates(args.templates)
    records = []
    for item in load_dataset_json(args.dataset):
        ref_summary = item["input"].get("oracle_reference_summaries", "")
        oa_content = item["input"].get("office_action", "")
        rec = build_messages(args.task, templates, item, ref_summary, oa_content)
        if rec["messages"][-1]["content"]:
            records.append(rec)
    random.shuffle(records)
    n = len(records)
    k = int(n * args.val_ratio)
    train = records[k:]
    val = records[:k]
    with open(os.path.join(args.outdir, f"{args.task}_train.jsonl"), "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.outdir, f"{args.task}_val.jsonl"), "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
if __name__ == "__main__":
    main()



