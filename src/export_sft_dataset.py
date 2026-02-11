import os
import json
import argparse
import random
def load_templates(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def build_messages(task_key, template, item, ref_summary, oa_content):
    system_prompt = template["system_prompt"]
    user_prompt = template["user_prompt_template"].format(
        current_claims=item["input"]["claims"],
        reference_summaries=ref_summary,
        office_action=oa_content,
        rebuttal=item["input"].get("rebuttal", "N/A"),
        claim_count=item["meta"].get("claim_count", 1),
    )
    if "rebuttal" in task_key.lower():
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
    parser.add_argument("--dataset", type=str, default="dataset/processed/test_set.jsonl")
    parser.add_argument("--templates", type=str, default="config/prompt_templates.json")
    parser.add_argument("--task", type=str, choices=["v1_initial_examination", "v1_high_fidelity_rebuttal"], required=True)
    parser.add_argument("--outdir", type=str, default="sft")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    templates = load_templates(args.templates)
    if args.task in templates.get("Office_Action_Generation", {}):
        template = templates["Office_Action_Generation"][args.task]
    else:
        template = templates["Rebuttal_Generation"][args.task]
    records = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            ref_summary = item["input"].get("oracle_reference_summaries", "")
            oa_content = item["input"].get("office_action", "")
            rec = build_messages(args.task, template, item, ref_summary, oa_content)
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
