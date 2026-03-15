# PatRe
[**🌐 Homepage**](https://patre.wangqiyao.me/) | [**🤗 Dataset**](https://huggingface.co/datasets/IPIntelligence/PatRe-Bench) | [**📖 arXiv**](https://arxiv.org/abs/) | [**GitHub**](https://github.com/AIforIP/PatRe)


This repo contains the evaluation code for the paper "[PatRe: A Large-Scale Full-Stage Patent Office Action and Rebuttal Dataset and Benchmark](https://arxiv.org/abs/)"

## 🔔News

- 🔥 [2026-02-16] Research Begining.

## 👀 Introduction



## 🚀 Quick Start

Requirements:
- Python 3.9+ recommended

Install dependencies:

```bash
pip install -r requirements.txt
export PYTHONPATH=$pwd
```

### API Configuration

Recommended: configure credentials via environment variables (avoid hardcoding keys into YAML).

Windows PowerShell (persistent):

```powershell
setx OPENAI_API_KEY "YOUR_KEY"
setx OPENAI_BASE_URL "YOUR_BASE_URL"
```

Windows CMD (current session only):

```bat
set OPENAI_API_KEY=YOUR_KEY
set OPENAI_BASE_URL=YOUR_BASE_URL
```

Linux/MacOS:

```shell
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_BASE_URL="YOUR_BASE_URL"
```

### Data & Paths

Configure paths in [eval_config.yaml](config/eval_config.yaml):
- `paths.test_set` (e.g., `dataset/processed/test_set_new.jsonl`)
- `paths.reference_pool` (e.g., `dataset/ReferencePool_Full`)
- `paths.results` (default: `results/`)

Build the aligned test set:

```bash
python src/data_alignment.py
```


## ✍️ Experiment

Run a minimal OA experiment (1 sample):

```bash
python run_exp.py --task oa --ipc A --num_samples 1
```

Generate-only (skip evaluation):

```bash
python run_exp.py --task oa --generate_only
```

Eval-only (re-score existing results):

```bash
python run_exp.py --eval_only --results_dir results/YOUR_EXPERIMENT 
```

Multi-round simulation (OA + rebuttal alternating):

```bash
python run_exp.py
```

### 📖 Configuration Tips

- `--task {oa|rebuttal|both}`: run OA only, rebuttal only, or full multi-round
- `--ipc A-H`: filter by IPC category
- `--num_samples N`: limit number of samples evaluated/generated
- `--generate_only`: generation without evaluation
- `--eval_only --results_dir DIR`: evaluate existing results
- `--eval_limit N`: evaluate only the first N round directories in eval-only mode
- `--test_mode {zero_shot|sft_model}`: override testing mode
- `--retrieval {no_ref|oracle_summary|bm25_retrieved}`: override retrieval setting
- `--backend {api|vllm|ollama}`: switch the model backend

### vLLM / Ollama Backends

To use vLLM, you must both:
- start a vLLM server exposing an OpenAI-compatible endpoint (usually under `/v1`)
- run PatRe with `--backend vllm`

Example (PatRe side):

```bash
python run_exp.py --backend vllm
```

The base URL for vLLM is configured in `models.vllm_base_url` in [eval_config.yaml](config/eval_config.yaml).

### Output Structure

PatRe writes results under the experiment directory, grouped by patent and rounds.

Typical layout (new structure):

```text
results/<experiment_name>/
  patent_<PATENT_ID>/
    round_01_oa/
      input/
        ai_input.json
      output/
        ai_output.txt
      eval/
        context.json
        scores.json
        scores_summary.json
```

Eval-only compatibility:
- If `eval/context.json` exists, it will be used.
- If only legacy `context.json` exists, it will be used.
- If no context file exists, PatRe will attempt to reconstruct evaluation context from `paths.test_set` and the round folder name.


## Troubleshooting

- Environment variables not applied on Windows: `setx` requires opening a new terminal; use `set` for current session.
- `FAILED_TO_GENERATE`: check rate limits, model name, context length; try `PATRE_DRY_RUN=1` to validate the pipeline.
- Similarity metrics are `null`: verify `rouge-score`, `sacrebleu`, `bert-score` are installed and that `ground_truth` is available.

## License

This repository is intended for academic/research use. Ensure you comply with dataset and model service licenses when distributing results.
