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
- `paths.test_set` (e.g., `dataset/processed/test_set_new.json` — a formatted JSON **array** of samples; legacy JSONL is still supported when loading)
- `paths.reference_pool` (e.g., `dataset/ReferencePool_Full`)
- `paths.results` (default: `results/`)

Build the aligned test set:

```bash
python src/data_alignment.py
```

**BM25 in alignment ([src/data_alignment.py](src/data_alignment.py))**: By default, alignment does **not** compute BM25 over the reference pool (faster, smaller `test_set_new.json`). To precompute BM25 summaries and write them to each sample’s `input.bm25_reference_summaries`, enable it with an environment variable:

```bash
PATRE_ENABLE_BM25_ALIGNMENT=1 python src/data_alignment.py
```

This loads [dataset/ReferencePool_Full](dataset/ReferencePool_Full), runs BM25 on claims (`BM25_TOP_K` in the script, default 5), and applies the same **temporal pool filter** as retrieval: only references whose parsed date is strictly before the sample’s `meta.oa_date` (see [src/ref_pool_dates.py](src/ref_pool_dates.py)). Requires `rank_bm25` (see [requirements.txt](requirements.txt)).

At inference with `--retrieval bm25_retrieved`, the evaluator merges these BM25 entries with examiner-oracle references into one deduplicated candidate pool (up to 10 items) for the model prompt—not BM25 alone.


## ✍️ Experiment

Run a minimal OA experiment (1 sample):

```bash
python run_exp.py --task oa --ipc A --num_samples 1
```

Generate-only (skip evaluation):

```bash
python run_exp.py --task oa --generate_only
```

Resume generation in an existing results directory (skip patents that already have non-empty `output/ai_output.txt`; with full pipeline, also skip when `eval/scores.json` exists). Re-run with the **same** `--task`, filters, and data slice as the original run.

```bash
python run_exp.py --task oa --generate_only --resume_dir results/YOUR_EXPERIMENT_DIR
```

Eval-only:

```bash
python run_exp.py --eval_only --results_dir results/YOUR_EXPERIMENT 
```

Force re-evaluation (overwrite by recomputing and writing new scores):

```bash
python run_exp.py --eval_only --results_dir results/YOUR_EXPERIMENT --eval_force
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
- **BM25 temporal filter**: for `bm25_retrieved`, the pool is restricted to ReferencePool documents whose parsed date is **strictly before** the sample’s `meta.oa_date` (same rule when precomputing with `PATRE_ENABLE_BM25_ALIGNMENT=1`). Which JSON keys are used for pool dates is documented in [src/ref_pool_dates.py](src/ref_pool_dates.py).
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

## Contact
