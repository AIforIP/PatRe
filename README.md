PatRe: Patent Review Experiment Runner

概述
- 多轮专利审查实验框架：按数据顺序交替运行 OA → Rebuttal → OA…
- 支持完整“生成 + 评估”、仅生成、仅评估（复用已有结果）三种模式
- 内置附加指标（Disposition/Statute/Citation/Formatting/Language Style 等）与轻量统计脚本
- 支持 ChatAnywhere(OpenAI 兼容) API；提供 Dry-Run 模式便于离线自测

目录结构（关键项）
- config/
  - eval_config.yaml：核心配置（数据路径、模型、参数）
  - prompt_templates.json：所有生成与评估的模板
- src/
  - evaluator.py：主流程（生成、评审、附加指标、输出）
  - model_backend.py：API/本地后端与 Dry-Run
  - metrics_library.py：BLEU/ROUGE/BERTScore 与规则指标
- tools/
  - regenerate_scores_summary.py：从 scores.json 重建 summary（无 LLM）
  - aggregate_scores.py：对 summary 聚合求均值（缺失自动跳过）
- run_exp.py：主入口（生成、评估、多轮）
- eval_only.py：仅评估已有 results

安装
- Python 3.9+ 建议
- 安装依赖

```bash
pip install -r requirements.txt
```

环境变量与安全
- 推荐以环境变量配置 API：
  - OPENAI_API_KEY=你的 ChatAnywhere API Key
  - OPENAI_BASE_URL=https://api.chatanywhere.tech/v1
- Windows PowerShell（永久）：

```powershell
setx OPENAI_API_KEY "你的key"
setx OPENAI_BASE_URL "https://api.chatanywhere.tech/v1"
```

- CMD 当前会话：

```bat
set OPENAI_API_KEY=你的key
set OPENAI_BASE_URL=https://api.chatanywhere.tech/v1
```



数据与路径
- 在 config/eval_config.yaml 里配置：
  - paths.test_set（如 dataset/processed/test_set_new.jsonl）
  - paths.reference_pool（如 dataset/ReferencePool_Full）
  - paths.results（默认 results/）

数据准备（生成 test_set_new.jsonl）
- 准备原始数据
  - 将各类别数据放在 dataset/PatTimeline_*_Category/cleaned_json_output 下
  - 将引用池放在 dataset/ReferencePool_Full 下（文件名形如 USXXXX.json）
- 生成对齐后的测试集

```bash
python src/data_alignment.py
```

- 预期输出
  - 控制台显示 “Successfully aligned N data samples.” 与保存路径
  - 文件写入 dataset/processed/test_set_new.jsonl
- 可选验证

```bash
python tools/test_data_aligner.py
```

- 若你更换了保存路径，可用脚本更新配置

```bash
python tools/set_testset_path.py --test-set dataset/processed/test_set_new.jsonl
```

快速开始
1) 最小验证一条样本（只跑 OA）：

```bash
python run_exp.py --task oa --ipc A --num_samples 1
```

2) 只生成（不打分）：

```bash
python run_exp.py --task oa --generate_only --num_samples 1
```

3) 只评估现有结果（前 5 条）：

```bash
python run_exp.py --eval_only --results_dir results/你的实验目录 --task both --eval_limit 5
```

4) 多轮（根据数据轮次自动交替）：

```bash
python run_exp.py --task both --ipc A --num_samples 3
```

常用参数
- --task {oa|rebuttal|both}：选择任务或多轮
- --ipc A-H：按 IPC 分类筛选
- --num_samples N：限制本次生成/评估的样本数
- --generate_only：仅生成不评估
- --eval_only --results_dir DIR：只对既有结果评估
- --eval_limit N：eval_only 模式限制只评 N 条 round
- --test_mode {zero_shot|sft_model}：覆盖测试模式
- --retrieval {no_ref|oracle_summary|bm25_retrieved}：覆盖检索设置
- --backend {api|vllm|ollama}：后端切换

评估输出
- 单轮目录：results/.../patent_xxx/round_k_{oa|rebuttal}/
  - scores.json：包含 LLM 打分、规则分、明细
  - scores_summary.json：面向阅读的轻量摘要（百分比与分值）
  - context.json / ai_output.txt：输入与输出快照
- 批量聚合：

```bash
python tools/regenerate_scores_summary.py --root results/你的实验目录 --overwrite
python tools/aggregate_scores.py --root results/你的实验目录 --filename scores_summary.json --out results/你的实验目录/metrics_mean_by_task_summary.csv --include_all_metrics
```

关键约定
- Rebuttal 是否运行取决于该轮是否存在可用的备注文本（remarks）；若该 OA 为 Notice of Allowance，数据对齐阶段会将该轮 rebuttal 置为 N/A，从而跳过该轮的 rebuttal。
- Point-wise Coverage、Constructiveness 等需要评审模板包含对应字段；若旧结果缺失，可在 eval-only 重跑生成这些字段。

可选依赖
- 若需要运行分析/可视化脚本（如 analyze_stats），可安装 pandas、scikit-learn、matplotlib、seaborn、sentence-transformers 等；这些不影响核心跑实验。

故障排查
- API key 未生效：Windows 使用 setx 后需重开终端；或使用 set 在当前窗口生效。
- 404 访问 base_url：浏览器直接访问 /v1 返回 404 正常；实际请求是 POST /v1/chat/completions。
- FAILED_TO_GENERATE：检查限流、模型名、上下文过长；或临时使用 PATRE_DRY_RUN=1 验证流程。

许可证
- 本项目用于学术与研究目的，实际分发请根据数据/模型服务协议合规处理。
