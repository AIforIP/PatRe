# PatRe 项目使用指南 (Usage Guide)

本项目旨在利用大语言模型（LLM）进行专利审查（Patent Review）的自动化评估与生成，支持多轮审查意见（Office Action）与反驳（Rebuttal）的交互模拟。

## 1. 环境准备

### 1.1 安装依赖
确保您的环境中安装了 Python 3.8+，并运行以下命令安装必要依赖：
```bash
pip install -r requirements.txt
```
*(注：如果根目录没有 requirements.txt，请确保安装了 `openai`, `yaml`, `rank_bm25`, `tqdm` 等常用库)*

### 1.2 API 配置
编辑 [eval_config.yaml](file:///d:/study/competition/LLM_based_IP_Review_Research/Project/PatRe/config/eval_config.yaml) 文件：
- `api_key`: 填入您的 LLM API Key。
- `base_url`: 填入 API 基础路径。
- `backend`: 选择后端类型（`api`, `ollama`, `vllm` 等）。

## 2. 核心功能运行

### 2.1 运行实验 (run_exp.py)
这是项目的主要入口脚本，支持多种模式：

- **单任务运行 (Zero-Shot)**:
  ```bash
  python run_exp.py --retrieval oracle_summary
  ```

- **多轮交互模拟 (Multi-round)**:
  ```bash
  python run_exp.py --run_multiround --retrieval bm25_retrieved
  ```
  该模式会根据数据中真实的 Rebuttal 轮数动态运行，并按照专利编号组织输出结果。

- **指定检索模式**:
  - `no_ref`: 不使用参考资料。
  - `oracle_summary`: 使用人工标注的文献摘要作为参考。
  - `bm25_retrieved`: 使用 BM25 算法从本地库中检索相关文献。

### 2.2 SFT 支持
- **导出数据集**: 运行 `src/export_sft_dataset.py` 将原始数据转换为 OpenAI SFT 格式。
- **启动训练**: 使用 `src/run_sft_openai.py` 提交训练任务（需配置 OpenAI API）。

## 3. 结果查看

实验结果将保存在 `results/` 目录下，结构如下：
```text
results/
└── patent_[ID]_App/               # 按专利 ID 分组
    ├── round_1_initial_examination/
    │   ├── input.json             # 输入提示词
    │   ├── prediction.txt         # 模型生成的审查意见/反驳
    │   ├── evaluation.json        # LLM-as-a-Judge 评分结果
    │   └── context.json           # 检索到的上下文
    └── summary.json               # 该专利各轮次的汇总指标
```

## 4. Git 提交指南

如果您需要将更改推送到 GitHub，请执行：
1. `git add .`
2. `git commit -m "您的提交信息"`
3. `git push origin main`

---
*如有疑问，请参考 [README.md](file:///d:/study/competition/LLM_based_IP_Review_Research/Project/PatRe/README.md) 或联系项目负责人。*
