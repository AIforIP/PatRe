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
建议使用环境变量配置 API（避免把 key 写进配置文件）：
- OPENAI_API_KEY：你的 API Key
- OPENAI_BASE_URL：https://api.chatanywhere.tech/v1

config/eval_config.yaml 主要用于设置数据路径、模型名与运行参数；不建议在其中硬编码 api_key。

## 2. 核心功能运行

### 2.1 运行实验 (run_exp.py)
这是项目的主要入口脚本，支持多种模式：

- **单任务运行**:
  ```bash
  python run_exp.py --task oa --retrieval oracle_summary
  ```

- **多轮交互模拟 (Multi-round)**:
  ```bash
  python run_exp.py --task both --retrieval bm25_retrieved
  ```
  该模式会根据数据中轮次交替运行 OA 与 Rebuttal，并按专利编号组织输出结果。

- **指定检索模式**:
  - `no_ref`: 不使用参考资料。
  - `oracle_summary`: 使用人工标注的文献摘要作为参考。
  - `bm25_retrieved`: 使用 BM25 算法从本地库中检索相关文献。

### 2.2 SFT 支持
- **导出数据集**: 运行 `src/export_sft_dataset.py` 将原始数据转换为 OpenAI SFT 格式。
- **启动训练**: 使用 `src/run_sft_openai.py` 提交训练任务（需配置 OpenAI API）。

## 3. 结果查看

实验结果将保存在 `results/` 目录下（顶层目录名包含模型/任务/检索/IPC/时间戳），结构如下：
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
提示：README.md 是最新的、可直接发布到 GitHub 的入口文档；本 GUIDE 用于补充说明，若与 README 冲突，以 README 为准。
