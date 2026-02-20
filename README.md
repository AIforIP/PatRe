# PatRe: LLM-based IP Review Research
# PatRe: 基于大模型的知识产权审查研究

## 1. 项目概览 (Project Overview)
**PatRe** 是一个致力于评估和改进大语言模型（LLMs）在知识产权（IP）审查领域表现的研究项目。核心关注点是自动化并分析**专利审查员（Patent Examiners）**与**申请人（Applicants）**之间的交互过程。

**PatRe** is a research project aiming to evaluate and improve Large Language Models (LLMs) in the domain of Intellectual Property (IP) Review. The primary focus is on automating and analyzing the interaction between **Patent Examiners** (Office Action Generation) and **Applicants** (Rebuttal/Amendment Generation).

- **目标会议 (Target Venue)**: NIPS 2026
- **核心任务**: 专利审查意见生成 (Office Action Generation) & 申诉/反驳生成 (Rebuttal Generation)

## 2. 任务定义 (Task Definition)
本项目目前的 Benchmark 包含两个主要任务：

### 任务 1：审查意见生成 (Review Generation / Office Action Generation)
- **目标 (Goal)**: 模拟专利审查员，基于当前权利要求（Claims）和现有技术（Prior Art）生成审查意见书（Office Action）。
- **输入 (Input)**: 当前权利要求 (Current Claims)，检索到的参考文献摘要 (Retrieved Reference Summaries/Prior Art)。
- **输出 (Output)**: 审查意见（包含 §102 新颖性, §103 创造性等驳回理由），最终决定（非最终驳回 Non-Final Rejection, 最终驳回 Final Rejection, 授权 Allowance）。
- **评估 (Evaluation)**:
  - 决定的准确性 (Accuracy of decision)
  - 驳回理由的合理性 (Rationality of rejection reasons)
  - 引用的可靠性 (Reliability of citations)

### 任务 2：反驳生成 (Rebuttal Generation / Amendment Generation)
- **目标 (Goal)**: 模拟专利律师/申请人，针对审查意见撰写反驳（Rebuttal）。
- **输入 (Input)**: 审查意见（文本、引用），当前权利要求。
- **输出 (Output)**: 申请人争辩（Arguments/Remarks），修改后的权利要求（Refined Claims，可选）。
- **评估 (Evaluation)**: 反驳在克服驳回方面的有效性 (Effectiveness of the rebuttal)。

*(未来计划: 审查结果预测 Office Action Prediction, 权利要求优化 Claims Refinement)*

## 3. 数据处理与格式 (Data Processing & Format)

### 数据处理流程 (Pipeline)
1.  **爬取 (Crawling)**:
    - 原始专利 (Initial Patent): 元数据、标题、摘要、权利要求、说明书。
    - 非最终驳回 (Non-Final Rejection): 审查意见文本、参考文献。
    - 申请人回复 (Applicant Response): 争辩文本 (Arguments/Remarks) 及 修改后的权利要求 (Refined Claims)。
    - 最终结果 (Final Rejection/Result)。
2.  **原子化 (Atomization)**:
    - 将数据拆分为原子单元（例如：针对单条 Claim 的处理，或单轮 Office Action 的处理）。
3.  **对齐 (Alignment)**:
    - 使用 `src/data_alignment.py` 脚本将原始 JSON 文件转换为标准化的 JSONL 格式。

### 数据格式 (Data Format)
处理后的数据文件通常位于 `data/processed/test_set.jsonl`，结构如下：
```json
{
  "id": "Patent_Number",
  "meta": {
    "ipc": "IPC_Category (分类号)",
    "claim_count": 10
  },
  "input": {
    "claims": "Claim text (权利要求文本)...",
    "oracle_reference_summaries": "Summary of cited references (参考文献摘要)...",
    "rebuttal": "Previous rebuttal (前一轮反驳，如有)...",
    "office_action": "Previous Office Action (前一轮审查意见，用于Rebuttal任务)..."
  },
  "ground_truth": {
    "oa_text": "Actual Office Action text (真实审查意见)...",
    "decision": "Non-Final Rejection (决定结果)"
  }
}
```

### 测试集构成 (Test Set Composition)
- **总量**: 约 480 个专利（IPC分类 A-H 每个类选择 60 个）。
- **分布**: 约 80% 最终被驳回 (Final Rejection)，20% 最终获得授权 (Allowance)。
- **规模**: 包含 960-1440 条审查意见（每个专利至少包含 2 轮 Office Action）。

## 4. 多轮审查机制 (Multi-Round Review Mechanism)

### 动态轮数支持
PatRe支持根据数据集中的实际审查轮数动态运行多轮审查流程：

- **自动分析**: 系统会自动分析测试数据集中的专利审查轮数分布
- **动态适应**: 根据每个专利的实际轮数运行相应次数的审查-反驳循环
- **灵活配置**: 支持1轮（仅初始审查）到N轮（完整审查-反驳链）的任意组合

### 运行模式
```bash
# 运行动态多轮审查（根据数据实际轮数）
python run_exp.py --run_multiround

# 运行固定3轮审查流程
python run_exp.py --run_chain3

# 运行指定轮数的审查流程
python run_exp.py --task initial_examination  # 第1轮：初始审查
python run_exp.py --task rebuttal  # 第2轮：申请人反驳
python run_exp.py --task response_to_rebuttal   # 第3轮：审查员回复
```

## 5. 实验与评估 (Experiment & Evaluation)

### 测试设置 (Test Settings)
实验配置主要在 `config/eval_config.yaml` 中定义。
- **模型 (Models)**: 支持 GPT-4o-mini, GPT-4o, DeepSeek-V3 等。
- **模式 (Modes)**:
  - `Zero-Shot`: 零样本直接生成。
  - `SFT`: 监督微调 (Supervised Fine-Tuning)。
- **检索设置 (Retrieval Settings)**:
  - `no_ref`: 不提供参考文献 (Blind Test)。
  - `oracle_summary`: 提供 Ground Truth 的文献摘要 (Oracle)。
  - `bm25_retrieved`: 使用 BM25 模拟检索结果 (Simulation)。
- **粒度 (Granularity)**: 全文生成 (Full Text) vs 单点 Claim 生成 (Atomic Claim)。

### 评估指标 (Evaluation Metrics)
- **传统评估 (OA)**: 使用 LLM-as-a-Judge 对生成的准确性 (Accuracy)、合理性 (Rationality)、可靠性 (Reliability)、格式规范性 (Formatting) 进行打分。
- **法条适用准确率 (Legal Statute Precision)**: 对比 Ground Truth，检查是否正确识别并适用 102/103。
- **技术特征映射准确度 (Feature Mapping Accuracy)**: 检查生成 OA 是否覆盖真实 OA 中关键区别技术特征点。
- **引用准确度 (Reference Citation Accuracy)**: 检查引用文献是否真实支撑所映射的技术特征与驳回理由。
- **逻辑连贯性得分 (Coherence Score)**: 由 `gpt-5-mini` 单独评估 OA 论证链条是否自洽、可说服。
- **PatentScore (1-5)**: 三个维度评分，分别为结构稳定性 (Structure Stability)、法律清晰度 (Legal Clarity)、语义逻辑 (Semantic Logic)。
- **Point-wise 评估**: 将生成结果与 Ground Truth 进行逐项对比，评估事实准确性、法律合规性、技术精确性等维度。
- **Pair-wise 比较**: 将模型生成结果与 Ground Truth 进行成对比较，判断优劣。
- **Claim-level 评估**: 对每个权利要求进行单独评估，包括权利要求特定准确性、要素覆盖率、现有技术映射等。

### 输出文件结构 (Output File Structure)
实验结果采用清晰的按专利分组层级结构：

```
results/
├── patent_12295695_App/
│   ├── round_1_initial_examination/
│   │   ├── input.json          # AI实际输入（claims + refs）
│   │   ├── prediction.txt      # AI生成的OA
│   │   ├── evaluation.json     # 裁判AI评分结果
│   │   └── context.json        # 完整背景信息（仅用于评估）
│   ├── round_2_rebuttal/
│   │   ├── input.json          # AI实际输入（OA + claims）
│   │   ├── prediction.txt      # AI生成的反驳
│   │   ├── evaluation.json     
│   │   └── context.json        
│   └── summary.json            # 该专利所有轮次汇总
├── patent_US20180233034A1/
│   └── ...                     # 同样结构
└── experiment_overview.json    # 整体实验统计
```

#### 单轮次输入文件示例 (input.json)
```json
{
  "exam_phase": "initial_examination",
  "ai_input": {
    "claims": "CLAIMS\nWhat is claimed is:\n1. A method of managing monitoring data...",
    "refs": "Ref US20180233034A1: . Abstract A vehicle control device includes...",
    "available_info": ["claims", "bm25_references"]
  },
  "restricted_info": ["ground_truth_oa", "rebuttal_history", "human_decisions"]
}
```

#### 评估文件示例 (evaluation.json)
```json
{
  "scoring_context": {
    "ai_prediction_file": "prediction.txt",
    "ground_truth": "DETAILED ACTION\nNotice of Pre-AIA or AIA Status...",
    "comparison_basis": "element_by_element_analysis"
  },
  "scores": {
    "accuracy": 8,
    "rationality": 7, 
    "reliability": 6,
    "formatting": 9
  },
  "detailed_feedback": "AI correctly identified §103 rejection but missed some prior art citations..."
}
```

## 5. 项目结构 (Project Structure)
```
PatRe/
├── config/                 # 配置文件
│   ├── eval_config.yaml    # 主实验配置文件
│   └── prompt_templates.json # OA/Rebuttal 生成用的 Prompt 模板
├── src/                    # 源代码
│   ├── OfficeActionGeneration/ # OA 生成相关文档/代码
│   ├── RebuttalGeneration/     # Rebuttal 生成相关文档/代码
│   ├── data_alignment.py   # 数据对齐与预处理脚本
│   ├── evaluator.py        # 评测引擎核心类
│   ├── model_backend.py    # LLM API 调用接口
│   └── utils.py            # 通用工具函数
├── dataset/                # 数据存储目录
├── results/                # 实验结果输出目录
├── run_exp.py              # 实验启动主入口
├── clean_data.py           # 数据清洗工具
└── README.md               # 项目文档
```

## 6. 运行指南 (How to Run)

<!-- ### 环境准备 (Prerequisites)
Python 3.8+，请安装依赖（如有 `requirements.txt`）：
```bash
pip install -r requirements.txt
``` -->

### 数据准备 (Data Preparation)
运行对齐脚本，准备测试集数据：
```bash
python src/data_alignment.py
```

### 运行实验 (Running Experiments)
运行主实验脚本 `run_exp.py`：
```bash
# 使用默认配置运行 (Run with default config)
python run_exp.py

# 指定特定模型 (Run for a specific model)
python run_exp.py --model gpt-4o-mini

# 指定特定 IPC 分类 (Run for a specific IPC category)
python run_exp.py --ipc A

# 运行动态多轮审查流程（根据数据实际轮数）
python run_exp.py --run_multiround

# 运行固定3轮审查流程
python run_exp.py --run_chain3

# 运行指定任务
python run_exp.py --task initial_examination      # 初始审查
python run_exp.py --task rebuttal   # 申请人反驳
python run_exp.py --task response_to_rebuttal     # 审查员回复

# 运行 point-wise 评估 (Run point-wise evaluation)
python run_exp.py --run_point_wise --model gpt-4o-mini

# 运行 claim-level 评估 (Run claim-level evaluation)
python run_exp.py --run_claim_level --model gpt-4o-mini

# 运行所有评估指标 (Run all evaluation metrics)
python run_exp.py --run_point_wise --run_pairwise --run_claim_level --model gpt-4o-mini

# 组合运行模式
python run_exp.py --run_multiround --test_mode zero_shot --retrieval bm25_retrieved
```
