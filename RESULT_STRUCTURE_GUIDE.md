# PatRe 结果文件结构说明

## 新的输出结构概览

```
results/
├── experiment_gpt-4o-mini_v1_initial_examination_20260209_1548/
│   ├── experiment_overview.json          # 实验整体概览
│   ├── patent_12295695_App/              # 专利1的文件夹
│   │   ├── patent_summary.json           # 该专利所有轮次汇总
│   │   └── round_1_v1_initial_examination/
│   │       ├── ai_input.json             # AI实际输入（考试时看到的内容）
│   │       ├── ai_output.txt             # AI生成的答案
│   │       ├── evaluation_context.json   # 评估用的完整上下文
│   │       └── scores.json               # 评分结果
│   └── patent_US20180233034A1/           # 专利2的文件夹
│       └── round_1_v1_initial_examination/
│           ├── ai_input.json
│           ├── ai_output.txt
│           ├── evaluation_context.json
│           └── scores.json
```

## 各文件详细说明

### 1. experiment_overview.json - 实验概览
```json
{
  "experiment_id": "gpt-4o-mini_v1_initial_examination_20260209_1548",
  "config_snapshot": {...},  // 完整的实验配置
  "timestamp": "2026-02-09T15:48:00",
  "total_patents": 2,
  "patent_list": ["12295695_App", "US20180233034A1"],
  "overall_metrics": {
    "accuracy": 7.5,
    "rationality": 6.8,
    "reliability": 6.2,
    "formatting": 8.1,
    "total_score_mean": 28.6,
    "num_samples_scored": 2
  }
}
```

### 2. patent_summary.json - 单个专利汇总
```json
{
  "patent_id": "12295695_App",
  "total_rounds": 1,
  "tasks_executed": ["v1_initial_examination"],
  "average_scores": {
    "accuracy": 8,
    "rationality": 7,
    "reliability": 6,
    "formatting": 9,
    "total_score_mean": 30
  }
}
```

### 3. ai_input.json - AI实际输入（关键！）
```json
{
  "claims": "CLAIMS\nWhat is claimed is:\n1. A method of managing monitoring data...",
  "refs": "Ref US20180233034A1: . Abstract A vehicle control device includes...",
  "task_type": "v1_initial_examination",
  "retrieval_setting": "bm25_retrieved",
  "available_info": ["claims", "refs", "task_type", "retrieval_setting"],
  "restricted_info": ["ground_truth_oa", "rebuttal_history", "human_decisions"]
}
```

### 4. ai_output.txt - AI生成的答案
```
**Status**: Non-Final Rejection

**Detailed Action**:
**A. Opening Statement with Claim Status**  
The current claims 1-20 have been examined...

[完整的AI生成内容]
```

### 5. evaluation_context.json - 评估上下文
```json
{
  "full_context": {
    "claims": "完整的权利要求文本...",
    "refs": "完整的参考文献...",
    "office_action": "人类审查员的真实OA（仅用于评估）",
    "rebuttal_history": "反驳历史（仅用于评估）"
  },
  "ground_truth": "人类审查员的真实答案...",
  "input_refs_preview": "参考文献前200字预览..."
}
```

### 6. scores.json - 评分结果
```json
{
  "scores": {
    "accuracy": 8,
    "rationality": 7,
    "reliability": 6,
    "formatting": 9
  },
  "total_score": 30,
  "justification": "AI correctly identified §103 rejection but missed some prior art citations..."
}
```

## 科研意义说明

### 考试过程 vs 评估过程分离
- **ai_input.json**: AI考试时实际看到的内容（确保公平性）
- **ai_output.txt**: AI独立生成的答案（零样本预测）
- **evaluation_context.json**: 评估时使用的完整上下文（包含"剧透"信息）

### 透明度提升
1. **明确标识限制信息**: `restricted_info` 字段清楚标明AI不应该看到的内容
2. **考试阶段标识**: `exam_phase` 区分纯预测和有上下文的生成
3. **输入输出分离**: AI的输入和输出文件完全独立

### 便于验证
- 其他研究者可以：
  1. 查看 `ai_input.json` 确认AI实际输入
  2. 运行相同输入验证 `ai_output.txt` 的可复现性
  3. 对比 `evaluation_context.json` 中的真实答案
  4. 检查 `scores.json` 中的评分细节

这种结构设计确保了实验的科学性和可验证性，完全符合您提出的科研规范要求。