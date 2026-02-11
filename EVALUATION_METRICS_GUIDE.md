# PatRe 评估指标使用指南

## 概述

PatRe 项目现在支持多种评估指标，包括传统的审查员/代理人评估以及新增的 point-wise、pairwise 和 claim-level 评估。

## 评估指标类型

### 1. 传统评估指标
- **examiner_judge**: 评估 Office Action 生成质量（准确性、合理性、可靠性、格式）
- **rebuttal_judge**: 评估 Rebuttal 生成质量（有效性、准确性、权利要求完整性、专业性）

### 2. 新增评估指标

#### Point-wise 评估
- **指标名称**: `point_wise_evaluator`
- **功能**: 将生成结果与真实值进行逐项对比
- **评估维度**:
  - Factual Accuracy: 事实准确性
  - Legal Compliance: 法律合规性
  - Technical Precision: 技术精确性
  - Argument Quality: 论证质量
  - Structure Coherence: 结构一致性

#### Pairwise 比较
- **指标名称**: `pairwise_evaluator`
- **功能**: 比较两个文本哪个更好
- **评估维度**:
  - Overall Quality: 整体质量
  - Factual Accuracy: 事实准确性
  - Legal Compliance: 法律合规性
  - Effectiveness: 有效性
  - Clarity: 清晰度

#### Claim-level 评估
- **指标名称**: `claim_level_evaluator`
- **功能**: 对每个权利要求进行单独评估
- **评估维度**:
  - Claim-Specific Accuracy: 权利要求特定准确性
  - Element Coverage: 要素覆盖率
  - Prior Art Mapping: 现有技术映射
  - Rejection Reasoning: 驳回推理
  - Technical Depth: 技术深度

## 使用方法

### 1. 运行单个评估指标

```bash
# 运行 point-wise 评估
python run_exp.py --run_point_wise --model gpt-4o-mini --task v1_initial_examination

# 运行 pairwise 比较
python run_exp.py --run_pairwise --model gpt-4o-mini --task v1_high_fidelity_rebuttal

# 运行 claim-level 评估
python run_exp.py --run_claim_level --model gpt-4o-mini --task v1_initial_examination
```

### 2. 组合运行多个评估指标

```bash
# 同时运行多个评估指标
python run_exp.py --run_point_wise --run_claim_level --model gpt-4o-mini --task v1_initial_examination

# 与多轮审查结合
python run_exp.py --run_multiround --run_point_wise --run_claim_level
```

### 3. 运行完整评估流程

```bash
# 完整评估（包含所有指标）
python run_exp.py --run_point_wise --run_pairwise --run_claim_level --model gpt-4o-mini --task v1_initial_examination
```

## 分析评估结果

运行以下命令分析评估结果：

```bash
python analyze_evaluation_metrics.py --result-file ./results/[your_experiment]/final_results.json --output-dir ./analysis_output
```

这将生成：
- 详细的评估报告
- 汇总指标图表
- 评分分布图

## 配置自定义评估指标

评估指标的配置在 `config/prompt_templates.json` 的 `Evaluation_Metrics` 部分：

```json
{
  "Evaluation_Metrics": {
    "point_wise_evaluator": {
      "description": "Point-wise evaluation comparing generated text with ground truth",
      "system_prompt": "You are a Senior Patent Attorney performing detailed point-wise comparison...",
      "user_prompt_template": "Compare the generated text with ground truth based on the given context..."
    }
  }
}
```

您可以根据需要修改提示模板来自定义评估标准。

## 评估维度说明

### 准确性 (Accuracy)
- 决策（驳回/允许）是否与事实相符
- 是否应用了正确的法律条款（102/103）

### 合理性 (Rationality)  
- 驳回理由是否逻辑合理
- 是否正确解释了权利要求限制

### 可靠性 (Reliability)
- 现有技术引用是否准确
- 引用的现有技术是否确实披露了所引述的要素

### 有效性 (Effectiveness)
- 论证是否具有说服力
- 是否正确解决了办公室行动中提出的具体问题

## 注意事项

1. **计算成本**: 新增的评估指标需要额外的 API 调用，会增加计算成本
2. **时间消耗**: 多重评估会显著增加运行时间
3. **结果存储**: 评估结果会存储在 `results` 目录下的相应实验文件夹中

## 实验设计建议

根据您的研究计划，建议的实验流程：

1. **2.2-2.8**: 数据集大规模爬取
2. **2.8-2.14**: TestSet 确定 + Metric 设计验证
3. **2.22-3.18**: 
   - Zero-shot 评估（使用所有评估指标）
   - SFT 模型训练
4. **3.19-4.1**: 结果分析 + 论文撰写

使用新增的评估指标可以更全面地评估模型性能，特别是在准确性、有效性和技术深度等方面。