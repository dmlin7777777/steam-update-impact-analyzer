# 批量推理引擎使用说明

## 概述

`batch_inference.py` 是一个完整的批量推理引擎，用于对游戏评论进行四个任务的预测：
- **情感分类** - BERT + 分类器（3 类：negative, neutral, positive）
- **主题识别** - BERT + 分类器（6 类）
- **风险评分** - 规则引擎（基于特征分布）
- **趋势告警** - LightGBM + TF-IDF（二分类：告警/正常）

## 模型架构

### 情感分类 (Sentiment)
- **模型类型**: DistilBertForSequenceClassification
- **模型路径**: `model/fps_sentiment_bert_model/bert_model_fps`
  - 完整的端到端模型，已包含分类头
  - 输入: 分词化的文本 (max_length=256)
  - 输出: 3 类概率 (negative, neutral, positive)
- **结构**: DistilBERT (distilled BERT) + 3-class classification head

### 主题识别 (Topic)
- **模型类型**: DistilBertForSequenceClassification
- **模型路径**: `model/fps_topic_6class_lda/best_bert_model_topic_category`
  - 完整的端到端模型，已包含分类头
  - 输入: 分词化的文本 (max_length=256)
  - 输出: 6 类概率（cheating, matchmaking_issues, monetization_concerns, multiplayer_features, technical_issues, user_interface）
- **标签映射**: `model/fps_topic_6class_lda/label_mapping_topic_category.json`
- **结构**: DistilBERT (distilled BERT) + 6-class classification head

### 趋势告警 (Trend Alert)
- **特征向量化**: `model/trend/final_vectorizer_oversample.joblib`
  - TF-IDF (max_features=450, ngram_range=(1,2))
  - 可选: 时间序列特征（rolling sentiment, Prophet）
- **分类器**: `model/trend/final_model_oversample.joblib`
  - LightGBM 二分类
  - 预设阈值: T=0.80（建议）

### 风险评分 (Risk)
- **配置**: `model/risk_scoring_system/risk_thresholds.json`
  - 规则引擎，基于已有的 risk_score 列分级

## 环境要求

```bash
pip install torch transformers pandas numpy scikit-learn lightgbm joblib
```

**GPU 支持** (可选，但推荐):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## 快速开始

### 1. 从特定时间窗口推理

```bash
python batch_inference.py \
  --genre fps \
  --start_date 2024-03-20 \
  --window_hours 48
```

- `--genre`: 游戏类型（目前支持 fps）
- `--start_date`: 起始日期 (YYYY-MM-DD)
- `--window_hours`: 时间窗口长度（小时），默认 48

### 2. 从自定义文件推理

```bash
python batch_inference.py \
  --genre fps \
  --input_file custom_reviews.parquet \
  --output_prefix custom_analysis
```

- `--input_file`: 输入 Parquet 文件路径
  - 必需列: `review_content_processed`, `timestamp`
  - 可选列: `rolling_24h_sentiment_mean`, `rolling_48h_sentiment_mean` 等（用于趋势告警）
- `--output_prefix`: 输出文件前缀

## 输出文件

执行推理后，在 `analysis_results/inference/` 生成以下文件：

### 1. 完整预测结果 (Parquet)
文件: `inference_fps_20240320_153000.parquet`
- 包含所有输入列
- 新增列:
  - `sentiment_pred`: 情感标签 (negative/neutral/positive)
  - `sentiment_proba_*`: 各类的概率
  - `topic_pred`: 主题标签
  - `topic_proba`: 最高概率值
  - `risk_level`: 风险等级 (low/medium/high/critical)
  - `trend_alert_pred`: 告警预测 (0/1)
  - `trend_alert_proba`: 告警概率

### 2. 汇总统计 (JSON)
文件: `inference_fps_20240320_153000_summary.json`
```json
{
  "metadata": {
    "genre": "fps",
    "n_samples": 15234,
    "timestamp": "20240320_153000"
  },
  "sentiment_distribution": {...},
  "topic_distribution": {...},
  "risk_distribution": {...},
  "trend_alert_rate": 0.0234
}
```

### 3. 高风险样本 (CSV)
文件: `inference_fps_20240320_153000_high_risk.csv`
- 仅包含风险等级为 "high" 或 "critical" 的样本
- 便于人工审核

## 推理流程

1. **加载模型**
   - 加载 BERT tokenizer 和模型（GPU/CPU）
   - 加载分类器头（joblib）
   - 加载风险配置和向量化器

2. **加载数据**
   - 从特征库或自定义文件读取评论
   - 可选时间窗口过滤

3. **情感分类**
   - 用 BERT tokenizer 编码评论 (max_length=512)
   - 批量处理 (batch_size=32)
   - 使用 [CLS] token 作为句子嵌入
   - 分类器输出 3 类概率

4. **主题识别**
   - 同情感分类流程
   - 分类器输出 6 类概率
   - 使用标签映射转换为标签名

5. **趋势告警**
   - TF-IDF 向量化评论文本
   - 可选: 拼接时间序列特征
   - LightGBM 预测二分类概率
   - 推荐阈值: 0.80

6. **风险评分**
   - 基于已有 risk_score 列分级
   - 阈值: low (<1.0), medium (1.0-2.0), high (2.0-3.0), critical (>3.0)

## 常见问题

### Q1: BERT 推理很慢，怎么加速？

**方案**：
1. 使用 GPU: 脚本自动检测 `torch.cuda.is_available()`
2. 减少批大小: 在代码中改 `batch_size = 16`
3. 使用知识蒸馏模型: 考虑使用 DistilBERT 或 MobileBERT（需重新训练）

### Q2: 缺少时间序列特征，趋势告警精度会下降吗？

**答**: 会。TF-IDF 仅提供文本信息，时间序列特征（rolling sentiment 等）提供上下文信息。
建议在输入文件中包含这些特征以获得更准确的预测。

特征列表:
- `rolling_24h_sentiment_mean` / `rolling_24h_sentiment_std`
- `rolling_48h_sentiment_mean` / `rolling_48h_sentiment_std`
- `rolling_72h_sentiment_mean` / `rolling_72h_sentiment_std`
- `prophet_yhat_normalized` / `prophet_yhat_upper_breach` / `prophet_yhat_lower_breach`
- `review_length`, `review_count_1h`

### Q3: 模型路径不对，如何修改？

在 `batch_inference.py` 中修改 `MODELS_DIR`:
```python
MODELS_DIR = Path('/path/to/your/models')
```

确保模型结构如下:
```
model/
├── fps_sentiment_bert_model/
│   ├── bert_model_fps/  (BERT 模型)
│   └── baseline_model_fps.joblib  (分类器)
├── fps_topic_6class_lda/
│   ├── best_bert_model_topic_category/  (BERT 模型)
│   └── best_baseline_model_topic_category.joblib  (分类器)
├── trend/
│   ├── final_model_oversample.joblib  (LightGBM)
│   └── final_vectorizer_oversample.joblib  (TF-IDF)
└── risk_scoring_system/
    └── risk_thresholds.json  (风险配置)
```

### Q4: 如何调整趋势告警的阈值？

在保存结果前，修改预测:
```python
threshold = 0.75  # 降低阈值 → 更多告警
df['trend_alert_pred'] = (df['trend_alert_proba'] >= threshold).astype(int)
```

推荐范围: 0.70 - 0.85

## 性能指标

| 任务 | 模型 | 批大小 | 单样本耗时* |
|------|------|--------|------------|
| 情感分类 | BERT + 分类器 | 32 | 2-3 ms |
| 主题识别 | BERT + 分类器 | 32 | 2-3 ms |
| 趋势告警 | LightGBM | 全部 | <1 ms |
| 风险评分 | 规则引擎 | 全部 | <0.1 ms |

*GPU (NVIDIA RTX 3090) 上的估计值

## 故障排除

### 错误: "torch/transformers not available"
```bash
pip install torch transformers
```

### 错误: "CUDA out of memory"
- 减少 batch_size (在代码中改为 16 或 8)
- 或使用 CPU (会慢 10-50 倍)

### 错误: "Model file not found"
- 检查模型路径是否正确
- 确保模型文件已保存

### 低推理精度
- 检查输入文本是否正确预处理
- 确保包含时间序列特征（如适用）
- 调整阈值

---

**最后更新**: 2024-10-26
**作者**: AI Assistant
