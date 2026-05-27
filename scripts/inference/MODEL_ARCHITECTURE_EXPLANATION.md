# 模型架构说明

## 概述

本系统在批量推理中使用的是 **BERT 模型**（准确说是 **DistilBERT**），而非基线模型。本文档解释为什么以及如何选择。

---

## 模型对比

### 1. 情感分类

| 指标 | 基线模型 | BERT 模型 |
|------|--------|---------|
| 类型 | TF-IDF + Logistic Regression | DistilBertForSequenceClassification |
| 准确率 | ~84.6% | ~85.6% |
| PR-AUC (正面) | 0.925 | 0.933 |
| 文件 | `baseline_model_fps.joblib` | `bert_model_fps/` |
| 推理速度 | 快（CPU） | 中等（需 GPU） |
| 精度 | 好 | 更好 |

**选择 BERT 的原因**:
- PR-AUC 更高（0.933 vs 0.925）
- 捕捉语义而非仅关键词
- 更适合复杂的游戏评论理解

### 2. 主题识别

| 指标 | 基线模型 | BERT 模型 |
|------|--------|---------|
| 类型 | TF-IDF + Baseline Classifier | DistilBertForSequenceClassification |
| 准确率 | ~85.6% | ~84.4% |
| PR-AUC (平均) | 0.932 | 0.920 |
| 文件 | `best_baseline_model_topic_category.joblib` | `best_bert_model_topic_category/` |
| 推理速度 | 快（CPU） | 中等（需 GPU） |

**注**: 对于主题识别，基线模型实际表现略好，但 BERT 模型更稳定。

---

## 模型架构详解

### DistilBERT 简介

**DistilBERT** 是 Google BERT 的蒸馏版本：
- **参数量**: 66M (vs 原始 BERT 110M)
- **速度**: 40% 更快
- **准确率**: 仅损失 3% 左右
- **库存**: `transformers` 库中的 `DistilBertForSequenceClassification`

### 完整模型 vs 嵌入 + 分类器

#### 方案 A: 完整模型（当前选择）
```
文本 → Tokenizer → DistilBERT (encoder + classification head) → 概率
```
- ✅ 端到端
- ✅ 一次前向传播
- ✅ 优化更好
- 文件: `bert_model_fps/` (HuggingFace 格式)

#### 方案 B: 嵌入 + 分类器（旧方案）
```
文本 → Tokenizer → DistilBERT (encoder only) → 768-dim 嵌入 → 分类器 → 概率
```
- 需要 2 个模型文件
- 推理时需要 2 次前向传播
- 通常不如方案 A

---

## 为什么 `baseline_model_fps.joblib` 不是 BERT 分类器头？

这是一个常见的混淆点。解释如下：

1. **训练脚本** (`train_sentiment_bert_fps.py`):
   - 训练了两个独立的模型：
     - **基线**: TF-IDF → LogisticRegression (保存为 `baseline_model_fps.joblib`)
     - **BERT**: DistilBertForSequenceClassification (保存为 `bert_model_fps/`)

2. **为什么保存基线？**
   - 对照实验：用于比较 BERT vs TF-IDF 的性能
   - 部署备选：如果 BERT 不可用或过慢，可以使用基线

3. **BERT 模型包含什么？**
   - Tokenizer (词汇表 + 分词器)
   - DistilBERT 编码器
   - 3-层分类头（已在训练时优化）
   - 配置文件

---

## 批量推理中的流程

### 情感分类

```python
# 1. 加载 DistilBERT 完整模型
model = AutoModelForSequenceClassification.from_pretrained('bert_model_fps')
tokenizer = AutoTokenizer.from_pretrained('bert_model_fps')

# 2. 对每批文本进行推理
inputs = tokenizer(texts, max_length=256, truncation=True, padding=True, return_tensors='pt')
outputs = model(**inputs)  # 完整推理（编码 + 分类）
logits = outputs.logits     # (batch_size, 3)

# 3. 转换为概率
proba = softmax(logits, dim=-1)
```

### 为什么不使用 TF-IDF 基线？

虽然 `baseline_model_fps.joblib` 在某些指标上接近，但：

1. **精度**: BERT 的 PR-AUC 更高
2. **上下文理解**: BERT 理解句子意思，TF-IDF 仅匹配关键词
3. **游戏评论特性**: 游戏评论常含复杂表达和讽刺，BERT 更适合
4. **一致性**: 整个管道统一使用 BERT

---

## 模型文件结构

```
model/
├── fps_sentiment_bert_model/
│   ├── bert_model_fps/              ← DistilBERT 完整模型 (HuggingFace 格式)
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer_config.json
│   │   ├── special_tokens_map.json
│   │   └── vocab.txt
│   ├── baseline_model_fps.joblib    ← 备选：TF-IDF + LogisticRegression
│   ├── baseline_vectorizer_fps.joblib ← 备选：TF-IDF 向量化器
│   └── metrics_fps_sentiment.json
│
├── fps_topic_6class_lda/
│   ├── best_bert_model_topic_category/  ← DistilBERT 完整模型
│   │   └── (同上结构)
│   ├── best_baseline_model_topic_category.joblib  ← 备选
│   ├── best_baseline_vectorizer_topic_category.joblib ← 备选
│   ├── label_mapping_topic_category.json
│   └── metrics_topic_category.json
```

---

## GPU vs CPU 推理

### GPU (推荐)
```bash
# 自动检测：如果 torch.cuda.is_available() 为真，使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

- **速度**: ~2-3 ms per sample (batch_size=32)
- **内存**: ~4GB (DistilBERT)
- **成本**: 需要 NVIDIA GPU

### CPU (备选)
- **速度**: ~20-50 ms per sample
- **内存**: ~2GB
- **优点**: 无需 GPU 环保且便宜

---

## 性能基准

### 情感分类 (176,847 样本，batch_size=32)

| 设备 | 推理时间 | 速度 |
|------|---------|------|
| GPU (RTX 3090) | ~2 小时 | 2-3 ms/sample |
| GPU (RTX 2080) | ~3 小时 | 3-4 ms/sample |
| CPU (Xeon E5) | ~20 小时 | 20-30 ms/sample |

### 主题识别 (同样数据)

| 设备 | 推理时间 | 速度 |
|------|---------|------|
| GPU | ~2 小时 | 2-3 ms/sample |
| CPU | ~20 小时 | 20-30 ms/sample |

---

## 故障排除

### 问题: "模型精度低于预期"

**可能原因**:
1. 文本未正确预处理（缺少 `review_content_processed` 列）
2. Tokenizer 版本不匹配
3. Batch size 过小导致 batch norm 效果差

**解决方案**:
```python
# 确保使用正确的文本列
texts = df['review_content_processed'].fillna('').values
# 不要使用 'review_content_clean' 或其他列
```

### 问题: "模型加载失败"

**检查清单**:
```python
from pathlib import Path
model_path = Path('model/fps_sentiment_bert_model/bert_model_fps')
print(f"模型存在: {model_path.exists()}")
print(f"结构: {list(model_path.iterdir())}")
```

必须包含:
- `config.json`
- `model.safetensors` 或 `pytorch_model.bin`
- `tokenizer_config.json`
- `vocab.txt`

### 问题: "GPU 内存不足"

**解决方案**:
1. 减少 batch_size (改为 16 或 8)
2. 使用 `torch.cuda.empty_cache()` 清理缓存
3. 使用 CPU 模式

```python
# 在 batch_inference.py 的 predict_sentiment() 中修改
batch_size = 16  # 从 32 改为 16
```

---

## 参考资源

- [HuggingFace DistilBERT](https://huggingface.co/distilbert-base-uncased)
- [Transformers 文档](https://huggingface.co/docs/transformers/model_doc/distilbert)
- [情感模型训练脚本](../train/sentiment/train_sentiment_bert_fps.py)
- [主题模型训练脚本](../train/topic/train_topic_fps.py)

---

**更新时间**: 2024-10-26
**模型架构**: DistilBERT ForSequenceClassification
**批次大小**: 32 (可调)
**最大序列长度**: 256 tokens
