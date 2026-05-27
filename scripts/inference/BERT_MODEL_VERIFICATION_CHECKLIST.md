# BERT 模型加载验证清单

## 目标
验证 `batch_inference.py` 中的 BERT 模型是否正确加载和推理。

---

## 1. 模型文件检查

### 情感分类模型
```bash
# Windows PowerShell
Test-Path "c:\Users\12932\Desktop\nus\BAP\model\fps_sentiment_bert_model\bert_model_fps\config.json"
Test-Path "c:\Users\12932\Desktop\nus\BAP\model\fps_sentiment_bert_model\bert_model_fps\model.safetensors"
Test-Path "c:\Users\12932\Desktop\nus\BAP\model\fps_sentiment_bert_model\bert_model_fps\tokenizer_config.json"
```

**预期**: 都返回 `True`

### 主题识别模型
```bash
Test-Path "c:\Users\12932\Desktop\nus\BAP\model\fps_topic_6class_lda\best_bert_model_topic_category\config.json"
Test-Path "c:\Users\12932\Desktop\nus\BAP\model\fps_topic_6class_lda\best_bert_model_topic_category\model.safetensors"
Test-Path "c:\Users\12932\Desktop\nus\BAP\model\fps_topic_6class_lda\best_bert_model_topic_category\tokenizer_config.json"
```

**预期**: 都返回 `True`

---

## 2. Python 环境检查

```bash
# 在项目根目录
cd c:\Users\12932\Desktop\nus\BAP

# 检查必要的包
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import joblib; print(f'Joblib: {joblib.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

**预期**: 所有包都已安装

---

## 3. GPU 检查（可选但推荐）

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**预期**: 如果有 NVIDIA GPU 返回 `True` 和 GPU 名称

---

## 4. 模型加载测试

### 方式 A: 使用测试脚本（推荐）

```bash
cd scripts\inference
python test_model_loading.py
```

**预期输出**:
```
================================================================================
测试模型加载
================================================================================
✅ BERT Tokenizer (Sentiment)
✅ BERT Model (Sentiment)
✅ BERT Tokenizer (Topic)
✅ BERT Model (Topic)
✅ Label Mapping (Topic)
✅ LightGBM Model (Trend)
✅ TF-IDF Vectorizer (Trend)
✅ Risk Config

================================================================================
✅ 所有模型加载成功！
================================================================================
```

### 方式 B: 手动测试

```python
from scripts.inference.batch_inference import BatchInferenceEngine

engine = BatchInferenceEngine(genre='fps')
engine.load_models()

# 检查模型是否加载
print("情感 Tokenizer:", 'sentiment_tokenizer' in engine.models)
print("情感 BERT 模型:", 'sentiment_model' in engine.models)
print("主题 Tokenizer:", 'topic_tokenizer' in engine.models)
print("主题 BERT 模型:", 'topic_model' in engine.models)
print("趋势模型:", 'trend' in engine.models)
```

**预期**: 所有输出都是 `True`

---

## 5. 快速推理测试

```python
import pandas as pd
from scripts.inference.batch_inference import BatchInferenceEngine

# 创建测试数据
test_df = pd.DataFrame({
    'review_content_processed': [
        'game is amazing best ever',
        'terrible lag and crashes',
        'average game nothing special'
    ],
    'timestamp': ['2024-10-01', '2024-10-01', '2024-10-01'],
    'risk_score': [0.5, 2.5, 1.0]
})

# 运行推理
engine = BatchInferenceEngine(genre='fps')
engine.load_models()

# 只测试情感分类（最快）
result = engine.predict_sentiment(test_df)
print(result[['review_content_processed', 'sentiment_pred', 'sentiment_proba_positive']])
```

**预期**: 
- 第 1 条评论: 正面 (`sentiment_pred` = positive)
- 第 2 条评论: 负面 (`sentiment_pred` = negative)
- 第 3 条评论: 中立 (`sentiment_pred` = neutral)

---

## 6. 完整批量推理测试

```bash
# 使用最近的 100 条评论进行推理
cd scripts\inference
python batch_inference.py --genre fps --start_date 2024-10-20 --window_hours 24 --output_prefix test
```

**预期输出**:
- 创建文件: `analysis_results/inference/inference_fps_*.parquet`
- 包含列: `sentiment_pred`, `topic_pred`, `risk_level`, `trend_alert_pred`
- 运行时间: ~10-20 分钟（GPU）

---

## 7. 输出验证

```bash
# 检查输出文件
ls -la analysis_results/inference/inference_fps_*.parquet

# 查看结果摘要
python -c "
import pandas as pd
from pathlib import Path

# 读取最新的结果文件
result_file = sorted(Path('analysis_results/inference').glob('inference_fps_*.parquet'))[-1]
df = pd.read_parquet(result_file)

print(f'样本数: {len(df)}')
print(f'\\n情感分布:\\n{df[\"sentiment_pred\"].value_counts()}')
print(f'\\n主题分布:\\n{df[\"topic_pred\"].value_counts()}')
print(f'\\n风险分布:\\n{df[\"risk_level\"].value_counts()}')
print(f'\\n趋势告警率: {df[\"trend_alert_pred\"].mean():.2%}')
"
```

**预期**: 数据合理分布，无 NaN 或异常值

---

## 8. 常见问题排除

### Q1: "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Q2: "OutOfMemoryError: CUDA out of memory"
解决方案（在 `batch_inference.py` 中修改）:
```python
# 第 225 行附近
batch_size = 16  # 改为更小的值（默认是 32）
```

### Q3: "FileNotFoundError: model not found"
检查模型路径:
```python
from pathlib import Path
model_dir = Path('model/fps_sentiment_bert_model/bert_model_fps')
print(f"模型存在: {model_dir.exists()}")
print(f"文件: {list(model_dir.iterdir())}")
```

### Q4: "推理结果都是同一个类"
可能原因:
- 输入文本为空 → 检查 `review_content_processed` 列
- 模型未训练好 → 重新训练
- Tokenizer 版本不匹配 → 重新加载

---

## 9. 性能基准

运行后记录这些指标：

| 阶段 | 耗时 | 速度 |
|------|------|------|
| 模型加载 | ___ ms | - |
| 情感分类 (1000 samples) | ___ s | ___  ms/sample |
| 主题识别 (1000 samples) | ___ s | ___ ms/sample |
| 趋势告警 (1000 samples) | ___ s | ___ ms/sample |
| 风险评分 (1000 samples) | ___ s | ___ ms/sample |
| **总计** | ___ s | - |

---

## 10. 检查清单

- [ ] 所有模型文件存在
- [ ] Python 环境已配置
- [ ] GPU 已检测（如有）
- [ ] 模型加载测试通过
- [ ] 快速推理测试通过
- [ ] 完整批量推理成功
- [ ] 输出文件有效
- [ ] 性能指标合理

---

## 验证完成

如果以上所有步骤都通过，说明：
✅ BERT 模型正确加载
✅ 推理流程正常工作
✅ 可以投入生产使用

---

**检查清单版本**: 1.0  
**最后更新**: 2024-10-26  
**关键文件**: `batch_inference.py`, `test_model_loading.py`
