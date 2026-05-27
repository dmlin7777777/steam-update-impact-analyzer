# 游戏评论分析系统集成指南

## 快速开始：端到端工作流

```
原始评论 → 特征工程 → 弱标签生成 → 模型训练 → 批量推理 → 监控告警
    ↓            ↓            ↓           ↓           ↓          ↓
  reviews    features/    weaklabels   models/   predictions  alerts/
   .csv      .parquet     .parquet      .pkl       .json      .json
```

**核心流程（5步）**：
1. **特征工程**：`feature_engineering.py` → 文本/情感/行为/趋势特征
2. **弱标签**：`weaklabeling.py` → 情感/主题/风险/趋势标签
3. **训练**：`train_*.py` → Baseline + Advanced（5折CV）
4. **推理**：`batch_inference.py` → 批量预测
5. **监控**：`update_monitoring_dashboard.py` → 48h仪表盘+告警

---

## 系统概述

**四大核心功能**：
- **情感分类**：3分类，Baseline 82%+, BERT 85%+
- **主题识别**：6分类，Baseline 75%+, BERT 80%+
- **风险评分**：4级（Method 4），减少75.9%误报
- **趋势告警**：二分类，基于12个趋势特征

**完成状态**：
- ✅ 情感/主题训练（176k样本，5折CV）
- ✅ 风险评分系统
- ✅ 趋势弱标签修复（7.82% alert rate）
- 🔄 趋势告警模型（准备就绪）

---

## 已完成任务

### 1. 情感分类
- **数据**：176,848样本，3分类（正/中/负）
- **Baseline**：TF-IDF+LR，W-F1: 82%+，训练5分钟
- **Advanced**：DistilBERT，W-F1: 85%+，训练5-6小时
- **输出**：`sentiment_*.pkl`，`*_report.json`，混淆矩阵

### 2. 主题识别
- **数据**：176,848样本，6分类（匹配/作弊/付费/UI/技术/多人）
- **Baseline**：TF-IDF+LR，W-F1: 0.75-0.78
- **Advanced**：DistilBERT，W-F1: 0.80-0.83
- **输出**：`topic_*.pkl`，`*_report.json`，`lda_topic_keywords.json`

### 3. 风险评分
- **方法**：Method 4 (Hybrid Adaptive)，规则引擎
- **阈值**：Critical≥3.0 (2.3%), High≥2.0 (14%), Medium≥1.0 (26%)
- **因素**：10个（toxic, 负面情感, bug, 平衡, 付费等）
- **输出**：`risk_thresholds.json`，`high_risk_samples_top1000.csv`

### 4. 趋势告警（准备中）
- **数据**：176,848样本，7.82% alert
- **触发器**：7种（异常分数, 情感差分24/48/72h, 评论速率变化）
- **标准**：≥3个触发器同时满足
- **Baseline**：TF-IDF+LR，PR-AUC: 0.60-0.70
- **Advanced**：TF-IDF+趋势特征+LightGBM，PR-AUC: 0.75-0.85

---

## 使用指南

### 训练流程

```bash
# 1. 特征工程
cd scripts/feature
python feature_engineering.py --genre fps

# 2. 弱标签生成
cd ../train
python weaklabeling.py

# 3. 模型训练
python train_sentiment_fps.py
python train_topic_fps.py
python train_trend_alert_fps.py  # 待运行

# 4. 风险分析
cd ../summary
python analyze_risk_score_thresholds.py
```

### 推理流程

```bash
# 1. 批量推理（预测情感/主题/风险/趋势）
cd scripts/inference
python batch_inference.py --genre fps --start_date 2024-03-20 --window_hours 48

# 2. 监控仪表盘（交互式可视化）
python update_monitoring_dashboard.py --update_date 2024-03-20

# 3. 危机告警（5条规则检测）
python crisis_alert.py --update_date 2024-03-20 --baseline_days 7
```

**输出文件**：
- `inference_fps_*.parquet` - 完整预测结果
- `dashboard_fps_*.html` - 交互式仪表盘（4图）
- `crisis_alert_fps_*.json/md` - 告警报告

**告警规则**：
1. 负面情感 > 60%
2. Critical风险 > 5%
3. 付费+负面 > 30%
4. 评论速率 > 基线3倍
5. 趋势+风险组合 ≥ 10条

详见：`scripts/inference/README.md`

---

## 性能对比

| 任务 | Baseline | Advanced | 提升 | 训练时间 |
|------|----------|----------|------|----------|
| 情感 | W-F1: 0.82 | W-F1: 0.85 | +3.7% | 5分钟 vs 5小时 |
| 主题 | W-F1: 0.75-0.78 | W-F1: 0.80-0.83 | +5-6% | 5分钟 vs 5小时 |
| 风险 | Method 3 | Method 4 | -75.9% 误报 | 秒级 |
| 趋势 | PR-AUC: 0.60-0.70 | PR-AUC: 0.75-0.85 | +15-25% | 10分钟 vs 30分钟 |

---

## 目录结构

```
BAP/
├── features/fps/                    # 特征文件
├── analysis_results/
│   ├── train/                       # 训练结果
│   │   ├── fps_sentiment_bert/
│   │   ├── fps_topic_6class_lda/
│   │   └── fps_trend_alert/
│   ├── risk_scoring_system/         # 风险评分
│   └── inference/                   # 推理结果（待创建）
├── config/
│   └── update_events.json           # 更新事件（待创建）
└── scripts/
    ├── feature/feature_engineering.py
    ├── train/
    │   ├── weaklabeling.py
    │   └── train_*.py
    ├── inference/                   # 推理脚本（待创建）
    │   ├── batch_inference.py
    │   ├── update_monitoring_dashboard.py
    │   └── crisis_alert.py
    └── summary/
```

---

## 后续计划

### 阶段一：模型训练（1天）
- [ ] 运行 `train_trend_alert_fps.py`（20-30分钟）
- [ ] 验证 PR-AUC ≥ 0.75 (Advanced)

### 阶段二：推理系统（✅ 已完成框架）
- [x] **批量推理引擎**：`batch_inference.py`
  - 加载4个任务的模型
  - 对指定时间窗口批量预测
  - 输出parquet + summary.json + high_risk.csv
- [x] **监控仪表盘**：`update_monitoring_dashboard.py`
  - 4个可视化：情感趋势、主题分布、风险时间线、关键统计
  - 交互式HTML（Plotly）
  - Top 20高风险评论
- [x] **危机告警系统**：`crisis_alert.py`
  - 5条告警规则（负面激增、高风险集中、付费争议、速率异常、综合告警）
  - 基线对比（7天默认）
  - JSON + Markdown报告
- [ ] **BERT模型适配**：当前情感/主题使用占位符，待Sentiment/Topic训练完成后适配
- [ ] **趋势模型集成**：待Trend Alert训练完成后集成

### 阶段三：告警系统（✅ 已完成框架）
- [x] **5条告警规则**：
  1. 负面情感 > 60%
  2. Critical风险 > 5%
  3. 付费主题+负面 > 30%
  4. 评论速率 > 基线3倍
  5. 趋势告警+高风险 ≥ 10条
- [x] **基线对比报告**：更新前N天 vs 更新后48h
- [x] **告警级别**：CRITICAL / HIGH / MEDIUM / LOW
- [ ] **通知集成**：Email / Slack推送（待实现）

### 阶段四：部署（探索性）
- [ ] FastAPI 推理服务
- [ ] 多任务学习（可选）

---

## MVP 最小可交付物

**目标**：1.5天快速验证"更新后48h监控"

**输出**：
- `dashboard_2024-03-20.html`（交互式仪表盘）
- `alert_2024-03-20.json`（告警详情）
- `comparison_report.md`（更新前后对比）

**特点**：
- ✅ 基于现有模型（无需等趋势训练）
- ✅ 批处理模式（非实时）
- ✅ 手动触发
- ⚠️ 简化告警规则（统计阈值）

---

## 关键配置

**特征工程**：
- TF-IDF: 5000特征
- BERT: distilbert-base-uncased
- Topic: LDA 8-topics → 6-class

**模型训练**：
- 随机种子: 42
- CV: 5折 TimeSeriesSplit
- BERT: lr=2e-5, batch=16, epochs=10

**风险评分**：
- Method 4: (p99+μ+2σ)/2 = 3.0
- 10个因素：toxic(1.0), 负面(1.0), bug(0.8), 平衡(0.8), 付费(0.8)等

**趋势告警**：
- 窗口: 24h/48h/72h
- 触发器: 7种（≥3同时满足）
- LightGBM: 200树, depth=8, lr=0.05

---

**文档版本：** v3.0  
**最后更新：** 2025-10-25  
**作者：** BAP 项目组

**更新日志**：
- v3.0: 精简版，聚焦核心流程与关键信息
- v2.2: 添加推理系统规划
- v2.1: 新增趋势告警任务
- v2.0: 重写为已完成工作版本
