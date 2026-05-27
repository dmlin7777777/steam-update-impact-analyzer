# 推理系统使用指南

## 📦 系统组件

推理系统包含3个核心脚本：

1. **batch_inference.py** - 批量推理引擎
2. **update_monitoring_dashboard.py** - 48小时监控仪表盘
3. **crisis_alert.py** - 危机告警系统

## 🚀 快速开始

### 前置条件

确保已训练好以下模型：
- ✅ 情感分类模型 (`sentiment_advanced_bert.pkl`)
- ✅ 主题识别模型 (`topic_advanced_bert.pkl`)
- 🔄 趋势告警模型 (`trend_alert_advanced.pkl`) - 可选

### 完整工作流

```bash
# 1. 批量推理（对指定时间窗口的评论进行预测）
cd scripts/inference
python batch_inference.py --genre fps --start_date 2024-03-20 --window_hours 48

# 2. 生成监控仪表盘（可视化分析）
python update_monitoring_dashboard.py --update_date 2024-03-20 --window_hours 48

# 3. 危机告警检测（自动化告警）
python crisis_alert.py --update_date 2024-03-20 --window_hours 48 --baseline_days 7
```

## 📋 详细说明

### 1. 批量推理引擎 (batch_inference.py)

**功能**：
- 加载训练好的模型
- 对指定时间窗口的评论进行四个任务的预测
- 输出结构化预测结果

**参数**：
- `--genre`: 游戏类型 (fps/strategy/leisure)
- `--start_date`: 起始日期 (YYYY-MM-DD)
- `--window_hours`: 时间窗口（小时），默认48
- `--input_file`: 自定义输入文件（可选）
- `--output_prefix`: 输出文件前缀

**示例**：
```bash
# 标准用法：分析2024-03-20更新后48小时
python batch_inference.py --genre fps --start_date 2024-03-20 --window_hours 48

# 自定义输入文件
python batch_inference.py --genre fps --input_file custom_reviews.parquet

# 72小时窗口
python batch_inference.py --genre fps --start_date 2024-03-20 --window_hours 72
```

**输出文件**：
```
analysis_results/inference/
├── inference_fps_20241025_170000.parquet        # 完整预测结果
├── inference_fps_20241025_170000_summary.json   # 汇总统计
└── inference_fps_20241025_170000_high_risk.csv  # 高风险样本
```

**输出字段**：
- `sentiment_pred`: 情感预测 (positive/neutral/negative)
- `sentiment_proba_*`: 情感概率分布
- `topic_pred`: 主题预测 (6个类别)
- `topic_proba`: 主题预测概率
- `risk_score`: 风险分数 (0-10)
- `risk_level`: 风险等级 (low/medium/high/critical)
- `trend_alert_pred`: 趋势告警预测 (0/1)
- `trend_alert_proba`: 趋势告警概率

---

### 2. 监控仪表盘 (update_monitoring_dashboard.py)

**功能**：
- 生成交互式HTML仪表盘
- 4个核心可视化：情感趋势、主题分布、风险时间线、关键统计
- 提取Top 20高风险评论

**参数**：
- `--update_date`: 更新日期 (YYYY-MM-DD)
- `--window_hours`: 监控窗口（小时），默认48
- `--genre`: 游戏类型
- `--inference_file`: 推理结果文件（可选，默认使用最新）
- `--output_prefix`: 输出文件前缀

**示例**：
```bash
# 标准用法
python update_monitoring_dashboard.py --update_date 2024-03-20

# 72小时监控
python update_monitoring_dashboard.py --update_date 2024-03-20 --window_hours 72

# 指定推理文件
python update_monitoring_dashboard.py --update_date 2024-03-20 \
    --inference_file ../inference/inference_fps_20241025_170000.parquet
```

**输出文件**：
```
analysis_results/dashboards/
├── dashboard_fps_20240320_48h.html              # 交互式仪表盘 ⭐
├── dashboard_fps_20240320_top_risk.csv          # Top 20高风险评论
└── dashboard_fps_20240320_summary.json          # 统计摘要
```

**仪表盘内容**：
1. **情感趋势图**：每小时情感分布堆叠面积图
2. **主题分布饼图**：6个主题的占比
3. **风险时间线**：平均/最大风险分数 + 高风险比例（双Y轴）
4. **关键统计表**：总评论数、负面比例、高风险比例、趋势告警数、平均风险分数

**查看方式**：
```bash
# 在浏览器中打开
start analysis_results/dashboards/dashboard_fps_20240320_48h.html  # Windows
open analysis_results/dashboards/dashboard_fps_20240320_48h.html   # Mac
```

---

### 3. 危机告警系统 (crisis_alert.py)

**功能**：
- 运行5条告警规则
- 对比基线数据
- 生成告警报告（JSON + Markdown）

**参数**：
- `--update_date`: 更新日期 (YYYY-MM-DD)
- `--window_hours`: 监控窗口（小时），默认48
- `--baseline_days`: 基线周期（天），默认7
- `--genre`: 游戏类型

**示例**：
```bash
# 标准用法：对比7天基线
python crisis_alert.py --update_date 2024-03-20

# 14天基线对比
python crisis_alert.py --update_date 2024-03-20 --baseline_days 14

# 72小时监控窗口
python crisis_alert.py --update_date 2024-03-20 --window_hours 72
```

**告警规则**：

| 规则 | 描述 | 阈值 | 严重级别 |
|------|------|------|----------|
| R1 | 负面情感激增 | 负面评论 > 60% | HIGH |
| R2 | 高风险集中 | Critical风险 > 5% | CRITICAL |
| R3 | 付费争议 | 付费主题+负面 > 30% | MEDIUM |
| R4 | 评论速率异常 | 评论量 > 基线3倍 | MEDIUM |
| R5 | 综合告警 | 趋势告警+高风险 ≥ 10条 | CRITICAL |

**输出文件**：
```
analysis_results/alerts/
├── crisis_alert_fps_20240320.json               # 完整告警数据
└── crisis_alert_fps_20240320.md                 # 人类可读摘要 ⭐
```

**告警级别**：
- 🔴 **CRITICAL**: 立即行动
- 🟠 **HIGH**: 24h内处理
- 🟡 **MEDIUM**: 密切关注
- 🟢 **LOW**: 正常

---

## 🎯 典型使用场景

### 场景1: 游戏大版本更新后监控

```bash
# 假设2024-03-20发布v2.5.0大更新

# Step 1: 运行批量推理（更新后48小时）
python batch_inference.py --genre fps --start_date 2024-03-20 --window_hours 48

# Step 2: 生成监控仪表盘
python update_monitoring_dashboard.py --update_date 2024-03-20

# Step 3: 危机告警检测（对比7天基线）
python crisis_alert.py --update_date 2024-03-20 --baseline_days 7

# Step 4: 查看结果
# - 打开仪表盘: analysis_results/dashboards/dashboard_fps_20240320_48h.html
# - 查看告警: analysis_results/alerts/crisis_alert_fps_20240320.md
```

### 场景2: 热修复补丁后快速验证

```bash
# 假设2024-04-15发布热修复，只需监控24小时

python batch_inference.py --genre fps --start_date 2024-04-15 --window_hours 24
python update_monitoring_dashboard.py --update_date 2024-04-15 --window_hours 24
python crisis_alert.py --update_date 2024-04-15 --window_hours 24
```

### 场景3: 自定义数据分析

```bash
# 使用自定义评论文件
python batch_inference.py --genre fps --input_file custom_reviews.parquet
python update_monitoring_dashboard.py --update_date 2024-03-20 \
    --inference_file analysis_results/inference/inference_fps_latest.parquet
```

---

## 📊 输出目录结构

```
BAP/
├── analysis_results/
│   ├── inference/              # 推理结果
│   │   ├── inference_fps_*.parquet
│   │   ├── *_summary.json
│   │   └── *_high_risk.csv
│   ├── dashboards/             # 监控仪表盘
│   │   ├── dashboard_fps_*.html
│   │   ├── *_top_risk.csv
│   │   └── *_summary.json
│   └── alerts/                 # 危机告警
│       ├── crisis_alert_fps_*.json
│       └── crisis_alert_fps_*.md
├── config/
│   └── update_events.json      # 更新事件配置
└── models/
    └── fps/                    # 训练好的模型
        ├── sentiment_advanced_bert.pkl
        ├── topic_advanced_bert.pkl
        └── trend_alert_advanced.pkl
```

---

## ⚙️ 配置文件

### 更新事件配置 (config/update_events.json)

用于管理游戏更新事件和告警阈值：

```json
{
  "update_events": [
    {
      "game_id": "fps_game_1",
      "update_date": "2024-03-20",
      "update_version": "v2.5.0",
      "update_type": "major",
      "description": "大型更新：新地图、武器平衡调整",
      "monitoring": {
        "enabled": true,
        "window_hours": 48,
        "baseline_days": 7
      },
      "alert_thresholds": {
        "negative_sentiment_pct": 0.60,
        "critical_risk_pct": 0.05
      }
    }
  ]
}
```

---

## 🔧 故障排查

### 1. 模型未找到

**问题**：`⚠️ 情感模型未找到`

**解决**：
- 检查模型是否已训练：`ls models/fps/`
- 训练模型：
  ```bash
  cd scripts/train
  python train_sentiment_fps.py
  python train_topic_fps.py
  ```

### 2. 推理结果为空

**问题**：`未找到推理结果`

**解决**：
- 先运行批量推理：`python batch_inference.py --genre fps --start_date 2024-03-20`
- 或指定输入文件：`python update_monitoring_dashboard.py --inference_file xxx.parquet`

### 3. 时间窗口无数据

**问题**：`样本数: 0`

**解决**：
- 检查起始日期是否在数据范围内
- 查看特征文件的时间范围：
  ```python
  import pandas as pd
  df = pd.read_parquet('features/fps/...parquet')
  print(df['timestamp'].min(), df['timestamp'].max())
  ```

---

## 📝 待完成功能

当前推理系统为**框架版本**，以下功能待实现：

- [ ] **BERT模型推理适配** - 情感和主题预测使用占位符
- [ ] **趋势模型集成** - 等待train_trend_alert_fps.py完成
- [ ] **实时推理API** - FastAPI服务
- [ ] **自动化调度** - 定时运行推理和告警
- [ ] **通知集成** - Email/Slack告警推送

---

## 🎓 最佳实践

1. **按顺序运行**：先推理 → 再仪表盘 → 最后告警
2. **保留历史记录**：所有输出文件带时间戳，方便对比
3. **调整阈值**：根据业务需求修改`crisis_alert.py`中的阈值
4. **定期验证**：每次更新后运行完整流程
5. **可视化优先**：先查看仪表盘，再看详细告警

---

**文档版本**: v1.0  
**最后更新**: 2025-10-25  
**作者**: BAP 项目组
