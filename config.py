from pathlib import Path

BASE_DIR = Path(__file__).parent

# 数据路径
DATA_LABEL_DIR = BASE_DIR / "data_label"
DATA_NOLABEL_DIR = BASE_DIR / "data_nolabel"
FEATURES_DIR = BASE_DIR / "features"
ANALYSIS_DIR = BASE_DIR / "analysis_results"
MODEL_DIR = BASE_DIR / "model"

# 各 genre 的组合评论文件
COMBINED_LABEL = {
    "fps":      DATA_LABEL_DIR / "combined" / "combined_fps_reviews.xlsx",
    "leisure":  DATA_LABEL_DIR / "combined" / "combined_leisure_reviews.xlsx",
    "strategy": DATA_LABEL_DIR / "combined" / "combined_strategy_reviews.xlsx",
}

NEWS_DIR = DATA_NOLABEL_DIR / "cleaned"

# 模型子目录（当前只有 fps）
MODEL_PATHS = {
    "fps": {
        "sentiment_bert":    MODEL_DIR / "fps_sentiment_bert_model" / "bert_model_fps",
        "sentiment_baseline": MODEL_DIR / "fps_sentiment_bert_model" / "baseline_model_fps.joblib",
        "sentiment_vec":     MODEL_DIR / "fps_sentiment_bert_model" / "baseline_vectorizer_fps.joblib",
        "topic_bert":        MODEL_DIR / "fps_topic_6class_lda" / "best_bert_model_topic_category",
        "topic_baseline":    MODEL_DIR / "fps_topic_6class_lda" / "best_baseline_model_topic_category.joblib",
        "topic_vec":         MODEL_DIR / "fps_topic_6class_lda" / "best_baseline_vectorizer_topic_category.joblib",
        "topic_label_map":   MODEL_DIR / "fps_topic_6class_lda" / "label_mapping_topic_category.json",
        "trend":             MODEL_DIR / "trend" / "final_model_oversample.joblib",
        "trend_vec":         MODEL_DIR / "trend" / "final_vectorizer_oversample.joblib",
        "risk_thresholds":   MODEL_DIR / "risk_scoring_system" / "risk_thresholds.json",
    }
}

# Feature store：预计算特征文件（含弱标签的最终版本）
FEATURE_STORE = {
    "fps":      FEATURES_DIR / "fps"      / "gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet",
    "leisure":  FEATURES_DIR / "leisure"  / "gpu_optimized_features_leisure_exclflagged_enhanced_features_with_weaklabels.parquet",
    "strategy": FEATURES_DIR / "strategy" / "gpu_optimized_features_strategy_exclflagged_enhanced_features_with_weaklabels.parquet",
}

# LLM 设置
LLM_MODEL = "claude-sonnet-4-6"          # LLM review agent 使用
LLM_MODEL_LIGHT = "claude-haiku-4-5-20251001"  # 轻量任务
LLM_REVIEW_THRESHOLD = 0.05   # flagged 占比超过 5% 触发 LLM 复查
RISK_GRAY_ZONE = (1.5, 3.0)   # 灰色地带触发 LLM 风险复核

# 推理设置
INFERENCE_WINDOW_HOURS = 48
BASELINE_DAYS = 7

# 告警阈值
ALERT_THRESHOLDS = {
    "negative_sentiment_pct": 0.60,
    "critical_risk_pct": 0.05,
    "review_rate_multiplier": 3.0,
}
