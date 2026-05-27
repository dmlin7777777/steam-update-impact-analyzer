"""
Sample Test: Trend Alert Models Performance
快速验证 Baseline vs Advanced 模型性能
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score, f1_score, roc_auc_score
)
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Import Prophet feature generator
try:
    from trend_prophet_features import generate_prophet_features
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet features not available")

# Configuration
SAMPLE_SIZE = 10000  # 10k样本快速测试
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
FEATURE_DIR = BASE_DIR / 'features' / 'fps'
INPUT_FILE = FEATURE_DIR / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet'

# Model parameters
TFIDF_MAX_FEATURES = 3000  # 减少特征数加快测试
TEST_SIZE = 0.2


def load_sample_data():
    """
    加载分层抽样数据
    """
    print("\n" + "="*80)
    print("加载样本数据")
    print("="*80)
    
    print(f"\n📁 读取: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    
    print(f"   总记录数: {len(df):,}")
    
    # 检查必需列
    if 'trend_alert_weak' not in df.columns:
        raise ValueError("'trend_alert_weak' column not found!")
    
    # 分层抽样（保持正负样本比例）
    positive_samples = df[df['trend_alert_weak'] == 1].sample(
        n=min(int(SAMPLE_SIZE * 0.1), (df['trend_alert_weak'] == 1).sum()),
        random_state=RANDOM_SEED
    )
    negative_samples = df[df['trend_alert_weak'] == 0].sample(
        n=min(SAMPLE_SIZE - len(positive_samples), (df['trend_alert_weak'] == 0).sum()),
        random_state=RANDOM_SEED
    )
    
    df_sample = pd.concat([positive_samples, negative_samples]).sample(
        frac=1, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    
    # 按时间排序（时序分割）
    df_sample = df_sample.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n✅ 样本大小: {len(df_sample):,}")
    print(f"   正样本: {(df_sample['trend_alert_weak'] == 1).sum():,} ({(df_sample['trend_alert_weak'] == 1).mean():.2%})")
    print(f"   负样本: {(df_sample['trend_alert_weak'] == 0).sum():,} ({(df_sample['trend_alert_weak'] == 0).mean():.2%})")
    
    return df_sample


def add_prophet_features(df):
    """
    添加Prophet特征
    """
    if not PROPHET_AVAILABLE:
        print("\n⚠️  跳过Prophet特征生成（库未安装）")
        return df
    
    print("\n" + "="*80)
    print("生成PROPHET特征")
    print("="*80)
    
    try:
        if 'vader_compound' in df.columns and 'timestamp' in df.columns:
            print(f"\n🔮 生成Prophet baseline特征...")
            prophet_features = generate_prophet_features(
                df,
                sentiment_col='vader_compound',
                timestamp_col='timestamp'
            )
            
            for col in prophet_features.columns:
                df[col] = prophet_features[col]
            
            print(f"\n✅ Prophet特征已添加:")
            for col in prophet_features.columns:
                non_zero = (df[col] != 0).sum()
                print(f"   - {col}: mean={df[col].mean():.4f}, non-zero={non_zero}/{len(df)}")
        else:
            print(f"\n⚠️  缺少必需列（vader_compound, timestamp）")
    except Exception as e:
        print(f"\n⚠️  Prophet特征生成失败: {e}")
    
    return df


def extract_features(df):
    """
    提取特征
    """
    print("\n" + "="*80)
    print("提取特征")
    print("="*80)
    
    # 1. 文本特征 (TF-IDF)
    print(f"\n📝 提取TF-IDF特征 (max_features={TFIDF_MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df['review_content_processed'].fillna('')).toarray()
    print(f"   TF-IDF shape: {X_text.shape}")
    
    # 2. 趋势特征（移除标签泄漏特征）
    trend_cols = [
        # ❌ REMOVED - 用于生成标签的触发器（数据泄漏）:
        # 'sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h',
        # 'comment_rate_change_24h', 'comment_rate_change_48h', 'comment_rate_change_72h',
        # 'ensemble_anomaly_score', 'if_anomaly_score', 'statistical_anomaly_score',
        
        # ✅ 滚动统计特征（描述性，非触发器）
        'sentiment_rolling_mean_24h', 'sentiment_rolling_mean_48h', 'sentiment_rolling_mean_72h',
        'sentiment_rolling_std_24h', 'sentiment_rolling_std_48h', 'sentiment_rolling_std_72h',
        'comment_rolling_mean_24h', 'comment_rolling_mean_48h', 'comment_rolling_mean_72h',
        
        # ✅ 基础统计（不直接参与标签生成）
        'vader_compound', 'votes_up', 'votes_funny', 'comment_count',
        'author_num_reviews', 'author_playtime_forever',
        
        # ✅ 风险特征（独立规则生成）
        'is_toxic', 'contains_bug_report', 'contains_balance_complaint',
        'contains_monetization_complaint', 'mentions_performance',
        
        # ✅ Prophet特征（基于历史基线，不参与标签）
        'prophet_baseline', 'prophet_deviation', 'prophet_lower_breach', 'prophet_upper_breach'
    ]
    
    available_trend_cols = [col for col in trend_cols if col in df.columns]
    X_trend = df[available_trend_cols].fillna(0).values
    
    print(f"\n📊 趋势特征: {len(available_trend_cols)} 个")
    prophet_cols = [col for col in available_trend_cols if col.startswith('prophet_')]
    if prophet_cols:
        print(f"   ✅ 包含 {len(prophet_cols)} 个Prophet特征")
    
    # 3. 组合特征
    X_combined = np.hstack([X_text, X_trend])
    
    print(f"\n✅ 特征提取完成:")
    print(f"   文本特征: {X_text.shape[1]}")
    print(f"   趋势特征: {X_trend.shape[1]}")
    print(f"   组合特征: {X_combined.shape[1]}")
    
    return X_text, X_combined, available_trend_cols


def train_and_evaluate(X_text, X_combined, y, split_idx):
    """
    训练并评估两个模型
    """
    # 时序分割（前80%训练，后20%测试）
    train_idx = slice(0, split_idx)
    test_idx = slice(split_idx, len(y))
    
    X_train_text, X_test_text = X_text[train_idx], X_text[test_idx]
    X_train_combined, X_test_combined = X_combined[train_idx], X_combined[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\n📊 数据分割:")
    print(f"   训练集: {len(y_train):,} ({y_train.sum():,} positive, {y_train.mean():.2%})")
    print(f"   测试集: {len(y_test):,} ({y_test.sum():,} positive, {y_test.mean():.2%})")
    
    # ============================================================
    # BASELINE MODEL: TF-IDF + LogisticRegression
    # ============================================================
    print("\n" + "="*80)
    print("训练 BASELINE 模型 (TF-IDF + LogisticRegression)")
    print("="*80)
    
    model_baseline = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    model_baseline.fit(X_train_text, y_train)
    y_pred_b = model_baseline.predict(X_test_text)
    y_pred_proba_b = model_baseline.predict_proba(X_test_text)[:, 1]
    
    # ============================================================
    # ADVANCED MODEL: Features + LightGBM
    # ============================================================
    print("\n" + "="*80)
    print("训练 ADVANCED 模型 (Features + LightGBM)")
    print("="*80)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model_advanced = lgb.LGBMClassifier(
        n_estimators=100,  # 减少迭代次数加快测试
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    
    model_advanced.fit(
        X_train_combined, y_train,
        eval_set=[(X_test_combined, y_test)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )
    
    y_pred_a = model_advanced.predict(X_test_combined)
    y_pred_proba_a = model_advanced.predict_proba(X_test_combined)[:, 1]
    
    # ============================================================
    # EVALUATION
    # ============================================================
    print("\n" + "="*80)
    print("性能对比")
    print("="*80)
    
    # Baseline metrics
    pr_auc_b = average_precision_score(y_test, y_pred_proba_b)
    roc_auc_b = roc_auc_score(y_test, y_pred_proba_b)
    f1_b = f1_score(y_test, y_pred_b)
    cm_b = confusion_matrix(y_test, y_pred_b)
    
    # Advanced metrics
    pr_auc_a = average_precision_score(y_test, y_pred_proba_a)
    roc_auc_a = roc_auc_score(y_test, y_pred_proba_a)
    f1_a = f1_score(y_test, y_pred_a)
    cm_a = confusion_matrix(y_test, y_pred_a)
    
    # Print comparison
    print(f"\n{'指标':<20} {'Baseline':<15} {'Advanced':<15} {'提升':<15}")
    print("="*65)
    print(f"{'PR-AUC':<20} {pr_auc_b:<15.4f} {pr_auc_a:<15.4f} {(pr_auc_a-pr_auc_b)*100:>+14.2f}%")
    print(f"{'ROC-AUC':<20} {roc_auc_b:<15.4f} {roc_auc_a:<15.4f} {(roc_auc_a-roc_auc_b)*100:>+14.2f}%")
    print(f"{'F1 Score':<20} {f1_b:<15.4f} {f1_a:<15.4f} {(f1_a-f1_b)*100:>+14.2f}%")
    
    # Confusion matrices
    print(f"\n📊 混淆矩阵对比:")
    print(f"\nBaseline:")
    print(f"   TN: {cm_b[0,0]:>6,}  FP: {cm_b[0,1]:>6,}")
    print(f"   FN: {cm_b[1,0]:>6,}  TP: {cm_b[1,1]:>6,}")
    print(f"   Precision: {cm_b[1,1]/(cm_b[1,1]+cm_b[0,1]):.4f}  Recall: {cm_b[1,1]/(cm_b[1,1]+cm_b[1,0]):.4f}")
    
    print(f"\nAdvanced:")
    print(f"   TN: {cm_a[0,0]:>6,}  FP: {cm_a[0,1]:>6,}")
    print(f"   FN: {cm_a[1,0]:>6,}  TP: {cm_a[1,1]:>6,}")
    print(f"   Precision: {cm_a[1,1]/(cm_a[1,1]+cm_a[0,1]):.4f}  Recall: {cm_a[1,1]/(cm_a[1,1]+cm_a[1,0]):.4f}")
    
    # Recall@K analysis
    print(f"\n📈 Recall@K 分析:")
    for k_pct in [0.01, 0.05, 0.10]:
        k = int(len(y_test) * k_pct)
        
        # Baseline
        top_k_idx_b = np.argsort(y_pred_proba_b)[-k:]
        recall_at_k_b = y_test[top_k_idx_b].sum() / y_test.sum()
        precision_at_k_b = y_test[top_k_idx_b].sum() / k
        
        # Advanced
        top_k_idx_a = np.argsort(y_pred_proba_a)[-k:]
        recall_at_k_a = y_test[top_k_idx_a].sum() / y_test.sum()
        precision_at_k_a = y_test[top_k_idx_a].sum() / k
        
        print(f"\n   Top {k_pct*100:.0f}% (k={k:,}):")
        print(f"      Baseline  - Recall: {recall_at_k_b:.4f}, Precision: {precision_at_k_b:.4f}")
        print(f"      Advanced  - Recall: {recall_at_k_a:.4f}, Precision: {precision_at_k_a:.4f}")
        print(f"      提升      - Recall: {(recall_at_k_a-recall_at_k_b)*100:+.2f}%, Precision: {(precision_at_k_a-precision_at_k_b)*100:+.2f}%")
    
    # Feature importance (Advanced model)
    if hasattr(model_advanced, 'feature_importances_'):
        print(f"\n🎯 Top 10 重要特征 (Advanced模型):")
        feature_importance = model_advanced.feature_importances_
        # Top 10
        top_10_idx = np.argsort(feature_importance)[-10:][::-1]
        for i, idx in enumerate(top_10_idx, 1):
            importance = feature_importance[idx]
            print(f"   {i:2d}. Feature {idx:4d}: {importance:.4f}")
    
    return {
        'baseline': {
            'pr_auc': pr_auc_b,
            'roc_auc': roc_auc_b,
            'f1': f1_b,
            'confusion_matrix': cm_b.tolist()
        },
        'advanced': {
            'pr_auc': pr_auc_a,
            'roc_auc': roc_auc_a,
            'f1': f1_a,
            'confusion_matrix': cm_a.tolist()
        }
    }


def main():
    print("="*80)
    print("SAMPLE TEST: TREND ALERT 模型性能测试")
    print("="*80)
    print(f"样本大小: {SAMPLE_SIZE:,}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 加载样本
    df = load_sample_data()
    
    # 2. 添加Prophet特征
    df = add_prophet_features(df)
    
    # 3. 提取特征
    X_text, X_combined, trend_feature_names = extract_features(df)
    y = df['trend_alert_weak'].values
    
    # 4. 训练和评估
    split_idx = int(len(df) * (1 - TEST_SIZE))
    results = train_and_evaluate(X_text, X_combined, y, split_idx)
    
    # 5. 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    pr_improvement = (results['advanced']['pr_auc'] - results['baseline']['pr_auc']) * 100
    
    print(f"\n✅ 样本测试完成: {len(df):,} 条评论")
    print(f"✅ Baseline PR-AUC: {results['baseline']['pr_auc']:.4f}")
    print(f"✅ Advanced PR-AUC: {results['advanced']['pr_auc']:.4f}")
    print(f"✅ 性能提升: {pr_improvement:+.2f}% PR-AUC")
    
    if pr_improvement > 5:
        print(f"\n🎯 Advanced模型显著优于Baseline，建议进行完整训练！")
        print(f"   运行: python scripts/train/train_trend_alert_fps.py")
    elif pr_improvement > 0:
        print(f"\n💡 Advanced模型略优于Baseline，可以进行完整训练验证")
    else:
        print(f"\n⚠️  Advanced模型未显示优势，建议检查特征工程")
    
    print(f"\n{'='*80}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
