"""
Trend Alert Training Script for FPS Genre

训练目标：
- 基于修复后的 trend_alert_weak (7.82% positive) 训练二分类模型
- 识别游戏评论的趋势异常（负面情绪激增、评论量异常等）

模型：
- Baseline: TF-IDF + LogisticRegression
- Advanced: Feature Engineering + LightGBM (利用趋势特征)

评估指标：
- Primary: PR-AUC (不平衡数据优选)
- Secondary: Recall@K, Precision, F1, ROC-AUC
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    precision_recall_curve, roc_auc_score, 
    average_precision_score, f1_score
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import Prophet feature generator (optional enhancement)
try:
    from trend_prophet_features import generate_prophet_features
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet features not available (trend_prophet_features.py not found)")

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
USE_PROPHET_FEATURES = True  # Set to False to disable Prophet enhancement

# Optimized classification threshold (based on threshold optimization analysis)
OPTIMAL_THRESHOLD = 0.87  # Maximizes F1 score (avg F1=0.55, P=0.58, R=0.53)

BASE_DIR = Path(r'c:\Users\12932\Desktop\nus\BAP')
FEATURES_DIR = BASE_DIR / 'features' / 'fps'
OUTPUT_DIR = BASE_DIR / 'analysis_results' / 'train' / 'fps_trend_alert'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters
TFIDF_MAX_FEATURES = 450  # Reduced from 5000 to 450 for feature reduction (450 TF-IDF + ~19 trend = ~469 total features)
N_SPLITS = 5
TOP_K_PERCENTAGES = [0.01, 0.05, 0.10]  # Top 1%, 5%, 10%

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_data():
    """
    Load feature data with trend_alert_weak labels
    """
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load weaklabeled features
    feature_file = FEATURES_DIR / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet'
    
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    
    df = pd.read_parquet(feature_file)
    print(f"\n✅ Loaded {len(df):,} samples from {feature_file.name}")
    
    # Check required columns
    required_cols = ['review_content_processed', 'trend_alert_weak', 'timestamp']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Sort by timestamp for time series split
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Distribution
    print(f"\n📊 Trend Alert Distribution:")
    print(df['trend_alert_weak'].value_counts())
    print(f"\nPercentage:")
    print(df['trend_alert_weak'].value_counts(normalize=True) * 100)
    
    positive_rate = df['trend_alert_weak'].mean()
    print(f"\n📈 Positive Rate: {positive_rate:.2%}")
    
    if positive_rate > 0.95 or positive_rate < 0.02:
        print(f"\n⚠️  WARNING: Extremely imbalanced labels ({positive_rate:.2%} positive)")
        print("   Consider regenerating weak labels or adjusting thresholds")
    
    return df


def extract_trend_features(df):
    """
    Extract trend-related features for training
    """
    print("\n" + "="*80)
    print("EXTRACTING TREND FEATURES")
    print("="*80)
    
    # Available trend features (from trend_anomaly_analysis.py output)
    trend_cols = [
        # ❌ REMOVED - 用于生成 trend_alert_weak 标签的触发器（会导致数据泄漏）:
        # 'sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h',
        # 'comment_rate_change_24h', 'comment_rate_change_48h', 'comment_rate_change_72h',
        # 'ensemble_anomaly_score', 'if_anomaly_score', 'statistical_anomaly_score',
        
        # ✅ 滚动统计特征（描述性统计，不是触发器）
        'sentiment_rolling_mean_24h', 'sentiment_rolling_mean_48h', 'sentiment_rolling_mean_72h',
        'sentiment_rolling_std_24h', 'sentiment_rolling_std_48h', 'sentiment_rolling_std_72h',
        'comment_rolling_mean_24h', 'comment_rolling_mean_48h', 'comment_rolling_mean_72h',
        
        # ✅ 基础统计特征
        'vader_compound', 'votes_up', 'votes_funny', 'comment_count',
        'author_num_reviews', 'author_playtime_forever',
        
        # ✅ 风险特征（独立规则生成，不参与trend_alert标签）
        'is_toxic', 'contains_bug_report', 'contains_balance_complaint',
        'contains_monetization_complaint', 'mentions_performance',
        
        # ✅ Prophet特征（基于历史基线预测）
        'prophet_baseline', 'prophet_deviation', 'prophet_lower_breach', 'prophet_upper_breach'
    ]
    
    # Filter available columns
    available_trend_cols = [col for col in trend_cols if col in df.columns]
    
    print(f"\n📊 Available Trend Features: {len(available_trend_cols)}/{len(trend_cols)}")
    
    missing_features = set(trend_cols) - set(available_trend_cols)
    if missing_features:
        print(f"   Missing: {missing_features}")
    
    # Check for Prophet features
    prophet_cols = [col for col in available_trend_cols if col.startswith('prophet_')]
    if prophet_cols:
        print(f"   ✅ Prophet features available: {prophet_cols}")
    else:
        print(f"   ⚠️  Prophet features not found")
    
    if len(available_trend_cols) < 5:
        print("\n⚠️  WARNING: Very few trend features available")
        print("   Consider running trend_anomaly_analysis.py first")
    
    # Fill missing values
    X_trend = df[available_trend_cols].fillna(0).values
    
    print(f"\n✅ Extracted {X_trend.shape[1]} trend features")
    print(f"   Shape: {X_trend.shape}")
    
    return X_trend, available_trend_cols


def train_baseline_model(df, X_text, y, train_idx, val_idx):
    """
    Baseline: TF-IDF + Logistic Regression (text only)
    """
    print("\n" + "="*80)
    print("TRAINING BASELINE MODEL (TF-IDF + LogisticRegression)")
    print("="*80)
    
    X_train_text, y_train = X_text[train_idx], y[train_idx]
    X_val_text, y_val = X_text[val_idx], y[val_idx]
    
    # Train with class weighting (due to imbalance)
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    model.fit(X_train_text, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_val_text)[:, 1]
    y_pred = model.predict(X_val_text)
    
    return model, y_pred, y_pred_proba


def find_best_threshold(y_true, y_proba, thresholds=None):
    """Search thresholds to maximize F1 on validation set.

    Returns: best_threshold, best_pred_binary, best_f1
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_f1 = -1.0
    best_t = 0.5
    best_pred = (y_proba >= best_t).astype(int)

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        try:
            f = f1_score(y_true, preds)
        except Exception:
            f = 0.0
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
            best_pred = preds

    return best_t, best_pred, best_f1


def train_advanced_model(X_combined, y, train_idx, val_idx):
    """
    Advanced: Feature Engineering + LightGBM (text + trend features)
    
    Args:
        threshold: Classification threshold (default 0.87, optimized for max F1)
    """
    print("\n" + "="*80)
    print("TRAINING ADVANCED MODEL (Features + LightGBM)")
    print("="*80)
    print(f"   Training LightGBM (will search threshold per-fold later)")
    
    X_train, y_train = X_combined[train_idx], y[train_idx]
    X_val, y_val = X_combined[val_idx], y[val_idx]
    
    # LightGBM with optimized parameters for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = lgb.LGBMClassifier(
        n_estimators=500,        # 大幅增加树数量
        max_depth=12,            # 增加树深度
        learning_rate=0.2,       # 提高学习率
        num_leaves=127,          # 增加叶子节点数
        min_child_samples=10,    # 进一步减少叶子节点最小样本数
        min_child_weight=1e-5,   # 大幅减少叶子节点最小权重
        reg_alpha=0.001,         # 大幅降低L1正则化
        reg_lambda=0.001,        # 大幅降低L2正则化
        scale_pos_weight=scale_pos_weight,  # 使用实际权重
        subsample=0.95,          # 增加采样率
        colsample_bytree=0.95,   # 增加特征采样率
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Predictions (probabilities) returned; thresholding done outside per-fold
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    return model, y_pred_proba


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """
    Comprehensive evaluation for trend alert
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*80}")
    
    # Classification metrics
    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['No Alert', 'Alert'],
                                digits=4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n🔢 Confusion Matrix:")
    print(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    
    # PR-AUC (primary metric for imbalanced data)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    print(f"\n⭐ PR-AUC: {pr_auc:.4f}")
    
    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        print(f"   ROC-AUC: {roc_auc:.4f}")
    except:
        roc_auc = None
        print(f"   ROC-AUC: N/A (single class in validation)")
    
    # F1 Score
    f1 = f1_score(y_true, y_pred)
    print(f"   F1 Score: {f1:.4f}")
    
    # Recall@K analysis
    print(f"\n📈 Recall@K Analysis:")
    for k_pct in TOP_K_PERCENTAGES:
        k = int(len(y_true) * k_pct)
        top_k_idx = np.argsort(y_pred_proba)[-k:]
        recall_at_k = y_true[top_k_idx].sum() / y_true.sum()
        precision_at_k = y_true[top_k_idx].sum() / k
        print(f"   Top {k_pct*100:.0f}% (k={k:,}): Recall={recall_at_k:.4f}, Precision={precision_at_k:.4f}")
    
    # Metrics dict
    metrics = {
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_true, y_pred, 
            target_names=['No Alert', 'Alert'],
            output_dict=True
        )
    }
    
    return metrics


def plot_pr_curve(y_true, y_pred_proba_baseline, y_pred_proba_advanced, fold_idx):
    """
    Plot Precision-Recall curve comparison
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Baseline
    precision_b, recall_b, _ = precision_recall_curve(y_true, y_pred_proba_baseline)
    pr_auc_b = average_precision_score(y_true, y_pred_proba_baseline)
    ax.plot(recall_b, precision_b, linewidth=2, 
            label=f'Baseline (PR-AUC={pr_auc_b:.4f})', color='blue')
    
    # Advanced
    precision_a, recall_a, _ = precision_recall_curve(y_true, y_pred_proba_advanced)
    pr_auc_a = average_precision_score(y_true, y_pred_proba_advanced)
    ax.plot(recall_a, precision_a, linewidth=2, 
            label=f'Advanced (PR-AUC={pr_auc_a:.4f})', color='red')
    
    # Baseline (no skill)
    no_skill = y_true.sum() / len(y_true)
    ax.axhline(no_skill, linestyle='--', color='gray', label=f'No Skill ({no_skill:.4f})')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title(f'Precision-Recall Curve - Fold {fold_idx+1}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    output_path = OUTPUT_DIR / f'pr_curve_fold{fold_idx+1}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved PR curve to: {output_path}")


def main():
    print("="*80)
    print("TREND ALERT TRAINING - FPS GENRE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Load data
    df = load_data()
    
    # 2. Generate Prophet features (optional enhancement)
    if USE_PROPHET_FEATURES and PROPHET_AVAILABLE:
        print("\n" + "="*80)
        print("GENERATING PROPHET BASELINE FEATURES")
        print("="*80)
        
        try:
            # Check if required columns exist
            if 'vader_compound' in df.columns and 'timestamp' in df.columns:
                print(f"\n🔮 Generating Prophet baseline features...")
                prophet_features = generate_prophet_features(
                    df,
                    sentiment_col='vader_compound',
                    timestamp_col='timestamp'
                )
                
                # Add to dataframe
                for col in prophet_features.columns:
                    df[col] = prophet_features[col]
                
                print(f"\n✅ Prophet Features Added:")
                for col in prophet_features.columns:
                    print(f"   - {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")
                    
                # Check breach rates
                if 'prophet_lower_breach' in df.columns:
                    lower_rate = df['prophet_lower_breach'].mean()
                    upper_rate = df['prophet_upper_breach'].mean()
                    print(f"\n📊 Breach Rates:")
                    print(f"   - Lower: {lower_rate:.2%} (expected ~2.5%)")
                    print(f"   - Upper: {upper_rate:.2%} (expected ~2.5%)")
            else:
                print(f"\n⚠️  Missing required columns for Prophet")
                print(f"   Required: vader_compound, timestamp")
                
        except Exception as e:
            print(f"\n⚠️  Prophet feature generation failed: {e}")
            print(f"   Continuing without Prophet features")
    
    # 3. Check if we have trend features or need to regenerate
    has_trend_features = 'sentiment_diff_24h' in df.columns
    
    if not has_trend_features:
        print("\n⚠️  WARNING: No trend features found in data!")
        print("   Please run trend_anomaly_analysis.py first to generate trend features.")
        print("   For now, will train with basic features only.")
    
    # 4. Prepare features
    print("\n" + "="*80)
    print("PREPARING FEATURES")
    print("="*80)
    
    # Text features (TF-IDF)
    print(f"\n📝 Extracting TF-IDF features (max_features={TFIDF_MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    X_text = vectorizer.fit_transform(df['review_content_processed'].fillna(''))
    print(f"   Shape: {X_text.shape}")
    
    # Trend features (if available)
    if has_trend_features:
        X_trend, trend_feature_names = extract_trend_features(df)
        # Combine text + trend features for advanced model
        X_combined = np.hstack([X_text.toarray(), X_trend])
        print(f"\n✅ Combined features shape: {X_combined.shape}")
    else:
        X_combined = X_text.toarray()
        trend_feature_names = []
    
    # Target
    y = df['trend_alert_weak'].values
    
    # 4. Time Series Cross-Validation
    print("\n" + "="*80)
    print(f"TIME SERIES CROSS-VALIDATION ({N_SPLITS} folds)")
    print("="*80)
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    baseline_metrics = []
    advanced_metrics = []

    # Store best models (based on PR-AUC). Initialize PR-AUC to -1 so any valid fold >=0 will replace it.
    best_baseline_model = None
    best_baseline_vectorizer = vectorizer
    best_baseline_pr_auc = -1.0

    best_advanced_model = None
    best_advanced_pr_auc = -1.0
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_text)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{N_SPLITS}")
        print(f"{'='*80}")
        print(f"Train: {len(train_idx):,} samples | Val: {len(val_idx):,} samples")
        print(f"Train positive: {y[train_idx].sum():,} ({y[train_idx].mean():.2%})")
        print(f"Val positive: {y[val_idx].sum():,} ({y[val_idx].mean():.2%})")
        
        # Train Baseline
        model_baseline, _, y_pred_proba_b = train_baseline_model(
            df, X_text, y, train_idx, val_idx
        )

        # Per-fold threshold search for baseline to maximize F1 on validation
        best_t_b, y_pred_b_best, best_f1_b = find_best_threshold(y[val_idx], y_pred_proba_b)

        metrics_b = evaluate_model(
            y[val_idx], y_pred_b_best, y_pred_proba_b,
            f"Baseline - Fold {fold_idx+1}"
        )
        # attach fold-specific info
        metrics_b['fold'] = fold_idx + 1
        metrics_b['best_threshold'] = float(best_t_b)
        metrics_b['best_f1'] = float(best_f1_b)
        baseline_metrics.append(metrics_b)

        # Save per-fold baseline model and vectorizer
        import joblib as _joblib
        baseline_fold_model_path = OUTPUT_DIR / f'baseline_model_fold{fold_idx+1}.joblib'
        baseline_fold_vectorizer_path = OUTPUT_DIR / f'baseline_vectorizer_fold{fold_idx+1}.joblib'
        _joblib.dump(model_baseline, baseline_fold_model_path)
        _joblib.dump(best_baseline_vectorizer, baseline_fold_vectorizer_path)
        # save fold metrics
        with open(OUTPUT_DIR / f'baseline_metrics_fold{fold_idx+1}.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_b, f, indent=2, ensure_ascii=False)

        # Save best baseline model (by PR-AUC)
        if metrics_b['pr_auc'] > best_baseline_pr_auc:
            best_baseline_pr_auc = metrics_b['pr_auc']
            best_baseline_model = model_baseline
            print(f"  💾 New best baseline model (PR-AUC: {best_baseline_pr_auc:.4f})")
        
        # Train Advanced
        model_advanced, y_pred_proba_a = train_advanced_model(
            X_combined, y, train_idx, val_idx
        )

        # Per-fold threshold search for advanced model
        best_t_a, y_pred_a_best, best_f1_a = find_best_threshold(y[val_idx], y_pred_proba_a)

        metrics_a = evaluate_model(
            y[val_idx], y_pred_a_best, y_pred_proba_a,
            f"Advanced - Fold {fold_idx+1}"
        )
        metrics_a['fold'] = fold_idx + 1
        metrics_a['best_threshold'] = float(best_t_a)
        metrics_a['best_f1'] = float(best_f1_a)
        advanced_metrics.append(metrics_a)

        # Save per-fold advanced model and metrics
        advanced_fold_model_path = OUTPUT_DIR / f'advanced_model_fold{fold_idx+1}.joblib'
        _joblib.dump(model_advanced, advanced_fold_model_path)
        with open(OUTPUT_DIR / f'advanced_metrics_fold{fold_idx+1}.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_a, f, indent=2, ensure_ascii=False)

        # Save best advanced model (by PR-AUC)
        if metrics_a['pr_auc'] > best_advanced_pr_auc:
            best_advanced_pr_auc = metrics_a['pr_auc']
            best_advanced_model = model_advanced
            print(f"  💾 New best advanced model (PR-AUC: {best_advanced_pr_auc:.4f})")
        
        # Plot PR curve
        plot_pr_curve(y[val_idx], y_pred_proba_b, y_pred_proba_a, fold_idx)
    
    # 5. Aggregate results
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)
    
    print(f"\n📊 BASELINE (TF-IDF + LogisticRegression)")
    print(f"   Mean PR-AUC: {np.mean([m['pr_auc'] for m in baseline_metrics]):.4f} ± "
          f"{np.std([m['pr_auc'] for m in baseline_metrics]):.4f}")
    print(f"   Mean F1:     {np.mean([m['f1_score'] for m in baseline_metrics]):.4f} ± "
          f"{np.std([m['f1_score'] for m in baseline_metrics]):.4f}")
    
    print(f"\n📊 ADVANCED (Features + LightGBM)")
    print(f"   Mean PR-AUC: {np.mean([m['pr_auc'] for m in advanced_metrics]):.4f} ± "
          f"{np.std([m['pr_auc'] for m in advanced_metrics]):.4f}")
    print(f"   Mean F1:     {np.mean([m['f1_score'] for m in advanced_metrics]):.4f} ± "
          f"{np.std([m['f1_score'] for m in advanced_metrics]):.4f}")
    
    improvement = (np.mean([m['pr_auc'] for m in advanced_metrics]) - 
                   np.mean([m['pr_auc'] for m in baseline_metrics])) * 100
    print(f"\n💡 Improvement: {improvement:+.2f}% PR-AUC")
    
    # 6. Save reports
    print("\n" + "="*80)
    print("SAVING REPORTS")
    print("="*80)
    
    report = {
        'metadata': {
            'task': 'trend_alert_classification',
            'genre': 'fps',
            'n_samples': len(df),
            'positive_rate': float(y.mean()),
            'n_folds': N_SPLITS,
            'optimized_threshold': OPTIMAL_THRESHOLD,  # Add threshold info
            'timestamp': datetime.now().isoformat()
        },
        'baseline': {
            'model': 'TF-IDF + LogisticRegression',
            'metrics': baseline_metrics,
            'mean_pr_auc': float(np.mean([m['pr_auc'] for m in baseline_metrics])),
            'std_pr_auc': float(np.std([m['pr_auc'] for m in baseline_metrics])),
            'mean_f1': float(np.mean([m['f1_score'] for m in baseline_metrics])),
        },
        'advanced': {
            'model': 'Features + LightGBM',
            'n_features': X_combined.shape[1],
            'trend_features': trend_feature_names,
            'metrics': advanced_metrics,
            'mean_pr_auc': float(np.mean([m['pr_auc'] for m in advanced_metrics])),
            'std_pr_auc': float(np.std([m['pr_auc'] for m in advanced_metrics])),
            'mean_f1': float(np.mean([m['f1_score'] for m in advanced_metrics])),
        },
        'improvement': {
            'pr_auc_diff': float(improvement / 100),
            'pr_auc_pct': float(improvement)
        }
    }
    
    report_path = OUTPUT_DIR / 'trend_alert_fps_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Report saved to: {report_path}")
    
    # Save best models
    print("\n" + "="*80)
    print("SAVING BEST MODELS")
    print("="*80)
    
    import joblib
    
    # Fallback: if no best model was selected (e.g., PR-AUC never exceeded initial 0),
    # use the last trained models from the final fold so we still persist a working model.
    # This prevents the script from saving nothing in cases where PR-AUC==0 for all folds
    # (highly imbalanced/single-class validation splits, etc.).
    if best_baseline_model is None and 'model_baseline' in locals():
        best_baseline_model = model_baseline
        # Use the last recorded PR-AUC as the best estimate (or 0.0 if metrics missing)
        try:
            best_baseline_pr_auc = baseline_metrics[-1]['pr_auc']
        except Exception:
            best_baseline_pr_auc = 0.0
        print(f"\n⚠️  No best baseline model selected by PR-AUC - using last trained baseline model as fallback (PR-AUC: {best_baseline_pr_auc:.4f})")

    if best_advanced_model is None and 'model_advanced' in locals():
        best_advanced_model = model_advanced
        try:
            best_advanced_pr_auc = advanced_metrics[-1]['pr_auc']
        except Exception:
            best_advanced_pr_auc = 0.0
        print(f"\n⚠️  No best advanced model selected by PR-AUC - using last trained advanced model as fallback (PR-AUC: {best_advanced_pr_auc:.4f})")
    
    # Save baseline model
    if best_baseline_model is not None:
        baseline_model_path = OUTPUT_DIR / 'best_baseline_model.joblib'
        baseline_vectorizer_path = OUTPUT_DIR / 'best_baseline_vectorizer.joblib'
        
        joblib.dump(best_baseline_model, baseline_model_path)
        joblib.dump(best_baseline_vectorizer, baseline_vectorizer_path)
        
        print(f"\n✅ Baseline model saved:")
        print(f"   Model: {baseline_model_path.name}")
        print(f"   Vectorizer: {baseline_vectorizer_path.name}")
        print(f"   Best PR-AUC: {best_baseline_pr_auc:.4f}")
    
    # Save advanced model
    if best_advanced_model is not None:
        advanced_model_path = OUTPUT_DIR / 'best_advanced_model.joblib'
        
        joblib.dump(best_advanced_model, advanced_model_path)
        
        print(f"\n✅ Advanced model saved:")
        print(f"   Model: {advanced_model_path.name}")
        print(f"   Best PR-AUC: {best_advanced_pr_auc:.4f}")
        print(f"   Features: {X_combined.shape[1]} ({len(trend_feature_names)} trend features)")
    
    print(f"\n{'='*80}")
    print(f"All models and reports saved to: {OUTPUT_DIR}")
    print(f"{'='*80}")
    
    print(f"\n{'='*80}")
    print(f"⏱️  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
