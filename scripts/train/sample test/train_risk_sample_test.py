"""
FPS Risk Sample Test - 5k samples
测试 Baseline (TF-IDF + LR) 和 Advanced (Features + LightGBM)

目标：
1. 验证 Risk 训练流程的可行性
2. 快速评估 Baseline vs Advanced 性能差异
3. 确认特征工程的有效性

Primary Metric: PR-AUC
Secondary Metrics: Recall@K, Precision, F1, FPR
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    precision_recall_curve, auc, average_precision_score,
    classification_report, confusion_matrix, 
    roc_auc_score, precision_score, recall_score, f1_score
)

# LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not installed. Advanced model will be skipped.")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# ============================
# Configuration
# ============================

# Paths
BASE_DIR = Path(r'c:\Users\12932\Desktop\nus\BAP')
FEATURES_DIR = BASE_DIR / 'features' / 'fps'
OUTPUT_DIR = BASE_DIR / 'analysis_results' / 'train' / 'fps_risk_sample_test'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data config
SAMPLE_SIZE = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature config
TEXT_FEATURE = 'review_content_processed'
TARGET = 'risk_label_weak'

# Numeric features (13 risk features + behavior features)
NUMERIC_FEATURES = [
    # Risk score components (avoid risk_score itself - data leakage)
    'vader_compound', 'vader_neg', 'vader_pos', 'vader_neu',
    'sentiment_score', 'controversial_sentiment',
    'recommendation_sentiment_mismatch',
    
    # Content flags
    'is_toxic', 'contains_bug_report', 'contains_balance_complaint',
    'contains_monetization_complaint', 'mentions_performance',
    
    # Anomaly features
    'ensemble_anomaly_score', 'ml_anomaly_score', 'statistical_anomaly_score',
    
    # Author behavior
    'author_playtime_at_review', 'author_num_games_owned',
    'author_num_reviews', 'votes_helpful', 'votes_funny',
    
    # Interaction
    'votes_up', 'votes_down', 'comment_count',
    
    # Additional helpful/unhelpful features
    'negative_but_helpful', 'positive_but_unhelpful'
]

# Columns to exclude (avoid data leakage)
EXCLUDE_COLS = [
    'risk_score',  # Direct leakage
    'risk_label_weak',  # Target itself
    'risk_label',  # If exists
    'timestamp', 'review_datetime',  # Time columns (for sorting only)
    'review_id', 'author_id', 'app_id',  # ID columns
    'review_content', 'review_content_clean'  # Raw text (use processed)
]

# ============================
# Helper Functions
# ============================

def recall_at_k(y_true, y_proba, k=0.1):
    """
    Calculate Recall@K (recall in top k% predictions)
    """
    n = len(y_true)
    top_k = int(n * k)
    
    # Get top-k indices
    top_indices = np.argsort(y_proba)[-top_k:]
    
    # Count TP in top-k
    tp = y_true.iloc[top_indices].sum() if isinstance(y_true, pd.Series) else y_true[top_indices].sum()
    total_positives = y_true.sum()
    
    recall = tp / total_positives if total_positives > 0 else 0
    
    return recall


def evaluate_binary_classifier(y_true, y_proba, model_name="Model", threshold=0.5):
    """
    Comprehensive evaluation for binary classification
    """
    results = {}
    
    # Primary: PR-AUC
    pr_auc = average_precision_score(y_true, y_proba)
    results['pr_auc'] = pr_auc
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_proba)
    results['roc_auc'] = roc_auc
    
    # Recall@K
    for k in [0.05, 0.10, 0.20]:
        recall_k = recall_at_k(y_true, y_proba, k=k)
        results[f'recall@{int(k*100)}%'] = recall_k
    
    # Binary predictions
    y_pred = (y_proba >= threshold).astype(int)
    
    # Standard metrics
    results['precision'] = precision_score(y_true, y_pred)
    results['recall'] = recall_score(y_true, y_pred)
    results['f1'] = f1_score(y_true, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    results['true_negatives'] = int(tn)
    results['false_positives'] = int(fp)
    results['false_negatives'] = int(fn)
    results['true_positives'] = int(tp)
    results['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return results


def find_best_threshold(y_true, y_proba, metric='f1'):
    """
    Find best threshold to maximize a specific metric
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    if metric == 'f1':
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
    elif metric == 'precision':
        best_idx = np.argmax(precision[:-1])  # Exclude last element
    elif metric == 'recall':
        best_idx = np.argmax(recall[:-1])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    return best_threshold


def plot_pr_curve(y_true, y_proba_baseline, y_proba_advanced, save_path):
    """
    Plot Precision-Recall curve comparison
    """
    # Calculate PR curves
    precision_bl, recall_bl, _ = precision_recall_curve(y_true, y_proba_baseline)
    precision_adv, recall_adv, _ = precision_recall_curve(y_true, y_proba_advanced)
    
    # Calculate PR-AUC
    pr_auc_bl = auc(recall_bl, precision_bl)
    pr_auc_adv = auc(recall_adv, precision_adv)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(recall_bl, precision_bl, label=f'Baseline (PR-AUC={pr_auc_bl:.4f})', linewidth=2)
    plt.plot(recall_adv, precision_adv, label=f'Advanced (PR-AUC={pr_auc_adv:.4f})', linewidth=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve: Risk Classification (Sample Test)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ PR Curve saved to: {save_path}")


def plot_score_distribution(y_true, y_proba_baseline, y_proba_advanced, save_path):
    """
    Plot score distribution for both models
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline
    axes[0].hist(y_proba_baseline[y_true==0], bins=30, alpha=0.6, label='Risk=0', color='blue')
    axes[0].hist(y_proba_baseline[y_true==1], bins=30, alpha=0.6, label='Risk=1', color='red')
    axes[0].set_title('Baseline: Score Distribution', fontsize=12)
    axes[0].set_xlabel('Predicted Probability', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Advanced
    axes[1].hist(y_proba_advanced[y_true==0], bins=30, alpha=0.6, label='Risk=0', color='blue')
    axes[1].hist(y_proba_advanced[y_true==1], bins=30, alpha=0.6, label='Risk=1', color='red')
    axes[1].set_title('Advanced: Score Distribution', fontsize=12)
    axes[1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Score Distribution saved to: {save_path}")


# ============================
# Main Training Pipeline
# ============================

def main():
    print("="*80)
    print("FPS RISK SAMPLE TEST - 5K SAMPLES")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Load data
    print("📂 Loading data...")
    feature_file = FEATURES_DIR / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet'
    
    if not feature_file.exists():
        # Try coordinated groups version
        feature_file = FEATURES_DIR / 'gpu_optimized_features_fps_exclflagged_enhanced_features_coordinated_groups.parquet'
    
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    
    df = pd.read_parquet(feature_file)
    print(f"✅ Loaded {len(df):,} samples from {feature_file.name}")
    
    # Check target column
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in data!")
    
    # Check class distribution
    class_dist = df[TARGET].value_counts()
    print(f"\n📊 Full Dataset Class Distribution:")
    print(f"  Risk=0 (Low):  {class_dist.get(0, 0):,} ({class_dist.get(0, 0)/len(df)*100:.2f}%)")
    print(f"  Risk=1 (High): {class_dist.get(1, 0):,} ({class_dist.get(1, 0)/len(df)*100:.2f}%)")
    
    # 2. Sample data (stratified)
    print(f"\n🎲 Sampling {SAMPLE_SIZE:,} samples (stratified)...")
    df_sample = df.groupby(TARGET, group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_SIZE // 2), random_state=RANDOM_STATE)
    ).reset_index(drop=True)
    
    print(f"✅ Sampled {len(df_sample):,} samples")
    
    sample_dist = df_sample[TARGET].value_counts()
    print(f"\n📊 Sample Class Distribution:")
    print(f"  Risk=0 (Low):  {sample_dist.get(0, 0):,} ({sample_dist.get(0, 0)/len(df_sample)*100:.2f}%)")
    print(f"  Risk=1 (High): {sample_dist.get(1, 0):,} ({sample_dist.get(1, 0)/len(df_sample)*100:.2f}%)")
    
    # 3. Train-test split (stratified)
    print(f"\n✂️ Splitting data (test_size={TEST_SIZE})...")
    train_df, test_df = train_test_split(
        df_sample, 
        test_size=TEST_SIZE, 
        stratify=df_sample[TARGET], 
        random_state=RANDOM_STATE
    )
    
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]
    
    # ============================
    # BASELINE: TF-IDF + Logistic Regression
    # ============================
    
    print("\n" + "="*80)
    print("BASELINE: TF-IDF + Logistic Regression")
    print("="*80)
    
    print("\n📝 Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(train_df[TEXT_FEATURE].fillna(''))
    X_test_vec = vectorizer.transform(test_df[TEXT_FEATURE].fillna(''))
    
    print(f"✅ TF-IDF shape: {X_train_vec.shape}")
    
    print("\n🤖 Training Logistic Regression...")
    clf_baseline = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='liblinear',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    clf_baseline.fit(X_train_vec, y_train)
    print("✅ Training complete!")
    
    # Predictions
    print("\n🔮 Making predictions...")
    y_proba_baseline = clf_baseline.predict_proba(X_test_vec)[:, 1]
    
    # Find best threshold
    best_threshold_baseline = find_best_threshold(y_test, y_proba_baseline, metric='f1')
    print(f"✅ Best threshold (F1): {best_threshold_baseline:.4f}")
    
    # Evaluate
    print("\n📊 Evaluating Baseline...")
    results_baseline = evaluate_binary_classifier(
        y_test, y_proba_baseline, 
        model_name="Baseline", 
        threshold=best_threshold_baseline
    )
    
    print("\n" + "-"*80)
    print("BASELINE RESULTS")
    print("-"*80)
    print(f"Primary Metric:")
    print(f"  PR-AUC:        {results_baseline['pr_auc']:.4f}")
    print(f"\nSecondary Metrics:")
    print(f"  ROC-AUC:       {results_baseline['roc_auc']:.4f}")
    print(f"  Precision:     {results_baseline['precision']:.4f}")
    print(f"  Recall:        {results_baseline['recall']:.4f}")
    print(f"  F1:            {results_baseline['f1']:.4f}")
    print(f"  FPR:           {results_baseline['false_positive_rate']:.4f}")
    print(f"\nRecall@K:")
    print(f"  Recall@5%:     {results_baseline['recall@5%']:.4f}")
    print(f"  Recall@10%:    {results_baseline['recall@10%']:.4f}")
    print(f"  Recall@20%:    {results_baseline['recall@20%']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {results_baseline['true_negatives']:4d}  FP: {results_baseline['false_positives']:4d}")
    print(f"  FN: {results_baseline['false_negatives']:4d}  TP: {results_baseline['true_positives']:4d}")
    
    # ============================
    # ADVANCED: Features + LightGBM
    # ============================
    
    if not LIGHTGBM_AVAILABLE:
        print("\n⚠️ Skipping Advanced model (LightGBM not available)")
        results_advanced = None
        y_proba_advanced = None
    else:
        print("\n" + "="*80)
        print("ADVANCED: Text + Tabular Features + LightGBM")
        print("="*80)
        
        # Check feature availability
        available_features = [f for f in NUMERIC_FEATURES if f in train_df.columns]
        missing_features = [f for f in NUMERIC_FEATURES if f not in train_df.columns]
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        
        print(f"\n✅ Using {len(available_features)} numeric features")
        
        # Feature engineering
        print("\n🔧 Engineering features...")
        
        # Text features (reduced to 5k for faster training)
        vectorizer_adv = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )
        
        X_train_text = vectorizer_adv.fit_transform(train_df[TEXT_FEATURE].fillna(''))
        X_test_text = vectorizer_adv.transform(test_df[TEXT_FEATURE].fillna(''))
        
        # Numeric features (standardized)
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(train_df[available_features].fillna(0))
        X_test_num = scaler.transform(test_df[available_features].fillna(0))
        
        # Combine (scipy sparse + numpy array)
        from scipy.sparse import hstack, csr_matrix
        X_train_combined = hstack([X_train_text, csr_matrix(X_train_num)])
        X_test_combined = hstack([X_test_text, csr_matrix(X_test_num)])
        
        print(f"✅ Combined features shape: {X_train_combined.shape}")
        
        # LightGBM
        print("\n🤖 Training LightGBM...")
        
        # Calculate scale_pos_weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'scale_pos_weight': scale_pos_weight
        }
        
        train_data = lgb.Dataset(X_train_combined, label=y_train)
        test_data = lgb.Dataset(X_test_combined, label=y_test, reference=train_data)
        
        model_advanced = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=50)]
        )
        
        print("✅ Training complete!")
        
        # Predictions
        print("\n🔮 Making predictions...")
        y_proba_advanced = model_advanced.predict(X_test_combined)
        
        # Find best threshold
        best_threshold_advanced = find_best_threshold(y_test, y_proba_advanced, metric='f1')
        print(f"✅ Best threshold (F1): {best_threshold_advanced:.4f}")
        
        # Evaluate
        print("\n📊 Evaluating Advanced...")
        results_advanced = evaluate_binary_classifier(
            y_test, y_proba_advanced, 
            model_name="Advanced", 
            threshold=best_threshold_advanced
        )
        
        print("\n" + "-"*80)
        print("ADVANCED RESULTS")
        print("-"*80)
        print(f"Primary Metric:")
        print(f"  PR-AUC:        {results_advanced['pr_auc']:.4f}")
        print(f"\nSecondary Metrics:")
        print(f"  ROC-AUC:       {results_advanced['roc_auc']:.4f}")
        print(f"  Precision:     {results_advanced['precision']:.4f}")
        print(f"  Recall:        {results_advanced['recall']:.4f}")
        print(f"  F1:            {results_advanced['f1']:.4f}")
        print(f"  FPR:           {results_advanced['false_positive_rate']:.4f}")
        print(f"\nRecall@K:")
        print(f"  Recall@5%:     {results_advanced['recall@5%']:.4f}")
        print(f"  Recall@10%:    {results_advanced['recall@10%']:.4f}")
        print(f"  Recall@20%:    {results_advanced['recall@20%']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {results_advanced['true_negatives']:4d}  FP: {results_advanced['false_positives']:4d}")
        print(f"  FN: {results_advanced['false_negatives']:4d}  TP: {results_advanced['true_positives']:4d}")
        
        # Feature importance (top 20)
        print("\n📊 Feature Importance (Top 20)...")
        
        # Get actual number of features from model
        importance = model_advanced.feature_importance(importance_type='gain')
        n_text_features = X_train_text.shape[1]
        n_num_features = len(available_features)
        
        feature_names = (
            [f'tfidf_{i}' for i in range(n_text_features)] +
            available_features
        )
        
        # Ensure lengths match
        assert len(feature_names) == len(importance), f"Feature name mismatch: {len(feature_names)} vs {len(importance)}"
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Features:")
        print(feature_importance_df.head(20).to_string(index=False))
        
        # Save feature importance
        feature_importance_df.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)
        print(f"\n✅ Feature importance saved to: {OUTPUT_DIR / 'feature_importance.csv'}")
    
    # ============================
    # Comparison & Visualization
    # ============================
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison = {
        'baseline': results_baseline,
        'advanced': results_advanced if results_advanced else {}
    }
    
    if results_advanced:
        improvement = {
            'pr_auc': (results_advanced['pr_auc'] - results_baseline['pr_auc']) / results_baseline['pr_auc'] * 100,
            'f1': (results_advanced['f1'] - results_baseline['f1']) / results_baseline['f1'] * 100,
            'recall@10%': (results_advanced['recall@10%'] - results_baseline['recall@10%']) / results_baseline['recall@10%'] * 100
        }
        
        print(f"\nPerformance Improvement (Advanced vs Baseline):")
        print(f"  PR-AUC:     {improvement['pr_auc']:+.2f}%")
        print(f"  F1:         {improvement['f1']:+.2f}%")
        print(f"  Recall@10%: {improvement['recall@10%']:+.2f}%")
        
        comparison['improvement'] = improvement
    
    # Save results
    with open(OUTPUT_DIR / 'sample_test_results.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR / 'sample_test_results.json'}")
    
    # Visualizations
    if results_advanced:
        print("\n📊 Generating visualizations...")
        
        # PR Curve
        plot_pr_curve(
            y_test, y_proba_baseline, y_proba_advanced,
            OUTPUT_DIR / 'pr_curve_comparison.png'
        )
        
        # Score distribution
        plot_score_distribution(
            y_test, y_proba_baseline, y_proba_advanced,
            OUTPUT_DIR / 'score_distribution.png'
        )
    
    # ============================
    # Summary
    # ============================
    
    print("\n" + "="*80)
    print("SAMPLE TEST SUMMARY")
    print("="*80)
    
    print(f"\n✅ Test completed successfully!")
    print(f"\n📊 Key Findings:")
    print(f"  - Baseline PR-AUC:  {results_baseline['pr_auc']:.4f}")
    if results_advanced:
        print(f"  - Advanced PR-AUC:  {results_advanced['pr_auc']:.4f} ({improvement['pr_auc']:+.2f}%)")
        print(f"  - Best Recall@10%:  {results_advanced['recall@10%']:.4f}")
    
    print(f"\n💡 Recommendation:")
    if results_advanced and results_advanced['pr_auc'] > 0.75:
        print(f"  ✅ Performance looks good! Proceed with full training (176k samples).")
    elif results_baseline['pr_auc'] > 0.70:
        print(f"  ⚠️ Baseline is acceptable, but Advanced could be improved.")
        print(f"     Consider feature engineering or trying different models.")
    else:
        print(f"  ⚠️ Performance below expectation. Review features and data quality.")
    
    print(f"\n📁 Output Directory: {OUTPUT_DIR}")
    print(f"\n⏱️ End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


if __name__ == "__main__":
    main()
