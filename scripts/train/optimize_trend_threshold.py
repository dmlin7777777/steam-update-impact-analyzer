"""
Threshold Optimization for Trend Alert Model

基于已训练的模型，优化分类阈值以最大化业务目标：
1. 最大化 F1 分数
2. 最大化 Precision@K (top K预测的准确率)
3. 在给定召回率下最大化精确率

输出：
- 最优阈值
- 各阈值下的性能指标
- PR曲线和阈值分析图
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score, 
    recall_score, confusion_matrix, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import argparse
import joblib

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(r'c:\Users\12932\Desktop\nus\BAP')
FEATURES_DIR = BASE_DIR / 'features' / 'fps'
OUTPUT_DIR = BASE_DIR / 'analysis_results' / 'train' / 'fps_trend_alert'

# Model parameters (same as training)
TFIDF_MAX_FEATURES = 450
N_SPLITS = 5

# Toggle oversampling (can be overridden by CLI)
OVERSAMPLE = True

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)


def load_data():
    """Load feature data"""
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    feature_file = FEATURES_DIR / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet'
    df = pd.read_parquet(feature_file)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n✅ Loaded {len(df):,} samples")
    print(f"   Positive rate: {df['trend_alert_weak'].mean():.2%}")
    
    return df


def extract_features(df):
    """Extract same features as training"""
    print("\n" + "="*80)
    print("EXTRACTING FEATURES")
    print("="*80)
    
    # Text features
    print(f"\n📝 TF-IDF features (max_features={TFIDF_MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    # Do NOT fit TF-IDF globally here; return raw texts and trend features so TF-IDF
    # can be fit on each fold's training partition to avoid leakage.
    X_text = None
    texts = df['review_content_processed'].fillna('')
    
    # Trend features
    trend_cols = [
        'sentiment_rolling_mean_24h', 'sentiment_rolling_mean_48h', 'sentiment_rolling_mean_72h',
        'sentiment_rolling_std_24h', 'sentiment_rolling_std_48h', 'sentiment_rolling_std_72h',
        'vader_compound', 'votes_up', 'votes_funny', 'comment_count',
        'is_toxic', 'contains_bug_report', 'contains_balance_complaint',
        'contains_monetization_complaint', 'mentions_performance',
        'prophet_baseline', 'prophet_deviation', 'prophet_lower_breach', 'prophet_upper_breach'
    ]
    
    available_trend_cols = [col for col in trend_cols if col in df.columns]
    X_trend = df[available_trend_cols].fillna(0).values
    
    # Combine
    y = df['trend_alert_weak'].values

    print(f"✅ Trend features: {X_trend.shape}")

    # Return trend matrix, raw texts, labels and available trend column names
    return X_trend, texts, y, available_trend_cols


def train_models_fold(X_trend, texts, y, train_idx, val_idx, fold_idx, oversample=True):
    """Train advanced (LightGBM) and baseline (LogisticRegression) on one fold
    Returns: lgb_model, lgb_proba, baseline_model, baseline_proba, y_val
    """
    # Prepare train/val splits for trend and text
    X_trend_train, X_trend_val = X_trend[train_idx], X_trend[val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]
    texts_train = texts.iloc[train_idx]
    texts_val = texts.iloc[val_idx]

    # --------------------------
    # Oversample positives in training split to mitigate imbalance
    # Strategy: simple duplication of positive examples to roughly balance classes.
    # This is done only on the training split; validation remains untouched.
    # --------------------------
    if oversample:
        pos_mask = (y_train == 1)
        n_pos = int(pos_mask.sum())
        n_neg = int((y_train == 0).sum())
        if n_pos > 0 and n_pos < n_neg:
            times = int(np.ceil(n_neg / n_pos))
            if times > 1:
                pos_idx_local = np.where(pos_mask)[0]
                rep_idx = np.concatenate([pos_idx_local] * (times - 1))
                # texts_train is a pandas Series; replicate positive texts
                texts_train = pd.concat([
                    texts_train.reset_index(drop=True),
                    texts_train.iloc[rep_idx].reset_index(drop=True)
                ], ignore_index=True)
                # replicate trend features and labels
                X_trend_train = np.vstack([X_trend_train, X_trend_train[rep_idx]])
                y_train = np.concatenate([y_train, y_train[rep_idx]])

    # Fit TF-IDF on (possibly oversampled) training text only (avoid leakage)
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    X_text_train = vectorizer.fit_transform(texts_train)
    X_text_val = vectorizer.transform(texts_val)

    # Save per-fold vectorizer
    import joblib as _joblib
    vec_path = OUTPUT_DIR / f'vectorizer_fold{fold_idx+1}{"_oversample" if oversample else ""}.joblib'
    _joblib.dump(vectorizer, vec_path)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.2,
        num_leaves=127,
        min_child_samples=10,
        min_child_weight=1e-5,
        reg_alpha=0.001,
        reg_lambda=0.001,
        scale_pos_weight=scale_pos_weight,
        subsample=0.95,
        colsample_bytree=0.95,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    
    # Combine text features (dense) with trend features
    X_train_text_array = X_text_train.toarray()
    X_val_text_array = X_text_val.toarray()
    X_train = np.hstack([X_train_text_array, X_trend_train])
    X_val = np.hstack([X_val_text_array, X_trend_val])

    # Split part of training data for calibration (if possible)
    try:
        X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
            X_train, y_train, test_size=0.10, stratify=y_train, random_state=RANDOM_SEED
        )
    except Exception:
        # Fallback: no stratify or too few positives, simple split
        X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
            X_train, y_train, test_size=0.10, random_state=RANDOM_SEED
        )

    model.fit(
        X_train_fit, y_train_fit,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Calibrate model using held-out part from training
    try:
        calibrated = CalibratedClassifierCV(base_estimator=model, method='isotonic', cv='prefit')
        calibrated.fit(X_cal, y_cal)
        lgb_proba = calibrated.predict_proba(X_val)[:, 1]
    except Exception:
        # If calibration fails (e.g., insufficient labels), use raw probabilities
        lgb_proba = model.predict_proba(X_val)[:, 1]
    # Baseline: LogisticRegression on text features only
    baseline = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    # For baseline, train on text features. Use calibration similarly.
    try:
        baseline.fit(X_train_text_array, y_train)
        # calibrate baseline
        try:
            baseline_cal = CalibratedClassifierCV(base_estimator=baseline, method='isotonic', cv='prefit')
            baseline_cal.fit(X_cal[:, :X_train_text_array.shape[1]], y_cal)
            baseline_proba = baseline_cal.predict_proba(X_val[:, :X_train_text_array.shape[1]])[:, 1]
        except Exception:
            baseline_proba = baseline.predict_proba(X_val[:, :X_train_text_array.shape[1]])[:, 1]
    except Exception:
        # Fallback to sparse arrays
        baseline.fit(X_text_train.toarray(), y_train)
        baseline_proba = baseline.predict_proba(X_text_val.toarray())[:, 1]

    return model, lgb_proba, baseline, baseline_proba, y_val


def find_optimal_thresholds(y_true, y_pred_proba):
    """Find optimal thresholds for different objectives"""
    print("\n" + "="*80)
    print("FINDING OPTIMAL THRESHOLDS")
    print("="*80)
    
    thresholds = np.linspace(0.1, 0.9, 81)  # 0.1 to 0.9 with 0.01 step
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    df_results = pd.DataFrame(results)
    
    # Find optimal thresholds for different objectives
    optimal_thresholds = {
        'max_f1': {
            'threshold': df_results.loc[df_results['f1'].idxmax(), 'threshold'],
            'f1': df_results['f1'].max(),
            'precision': df_results.loc[df_results['f1'].idxmax(), 'precision'],
            'recall': df_results.loc[df_results['f1'].idxmax(), 'recall']
        },
        'recall_80': {
            'threshold': df_results[df_results['recall'] >= 0.80]['threshold'].max() 
                        if (df_results['recall'] >= 0.80).any() else None,
            'f1': df_results[df_results['recall'] >= 0.80]['f1'].max() 
                  if (df_results['recall'] >= 0.80).any() else None,
            'precision': df_results[df_results['recall'] >= 0.80]['precision'].max() 
                        if (df_results['recall'] >= 0.80).any() else None,
            'recall': 0.80
        },
        'precision_50': {
            'threshold': df_results[df_results['precision'] >= 0.50]['threshold'].min() 
                        if (df_results['precision'] >= 0.50).any() else None,
            'f1': df_results[df_results['precision'] >= 0.50]['f1'].max() 
                  if (df_results['precision'] >= 0.50).any() else None,
            'precision': 0.50,
            'recall': df_results[df_results['precision'] >= 0.50]['recall'].max() 
                     if (df_results['precision'] >= 0.50).any() else None
        }
    }
    
    return df_results, optimal_thresholds


def precision_recall_at_k(y_true, y_score, ks=[10, 50, 100], percentiles=[0.01, 0.05, 0.1]):
    """Compute precision@K and recall@K for given absolute Ks and percentile-based Ks.
    Returns dict with keys like p_at_10, r_at_10, p_at_1pct, r_at_1pct, etc."""
    res = {}
    n = len(y_true)
    order = np.argsort(-np.asarray(y_score))
    y_sorted = np.asarray(y_true)[order]

    for k in ks:
        if k <= 0:
            continue
        k_actual = min(int(k), n)
        top = y_sorted[:k_actual]
        p = float(top.sum()) / max(1, k_actual)
        r = float(top.sum()) / max(1, int(np.sum(y_true))) if np.sum(y_true) > 0 else None
        res[f'precision_at_{k}'] = p
        res[f'recall_at_{k}'] = r

    for pct in percentiles:
        k = max(1, int(np.ceil(pct * n)))
        top = y_sorted[:k]
        p = float(top.sum()) / max(1, k)
        r = float(top.sum()) / max(1, int(np.sum(y_true))) if np.sum(y_true) > 0 else None
        key = f'{int(pct*100)}pct'
        res[f'precision_at_{key}'] = p
        res[f'recall_at_{key}'] = r

    return res


def plot_threshold_analysis(df_results, optimal_thresholds, fold_idx, model_name='advanced'):
    """Plot threshold analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Precision-Recall-F1 vs Threshold
    ax = axes[0, 0]
    ax.plot(df_results['threshold'], df_results['precision'], 
            label='Precision', linewidth=2, color='blue')
    ax.plot(df_results['threshold'], df_results['recall'], 
            label='Recall', linewidth=2, color='green')
    ax.plot(df_results['threshold'], df_results['f1'], 
            label='F1', linewidth=2, color='red')
    
    # Mark optimal F1
    opt_f1 = optimal_thresholds['max_f1']
    ax.axvline(opt_f1['threshold'], linestyle='--', color='red', alpha=0.5,
               label=f"Optimal F1={opt_f1['f1']:.3f} @ {opt_f1['threshold']:.2f}")
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Precision, Recall, F1 vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. TP/FP/TN/FN counts
    ax = axes[0, 1]
    ax.plot(df_results['threshold'], df_results['tp'], label='TP', linewidth=2, color='green')
    ax.plot(df_results['threshold'], df_results['fp'], label='FP', linewidth=2, color='red')
    ax.plot(df_results['threshold'], df_results['fn'], label='FN', linewidth=2, color='orange')
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix Components vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Precision-Recall tradeoff
    ax = axes[1, 0]
    ax.plot(df_results['recall'], df_results['precision'], linewidth=2, color='purple')
    
    # Mark key points
    for key, opt in optimal_thresholds.items():
        if opt['threshold'] is not None:
            ax.plot(opt['recall'], opt['precision'], 'o', markersize=10,
                   label=f"{key}: T={opt['threshold']:.2f}")
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. F1 vs Threshold (detailed)
    ax = axes[1, 1]
    ax.plot(df_results['threshold'], df_results['f1'], linewidth=3, color='darkred')
    ax.fill_between(df_results['threshold'], 0, df_results['f1'], alpha=0.3, color='red')
    
    # Mark optimal
    opt_f1 = optimal_thresholds['max_f1']
    ax.axvline(opt_f1['threshold'], linestyle='--', color='black', linewidth=2)
    ax.axhline(opt_f1['f1'], linestyle='--', color='black', linewidth=2, alpha=0.5)
    ax.plot(opt_f1['threshold'], opt_f1['f1'], 'o', markersize=15, color='gold', 
            markeredgecolor='black', markeredgewidth=2,
            label=f"Max F1={opt_f1['f1']:.3f} @ T={opt_f1['threshold']:.2f}")
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'threshold_analysis_{model_name}_fold{fold_idx+1}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved threshold analysis to: {output_path}")


def main():
    print("="*80)
    print("THRESHOLD OPTIMIZATION - TREND ALERT")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # parse CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--oversample', type=lambda x: x.lower() in ('1','true','yes'), default=None,
                        help='Enable/disable oversampling per fold (true/false). If omitted, default from script OVERSAMPLE is used.')
    args = parser.parse_args()

    # determine oversample behavior
    global OVERSAMPLE
    if args.oversample is not None:
        OVERSAMPLE = bool(args.oversample)

    # Load data
    df = load_data()
    X_trend, texts, y, trend_feature_names = extract_features(df)
    
    # Time Series CV
    print("\n" + "="*80)
    print(f"TIME SERIES CROSS-VALIDATION ({N_SPLITS} folds)")
    print("="*80)
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    all_fold_results = []
    all_optimal_thresholds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_trend)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{N_SPLITS}")
        print(f"{'='*80}")
        
        # Train both models (advanced + baseline)
        print("Training advanced and baseline models...")
        lgb_model, lgb_proba, baseline_model, baseline_proba, y_val = train_models_fold(
            X_trend, texts, y, train_idx, val_idx, fold_idx, oversample=OVERSAMPLE
        )

        # Find optimal thresholds for advanced
        df_results_adv, optimal_thresholds_adv = find_optimal_thresholds(y_val, lgb_proba)
        print("\n📊 ADVANCED MODEL OPTIMAL THRESHOLDS:")
        for objective, opt in optimal_thresholds_adv.items():
            if opt['threshold'] is not None:
                print(f"\n{objective.upper()}: T={opt['threshold']:.3f}, F1={opt['f1']:.4f}, P={opt['precision']:.4f}, R={opt['recall']:.4f}")
            else:
                print(f"\n{objective.upper()}: Not achievable with current predictions")

        # Find optimal thresholds for baseline
        df_results_base, optimal_thresholds_base = find_optimal_thresholds(y_val, baseline_proba)
        print("\n📊 BASELINE MODEL OPTIMAL THRESHOLDS:")
        for objective, opt in optimal_thresholds_base.items():
            if opt['threshold'] is not None:
                print(f"\n{objective.upper()}: T={opt['threshold']:.3f}, F1={opt['f1']:.4f}, P={opt['precision']:.4f}, R={opt['recall']:.4f}")
            else:
                print(f"\n{objective.upper()}: Not achievable with current predictions")

        # Plot analysis for both
        plot_threshold_analysis(df_results_adv, optimal_thresholds_adv, fold_idx, model_name='advanced')
        plot_threshold_analysis(df_results_base, optimal_thresholds_base, fold_idx, model_name='baseline')

        # Record fold results
        all_fold_results.append({
            'fold': fold_idx+1,
            'advanced': df_results_adv.to_dict(orient='records'),
            'baseline': df_results_base.to_dict(orient='records')
        })
        all_optimal_thresholds.append({
            'fold': fold_idx+1,
            'advanced': optimal_thresholds_adv,
            'baseline': optimal_thresholds_base
        })
    
    # Aggregate results across folds
    print("\n" + "="*80)
    print("SUMMARY ACROSS ALL FOLDS")
    print("="*80)
    
    print("\n📊 AVERAGE OPTIMAL THRESHOLDS:")
    
    # Compute average optimal thresholds per model type
    for model_type in ['advanced', 'baseline']:
        print(f"\n📊 AVERAGE OPTIMAL THRESHOLDS - {model_type.upper()}:")
        for objective in ['max_f1', 'recall_80', 'precision_50']:
            # collect valid folds for this model and objective
            valid = [opt[model_type][objective] for opt in all_optimal_thresholds 
                     if opt[model_type][objective]['threshold'] is not None]
            if valid:
                avg_threshold = np.mean([v['threshold'] for v in valid])
                avg_f1 = np.mean([v['f1'] for v in valid])
                avg_precision = np.mean([v['precision'] for v in valid])
                avg_recall = np.mean([v['recall'] for v in valid])
                print(f"\n{objective.upper()}:")
                print(f"  Avg Threshold: {avg_threshold:.3f}")
                print(f"  Avg F1:        {avg_f1:.4f}")
                print(f"  Avg Precision: {avg_precision:.4f}")
                print(f"  Avg Recall:    {avg_recall:.4f}")
    
    # Save results
    # Build results summary including both advanced and baseline optimal thresholds per fold
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'n_folds': N_SPLITS,
        'fold_results': []
    }

    for opt in all_optimal_thresholds:
        fold_entry = {'fold': opt['fold'], 'advanced': {}, 'baseline': {}}
        for model_type in ['advanced', 'baseline']:
            for k, v in opt[model_type].items():
                fold_entry[model_type][k] = {
                    'threshold': float(v['threshold']) if v['threshold'] is not None else None,
                    'f1': float(v['f1']) if v['f1'] is not None else None,
                    'precision': float(v['precision']) if v['precision'] is not None else None,
                    'recall': float(v['recall']) if v['recall'] is not None else None
                }
        results_summary['fold_results'].append(fold_entry)
    
    suffix = '_oversample' if OVERSAMPLE else '_no_oversample'
    output_path = OUTPUT_DIR / f'threshold_optimization_results{suffix}.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # -------------------------
    # Final training on full data and artifact saving
    # -------------------------
    print('\n' + '='*80)
    print('FINAL MODEL TRAINING & ARTIFACT SAVING')
    print('='*80)

    # Time-based holdout: last 10% as holdout
    n_samples = X_trend.shape[0]
    holdout_start = int(n_samples * 0.9)
    X_trend_train_full = X_trend[:holdout_start]
    X_trend_holdout = X_trend[holdout_start:]
    texts_train_full = texts.iloc[:holdout_start]
    texts_holdout = texts.iloc[holdout_start:]
    y_train_full = y[:holdout_start]
    y_holdout = y[holdout_start:]

    # Optionally oversample training positives
    if OVERSAMPLE:
        pos_mask = (y_train_full == 1)
        n_pos = int(pos_mask.sum())
        n_neg = int((y_train_full == 0).sum())
        if n_pos > 0 and n_pos < n_neg:
            times = int(np.ceil(n_neg / n_pos))
            if times > 1:
                pos_idx_local = np.where(pos_mask)[0]
                rep_idx = np.concatenate([pos_idx_local] * (times - 1))
                texts_train_full = pd.concat([
                    texts_train_full.reset_index(drop=True),
                    texts_train_full.iloc[rep_idx].reset_index(drop=True)
                ], ignore_index=True)
                X_trend_train_full = np.vstack([X_trend_train_full, X_trend_train_full[rep_idx]])
                y_train_full = np.concatenate([y_train_full, y_train_full[rep_idx]])

    # Fit TF-IDF on full training texts
    final_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    X_text_train_full = final_vectorizer.fit_transform(texts_train_full)
    X_text_holdout = final_vectorizer.transform(texts_holdout)

    # Save final vectorizer
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_vec_path = OUTPUT_DIR / f'final_vectorizer{"_oversample" if OVERSAMPLE else ""}.joblib'
    joblib.dump(final_vectorizer, final_vec_path)

    # Combine features
    X_train_text_array = X_text_train_full.toarray()
    X_hold_text_array = X_text_holdout.toarray()
    X_train_full = np.hstack([X_train_text_array, X_trend_train_full])
    X_hold = np.hstack([X_hold_text_array, X_trend_holdout])

    # Train final LightGBM
    scale_pos_weight_full = (y_train_full == 0).sum() / max(1, (y_train_full == 1).sum())
    final_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=12,
        learning_rate=0.2,
        num_leaves=127,
        min_child_samples=10,
        min_child_weight=1e-5,
        reg_alpha=0.001,
        reg_lambda=0.001,
        scale_pos_weight=scale_pos_weight_full,
        subsample=0.95,
        colsample_bytree=0.95,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )

    # Calibration split from training (to create calibrator)
    try:
        X_fit_full, X_cal_full, y_fit_full, y_cal_full = train_test_split(
            X_train_full, y_train_full, test_size=0.10, stratify=y_train_full, random_state=RANDOM_SEED
        )
    except Exception:
        X_fit_full, X_cal_full, y_fit_full, y_cal_full = train_test_split(
            X_train_full, y_train_full, test_size=0.10, random_state=RANDOM_SEED
        )

    final_model.fit(X_fit_full, y_fit_full)

    # Calibrate
    final_calibrator = None
    try:
        final_calibrator = CalibratedClassifierCV(base_estimator=final_model, method='isotonic', cv='prefit')
        final_calibrator.fit(X_cal_full, y_cal_full)
        holdout_proba = final_calibrator.predict_proba(X_hold)[:, 1]
    except Exception:
        holdout_proba = final_model.predict_proba(X_hold)[:, 1]

    # Save final model and calibrator
    final_model_path = OUTPUT_DIR / f'final_model{"_oversample" if OVERSAMPLE else ""}.joblib'
    joblib.dump(final_model, final_model_path)
    if final_calibrator is not None:
        joblib.dump(final_calibrator, OUTPUT_DIR / f'final_calibrator{"_oversample" if OVERSAMPLE else ""}.joblib')

    # Evaluate on holdout at chosen threshold
    chosen_threshold = 0.80
    y_hold_pred = (holdout_proba >= chosen_threshold).astype(int)
    final_precision = precision_score(y_holdout, y_hold_pred, zero_division=0)
    final_recall = recall_score(y_holdout, y_hold_pred, zero_division=0)
    final_f1 = f1_score(y_holdout, y_hold_pred, zero_division=0)

    # compute precision@K / recall@K
    pr_at_k = precision_recall_at_k(y_holdout, holdout_proba, ks=[10,50,100], percentiles=[0.01,0.05,0.1])

    # Build manifest
    manifest = {
        'timestamp': datetime.now().isoformat(),
        'n_samples_total': int(n_samples),
        'holdout_size': int(len(y_holdout)),
        'oversample_used': bool(OVERSAMPLE),
        'chosen_threshold': float(chosen_threshold),
        'holdout_metrics': {
            'precision': float(final_precision),
            'recall': float(final_recall),
            'f1': float(final_f1),
        },
        'precision_recall_at_k': pr_at_k,
        'artifacts': {
            'final_vectorizer': str(final_vec_path),
            'final_model': str(final_model_path),
            'final_calibrator': str(OUTPUT_DIR / f'final_calibrator{"_oversample" if OVERSAMPLE else ""}.joblib') if final_calibrator is not None else None
        }
    }

    manifest_path = OUTPUT_DIR / f'final_model_manifest{"_oversample" if OVERSAMPLE else ""}.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Final artifacts saved. Manifest: {manifest_path}")
    
    print(f"\n{'='*80}")
    print(f"⏱️  End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
