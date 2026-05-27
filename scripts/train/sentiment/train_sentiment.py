"""
Train Baseline vs Advanced models for sentiment classification.

Baseline: TF-IDF on text + Logistic Regression
Advanced: Numerical features (excluding sentiment-related) + LightGBM

Outputs to: analysis_results/train/ab_weaklabels/sentiment/
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import joblib

ROOT = Path(r"c:/Users/12932/Desktop/nus/BAP")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARQUET = ROOT / r"features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet"
OUT_DIR = ROOT / r"analysis_results/train/ab_weaklabels/sentiment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Try import LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    LGB_AVAILABLE = False


def _make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (str, type(None), bool, int, float)):
        return obj
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return obj.tolist()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, np.generic):
                try:
                    nk = k.item()
                except Exception:
                    nk = str(k)
            else:
                nk = k
            if not isinstance(nk, (str, int, float, bool, type(None))):
                nk = str(nk)
            out[nk] = _make_json_serializable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _make_json_serializable(obj.tolist())
    try:
        return obj.item()
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


def select_feature_columns_sentiment(df: pd.DataFrame) -> list[str]:
    """Select numerical features excluding sentiment-related and ID columns."""
    drop_like = {
        'sentiment_label', 'risk_label', 'risk_label_weak', 'trend_alert_weak',
        'anomaly_label_weak', 'is_anomaly_weak', 'is_coordinated', 'is_coordinated_auto',
        'appid', 'SteamID', 'review_id', 'review_content', 'review_content_processed',
        'review_datetime', 'timestamp',
        # exclude sentiment-derived features to avoid leakage
        'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
        'vader_compound_z', 'sentiment_x_reputation', 'sentiment_x_bug',
        'controversial_sentiment', 'recommendation_sentiment_mismatch',
        # rolling sentiment features should also be excluded if they're derived from sentiment
        'sentiment_rolling_mean_24h', 'sentiment_rolling_std_24h',
        'sentiment_rolling_min_24h', 'sentiment_rolling_max_24h',
        'sentiment_rolling_mean_48h', 'sentiment_rolling_std_48h',
        'sentiment_rolling_min_48h', 'sentiment_rolling_max_48h',
        'sentiment_rolling_mean_72h', 'sentiment_rolling_std_72h',
        'sentiment_rolling_min_72h', 'sentiment_rolling_max_72h',
        'sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h',
        'sentiment_acceleration_24h', 'sentiment_acceleration_48h', 'sentiment_acceleration_72h',
        'sentiment_trend_direction', 'sentiment_volatility',
        'is_sentiment_local_peak', 'is_sentiment_local_trough',
        'avg_sentiment_in_24h', 'avg_sentiment_in_48h', 'avg_sentiment_in_72h',
        # Additional missed features that leak sentiment
        'sentiment_category',  # direct mapping from vader_compound
        'negative_but_helpful', 'positive_but_unhelpful',  # based on vader_compound interaction
    }
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in nums if c not in drop_like]
    return features


def train_sentiment():
    print("Loading data...")
    df = pd.read_parquet(PARQUET)
    
    target = 'sentiment_label'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in parquet. Available columns: {df.columns.tolist()}")
    
    # Drop NA targets
    mask = df[target].notna()
    df2 = df.loc[mask].copy()
    y = df2[target].astype(int).to_numpy()
    
    print(f"Total samples with sentiment_label: {len(df2)}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Sort by timestamp if available
    if 'timestamp' in df2.columns:
        df2 = df2.sort_values('timestamp').reset_index(drop=True)
    
    # Text column for baseline
    text_col = 'review_content_processed' if 'review_content_processed' in df2.columns else 'review_content'
    if text_col not in df2.columns:
        raise ValueError(f"No text column found. Need '{text_col}' for baseline model.")
    texts = df2[text_col].fillna('').astype(str).tolist()
    
    # Numerical features for advanced
    Xcols = select_feature_columns_sentiment(df2)
    X_num = df2[Xcols].to_numpy()
    
    print(f"Using {len(Xcols)} numerical features for Advanced model")
    print(f"Using '{text_col}' for Baseline model")
    
    # TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=5)
    
    baseline_metrics = []
    advanced_metrics = []
    per_fold = []
    fold = 0
    
    for train_idx, test_idx in tss.split(X_num):
        fold += 1
        print(f"\n=== Fold {fold} ===")
        
        y_train, y_test = y[train_idx], y[test_idx]
        
        # --- Baseline: TF-IDF + LogisticRegression ---
        texts_train = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3)
        X_train_tfidf = vectorizer.fit_transform(texts_train)
        X_test_tfidf = vectorizer.transform(texts_test)
        
        clf_base = LogisticRegression(
            max_iter=200,
            solver='liblinear',
            class_weight='balanced',
            random_state=RANDOM_STATE,
            multi_class='ovr'
        )
        clf_base.fit(X_train_tfidf, y_train)
        pred_base = clf_base.predict(X_test_tfidf)
        
        # --- Advanced: Numerical features + LightGBM ---
        X_train_num, X_test_num = X_num[train_idx], X_num[test_idx]
        
        # Preprocessing in fold
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train_adv = scaler.fit_transform(imputer.fit_transform(X_train_num))
        X_test_adv = scaler.transform(imputer.transform(X_test_num))
        
        if LGB_AVAILABLE:
            clf_adv = lgb.LGBMClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight='balanced'
            )
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf_adv = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        
        clf_adv.fit(X_train_adv, y_train)
        pred_adv = clf_adv.predict(X_test_adv)
        
        # --- Metrics ---
        def compute_metrics(y_true, y_pred, label=''):
            acc = accuracy_score(y_true, y_pred)
            prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_true, y_pred)
            
            return {
                'accuracy': float(acc),
                'precision_macro': float(prec_macro),
                'recall_macro': float(rec_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(prec_weighted),
                'recall_weighted': float(rec_weighted),
                'f1_weighted': float(f1_weighted),
                'confusion_matrix': cm.tolist(),
            }
        
        base_res = compute_metrics(y_test, pred_base, 'Baseline')
        adv_res = compute_metrics(y_test, pred_adv, 'Advanced')
        
        print(f"Baseline  - Accuracy: {base_res['accuracy']:.4f}, Weighted F1: {base_res['f1_weighted']:.4f}, Macro F1: {base_res['f1_macro']:.4f}")
        print(f"Advanced  - Accuracy: {adv_res['accuracy']:.4f}, Weighted F1: {adv_res['f1_weighted']:.4f}, Macro F1: {adv_res['f1_macro']:.4f}")
        
        per_fold.append({
            'fold': fold,
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'class_counts_train': dict(zip(*np.unique(y_train, return_counts=True))),
            'class_counts_test': dict(zip(*np.unique(y_test, return_counts=True))),
            'baseline': base_res,
            'advanced': adv_res,
        })
        
        baseline_metrics.append(base_res)
        advanced_metrics.append(adv_res)
    
    # Aggregate metrics
    def agg(mlist):
        out = {}
        keys = set().union(*[m.keys() for m in mlist])
        for k in keys:
            if k == 'confusion_matrix':
                continue  # skip aggregating confusion matrices
            vals = [m[k] for m in mlist if k in m and m[k] is not None]
            out[k] = float(np.mean(vals)) if vals else None
        return out
    
    metrics = {
        'target': target,
        'n_rows': int(df.shape[0]),
        'n_samples_used': int(len(df2)),
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'features_advanced': Xcols,
        'baseline_cv': agg(baseline_metrics),
        'advanced_cv': agg(advanced_metrics),
        'per_fold': per_fold,
    }
    
    # Save metrics
    with open(OUT_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(_make_json_serializable(metrics), f, ensure_ascii=False, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Baseline  CV - Accuracy: {metrics['baseline_cv']['accuracy']:.4f}, Weighted F1: {metrics['baseline_cv']['f1_weighted']:.4f}, Macro F1: {metrics['baseline_cv']['f1_macro']:.4f}")
    print(f"Advanced  CV - Accuracy: {metrics['advanced_cv']['accuracy']:.4f}, Weighted F1: {metrics['advanced_cv']['f1_weighted']:.4f}, Macro F1: {metrics['advanced_cv']['f1_macro']:.4f}")
    print(f"\nMetrics saved to: {OUT_DIR / 'metrics.json'}")
    
    # Train final models on full data and save
    print("\nTraining final models on full dataset...")
    
    # Final baseline
    vectorizer_final = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3)
    X_tfidf_full = vectorizer_final.fit_transform(texts)
    clf_base_final = LogisticRegression(
        max_iter=200,
        solver='liblinear',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        multi_class='ovr'
    )
    clf_base_final.fit(X_tfidf_full, y)
    joblib.dump({'vectorizer': vectorizer_final, 'model': clf_base_final}, OUT_DIR / 'baseline_model_sentiment.joblib')
    
    # Final advanced
    imputer_final = SimpleImputer(strategy='median')
    scaler_final = StandardScaler()
    X_adv_full = scaler_final.fit_transform(imputer_final.fit_transform(X_num))
    
    if LGB_AVAILABLE:
        clf_adv_final = lgb.LGBMClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
    else:
        clf_adv_final = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    
    clf_adv_final.fit(X_adv_full, y)
    joblib.dump({
        'imputer': imputer_final,
        'scaler': scaler_final,
        'model': clf_adv_final,
        'features': Xcols
    }, OUT_DIR / 'advanced_model_sentiment.joblib')
    
    print(f"Models saved to: {OUT_DIR}")
    print("\n✓ Sentiment training complete!")


if __name__ == '__main__':
    train_sentiment()
