import argparse
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == '.parquet':
        return pd.read_parquet(p)
    if p.suffix.lower() in ('.xls', '.xlsx'):
        return pd.read_excel(p)
    # fallback to csv
    return pd.read_csv(p)


def numeric_feature_matrix(df: pd.DataFrame, exclude: list):
    df_num = df.select_dtypes(include=[np.number]).copy()
    # drop target if present
    for e in exclude:
        if e in df_num.columns:
            df_num = df_num.drop(columns=[e])
    # if no numeric features, raise
    if df_num.shape[1] == 0:
        raise ValueError("No numeric features available. Consider using --use_text or provide numeric features.")
    return df_num


def main():
    # Explicit parameters (edit these variables as needed)
    INPUT = r"C:\Users\12932\Desktop\nus\BAP\features\fps\gpu_optimized_features_fps_exclflagged_enhanced_features.parquet"
    TARGET = "sentiment_label"  # <- change to your target column
    TIME_COL = 'review_date'
    USE_TEXT = False
    N_SPLITS = 3
    OUTPUT_DIR = r'analysis_results/train/baseline'
    DROP_CLEANED = True
    SAMPLE_WEIGHT_COL = None
    CLASS_WEIGHT = 'balanced'  # or 'none'

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataframe(INPUT)
    if TIME_COL not in df.columns:
        raise KeyError(f"Time column {TIME_COL} not found in input file")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    if TARGET not in df.columns:
        raise KeyError(f"Target column {TARGET} not found in input file")

    # optional drop of rows flagged by cleaning
    if DROP_CLEANED and 'is_removed_cleaning' in df.columns:
        before = len(df)
        df = df[df['is_removed_cleaning'] != 1].copy()
        print(f"Dropped {before - len(df)} rows with is_removed_cleaning==1")

    if SAMPLE_WEIGHT_COL and SAMPLE_WEIGHT_COL in df.columns:
        sample_weights = df[SAMPLE_WEIGHT_COL].values
    else:
        sample_weights = None

    y = df[TARGET].values

    # Build features
    if USE_TEXT and 'review_content_clean' in df.columns:
        print('Using text TF-IDF features')
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        X_text = vectorizer.fit_transform(df['review_content_clean'].fillna(''))
        # Train / CV using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        f1s = []
        for train_idx, test_idx in tscv.split(X_text):
            Xtr, Xte = X_text[train_idx], X_text[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            # compute sample weight for imbalance if requested
            if sample_weights is not None:
                sw_tr = sample_weights[train_idx]
            else:
                sw_tr = None
            cw = None if CLASS_WEIGHT == 'none' else 'balanced'
            clf = LogisticRegression(max_iter=1000, random_state=42, class_weight=cw)
            clf.fit(Xtr, ytr)
            preds = clf.predict(Xte)
            f1s.append(f1_score(yte, preds, average='weighted'))
        metrics = {
            'cv_f1_mean': float(np.mean(f1s)),
            'cv_f1_std': float(np.std(f1s)),
            'n_samples': int(len(df))
        }
        # final model on full data
        cw = None if CLASS_WEIGHT == 'none' else 'balanced'
        final_clf = LogisticRegression(max_iter=1000, random_state=42, class_weight=cw)
        if sample_weights is not None:
            final_clf.fit(X_text, y, sample_weight=sample_weights)
        else:
            final_clf.fit(X_text, y)
        joblib.dump({'model': final_clf, 'vectorizer': vectorizer}, os.path.join(OUTPUT_DIR, f'baseline_{TARGET}.joblib'))

    else:
        print('Using numeric features pipeline (median impute + scaler + LogisticRegression)')
        X = numeric_feature_matrix(df, exclude=[TARGET])
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)
        f1s = []
        for train_idx, test_idx in tscv.split(X):
            Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight=(None if CLASS_WEIGHT=='none' else 'balanced')))
            ])
            if sample_weights is not None:
                sw_tr = sample_weights[train_idx]
                pipe.fit(Xtr, ytr, clf__sample_weight=sw_tr)
            else:
                pipe.fit(Xtr, ytr)
            preds = pipe.predict(Xte)
            f1s.append(f1_score(yte, preds, average='weighted'))
        metrics = {
            'cv_f1_mean': float(np.mean(f1s)),
            'cv_f1_std': float(np.std(f1s)),
            'n_samples': int(len(df)),
            'n_features': int(X.shape[1])
        }
        # final model on full data
        final_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight=(None if CLASS_WEIGHT=='none' else 'balanced')))
        ])
        if sample_weights is not None:
            final_pipe.fit(X, y, clf__sample_weight=sample_weights)
        else:
            final_pipe.fit(X, y)
        joblib.dump(final_pipe, os.path.join(OUTPUT_DIR, f'baseline_{TARGET}.joblib'))

    # save metrics
    # classification report on full data
    try:
        if 'final_pipe' in locals():
            preds_full = final_pipe.predict(X)
        else:
            preds_full = final_clf.predict(X_text) if USE_TEXT else final_clf.predict(X)
        report = classification_report(y, preds_full, output_dict=True)
        metrics['classification_report'] = report
    except Exception:
        pass

    with open(os.path.join(OUTPUT_DIR, f'metrics_baseline_{TARGET}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print('Done. Metrics saved to', os.path.join(OUTPUT_DIR, f'metrics_baseline_{TARGET}.json'))


if __name__ == '__main__':
    main()
