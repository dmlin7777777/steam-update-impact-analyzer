import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False
    # fallback to sklearn's HistGradientBoosting
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
def main():
    # Explicit parameters (edit these variables as needed)
    INPUT = r"C:\Users\12932\Desktop\nus\BAP\features\fps\gpu_optimized_features_fps_exclflagged_enhanced_features.parquet"
    TARGET = "sentiment_label"  # <- change to your target column
    TIME_COL = 'review_date'
    N_SPLITS = 3
    OUTPUT_DIR = r'analysis_results/train/advanced'
    N_ESTIMATORS = 1000
    EARLY_STOPPING = 50
    DROP_CLEANED = True
    SAMPLE_WEIGHT_COL = None
    USE_CLASS_WEIGHT = False

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
        sample_weights_full = df[SAMPLE_WEIGHT_COL].values
    else:
        sample_weights_full = None

    X, y = prepare_matrix(df, TARGET)
    if USE_CLASS_WEIGHT:
        # compute per-sample weights inversely proportional to class frequency
        sample_weights_full = compute_sample_weight(class_weight='balanced', y=y)
    if X.shape[1] == 0:
        raise ValueError("No numeric features found for LightGBM. Precompute numeric features or use baseline with --use_text")

    # simple impute + scale for HGB fallback; LightGBM accepts raw X
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    metrics = {'folds': []}
    feature_importances = np.zeros(X.shape[1], dtype=float)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_imp)):
        Xtr, Xte = X_imp.iloc[train_idx], X_imp.iloc[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        if sample_weights_full is not None:
            sw_tr = sample_weights_full[train_idx]
            sw_te = sample_weights_full[test_idx]
        else:
            sw_tr = None
            sw_te = None

        if LGB_AVAILABLE:
            params = {
                'objective': 'multiclass' if len(np.unique(y)) > 2 else 'binary',
                'num_class': len(np.unique(y)) if len(np.unique(y)) > 2 else None,
                'random_state': 42,
                'verbosity': -1
            }
            model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, **{k:v for k,v in params.items() if v is not None})
            # pass sample_weight if provided
            if sw_tr is not None:
                model.fit(Xtr, ytr, sample_weight=sw_tr, eval_set=[(Xte, yte)], sample_weight_eval_set=[sw_te], early_stopping_rounds=EARLY_STOPPING, verbose=False)
            else:
                model.fit(Xtr, ytr, eval_set=[(Xte, yte)], early_stopping_rounds=EARLY_STOPPING, verbose=False)
            preds = model.predict(Xte)
            if hasattr(model, 'feature_importances_'):
                feature_importances += model.feature_importances_
        else:
            # fallback
            model = HGB(random_state=42, max_iter=200)
            if sw_tr is not None:
                model.fit(Xtr, ytr, sample_weight=sw_tr)
            else:
                model.fit(Xtr, ytr)
            preds = model.predict(Xte)
            try:
                feature_importances += getattr(model, 'feature_importances_', np.zeros(X.shape[1]))
            except Exception:
                pass

        fold_f1 = float(f1_score(yte, preds, average='weighted'))
        metrics['folds'].append({'fold': int(fold), 'f1': fold_f1})

    # average feature importance
    feature_importances = feature_importances / max(1, len(metrics['folds']))
    fi_df = pd.DataFrame({'feature': X.columns.tolist(), 'importance': feature_importances})
    fi_df = fi_df.sort_values('importance', ascending=False)
    fi_path = os.path.join(OUTPUT_DIR, f'feature_importances_{TARGET}.csv')
    fi_df.to_csv(fi_path, index=False)

    # train final model on all data
    if LGB_AVAILABLE:
        final = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, random_state=42)
        if sample_weights_full is not None:
            final.fit(X_imp, y, sample_weight=sample_weights_full)
        else:
            final.fit(X_imp, y)
    else:
        final = HGB(random_state=42, max_iter=200)
        if sample_weights_full is not None:
            final.fit(X_imp, y, sample_weight=sample_weights_full)
        else:
            final.fit(X_imp, y)

    joblib.dump({'model': final, 'imputer_columns': X.columns.tolist()}, os.path.join(OUTPUT_DIR, f'advanced_{TARGET}.joblib'))

    # summarize metrics
    metrics['cv_f1_mean'] = float(np.mean([f['f1'] for f in metrics['folds']]))
    metrics['cv_f1_std'] = float(np.std([f['f1'] for f in metrics['folds']]))
    metrics['n_samples'] = int(len(df))
    metrics['n_features'] = int(X.shape[1])

    # attempt classification report on full data
    try:
        preds_full = final.predict(X_imp)
        metrics['classification_report'] = classification_report(y, preds_full, output_dict=True)
    except Exception:
        pass

    with open(os.path.join(OUTPUT_DIR, f'metrics_advanced_{TARGET}.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print('Done. Model + metrics + feature importances saved to', OUTPUT_DIR)


if __name__ == '__main__':
    main()
