"""
Compute per-fold thresholds that select top-k candidates (k = K_FRAC * n_test) and report precision@k/recall@k.
Writes: analysis_results/train/ab_weaklabels/debug_risk/thresholds_topk.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    LGB_AVAILABLE = False

ROOT = Path(r"c:/Users/12932/Desktop/nus/BAP")
PARQUET = ROOT / r"features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet"
OUT_DIR = ROOT / r"analysis_results/train/ab_weaklabels/debug_risk"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# make sure project root is on sys.path so we can import project modules
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import select_feature_columns and K_FRAC from run_ab_weaklabels
from scripts.train.run_ab_weaklabels import select_feature_columns, K_FRAC, RANDOM_STATE


def compute():
    df = pd.read_parquet(PARQUET)
    target = 'risk_label_weak'
    if target not in df.columns:
        raise SystemExit('Target not in parquet')
    # drop NA
    df2 = df[df[target].notna()].copy()
    df2 = df2.sort_values('timestamp') if 'timestamp' in df2.columns else df2
    Xcols = select_feature_columns(df2, target_col=target)
    X = df2[Xcols].to_numpy()
    y = df2[target].astype(int).to_numpy()

    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    tss = TimeSeriesSplit(n_splits=5)

    per_fold = []
    fold = 0
    for train_idx, test_idx in tss.split(X):
        fold += 1
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_imp = imputer.fit_transform(X_train_raw)
        X_train = scaler.fit_transform(X_train_imp)
        X_test_imp = imputer.transform(X_test_raw)
        X_test = scaler.transform(X_test_imp)

        is_multiclass = len(np.unique(y_train)) > 2

        # Train advanced model
        if LGB_AVAILABLE:
            if not is_multiclass:
                n_pos = int(np.sum(y_train == 1))
                n_neg = int(np.sum(y_train == 0))
                scale = float(n_neg / n_pos) if (n_pos and n_pos > 0) else 1.0
            else:
                scale = 1.0
            clf = lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, scale_pos_weight=scale)
            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_test)[:,1] if not is_multiclass else clf.predict_proba(X_test)
        else:
            clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
            clf.fit(X_train, y_train)
            prob = clf.predict_proba(X_test)[:,1] if not is_multiclass else clf.predict_proba(X_test)

        # only for binary
        if is_multiclass:
            raise SystemExit('Unexpected multiclass for risk target')

        n = len(y_test)
        k = max(1, int(np.ceil(K_FRAC * n)))
        order = np.argsort(-prob)
        topk = order[:k]
        tp = int(np.sum(y_test[topk] == 1))
        total_pos = int(np.sum(y_test == 1))
        precision_at_k = tp / k
        recall_at_k = tp / total_pos if total_pos > 0 else None
        thresh_k = float(prob[order[k-1]]) if k-1 < len(order) else float(prob[order[-1]])

        per_fold.append({
            'fold': fold,
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'k': int(k),
            'precision_at_k': float(precision_at_k),
            'recall_at_k': float(recall_at_k) if recall_at_k is not None else None,
            'threshold_k': thresh_k
        })

    # aggregate
    mean_prec = float(np.mean([p['precision_at_k'] for p in per_fold]))
    mean_rec = float(np.mean([p['recall_at_k'] for p in per_fold if p['recall_at_k'] is not None]))
    mean_thresh = float(np.mean([p['threshold_k'] for p in per_fold]))

    out = {'per_fold': per_fold, 'mean_precision_at_k': mean_prec, 'mean_recall_at_k': mean_rec, 'mean_threshold_k': mean_thresh, 'k_frac': float(K_FRAC)}
    with open(OUT_DIR / 'thresholds_topk.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('Wrote', OUT_DIR / 'thresholds_topk.json')
    print(json.dumps(out, indent=2))

if __name__ == '__main__':
    compute()
