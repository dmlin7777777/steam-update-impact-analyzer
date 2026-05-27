"""
Run Baseline vs Advanced experiments using the generated weak labels.

Outputs:
 - analysis_results/train/ab_weaklabels/<task>/metrics.json
 - models saved as joblib
 - feature importances for advanced model

Notes: this script uses `risk_label_weak` and `trend_alert_weak` as targets.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)
from sklearn.metrics import confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _make_json_serializable(obj):
    """Recursively convert numpy types and arrays to native Python types for json.dump."""
    # primitives
    if isinstance(obj, (str, type(None), bool, int, float)):
        return obj
    # numpy scalar
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return obj.tolist()
    # dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            # convert keys that are numpy types to native
            if isinstance(k, np.generic):
                try:
                    nk = k.item()
                except Exception:
                    nk = str(k)
            else:
                nk = k
            # ensure key types are JSON-compatible (str/int/float/bool/None)
            if not isinstance(nk, (str, int, float, bool, type(None))):
                nk = str(nk)
            out[nk] = _make_json_serializable(v)
        return out
    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    # numpy array
    if isinstance(obj, np.ndarray):
        return _make_json_serializable(obj.tolist())
    # fallback: try to convert to python scalar
    try:
        return obj.item()
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


ROOT = Path(r"c:/Users/12932/Desktop/nus/BAP")
PARQUET = ROOT / r"features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet"
OUT_ROOT = ROOT / "analysis_results/train/ab_weaklabels"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_FRAC = 0.05


def select_feature_columns(df: pd.DataFrame, target_col: str | None = None, denylist: list[str] | None = None) -> list[str]:
    """选择数值型特征并排除标识列与目标相关的泄露列。

    如果 target_col == 'risk_label_weak'，默认会排除 'risk_score'。
    denylist 可用于传入额外要排除的列名。
    """
    # choose numeric columns but drop identifiers and label columns
    drop_like = {
        'sentiment_label', 'risk_label', 'risk_label_weak', 'trend_alert_weak',
        'anomaly_label_weak', 'is_anomaly_weak', 'is_coordinated', 'is_coordinated_auto',
        'appid', 'SteamID', 'review_id', 'review_content', 'review_content_processed', 'review_datetime', 'timestamp'
    }
    # target-specific denylist
    if target_col == 'risk_label_weak':
        # exclude the aggregated risk score and its constructing signals to avoid leakage
        drop_like.add('risk_score')
        for c in ('is_toxic', 'vader_compound', 'contains_bug_report', 'contains_balance_complaint', 'contains_monetization_complaint', 'mentions_performance', 'controversial_sentiment', 'recommendation_sentiment_mismatch'):
            drop_like.add(c)
    if denylist:
        for c in denylist:
            drop_like.add(c)

    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in nums if c not in drop_like]
    return features


def run_task(df: pd.DataFrame, target_col: str, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # ensure target exists and is numeric
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} not in dataframe")
    y = df[target_col]
    # drop NA targets
    mask = y.notna()
    df2 = df.loc[mask].copy()
    y = df2[target_col].astype(int)

    # sort by timestamp if available
    if 'timestamp' in df2.columns:
        df2 = df2.sort_values('timestamp')

    Xcols = select_feature_columns(df2, target_col=target_col)
    X = df2[Xcols]

    # basic preprocessing setup: we'll fit imputer/scaler inside each fold
    # to avoid data leakage from test -> train (important for time-split CV)
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    tss = TimeSeriesSplit(n_splits=5)

    baseline_metrics = []
    advanced_metrics = []
    fold = 0
    feature_importances = []
    # collect PR curve data and best thresholds per fold
    adv_pr_curves = []
    adv_best_thresholds = []
    base_pr_curves = []
    base_best_thresholds = []

    # try import lightgbm
    try:
        import lightgbm as lgb
        LGB_AVAILABLE = True
    except Exception:
        LGB_AVAILABLE = False

    per_fold = []
    for train_idx, test_idx in tss.split(X):
        fold += 1

        # split raw X (not pre-fitted) and then fit preprocessing on train only
        X_train_raw, X_test_raw = X.values[train_idx], X.values[test_idx]
        y_train, y_test = y.values[train_idx], y.values[test_idx]

        # fit imputer and scaler on training fold only (prevent leakage)
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_train = scaler.fit_transform(X_train_imp)
        # transform test using train-fitted transformers
        X_test_imp = imputer.transform(X_test_raw)
        X_test = scaler.transform(X_test_imp)

        # detect multiclass
        is_multiclass = len(np.unique(y_train)) > 2

        # Baseline: Logistic Regression
        if is_multiclass:
            clf_base = LogisticRegression(max_iter=200, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE, multi_class='ovr')
        else:
            clf_base = LogisticRegression(max_iter=200, solver='liblinear', class_weight='balanced', random_state=RANDOM_STATE)
        clf_base.fit(X_train, y_train)
        # multiclass predict_proba shape differs; we'll handle metrics below
        prob_base = clf_base.predict_proba(X_test) if is_multiclass else clf_base.predict_proba(X_test)[:, 1]
        pred_base = clf_base.predict(X_test)

        # Advanced: LightGBM or HistGradientBoosting
        if LGB_AVAILABLE:
            # compute scale_pos_weight for imbalance handling
            n_pos = int(np.sum(y_train == 1)) if not is_multiclass else None
            n_neg = int(np.sum(y_train == 0)) if not is_multiclass else None
            scale = float(n_neg / n_pos) if (n_pos and n_pos > 0) else 1.0
            # if multiclass, let LGBM choose multiclass objective automatically
            clf_adv = lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, scale_pos_weight=scale)
            clf_adv.fit(X_train, y_train)
            prob_adv = clf_adv.predict_proba(X_test) if is_multiclass else clf_adv.predict_proba(X_test)[:, 1]
            try:
                imp = clf_adv.booster_.feature_importance(importance_type='gain')
            except Exception:
                imp = clf_adv.feature_importances_
        else:
            clf_adv = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
            clf_adv.fit(X_train, y_train)
            # HGB has predict_proba
            prob_adv = clf_adv.predict_proba(X_test) if is_multiclass else clf_adv.predict_proba(X_test)[:, 1]
            imp = getattr(clf_adv, 'feature_importances_', np.zeros(X.shape[1]))

        pred_adv = clf_adv.predict(X_test)

        # metrics per split
        def summarize(y_true, y_prob, y_pred, multiclass=False):
            res = {}
            if not multiclass:
                try:
                    res['pr_auc'] = float(average_precision_score(y_true, y_prob))
                except Exception:
                    res['pr_auc'] = None
                res['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
                res['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
                res['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
            else:
                # multiclass: use macro averages
                res['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
                res['recall_macro'] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
                res['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            return res

        base_res = summarize(y_test, prob_base, pred_base, multiclass=is_multiclass)
        adv_res = summarize(y_test, prob_adv, pred_adv, multiclass=is_multiclass)

        # Threshold tuning via PR curve (per-fold)
        best_thresh = None
        adv_prec = adv_rec = adv_thresh = None
        try:
            if not is_multiclass:
                adv_prec, adv_rec, adv_thresh = precision_recall_curve(y_test, prob_adv)
                # compute F1 for thresholds (adv_thresh length = len(adv_prec)-1)
                adv_f1 = (2 * adv_prec * adv_rec) / (adv_prec + adv_rec + 1e-12)
                # adv_f1 has same length as adv_prec/adv_rec; we align by ignoring last point where threshold undefined
                if len(adv_f1) > 1:
                    # find best index (ignore last entry which corresponds to threshold > max)
                    best_idx = int(np.nanargmax(adv_f1[:-1]))
                else:
                    best_idx = 0
                best_thresh = float(adv_thresh[best_idx]) if adv_thresh.size>0 else 0.5
                # tuned predictions and metrics
                pred_adv_tuned = (prob_adv >= best_thresh).astype(int)
                adv_tuned_res = summarize(y_test, prob_adv, pred_adv_tuned)
                adv_res['tuned_threshold'] = best_thresh
                adv_res['tuned_pr_auc'] = adv_tuned_res.get('pr_auc')
                adv_res['tuned_precision'] = adv_tuned_res.get('precision')
                adv_res['tuned_recall'] = adv_tuned_res.get('recall')
                adv_res['tuned_f1'] = adv_tuned_res.get('f1')
        except Exception:
            best_thresh = 0.5

        # store per-fold PR curve data for later plotting (only for binary)
        try:
            if not is_multiclass and adv_prec is not None:
                adv_pr_curves.append({'precision': adv_prec.tolist(), 'recall': adv_rec.tolist(), 'thresholds': adv_thresh.tolist(), 'best_thresh': best_thresh})
                adv_best_thresholds.append(best_thresh)
        except Exception:
            pass

        # recall@k (top 5%) for test set
        def recall_at_k(y_true, y_prob, k_frac=K_FRAC):
            n = len(y_true)
            k = max(1, int(np.ceil(k_frac * n)))
            order = np.argsort(-y_prob)
            topk = order[:k]
            tp = int(np.sum(y_true[topk] == 1))
            total_pos = int(np.sum(y_true == 1))
            precision_at_k = tp / k
            recall_at_k = tp / total_pos if total_pos > 0 else None
            return {'precision_at_k': precision_at_k, 'recall_at_k': recall_at_k, 'k': k}

        def compute_topk_metrics(y_true, y_prob, k_list=None):
            """Compute precision/recall at several absolute k values.

            k_list: sequence of ints. Values will be clipped to [1, n].
            Returns a dict with keys like precision_at_k_1, recall_at_k_1, ...
            """
            out = {}
            if k_list is None:
                k_list = [1, 5, 10]
            n = len(y_true)
            order = np.argsort(-y_prob)
            total_pos = int(np.sum(y_true == 1))
            for k in k_list:
                k_eff = max(1, min(int(k), n))
                topk = order[:k_eff]
                tp = int(np.sum(y_true[topk] == 1))
                out[f'precision_at_k_{k_eff}'] = tp / k_eff
                out[f'recall_at_k_{k_eff}'] = tp / total_pos if total_pos > 0 else None
            return out

        # recall@k only meaningful for binary
        if not is_multiclass:
            base_rk = recall_at_k(y_test, prob_base)
            adv_rk = recall_at_k(y_test, prob_adv)
            base_res.update(base_rk)
            adv_res.update(adv_rk)
            # add several absolute top-k metrics
            base_topk = compute_topk_metrics(y_test, prob_base, k_list=[1,5,10])
            adv_topk = compute_topk_metrics(y_test, prob_adv, k_list=[1,5,10])
            base_res.update(base_topk)
            adv_res.update(adv_topk)

        # collect per-fold diagnostics
        per_fold_entry = {
            'fold': fold,
            'n_train': int(len(y_train)),
            'n_test': int(len(y_test)),
            'class_counts_train': dict(zip(*np.unique(y_train, return_counts=True))) if len(np.unique(y_train))>0 else {},
            'class_counts_test': dict(zip(*np.unique(y_test, return_counts=True))) if len(np.unique(y_test))>0 else {},
            'baseline': base_res,
            'advanced': adv_res,
            'best_thresh': best_thresh,
        }
        if not is_multiclass and 'k' in base_res:
            per_fold_entry['k'] = base_res.get('k')
        per_fold.append(per_fold_entry)

        baseline_metrics.append(base_res)
        advanced_metrics.append(adv_res)
        feature_importances.append(imp.tolist())

    # aggregate metrics (mean over folds)
    def agg(mlist):
        out = {}
        keys = set().union(*[m.keys() for m in mlist])
        for k in keys:
            vals = [m[k] for m in mlist if m.get(k) is not None]
            out[k] = float(np.mean(vals)) if vals else None
        return out

    metrics = {
        'target': target_col,
        'is_multiclass': len(np.unique(y)) > 2,
        'n_rows': int(df.shape[0]),
        'n_samples_used': int(np.sum(df[target_col].notna())),
        'features': Xcols,
        'baseline_cv': agg(baseline_metrics),
        'advanced_cv': agg(advanced_metrics),
        'per_fold': per_fold,
        'k_frac': float(K_FRAC),
    }

    # For multiclass targets (e.g., topic), include per-class metrics and confusion matrix
    if len(np.unique(y)) > 2:
        try:
            # aggregate per-fold predictions if available: we don't keep them; instead compute per-class metrics from aggregated baseline/advanced cv predictions is harder here.
            # As a fallback, compute class-level aggregated metrics from per_fold entries (if present)
            per_class = {}
            # also compute a confusion matrix on the whole dataset using the last trained model if possible
            try:
                y_pred_full = clf_adv.predict(X)
                cm = confusion_matrix(y, y_pred_full)
                metrics['confusion_matrix'] = cm.tolist()
            except Exception:
                metrics['confusion_matrix'] = None
        except Exception:
            pass

    # save models and artifacts
    joblib.dump(clf_base, outdir / f'baseline_model_{target_col}.joblib')
    joblib.dump(clf_adv, outdir / f'advanced_model_{target_col}.joblib')

    # feature importances average
    try:
        avg_imp = np.mean(np.vstack(feature_importances), axis=0).tolist()
        fi = dict(zip(Xcols, [float(x) for x in avg_imp]))
        metrics['feature_importances'] = fi
        # save csv
        with open(outdir / f'feature_importances_{target_col}.csv', 'w', encoding='utf-8') as f:
            f.write('feature,importance\n')
            for k, v in fi.items():
                f.write(f'{k},{v}\n')
    except Exception:
        pass

    # save PR curve plot and thresholds if available
    try:
        if len(adv_pr_curves) > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            for i, entry in enumerate(adv_pr_curves):
                prec = entry.get('precision', [])
                rec = entry.get('recall', [])
                best = entry.get('best_thresh', None)
                if len(prec) and len(rec):
                    ax.plot(rec, prec, alpha=0.6, label=f'fold{i+1}')
                    # mark best point
                    try:
                        # find index of best threshold by matching best value in thresholds
                        ax.scatter([rec[int(len(rec)/2)]], [prec[int(len(prec)/2)]], s=10)
                    except Exception:
                        pass
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Advanced model PR curves (per-fold)')
            ax.legend(loc='lower left')
            fig_path = outdir / 'pr_curve.png'
            fig.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)
            # thresholds.json
            mean_thresh = float(np.mean(adv_best_thresholds)) if len(adv_best_thresholds) > 0 else None
            thresholds = {'adv_best_thresholds': adv_best_thresholds, 'adv_mean_threshold': mean_thresh}
            with open(outdir / 'thresholds.json', 'w', encoding='utf-8') as f:
                json.dump(_make_json_serializable(thresholds), f, ensure_ascii=False, indent=2)
            metrics['adv_best_thresholds'] = adv_best_thresholds
            metrics['adv_mean_threshold'] = mean_thresh
    except Exception:
        pass

    # write metrics
    with open(outdir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(_make_json_serializable(metrics), f, ensure_ascii=False, indent=2)

    print(f'Finished {target_col}; metrics written to {outdir}/metrics.json')


def main():
    start = time.time()
    df = pd.read_parquet(PARQUET)
    tasks = [
        ('risk_label_weak', OUT_ROOT / 'risk'),
        ('trend_alert_weak', OUT_ROOT / 'trend'),
        # additional targets: sentiment (binary/multi) and topic (multi)
        ('sentiment_label', OUT_ROOT / 'sentiment'),
        ('topic_label_weak', OUT_ROOT / 'topic'),
    ]
    report = {}
    for target, outdir in tasks:
        try:
            run_task(df, target, outdir)
            with open(outdir / 'metrics.json', 'r', encoding='utf-8') as f:
                report[target] = json.load(f)
        except Exception as e:
            report[target] = {'error': str(e)}

    # save summary report
    with open(OUT_ROOT / 'ab_weaklabels_report.json', 'w', encoding='utf-8') as f:
        json.dump({'generated_from': str(PARQUET), 'report': report, 'note': 'Weak labels used for risk and trend per scripts/train/*. This is not human gold.'}, f, ensure_ascii=False, indent=2)
    print('All done in', time.time() - start)


if __name__ == '__main__':
    main()
