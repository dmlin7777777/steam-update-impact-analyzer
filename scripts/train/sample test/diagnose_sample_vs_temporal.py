"""
Diagnose why small sample (10k, recent) performs well but full sample (170k, all time) performs poorly.

Tests 3 sampling strategies:
1. Recent only (tail 10k) - your current quick validation
2. Random sample (10k from all time)
3. Full dataset (all 170k)

Goal: Isolate whether issue is sample size or temporal coverage.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report

ROOT = Path(r"c:/Users/12932/Desktop/nus/BAP")
PARQUET = ROOT / r"features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def analyze_temporal_bins(df, target, name):
    """Analyze class distribution across time bins."""
    print(f"\n[{name}] Temporal Distribution Analysis:")
    
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    bin_size = len(df_sorted) // 5
    
    print(f"{'Bin':<8} {'Date Range':<35} {'Neg %':>8} {'Neu %':>8} {'Pos %':>8} {'Samples':>10}")
    print("-" * 85)
    
    for i in range(5):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < 4 else len(df_sorted)
        bin_df = df_sorted.iloc[start_idx:end_idx]
        
        bin_start = pd.to_datetime(bin_df['timestamp'].min(), unit='s').strftime('%Y-%m-%d')
        bin_end = pd.to_datetime(bin_df['timestamp'].max(), unit='s').strftime('%Y-%m-%d')
        
        dist = bin_df[target].value_counts(normalize=True).sort_index()
        neg_pct = dist.get(0, 0) * 100
        neu_pct = dist.get(1, 0) * 100
        pos_pct = dist.get(2, 0) * 100
        
        print(f"Bin {i+1}    {bin_start} - {bin_end}   {neg_pct:7.2f}%  {neu_pct:7.2f}%  {pos_pct:7.2f}%  {len(bin_df):>10,}")
    
    # Calculate variance
    bin_dists = []
    for i in range(5):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < 4 else len(df_sorted)
        bin_df = df_sorted.iloc[start_idx:end_idx]
        dist = bin_df[target].value_counts(normalize=True).sort_index()
        bin_dists.append([dist.get(0, 0), dist.get(1, 0), dist.get(2, 0)])
    
    bin_dists = np.array(bin_dists)
    variance = bin_dists.std(axis=0).mean()
    print(f"\nMean class distribution variance: {variance:.4f}")
    
    return variance


def run_baseline_tscv(texts, y, n_splits=3, name=""):
    """Run Baseline with TimeSeriesSplit."""
    print(f"\n[{name}] Running Baseline with TimeSeriesSplit ({n_splits} folds)...")
    
    tss = TimeSeriesSplit(n_splits=n_splits)
    fold_f1s = []
    fold = 0
    
    for train_idx, test_idx in tss.split(range(len(texts)), y):
        fold += 1
        
        texts_train = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Show distribution shift
        train_dist = np.bincount(y_train, minlength=3) / len(y_train) * 100
        test_dist = np.bincount(y_test, minlength=3) / len(y_test) * 100
        shift = np.abs(train_dist - test_dist).sum()
        
        print(f"  Fold {fold}: Train [{train_dist[0]:.1f}%, {train_dist[1]:.1f}%, {train_dist[2]:.1f}%] → "
              f"Test [{test_dist[0]:.1f}%, {test_dist[1]:.1f}%, {test_dist[2]:.1f}%] (shift: {shift:.1f}%)")
        
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
        X_train_tfidf = vectorizer.fit_transform(texts_train)
        X_test_tfidf = vectorizer.transform(texts_test)
        
        clf = LogisticRegression(
            max_iter=200,
            solver='liblinear',
            class_weight='balanced',
            random_state=RANDOM_STATE,
            multi_class='ovr'
        )
        clf.fit(X_train_tfidf, y_train)
        pred = clf.predict(X_test_tfidf)
        
        f1_w = f1_score(y_test, pred, average='weighted')
        f1_m = f1_score(y_test, pred, average='macro')
        acc = (pred == y_test).mean()
        
        fold_f1s.append(f1_w)
        print(f"    → Acc: {acc:.4f}, W-F1: {f1_w:.4f}, M-F1: {f1_m:.4f}")
    
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    print(f"  → Mean W-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    
    return mean_f1, std_f1


def main():
    print("="*80)
    print("SAMPLE SIZE vs TEMPORAL COVERAGE DIAGNOSIS")
    print("="*80)
    print("\nHypothesis: Small recent sample works well, but full historical data has drift\n")
    
    # Load full data
    print("Loading full dataset...")
    df = pd.read_parquet(PARQUET)
    
    target = 'sentiment_label'
    mask = df[target].notna()
    df_full = df.loc[mask].copy()
    
    # Sort by timestamp (CRITICAL for temporal analysis)
    if 'timestamp' not in df_full.columns:
        raise ValueError("No timestamp column!")
    
    df_full = df_full.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Full dataset: {len(df_full):,} samples")
    print(f"Time range: {pd.to_datetime(df_full['timestamp'].min(), unit='s')} to "
          f"{pd.to_datetime(df_full['timestamp'].max(), unit='s')}")
    print(f"Class distribution: {dict(zip(*np.unique(df_full[target].values, return_counts=True)))}")
    
    text_col = 'review_content_processed' if 'review_content_processed' in df_full.columns else 'review_content'
    
    # ===== Strategy 1: Recent 10k (your quick validation) =====
    print("\n" + "="*80)
    print("STRATEGY 1: Recent 10k samples (TAIL)")
    print("="*80)
    
    df_recent = df_full.tail(10000).reset_index(drop=True)
    y_recent = df_recent[target].astype(int).to_numpy()
    texts_recent = df_recent[text_col].fillna('').astype(str).tolist()
    
    print(f"\nSamples: {len(df_recent):,}")
    min_time = pd.to_datetime(df_recent['timestamp'].min(), unit='s')
    max_time = pd.to_datetime(df_recent['timestamp'].max(), unit='s')
    print(f"Time range: {min_time} to {max_time}")
    time_span_days_recent = (max_time - min_time).total_seconds() / 86400
    print(f"Time span: {time_span_days_recent:.0f} days")
    
    var_recent = analyze_temporal_bins(df_recent, target, "Recent 10k")
    mean_recent, std_recent = run_baseline_tscv(texts_recent, y_recent, n_splits=3, name="Recent 10k")
    
    # ===== Strategy 2: Random 10k (from all time) =====
    print("\n" + "="*80)
    print("STRATEGY 2: Random 10k samples (SAMPLE)")
    print("="*80)
    
    df_random = df_full.sample(n=10000, random_state=RANDOM_STATE).sort_values('timestamp').reset_index(drop=True)
    y_random = df_random[target].astype(int).to_numpy()
    texts_random = df_random[text_col].fillna('').astype(str).tolist()
    
    print(f"\nSamples: {len(df_random):,}")
    min_time = pd.to_datetime(df_random['timestamp'].min(), unit='s')
    max_time = pd.to_datetime(df_random['timestamp'].max(), unit='s')
    print(f"Time range: {min_time} to {max_time}")
    time_span_days_random = (max_time - min_time).total_seconds() / 86400
    print(f"Time span: {time_span_days_random:.0f} days")
    
    var_random = analyze_temporal_bins(df_random, target, "Random 10k")
    mean_random, std_random = run_baseline_tscv(texts_random, y_random, n_splits=3, name="Random 10k")
    
    # ===== Strategy 3: Full dataset (subsample for speed) =====
    print("\n" + "="*80)
    print("STRATEGY 3: Full dataset (50k subsample for speed)")
    print("="*80)
    
    # Use systematic sampling to preserve temporal structure
    step = len(df_full) // 50000
    df_full_sub = df_full.iloc[::step].reset_index(drop=True)
    y_full = df_full_sub[target].astype(int).to_numpy()
    texts_full = df_full_sub[text_col].fillna('').astype(str).tolist()
    
    print(f"\nSamples: {len(df_full):,}")
    min_time = pd.to_datetime(df_full['timestamp'].min(), unit='s')
    max_time = pd.to_datetime(df_full['timestamp'].max(), unit='s')
    print(f"Time range: {min_time} to {max_time}")
    time_span_days_full = (max_time - min_time).total_seconds() / 86400
    print(f"Time span: {time_span_days_full:.0f} days")
    
    var_full = analyze_temporal_bins(df_full_sub, target, "Full dataset")
    mean_full, std_full = run_baseline_tscv(texts_full, y_full, n_splits=3, name="Full dataset")
    
    # ===== Comparison =====
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Strategy':<25} {'Samples':>10} {'Time Span':>12} {'Variance':>10} {'Mean W-F1':>12} {'Std':>8}")
    print("-" * 90)
    print(f"{'Recent 10k (TAIL)':<25} {len(df_recent):>10,} {time_span_days_recent:>11.0f}d {var_recent:>10.4f} {mean_recent:>12.4f} {std_recent:>8.4f}")
    
    print(f"{'Random 10k (SAMPLE)':<25} {len(df_random):>10,} {time_span_days_random:>11.0f}d {var_random:>10.4f} {mean_random:>12.4f} {std_random:>8.4f}")
    
    print(f"{'Full dataset (50k sub)':<25} {len(df_full_sub):>10,} {time_span_days_full:>11.0f}d {var_full:>10.4f} {mean_full:>12.4f} {std_full:>8.4f}")
    print("-" * 90)
    
    # Analysis
    print(f"\n{'='*80}")
    print("ROOT CAUSE ANALYSIS")
    print(f"{'='*80}\n")
    
    perf_drop_random = mean_recent - mean_random
    perf_drop_full = mean_recent - mean_full
    
    print(f"Performance drop from Recent 10k:")
    print(f"  → Random 10k:  {perf_drop_random:+.4f} ({perf_drop_random/mean_recent*100:+.1f}%)")
    print(f"  → Full dataset: {perf_drop_full:+.4f} ({perf_drop_full/mean_recent*100:+.1f}%)")
    
    if perf_drop_full > 0.10:
        print("\n⚠️  CONFIRMED: Full dataset has SIGNIFICANT performance degradation (>10%)")
        print("\nRoot cause:")
        if var_full > var_recent * 1.5:
            print("  ✓ Higher temporal variance in full dataset")
            print(f"    - Recent variance: {var_recent:.4f}")
            print(f"    - Full variance:   {var_full:.4f} ({var_full/var_recent:.1f}x higher)")
        
        if perf_drop_random > 0.05:
            print("  ✓ Random sampling across full time range also degrades performance")
            print("    - Not just about sample size, but TIME COVERAGE")
        
        print("\nRecommendations:")
        print("  1. Use RECENT DATA ONLY for training (e.g., last 6-12 months)")
        print("  2. Discard old data that has different distribution")
        print("  3. Retrain model regularly as new data comes in")
        print("  4. Add temporal features (months_since_release, etc.)")
        
    elif perf_drop_full > 0.05:
        print("\n⚠️  MODERATE degradation in full dataset (5-10%)")
        print("\nRecommendations:")
        print("  1. Consider using full dataset but with temporal features")
        print("  2. Monitor performance on recent data separately")
        print("  3. Retrain periodically (quarterly)")
    
    else:
        print("\n✓ Full dataset performance is acceptable")
        print("  → Safe to use all available data")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}\n")
    
    if perf_drop_full > 0.10:
        print("OPTION A (RECOMMENDED): Train on recent data only")
        print("  - Filter to last 12 months of data")
        print("  - Expect performance similar to quick validation (~80% W-F1)")
        print("  - Set up retraining pipeline (monthly or quarterly)")
        print("")
        print("OPTION B: Use all data with temporal awareness")
        print("  - Add 'months_since_first_review' feature")
        print("  - Add 'year', 'quarter' categorical features")
        print("  - Use sample weights (down-weight old data)")
        print("")
        print("OPTION C: Hybrid approach")
        print("  - Train separate models for 'old' and 'recent' periods")
        print("  - Route predictions based on data age")
    else:
        print("✓ Safe to proceed with full dataset training")
        print("  Run: python scripts/train/train_sentiment_per_genre.py")


if __name__ == '__main__':
    main()
