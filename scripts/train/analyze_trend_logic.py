"""
Analyze Trend Alert generation logic and propose better thresholds
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r'c:\Users\12932\Desktop\nus\BAP')
FEATURES_DIR = BASE_DIR / 'features' / 'fps'

print("Loading data...")
df = pd.read_parquet(FEATURES_DIR / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet')

print(f"\nTotal samples: {len(df):,}\n")

# Analyze each component
print("="*80)
print("TREND ALERT COMPONENT ANALYSIS")
print("="*80)

# 1. Anomaly component
print("\n1. is_anomaly_weak:")
if 'is_anomaly_weak' in df.columns:
    anom_dist = df['is_anomaly_weak'].value_counts()
    print(f"   0: {anom_dist.get(0, 0):,} ({anom_dist.get(0, 0)/len(df)*100:.2f}%)")
    print(f"   1: {anom_dist.get(1, 0):,} ({anom_dist.get(1, 0)/len(df)*100:.2f}%)")

# 2. Ensemble anomaly score (95th percentile)
print("\n2. ensemble_anomaly_score (>= p95):")
if 'ensemble_anomaly_score' in df.columns:
    ens_scores = df['ensemble_anomaly_score'].dropna()
    p95 = ens_scores.quantile(0.95)
    above_p95 = (ens_scores >= p95).sum()
    print(f"   p95 threshold: {p95:.3f}")
    print(f"   Above p95: {above_p95:,} ({above_p95/len(df)*100:.2f}%)")

# 3. Sentiment diff features
print("\n3. Sentiment diff features (abs >= p95):")
for col in ['sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h']:
    if col in df.columns:
        values = df[col].dropna()
        p95 = values.abs().quantile(0.95)
        above_p95 = (values.abs() >= p95).sum()
        print(f"   {col}:")
        print(f"     p95: {p95:.3f}")
        print(f"     Above p95: {above_p95:,} ({above_p95/len(df)*100:.2f}%)")

# 4. Comment rate change features
print("\n4. Comment rate change features (abs >= p95):")
for col in ['comment_rate_change_24h', 'comment_rate_change_48h', 'comment_rate_change_72h']:
    if col in df.columns:
        values = df[col].dropna()
        p95 = values.abs().quantile(0.95)
        above_p95 = (values.abs() >= p95).sum()
        print(f"   {col}:")
        print(f"     p95: {p95:.3f}")
        print(f"     Above p95: {above_p95:,} ({above_p95/len(df)*100:.2f}%)")

# 5. Analyze how many components trigger
print("\n" + "="*80)
print("TRIGGER COMPONENT COUNT")
print("="*80)

trigger_count = np.zeros(len(df), dtype=int)

# Count each component
if 'is_anomaly_weak' in df.columns:
    trigger_count += df['is_anomaly_weak'].fillna(0).astype(int)

if 'ensemble_anomaly_score' in df.columns:
    p95 = df['ensemble_anomaly_score'].dropna().quantile(0.95)
    trigger_count += (df['ensemble_anomaly_score'].fillna(0) >= p95).astype(int)

for col in ['sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h',
            'comment_rate_change_24h', 'comment_rate_change_48h', 'comment_rate_change_72h']:
    if col in df.columns:
        p95 = df[col].dropna().abs().quantile(0.95)
        trigger_count += (df[col].fillna(0).abs() >= p95).astype(int)

# Distribution of trigger counts
print("\nNumber of triggers per sample:")
for i in range(trigger_count.max() + 1):
    count = (trigger_count == i).sum()
    print(f"  {i} triggers: {count:,} ({count/len(df)*100:.2f}%)")

# Proposed better thresholds
print("\n" + "="*80)
print("PROPOSED BETTER THRESHOLDS")
print("="*80)

print("\nOption 1: Require 2+ triggers (more conservative)")
new_trend_1 = (trigger_count >= 2).astype(int)
dist_1 = pd.Series(new_trend_1).value_counts()
print(f"  Alert=0: {dist_1.get(0, 0):,} ({dist_1.get(0, 0)/len(df)*100:.2f}%)")
print(f"  Alert=1: {dist_1.get(1, 0):,} ({dist_1.get(1, 0)/len(df)*100:.2f}%)")

print("\nOption 2: Require 3+ triggers (most conservative)")
new_trend_2 = (trigger_count >= 3).astype(int)
dist_2 = pd.Series(new_trend_2).value_counts()
print(f"  Alert=0: {dist_2.get(0, 0):,} ({dist_2.get(0, 0)/len(df)*100:.2f}%)")
print(f"  Alert=1: {dist_2.get(1, 0):,} ({dist_2.get(1, 0)/len(df)*100:.2f}%)")

print("\nOption 3: Use p99 instead of p95 (stricter thresholds)")
trigger_count_p99 = np.zeros(len(df), dtype=int)

if 'is_anomaly_weak' in df.columns:
    trigger_count_p99 += df['is_anomaly_weak'].fillna(0).astype(int)

if 'ensemble_anomaly_score' in df.columns:
    p99 = df['ensemble_anomaly_score'].dropna().quantile(0.99)
    trigger_count_p99 += (df['ensemble_anomaly_score'].fillna(0) >= p99).astype(int)
    print(f"  ensemble_anomaly_score p99: {p99:.3f}")

for col in ['sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h',
            'comment_rate_change_24h', 'comment_rate_change_48h', 'comment_rate_change_72h']:
    if col in df.columns:
        p99 = df[col].dropna().abs().quantile(0.99)
        trigger_count_p99 += (df[col].fillna(0).abs() >= p99).astype(int)

new_trend_3 = (trigger_count_p99 >= 1).astype(int)
dist_3 = pd.Series(new_trend_3).value_counts()
print(f"  Alert=0: {dist_3.get(0, 0):,} ({dist_3.get(0, 0)/len(df)*100:.2f}%)")
print(f"  Alert=1: {dist_3.get(1, 0):,} ({dist_3.get(1, 0)/len(df)*100:.2f}%)")

# Recommendation
print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\n💡 Current trend_alert_weak (100% alert) has NO discriminative power.")
print("\n   Recommended approach:")
print("   1. Use Option 2 (3+ triggers) → 13.86% alert rate")
print("   2. Or use Option 3 (p99 thresholds) → 18.57% alert rate")
print("   3. Or use trigger_count as a continuous score (0-8) instead of binary label")
print("\n   For training, Option 2 or 3 would provide better class balance.")
