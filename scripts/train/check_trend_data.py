"""
Quick check: Trend Alert data distribution
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(r'c:\Users\12932\Desktop\nus\BAP')
FEATURES_DIR = BASE_DIR / 'features' / 'fps'

print("Loading data...")
df = pd.read_parquet(FEATURES_DIR / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet')

print(f"\nTotal samples: {len(df):,}")

if 'trend_alert_weak' in df.columns:
    dist = df['trend_alert_weak'].value_counts()
    print(f"\nTrend Alert Distribution:")
    print(f"  0 (No Alert):  {dist.get(0, 0):,} ({dist.get(0, 0)/len(df)*100:.2f}%)")
    print(f"  1 (Alert):     {dist.get(1, 0):,} ({dist.get(1, 0)/len(df)*100:.2f}%)")
    
    # Check related features
    print(f"\nRelated Features Availability:")
    trend_features = [
        'is_anomaly_weak', 'ensemble_anomaly_score',
        'sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h',
        'comment_rate_change_24h', 'comment_rate_change_48h', 'comment_rate_change_72h'
    ]
    
    for feat in trend_features:
        if feat in df.columns:
            non_null = df[feat].notna().sum()
            print(f"  {feat}: {non_null:,} ({non_null/len(df)*100:.2f}%)")
        else:
            print(f"  {feat}: NOT FOUND")
else:
    print("\n⚠️ trend_alert_weak not found in data!")
