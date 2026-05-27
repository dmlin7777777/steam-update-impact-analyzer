"""
检查 Risk 标签分布
"""
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r'C:\Users\12932\Desktop\nus\BAP')
DATA_PATH = BASE_DIR / 'features' / 'fps' / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet'

df = pd.read_parquet(DATA_PATH)

print("="*80)
print("FPS RISK LABEL ANALYSIS")
print("="*80)

print(f"\nTotal samples: {len(df):,}")

print("\nRisk Label Distribution:")
print(df['risk_label_weak'].value_counts().sort_index())
risk_ratio = df['risk_label_weak'].mean()
print(f"\nRisk ratio (positive class): {risk_ratio:.4f} ({risk_ratio*100:.2f}%)")

print("\nRisk Score Statistics:")
print(df['risk_score'].describe())

print("\nRisk Score Distribution by Label:")
for label in [0, 1]:
    subset = df[df['risk_label_weak'] == label]['risk_score']
    print(f"\nLabel {label}:")
    print(f"  Mean: {subset.mean():.4f}")
    print(f"  Median: {subset.median():.4f}")
    print(f"  Range: [{subset.min():.2f}, {subset.max():.2f}]")

# Check key features
print("\nKey Risk Features Availability:")
risk_features = [
    'is_toxic', 'vader_compound', 'contains_bug_report',
    'contains_balance_complaint', 'contains_monetization_complaint',
    'mentions_performance', 'controversial_sentiment',
    'recommendation_sentiment_mismatch', 'negative_but_helpful',
    'positive_but_unhelpful', 'ensemble_anomaly_score',
    'ml_anomaly_score', 'statistical_anomaly_score'
]

for feat in risk_features:
    if feat in df.columns:
        coverage = df[feat].notna().sum() / len(df)
        print(f"  ✓ {feat:40s}: {coverage*100:5.1f}% coverage")
    else:
        print(f"  ✗ {feat:40s}: Missing")
