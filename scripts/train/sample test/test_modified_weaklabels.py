"""
Test: Apply modified weaklabeling logic to FPS data and verify trend_alert_weak distribution
"""

import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.weaklabeling import generate_weak_labels

BASE_DIR = Path(r'c:\Users\12932\Desktop\nus\BAP')
FEATURES_DIR = BASE_DIR / 'features' / 'fps'

print("Loading FPS features...")
df = pd.read_parquet(FEATURES_DIR / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet')

print(f"Total samples: {len(df):,}\n")

# Drop old weak labels to force regeneration
print("Removing old weak labels...")
old_weak_labels = ['risk_label_weak', 'trend_alert_weak', 'anomaly_label_weak', 'is_anomaly_weak']
for col in old_weak_labels:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        print(f"  Dropped: {col}")

# Apply weak labels with modified logic
print("\nApplying modified weak labels...")
df = generate_weak_labels(df)

print("\n" + "="*80)
print("RESULTS")
print("="*80)

# Check trend_alert_weak
if 'trend_alert_weak' in df.columns:
    dist = df['trend_alert_weak'].value_counts()
    print(f"\n✅ trend_alert_weak (NEW):")
    print(f"   0 (No Alert): {dist.get(0, 0):,} ({dist.get(0, 0)/len(df)*100:.2f}%)")
    print(f"   1 (Alert):    {dist.get(1, 0):,} ({dist.get(1, 0)/len(df)*100:.2f}%)")
else:
    print("\n❌ trend_alert_weak not generated!")

# Check trigger_count
if 'trend_trigger_count' in df.columns:
    trigger_dist = df['trend_trigger_count'].value_counts().sort_index()
    print(f"\n✅ trend_trigger_count distribution:")
    for count, num_samples in trigger_dist.items():
        print(f"   {count} triggers: {num_samples:,} ({num_samples/len(df)*100:.2f}%)")
else:
    print("\n⚠️ trend_trigger_count not generated")

# Check risk_label_weak (should be unchanged)
if 'risk_label_weak' in df.columns:
    dist = df['risk_label_weak'].value_counts()
    print(f"\n✅ risk_label_weak (should be ~42% positive):")
    print(f"   0 (Low Risk):  {dist.get(0, 0):,} ({dist.get(0, 0)/len(df)*100:.2f}%)")
    print(f"   1 (High Risk): {dist.get(1, 0):,} ({dist.get(1, 0)/len(df)*100:.2f}%)")
else:
    print("\n❌ risk_label_weak not generated!")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if 'trend_alert_weak' in df.columns:
    alert_rate = df['trend_alert_weak'].sum() / len(df) * 100
    if 5 <= alert_rate <= 15:
        print(f"\n✅ SUCCESS! Alert rate = {alert_rate:.2f}% (target: 5-15%)")
        print("   Trend alert weak label has good discriminative power.")
    elif alert_rate > 95:
        print(f"\n❌ FAILED! Alert rate = {alert_rate:.2f}% (too high, no discriminative power)")
    else:
        print(f"\n⚠️ BORDERLINE: Alert rate = {alert_rate:.2f}%")
else:
    print("\n❌ FAILED! trend_alert_weak not generated")
