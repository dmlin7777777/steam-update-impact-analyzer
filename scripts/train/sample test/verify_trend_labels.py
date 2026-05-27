import pandas as pd
from pathlib import Path

base_dir = Path(r'c:\Users\12932\Desktop\nus\BAP')
df = pd.read_parquet(base_dir / 'features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet')

print('='*80)
print('TREND ALERT WEAK LABEL VERIFICATION')
print('='*80)
print(f'\nTotal samples: {len(df):,}')
print(f'\nTrend Alert Distribution:')
print(df['trend_alert_weak'].value_counts().sort_index())
print(f'\nPercentage:')
print(df['trend_alert_weak'].value_counts(normalize=True).sort_index() * 100)
print(f'\nPositive Rate: {df["trend_alert_weak"].mean():.2%}')

if 'trend_trigger_count' in df.columns:
    print(f'\nTrigger Count Stats:')
    print(df['trend_trigger_count'].describe())
    print(f'\nTrigger Count Distribution:')
    print(df['trend_trigger_count'].value_counts().sort_index())
