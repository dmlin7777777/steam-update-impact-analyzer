"""
Sample Test: Trend Alert with Prophet Features
Quick validation with small dataset before full training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import Prophet feature generator
try:
    from trend_prophet_features import generate_prophet_features
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet features not available")

# Configuration
SAMPLE_SIZE = 5000  # Small sample for quick test
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
FEATURE_DIR = BASE_DIR / 'features' / 'fps'
INPUT_FILE = FEATURE_DIR / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet'


def load_sample_data():
    """
    Load small sample of FPS data for testing
    """
    print("\n" + "="*80)
    print("LOADING SAMPLE DATA")
    print("="*80)
    
    print(f"\n📁 Reading: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    
    print(f"   Total records: {len(df):,}")
    
    # Check for weak labels
    if 'trend_alert_weak' not in df.columns:
        print("\n⚠️  WARNING: 'trend_alert_weak' column not found!")
        print("   Please run generate_trend_weak_labels_fps.py first")
        return None
    
    # Sample stratified by weak labels
    positive_samples = df[df['trend_alert_weak'] == 1].sample(
        n=min(int(SAMPLE_SIZE * 0.1), (df['trend_alert_weak'] == 1).sum()),
        random_state=RANDOM_SEED
    )
    negative_samples = df[df['trend_alert_weak'] == 0].sample(
        n=min(SAMPLE_SIZE - len(positive_samples), (df['trend_alert_weak'] == 0).sum()),
        random_state=RANDOM_SEED
    )
    
    df_sample = pd.concat([positive_samples, negative_samples]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"\n✅ Sample Size: {len(df_sample):,}")
    print(f"   Positive: {(df_sample['trend_alert_weak'] == 1).sum():,} ({(df_sample['trend_alert_weak'] == 1).mean():.2%})")
    print(f"   Negative: {(df_sample['trend_alert_weak'] == 0).sum():,} ({(df_sample['trend_alert_weak'] == 0).mean():.2%})")
    
    return df_sample


def test_prophet_features(df):
    """
    Test Prophet feature generation on sample data
    """
    print("\n" + "="*80)
    print("TESTING PROPHET FEATURE GENERATION")
    print("="*80)
    
    if not PROPHET_AVAILABLE:
        print("\n⚠️  Prophet not available - skipping Prophet test")
        return df
    
    # Check required columns
    required_cols = ['vader_compound', 'timestamp']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        print(f"\n⚠️  Missing required columns: {missing}")
        return df
    
    # Check date range
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    date_range = (df['timestamp'].max() - df['timestamp'].min()).days
    print(f"\n📅 Date Range: {date_range} days")
    print(f"   Start: {df['timestamp'].min()}")
    print(f"   End: {df['timestamp'].max()}")
    
    if date_range < 30:
        print(f"\n⚠️  Date range too short for Prophet ({date_range} < 30 days)")
        print(f"   Prophet features will be zero-filled")
    
    # Generate Prophet features
    print(f"\n🔮 Generating Prophet features...")
    try:
        prophet_features = generate_prophet_features(
            df,
            sentiment_col='vader_compound',
            timestamp_col='timestamp'
        )
        
        # Add to dataframe
        for col in prophet_features.columns:
            df[col] = prophet_features[col]
        
        print(f"\n✅ Prophet Features Generated:")
        print(f"\n{'Feature':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 73)
        
        for col in prophet_features.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            min_val = df[col].min()
            max_val = df[col].max()
            print(f"{col:<25} {mean_val:>11.4f} {std_val:>11.4f} {min_val:>11.4f} {max_val:>11.4f}")
        
        # Check breach rates
        if 'prophet_lower_breach' in df.columns:
            lower_rate = df['prophet_lower_breach'].mean()
            upper_rate = df['prophet_upper_breach'].mean()
            total_breach = (df['prophet_lower_breach'] | df['prophet_upper_breach']).mean()
            
            print(f"\n📊 Breach Statistics:")
            print(f"   Lower Breach: {lower_rate:.2%} (expected ~2.5%)")
            print(f"   Upper Breach: {upper_rate:.2%} (expected ~2.5%)")
            print(f"   Total Breach: {total_breach:.2%} (expected ~5%)")
        
        # Check deviation distribution
        if 'prophet_deviation' in df.columns:
            print(f"\n📈 Deviation Distribution:")
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            print(f"   Percentile: " + " ".join([f"{p:>6}%" for p in percentiles]))
            print(f"   Value:      " + " ".join([f"{df['prophet_deviation'].quantile(p/100):>7.3f}" for p in percentiles]))
        
        # Alert correlation
        if 'trend_alert_weak' in df.columns:
            print(f"\n🎯 Correlation with Trend Alert:")
            for col in prophet_features.columns:
                corr = df[col].corr(df['trend_alert_weak'])
                print(f"   {col:<25}: {corr:>7.4f}")
        
        return df
        
    except Exception as e:
        print(f"\n❌ Prophet feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        return df


def check_trend_features(df):
    """
    Check availability of trend features
    """
    print("\n" + "="*80)
    print("CHECKING TREND FEATURES")
    print("="*80)
    
    expected_features = [
        'sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h',
        'comment_rate_change_24h', 'comment_rate_change_48h', 'comment_rate_change_72h',
        'ensemble_anomaly_score', 'vader_compound', 'is_toxic'
    ]
    
    available = [f for f in expected_features if f in df.columns]
    missing = [f for f in expected_features if f not in df.columns]
    
    print(f"\n✅ Available: {len(available)}/{len(expected_features)} features")
    
    if missing:
        print(f"\n⚠️  Missing features:")
        for f in missing:
            print(f"   - {f}")
        print(f"\n   Run trend_anomaly_analysis.py to generate missing features")
    
    # Check Prophet features
    prophet_features = [col for col in df.columns if col.startswith('prophet_')]
    if prophet_features:
        print(f"\n🔮 Prophet features: {len(prophet_features)}")
        for f in prophet_features:
            print(f"   - {f}")
    else:
        print(f"\n⚠️  No Prophet features found")
    
    return available


def main():
    print("="*80)
    print("SAMPLE TEST: TREND ALERT WITH PROPHET FEATURES")
    print("="*80)
    print(f"Sample Size: {SAMPLE_SIZE:,}")
    print(f"Random Seed: {RANDOM_SEED}")
    
    # Load sample
    df = load_sample_data()
    
    if df is None:
        print("\n❌ Failed to load data")
        return
    
    # Test Prophet features
    df = test_prophet_features(df)
    
    # Check all trend features
    available_features = check_trend_features(df)
    
    # Summary
    print("\n" + "="*80)
    print("SAMPLE TEST SUMMARY")
    print("="*80)
    
    print(f"\n✅ Sample loaded: {len(df):,} reviews")
    print(f"✅ Alert rate: {df['trend_alert_weak'].mean():.2%}")
    print(f"✅ Trend features: {len(available_features)}")
    
    prophet_features = [col for col in df.columns if col.startswith('prophet_')]
    if prophet_features:
        print(f"✅ Prophet features: {len(prophet_features)}")
    else:
        print(f"⚠️  No Prophet features generated")
    
    # Check if ready for training
    if len(available_features) >= 5 and 'trend_alert_weak' in df.columns:
        print(f"\n🎯 Ready for Training!")
        print(f"   Run: python scripts/train/train_trend_alert_fps.py")
    else:
        print(f"\n⚠️  Not ready for training")
        print(f"   Generate trend features first: python scripts/feature/trend_anomaly_analysis.py")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
