"""
Prophet-based Trend Forecasting Features

目的：
- 使用 Prophet 预测情感基线，生成"偏差特征"
- 作为 LightGBM 的额外输入特征，增强异常检测能力

原理：
1. Prophet 学习历史趋势+季节性
2. 预测"正常情况下"的情感值
3. 实际值与预测值的偏差 → 异常信号

输出特征：
- prophet_baseline: Prophet 预测的情感基线
- prophet_deviation: 实际值 - 预测值
- prophet_lower_breach: 是否突破置信区间下界（布尔）
- prophet_upper_breach: 是否突破置信区间上界（布尔）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("⚠️  Prophet not installed. Run: pip install prophet")


def generate_prophet_features(df, sentiment_col='vader_compound', timestamp_col='timestamp'):
    """
    Generate Prophet-based anomaly detection features
    
    Args:
        df: DataFrame with timestamp and sentiment columns
        sentiment_col: Column name for sentiment scores
        timestamp_col: Column name for timestamps
    
    Returns:
        DataFrame with additional Prophet features
    """
    if not PROPHET_AVAILABLE:
        print("⚠️  Prophet not available, returning empty features")
        return pd.DataFrame({
            'prophet_baseline': [0] * len(df),
            'prophet_deviation': [0] * len(df),
            'prophet_lower_breach': [0] * len(df),
            'prophet_upper_breach': [0] * len(df)
        })
    
    print("\n" + "="*80)
    print("GENERATING PROPHET BASELINE FEATURES")
    print("="*80)
    
    # Prepare data for Prophet (daily aggregation)
    df_sorted = df.sort_values(timestamp_col).copy()
    df_sorted['date'] = pd.to_datetime(df_sorted[timestamp_col]).dt.date
    
    # Daily average sentiment
    daily_sentiment = df_sorted.groupby('date')[sentiment_col].agg(['mean', 'count']).reset_index()
    daily_sentiment.columns = ['ds', 'y', 'count']
    daily_sentiment['ds'] = pd.to_datetime(daily_sentiment['ds'])
    
    print(f"\n📊 Daily Sentiment Aggregation:")
    print(f"   Total days: {len(daily_sentiment)}")
    print(f"   Date range: {daily_sentiment['ds'].min()} to {daily_sentiment['ds'].max()}")
    print(f"   Avg reviews per day: {daily_sentiment['count'].mean():.1f}")
    
    # Minimum data requirement
    # if len(daily_sentiment) < 30:
    #     print(f"\n⚠️  WARNING: Only {len(daily_sentiment)} days of data (< 30)")
    #     print("   Prophet may not work well. Returning zero features.")
    #     return pd.DataFrame({
    #         'prophet_baseline': [0] * len(df),
    #         'prophet_deviation': [0] * len(df),
    #         'prophet_lower_breach': [0] * len(df),
    #         'prophet_upper_breach': [0] * len(df)
    #     })
    
    # Train Prophet (suppress output)
    print("\n🔮 Training Prophet model...")
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True if len(daily_sentiment) > 365 else False,
        interval_width=0.95,  # 95% confidence interval
        changepoint_prior_scale=0.05,  # Lower = smoother trend
        seasonality_prior_scale=10.0
    )
    
    # Fit on 80% of data (to avoid overfitting to recent data)
    train_size = int(len(daily_sentiment) * 0.8)
    train_data = daily_sentiment.iloc[:train_size]
    
    model.fit(train_data)
    print(f"   Trained on {len(train_data)} days")
    
    # Predict for all dates (including "future" = validation set)
    forecast = model.predict(daily_sentiment[['ds']])
    
    # Merge back to daily data
    daily_sentiment = daily_sentiment.merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
        on='ds', 
        how='left'
    )
    
    # Calculate deviations
    daily_sentiment['deviation'] = daily_sentiment['y'] - daily_sentiment['yhat']
    daily_sentiment['lower_breach'] = (daily_sentiment['y'] < daily_sentiment['yhat_lower']).astype(int)
    daily_sentiment['upper_breach'] = (daily_sentiment['y'] > daily_sentiment['yhat_upper']).astype(int)
    
    print(f"\n📈 Prophet Forecast Stats:")
    print(f"   Mean baseline: {daily_sentiment['yhat'].mean():.4f}")
    print(f"   Mean deviation: {daily_sentiment['deviation'].mean():.4f} ± {daily_sentiment['deviation'].std():.4f}")
    print(f"   Lower breaches: {daily_sentiment['lower_breach'].sum()} days ({daily_sentiment['lower_breach'].mean()*100:.1f}%)")
    print(f"   Upper breaches: {daily_sentiment['upper_breach'].sum()} days ({daily_sentiment['upper_breach'].mean()*100:.1f}%)")
    
    # Map back to original reviews (broadcast daily values to all reviews on that day)
    df_sorted['date'] = pd.to_datetime(df_sorted[timestamp_col]).dt.date
    
    prophet_features = df_sorted[['date']].merge(
        daily_sentiment[['ds', 'yhat', 'deviation', 'lower_breach', 'upper_breach']],
        left_on='date',
        right_on=daily_sentiment['ds'].dt.date,
        how='left'
    )
    
    # Handle missing values (if any)
    prophet_features = prophet_features.fillna({
        'yhat': daily_sentiment['yhat'].mean(),
        'deviation': 0,
        'lower_breach': 0,
        'upper_breach': 0
    })
    
    # Return as DataFrame (maintain original order)
    result = pd.DataFrame({
        'prophet_baseline': prophet_features['yhat'].values,
        'prophet_deviation': prophet_features['deviation'].values,
        'prophet_lower_breach': prophet_features['lower_breach'].values,
        'prophet_upper_breach': prophet_features['upper_breach'].values
    })
    
    print(f"\n✅ Generated {result.shape[1]} Prophet features for {len(result)} reviews")
    
    return result


def add_prophet_features_to_data(df, sentiment_col='vader_compound', timestamp_col='timestamp'):
    """
    Convenience function to add Prophet features to existing DataFrame
    """
    prophet_feats = generate_prophet_features(df, sentiment_col, timestamp_col)
    
    # Concatenate
    df_enhanced = pd.concat([df.reset_index(drop=True), prophet_feats], axis=1)
    
    return df_enhanced


if __name__ == "__main__":
    # Test with sample data
    from pathlib import Path
    
    BASE_DIR = Path(r'c:\Users\12932\Desktop\nus\BAP')
    FEATURES_DIR = BASE_DIR / 'features' / 'fps'
    
    feature_file = r'C:\Users\12932\Desktop\nus\BAP\test\fps\gpu_optimized_features_fps_inclflagged_enhanced_features_with_weaklabels.parquet'
    
    # if feature_file.exists():
    print("Testing Prophet feature generation...")
    df = pd.read_parquet(feature_file)
        
        # Generate features
    df_enhanced = add_prophet_features_to_data(df)
        
    print(f"\n✅ Original shape: {df.shape}")
    print(f"✅ Enhanced shape: {df_enhanced.shape}")
    print(f"\nNew columns:")
    print(df_enhanced[['prophet_baseline', 'prophet_deviation', 'prophet_lower_breach', 'prophet_upper_breach']].describe())
    # else:
        # print(f"Feature file not found: {feature_file}")
