"""
Diagnose temporal distribution issues across multiple games/genres.

Checks:
1. Time range for each genre
2. Class distribution over time
3. Overlap/gap analysis
4. VADER sentiment stability over time
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

ROOT = Path(r"c:/Users/12932/Desktop/nus/BAP")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARQUET_FILES = {
    'fps': ROOT / r"features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet",
    'leisure': ROOT / r"features/leisure/gpu_optimized_features_leisure_exclflagged_enhanced_features_with_weaklabels.parquet",
    'strategy': ROOT / r"features/strategy/gpu_optimized_features_strategy_exclflagged_enhanced_features_with_weaklabels.parquet",
}

OUT_DIR = ROOT / r"analysis_results/train/temporal_diagnosis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def diagnose_genre(genre: str, parquet_path: Path):
    """Diagnose temporal patterns for a single genre."""
    print(f"\n{'='*60}")
    print(f"Genre: {genre.upper()}")
    print(f"{'='*60}")
    
    df = pd.read_parquet(parquet_path)
    
    # Basic stats
    print(f"\nTotal samples: {len(df):,}")
    
    # Check for timestamp
    if 'timestamp' not in df.columns:
        print("⚠️  No 'timestamp' column found!")
        return None
    
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Time range
    min_date = df['datetime'].min()
    max_date = df['datetime'].max()
    time_span = (max_date - min_date).days
    
    print(f"\nTime Range:")
    print(f"  Start: {min_date}")
    print(f"  End:   {max_date}")
    print(f"  Span:  {time_span} days ({time_span/365.25:.1f} years)")
    
    # Sentiment label distribution
    target = 'sentiment_label'
    if target in df.columns:
        mask = df[target].notna()
        df_valid = df.loc[mask].copy()
        
        print(f"\nSentiment Label Distribution:")
        counts = df_valid[target].value_counts().sort_index()
        for label, count in counts.items():
            label_name = ['negative', 'neutral', 'positive'][int(label)]
            pct = count / len(df_valid) * 100
            print(f"  {label_name:8s} ({label}): {count:6,} ({pct:5.2f}%)")
        
        # Temporal drift analysis: split into 5 time bins
        df_valid = df_valid.sort_values('datetime')
        bin_size = len(df_valid) // 5
        
        print(f"\nTemporal Drift Analysis (5 equal-size bins):")
        print(f"{'Bin':<10} {'Date Range':<35} {'Negative %':>12} {'Neutral %':>12} {'Positive %':>12}")
        print("-" * 85)
        
        for i in range(5):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < 4 else len(df_valid)
            bin_df = df_valid.iloc[start_idx:end_idx]
            
            bin_start = bin_df['datetime'].min().strftime('%Y-%m-%d')
            bin_end = bin_df['datetime'].max().strftime('%Y-%m-%d')
            
            dist = bin_df[target].value_counts(normalize=True).sort_index()
            neg_pct = dist.get(0, 0) * 100
            neu_pct = dist.get(1, 0) * 100
            pos_pct = dist.get(2, 0) * 100
            
            print(f"Bin {i+1:1d}     {bin_start} - {bin_end}   {neg_pct:11.2f}%  {neu_pct:11.2f}%  {pos_pct:11.2f}%")
        
        # VADER sentiment drift (if available)
        if 'vader_compound' in df_valid.columns:
            print(f"\nVADER Compound Score over time:")
            print(f"{'Bin':<10} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
            print("-" * 50)
            
            for i in range(5):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < 4 else len(df_valid)
                bin_df = df_valid.iloc[start_idx:end_idx]
                
                vader_mean = bin_df['vader_compound'].mean()
                vader_std = bin_df['vader_compound'].std()
                vader_min = bin_df['vader_compound'].min()
                vader_max = bin_df['vader_compound'].max()
                
                print(f"Bin {i+1:1d}     {vader_mean:10.4f} {vader_std:10.4f} {vader_min:10.4f} {vader_max:10.4f}")
    
    return {
        'genre': genre,
        'n_samples': len(df),
        'min_date': min_date,
        'max_date': max_date,
        'time_span_days': time_span,
    }


def analyze_cross_genre_overlap(summary_list):
    """Analyze time overlap between genres."""
    print(f"\n{'='*60}")
    print("Cross-Genre Temporal Overlap Analysis")
    print(f"{'='*60}\n")
    
    for i, s1 in enumerate(summary_list):
        for s2 in summary_list[i+1:]:
            overlap_start = max(s1['min_date'], s2['min_date'])
            overlap_end = min(s1['max_date'], s2['max_date'])
            
            if overlap_start < overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                print(f"{s1['genre']:8s} ∩ {s2['genre']:8s}: {overlap_days:5,} days ({overlap_start.date()} - {overlap_end.date()})")
            else:
                print(f"{s1['genre']:8s} ∩ {s2['genre']:8s}: NO OVERLAP")


def recommend_strategy(summary_list):
    """Recommend training strategy based on diagnosis."""
    print(f"\n{'='*60}")
    print("RECOMMENDED TRAINING STRATEGY")
    print(f"{'='*60}\n")
    
    # Check time span variance
    time_spans = [s['time_span_days'] for s in summary_list]
    max_span = max(time_spans)
    min_span = min(time_spans)
    span_ratio = max_span / min_span if min_span > 0 else float('inf')
    
    print(f"Time span variance: {min_span} - {max_span} days (ratio: {span_ratio:.1f}x)")
    
    if span_ratio > 3:
        print("\n⚠️  HIGH TEMPORAL VARIANCE detected!")
        print("\nOption 1 (RECOMMENDED): Genre-Specific Models")
        print("  - Train separate model for each genre")
        print("  - Pros: Each model handles genre-specific temporal patterns")
        print("  - Cons: Need to maintain 3 models")
        print("  - Implementation: Use existing per-genre parquet files")
        
        print("\nOption 2: Time-Aligned Training")
        print("  - Use only the overlapping time period")
        print("  - Pros: Fair comparison, consistent distribution")
        print("  - Cons: Discard a lot of data")
        
        print("\nOption 3: Multi-Task Learning with Genre Embedding")
        print("  - Add genre as a feature/embedding")
        print("  - Train single model with genre-aware attention")
        print("  - Pros: Leverage all data, genre transfer learning")
        print("  - Cons: More complex architecture")
    else:
        print("\n✓ Temporal variance is acceptable")
        print("\nOption 1 (RECOMMENDED): Pooled Training")
        print("  - Combine all genres, sort by timestamp")
        print("  - Use TimeSeriesSplit for CV")
        print("  - Add 'appid' or 'genre' as categorical feature")
        
        print("\nOption 2: Stratified by Genre")
        print("  - Ensure each CV fold has balanced genre representation")
        print("  - Use GroupTimeSeriesSplit (group by genre)")
    
    print(f"\n{'='*60}")
    print("IMPLEMENTATION GUIDE")
    print(f"{'='*60}\n")
    
    print("For QUICK TEST (Option B - Diagnose before full training):")
    print("  1. Run this diagnosis script ✓ (you're here)")
    print("  2. Check temporal drift in each genre (above)")
    print("  3. Decide: Pooled vs Per-Genre training")
    print("  4. Run quick validation on chosen strategy")
    print("  5. If good, run full 5-fold CV")
    
    print("\nNext immediate steps:")
    print("  A. Train per-genre models (safest, recommended for production)")
    print("     - Use fps/leisure/strategy parquets separately")
    print("     - Each gets its own metrics.json")
    print("  ")
    print("  B. Create pooled dataset with genre embedding")
    print("     - Combine all 3 parquets")
    print("     - Add 'genre_fps', 'genre_leisure', 'genre_strategy' one-hot features")
    print("     - Train single model on 50w samples")


def main():
    print("Temporal Distribution Diagnosis for Multi-Genre Data")
    print("=" * 60)
    
    summary_list = []
    
    for genre, parquet_path in PARQUET_FILES.items():
        if parquet_path.exists():
            summary = diagnose_genre(genre, parquet_path)
            if summary:
                summary_list.append(summary)
        else:
            print(f"\n⚠️  File not found: {parquet_path}")
    
    if len(summary_list) > 1:
        analyze_cross_genre_overlap(summary_list)
    
    recommend_strategy(summary_list)
    
    print(f"\n{'='*60}")
    print(f"Diagnosis complete! Check output above for recommendations.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
