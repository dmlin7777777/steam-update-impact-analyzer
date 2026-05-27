"""
Extract LDA Topic Keywords from FPS Data
=========================================
从 FPS 数据中提取 8 个 LDA 主题的关键词

输出：
1. 每个主题的 top-10 关键词
2. 主题 → topic_category 映射
3. 主题分布统计
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter

# Paths
BASE_DIR = Path(r'C:\Users\12932\Desktop\nus\BAP')
DATA_PATH = BASE_DIR / 'features' / 'fps' / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet'
OUTPUT_PATH = BASE_DIR / 'analysis_results' / 'data_summary' / 'lda_topic_keywords.json'
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def extract_topic_keywords_from_reviews(df, n_keywords=10):
    """从评论文本中提取每个主题的高频词作为关键词"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    topic_keywords = {}
    
    for topic_id in range(8):
        # 获取该主题的所有评论
        topic_reviews = df[df['dominant_topic'] == topic_id]['review_content_processed'].dropna()
        
        if len(topic_reviews) < 10:
            topic_keywords[topic_id] = []
            continue
        
        # 使用 TF-IDF 提取关键词
        vectorizer = TfidfVectorizer(
            max_features=n_keywords,
            stop_words='english',
            ngram_range=(1, 2),  # unigram + bigram
            min_df=5
        )
        
        try:
            vectorizer.fit(topic_reviews)
            keywords = vectorizer.get_feature_names_out()
            topic_keywords[topic_id] = keywords.tolist()
        except Exception as e:
            print(f"  Warning: Failed to extract keywords for topic {topic_id}: {e}")
            topic_keywords[topic_id] = []
    
    return topic_keywords

def main():
    print("="*80)
    print("EXTRACTING LDA TOPIC KEYWORDS FROM FPS DATA")
    print("="*80)
    
    # Load data
    print(f"\nLoading data from {DATA_PATH.name}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"✓ Loaded {len(df):,} samples")
    
    # Topic distribution
    print("\n" + "="*80)
    print("TOPIC DISTRIBUTION")
    print("="*80)
    topic_dist = df['dominant_topic'].value_counts().sort_index()
    for topic_id, count in topic_dist.items():
        pct = count / len(df) * 100
        print(f"Topic {topic_id}: {count:7,} samples ({pct:5.2f}%)")
    
    # Topic → Category mapping
    print("\n" + "="*80)
    print("TOPIC → CATEGORY MAPPING")
    print("="*80)
    topic_category_map = df.groupby('dominant_topic')['topic_category'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
    ).to_dict()
    
    for topic_id, category in topic_category_map.items():
        count = (df['dominant_topic'] == topic_id).sum()
        pct = count / len(df) * 100
        print(f"Topic {topic_id} → {category:30s} ({count:6,}, {pct:5.2f}%)")
    
    # Extract keywords using TF-IDF
    print("\n" + "="*80)
    print("EXTRACTING TOPIC KEYWORDS (TF-IDF)")
    print("="*80)
    topic_keywords = extract_topic_keywords_from_reviews(df, n_keywords=15)
    
    # Display keywords
    print("\n" + "="*80)
    print("TOP KEYWORDS PER TOPIC")
    print("="*80)
    
    for topic_id in range(8):
        category = topic_category_map.get(topic_id, 'unknown')
        count = (df['dominant_topic'] == topic_id).sum()
        pct = count / len(df) * 100
        keywords = topic_keywords.get(topic_id, [])
        
        print(f"\nTopic {topic_id}: {category.upper()}")
        print(f"  Samples: {count:,} ({pct:.2f}%)")
        print(f"  Keywords: {', '.join(keywords[:10])}")
    
    # Analyze topic probability features
    print("\n" + "="*80)
    print("ANALYZING TOPIC PROBABILITY FEATURES")
    print("="*80)
    
    topic_prob_cols = [f'topic_{i}_prob' for i in range(8)]
    if all(col in df.columns for col in topic_prob_cols):
        print("\nAverage topic probabilities:")
        for topic_id in range(8):
            avg_prob = df[f'topic_{topic_id}_prob'].mean()
            print(f"  Topic {topic_id}: {avg_prob:.4f}")
    
    # Save results
    results = {
        'n_topics': 8,
        'n_samples': len(df),
        'topic_distribution': {int(k): int(v) for k, v in topic_dist.items()},
        'topic_category_mapping': {int(k): v for k, v in topic_category_map.items()},
        'topic_keywords': {int(k): v for k, v in topic_keywords.items()},
        'topic_details': []
    }
    
    for topic_id in range(8):
        category = topic_category_map.get(topic_id, 'unknown')
        count = int((df['dominant_topic'] == topic_id).sum())
        pct = float(count / len(df) * 100)
        keywords = topic_keywords.get(topic_id, [])
        
        results['topic_details'].append({
            'topic_id': topic_id,
            'category': category,
            'count': count,
            'percentage': round(pct, 2),
            'top_keywords': keywords[:10]
        })
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to {OUTPUT_PATH}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Topic':6s} {'Category':30s} {'Count':>10s} {'%':>6s} {'Top Keywords'}")
    print("-" * 100)
    for detail in results['topic_details']:
        keywords_str = ', '.join(detail['top_keywords'][:5])
        print(f"{detail['topic_id']:6d} {detail['category']:30s} {detail['count']:10,} {detail['percentage']:5.1f}% {keywords_str}")

if __name__ == '__main__':
    main()
