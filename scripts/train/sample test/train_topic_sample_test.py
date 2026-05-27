"""
FPS Topic Classification - Quick Sample Test
=============================================
快速样本测试：对比 8-class vs 6-class 性能，Baseline vs Advanced

配置:
- 采样: 5,000 样本 (per task, 减少以支持 BERT)
- 模型: 
  * Baseline: TF-IDF + LogisticRegression
  * Advanced: DistilBERT (2 epochs, 快速测试)
- CV: 3折 TimeSeriesSplit
- 预计时间: 15-20 分钟（含 BERT 训练）

目的:
1. 验证 LDA 映射的 6-class 性能
2. 对比 Baseline vs BERT 性能提升
3. 检查少数类 (monetization_concerns) 的性能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    precision_score, recall_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from scipy.special import softmax
from tqdm import tqdm
import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(r'C:\Users\12932\Desktop\nus\BAP')
DATA_PATH = BASE_DIR / 'features' / 'fps' / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet'
OUTPUT_DIR = BASE_DIR / 'analysis_results' / 'train' / 'fps_topic_sample_test'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sampling
SAMPLE_SIZE = 5000  # 5k samples per task (reduced for BERT)
RANDOM_STATE = 42

# Model - Baseline
MAX_FEATURES = 5000  # Reduced for speed
NGRAM_RANGE = (1, 2)  # Unigram + Bigram
N_SPLITS = 3  # 3-fold CV

# Model - BERT
MAX_LENGTH = 256
BATCH_SIZE = 16
BERT_EPOCHS = 2  # Quick test
LEARNING_RATE = 2e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Text columns
TEXT_COLUMN = 'review_content_processed'  # Use processed text for TF-IDF
TEXT_COLUMN_RAW = 'review_content_clean'   # Use clean text for BERT
TARGET_8CLASS = 'dominant_topic'
TARGET_6CLASS = 'topic_category'  # Now 6 classes after LDA mapping

print("="*80)
print("FPS TOPIC CLASSIFICATION - SAMPLE TEST (Baseline + BERT)")
print("="*80)
print(f"Sample size: {SAMPLE_SIZE:,} per task")
print(f"CV folds: {N_SPLITS}")
print(f"Baseline: TF-IDF (max {MAX_FEATURES}, {NGRAM_RANGE}) + LogisticRegression")
print(f"Advanced: DistilBERT ({BERT_EPOCHS} epochs)")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Data Preprocessing - LDA-based Topic Mapping
# ============================================================================

def remap_topics_by_lda_keywords(df):
    """
    基于 LDA 关键词重新映射主题类别
    
    合并规则：
    1. Topic 1 (cod) + Topic 3 (csgo) → matchmaking_issues
    2. Topic 5 (cheater) + Topic 6 (anti cheat) → cheating
    
    最终 6 个类别
    """
    if 'dominant_topic' not in df.columns:
        return df
    
    # Backup original
    if 'topic_category' in df.columns:
        df['topic_category_original'] = df['topic_category'].copy()
    
    # LDA topic → new category mapping
    topic_mapping = {
        0: 'multiplayer_features',
        1: 'matchmaking_issues',
        2: 'monetization_concerns',
        3: 'matchmaking_issues',
        4: 'technical_issues',
        5: 'cheating',
        6: 'cheating',
        7: 'user_interface'
    }
    
    # Apply mapping
    df['topic_category'] = df['dominant_topic'].map(topic_mapping)
    df['topic_category'] = df['topic_category'].fillna('other')
    
    return df

# ============================================================================
# Data Loading
# ============================================================================

def load_and_sample_data(target_column, sample_size):
    """Load data and stratified sample"""
    print(f"\n{'='*80}")
    print(f"LOADING DATA: {target_column}")
    print(f"{'='*80}")
    
    df = pd.read_parquet(DATA_PATH)
    print(f"✓ Total samples: {len(df):,}")
    
    # Apply LDA-based topic remapping
    df = remap_topics_by_lda_keywords(df)
    
    # Remove empty text
    df = df[df[TEXT_COLUMN].notna() & (df[TEXT_COLUMN].str.strip() != '')].copy()
    print(f"✓ After removing empty: {len(df):,}")
    
    # Sort by timestamp for time-based split
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Stratified sampling (maintain class distribution)
    if len(df) > sample_size:
        df_sample = df.groupby(target_column, group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df))), random_state=RANDOM_STATE)
        ).reset_index(drop=True)
        
        # If still too many, randomly sample
        if len(df_sample) > sample_size:
            df_sample = df_sample.sample(sample_size, random_state=RANDOM_STATE)
        
        print(f"✓ Sampled: {len(df_sample):,} (stratified)")
    else:
        df_sample = df
        print(f"✓ Using all data: {len(df_sample):,}")
    
    # Re-sort by timestamp
    df_sample = df_sample.sort_values('timestamp').reset_index(drop=True)
    
    return df_sample

def encode_target(df, target_column):
    """Encode target column"""
    if target_column == TARGET_8CLASS:
        # Already numeric (0-7)
        df['target'] = df[target_column].astype(int)
        label_mapping = {i: f'topic_{i}' for i in range(8)}
        n_classes = 8
    else:  # TARGET_5CLASS
        # Need encoding
        label_encoder = LabelEncoder()
        df['target'] = label_encoder.fit_transform(df[target_column])
        label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        n_classes = len(label_encoder.classes_)
    
    # Class distribution
    print(f"\n✓ Class distribution ({target_column}):")
    class_dist = df['target'].value_counts().sort_index()
    for idx, count in class_dist.items():
        pct = count / len(df) * 100
        label_name = label_mapping[idx]
        print(f"  [{idx}] {label_name:30s}: {count:5,} ({pct:5.2f}%)")
    
    return df, label_mapping, n_classes

# ============================================================================
# Training
# ============================================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, n_classes):
    """Train TF-IDF + LR and evaluate"""
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words='english',
        min_df=2
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Train
    clf = LogisticRegression(
        max_iter=500,
        class_weight=class_weight_dict,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=RANDOM_STATE
    )
    clf.fit(X_train_vec, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_vec)
    
    # Metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'per_class_f1': f1_score(y_test, y_pred, average=None, zero_division=0).tolist(),
        'per_class_precision': precision_score(y_test, y_pred, average=None, zero_division=0).tolist(),
        'per_class_recall': recall_score(y_test, y_pred, average=None, zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    return metrics

# ============================================================================
# BERT Model
# ============================================================================

class ReviewDataset(Dataset):
    """Dataset for BERT"""
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_bert_epoch(model, dataloader, optimizer, device):
    """Train BERT for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate_bert(model, dataloader, device):
    """Evaluate BERT"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def train_and_evaluate_bert(X_train, y_train, X_test, y_test, n_classes):
    """Train DistilBERT and evaluate"""
    
    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Datasets
    train_dataset = ReviewDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = ReviewDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=n_classes
    ).to(DEVICE)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Train
    for epoch in range(BERT_EPOCHS):
        train_loss = train_bert_epoch(model, train_loader, optimizer, DEVICE)
        print(f"    Epoch {epoch+1}/{BERT_EPOCHS}: Loss = {train_loss:.4f}")
    
    # Evaluate
    y_pred, y_true = evaluate_bert(model, test_loader, DEVICE)
    
    # Metrics
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'per_class_f1': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'per_class_precision': precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'per_class_recall': recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    return metrics

def run_cv_test(df, target_column, label_mapping, n_classes, use_bert=False):
    """Run 3-fold CV test"""
    model_type = "BERT" if use_bert else "Baseline"
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION: {target_column} ({model_type})")
    print(f"{'='*80}")
    
    if use_bert:
        X = df[TEXT_COLUMN_RAW].values  # Use clean text for BERT
    else:
        X = df[TEXT_COLUMN].values  # Use processed text for TF-IDF
    
    y = df['target'].values
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/{N_SPLITS}: Train={len(train_idx):,}, Test={len(test_idx):,}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if use_bert:
            metrics = train_and_evaluate_bert(X_train, y_train, X_test, y_test, n_classes)
        else:
            metrics = train_and_evaluate(X_train, y_train, X_test, y_test, n_classes)
        
        fold_metrics.append(metrics)
        
        print(f"  Acc: {metrics['accuracy']:.4f}, W-F1: {metrics['f1_weighted']:.4f}, M-F1: {metrics['f1_macro']:.4f}")
    
    # Aggregate
    avg_metrics = {
        'accuracy': {'mean': np.mean([m['accuracy'] for m in fold_metrics]),
                    'std': np.std([m['accuracy'] for m in fold_metrics])},
        'f1_weighted': {'mean': np.mean([m['f1_weighted'] for m in fold_metrics]),
                       'std': np.std([m['f1_weighted'] for m in fold_metrics])},
        'f1_macro': {'mean': np.mean([m['f1_macro'] for m in fold_metrics]),
                    'std': np.std([m['f1_macro'] for m in fold_metrics])},
    }
    
    # Per-class averages
    avg_metrics['per_class_f1'] = {
        'mean': np.mean([m['per_class_f1'] for m in fold_metrics], axis=0).tolist(),
        'std': np.std([m['per_class_f1'] for m in fold_metrics], axis=0).tolist()
    }
    
    # Display summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {target_column}")
    print(f"{'='*80}")
    print(f"Accuracy:    {avg_metrics['accuracy']['mean']:.4f} ± {avg_metrics['accuracy']['std']:.4f}")
    print(f"Weighted F1: {avg_metrics['f1_weighted']['mean']:.4f} ± {avg_metrics['f1_weighted']['std']:.4f}")
    print(f"Macro F1:    {avg_metrics['f1_macro']['mean']:.4f} ± {avg_metrics['f1_macro']['std']:.4f}")
    
    print(f"\nPer-class F1:")
    for idx in range(n_classes):
        label = label_mapping[idx]
        mean_f1 = avg_metrics['per_class_f1']['mean'][idx]
        std_f1 = avg_metrics['per_class_f1']['std'][idx]
        print(f"  [{idx}] {label:30s}: {mean_f1:.4f} ± {std_f1:.4f}")
    
    return avg_metrics, fold_metrics

# ============================================================================
# Main
# ============================================================================

def main():
    results = {}
    
    # ========================================================================
    # Task A: 8-class (dominant_topic)
    # ========================================================================
    print("\n" + "#"*80)
    print("TASK A: 8-CLASS CLASSIFICATION (dominant_topic)")
    print("#"*80)
    
    df_8class = load_and_sample_data(TARGET_8CLASS, SAMPLE_SIZE)
    df_8class, label_map_8, n_classes_8 = encode_target(df_8class, TARGET_8CLASS)
    
    # Baseline
    print("\n" + "-"*80)
    print("BASELINE: TF-IDF + LogisticRegression")
    print("-"*80)
    avg_8_baseline, folds_8_baseline = run_cv_test(df_8class, TARGET_8CLASS, label_map_8, n_classes_8, use_bert=False)
    
    # BERT
    print("\n" + "-"*80)
    print("ADVANCED: DistilBERT")
    print("-"*80)
    avg_8_bert, folds_8_bert = run_cv_test(df_8class, TARGET_8CLASS, label_map_8, n_classes_8, use_bert=True)
    
    results['8_class'] = {
        'target': TARGET_8CLASS,
        'n_samples': len(df_8class),
        'n_classes': n_classes_8,
        'label_mapping': label_map_8,
        'baseline': {'cv_metrics': avg_8_baseline, 'fold_metrics': folds_8_baseline},
        'bert': {'cv_metrics': avg_8_bert, 'fold_metrics': folds_8_bert}
    }
    
    # ========================================================================
    # Task B: 6-class (topic_category with LDA mapping)
    # ========================================================================
    print("\n" + "#"*80)
    print("TASK B: 6-CLASS CLASSIFICATION (topic_category, LDA-mapped)")
    print("#"*80)
    
    df_6class = load_and_sample_data(TARGET_6CLASS, SAMPLE_SIZE)
    df_6class, label_map_6, n_classes_6 = encode_target(df_6class, TARGET_6CLASS)
    
    # Baseline
    print("\n" + "-"*80)
    print("BASELINE: TF-IDF + LogisticRegression")
    print("-"*80)
    avg_6_baseline, folds_6_baseline = run_cv_test(df_6class, TARGET_6CLASS, label_map_6, n_classes_6, use_bert=False)
    
    # BERT
    print("\n" + "-"*80)
    print("ADVANCED: DistilBERT")
    print("-"*80)
    avg_6_bert, folds_6_bert = run_cv_test(df_6class, TARGET_6CLASS, label_map_6, n_classes_6, use_bert=True)
    
    results['6_class'] = {
        'target': TARGET_6CLASS,
        'n_samples': len(df_6class),
        'n_classes': n_classes_6,
        'label_mapping': label_map_6,
        'baseline': {'cv_metrics': avg_6_baseline, 'fold_metrics': folds_6_baseline},
        'bert': {'cv_metrics': avg_6_bert, 'fold_metrics': folds_6_bert}
    }
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    print(f"\n{'Task':<30s} {'Model':<15s} {'W-F1':<12s} {'M-F1':<12s} {'Acc':<12s}")
    print("-"*80)
    
    # 8-class
    print(f"{'8-class (dominant_topic)':<30s} {'Baseline':<15s} {avg_8_baseline['f1_weighted']['mean']:>6.4f} ± {avg_8_baseline['f1_weighted']['std']:<4.4f} {avg_8_baseline['f1_macro']['mean']:>6.4f} ± {avg_8_baseline['f1_macro']['std']:<4.4f} {avg_8_baseline['accuracy']['mean']:>6.4f}")
    print(f"{'':30s} {'BERT':<15s} {avg_8_bert['f1_weighted']['mean']:>6.4f} ± {avg_8_bert['f1_weighted']['std']:<4.4f} {avg_8_bert['f1_macro']['mean']:>6.4f} ± {avg_8_bert['f1_macro']['std']:<4.4f} {avg_8_bert['accuracy']['mean']:>6.4f}")
    
    # 6-class
    print(f"{'6-class (LDA-mapped)':<30s} {'Baseline':<15s} {avg_6_baseline['f1_weighted']['mean']:>6.4f} ± {avg_6_baseline['f1_weighted']['std']:<4.4f} {avg_6_baseline['f1_macro']['mean']:>6.4f} ± {avg_6_baseline['f1_macro']['std']:<4.4f} {avg_6_baseline['accuracy']['mean']:>6.4f}")
    print(f"{'':30s} {'BERT':<15s} {avg_6_bert['f1_weighted']['mean']:>6.4f} ± {avg_6_bert['f1_weighted']['std']:<4.4f} {avg_6_bert['f1_macro']['mean']:>6.4f} ± {avg_6_bert['f1_macro']['std']:<4.4f} {avg_6_bert['accuracy']['mean']:>6.4f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Best overall
    best_wf1 = max(
        avg_8_baseline['f1_weighted']['mean'],
        avg_8_bert['f1_weighted']['mean'],
        avg_6_baseline['f1_weighted']['mean'],
        avg_6_bert['f1_weighted']['mean']
    )
    
    if best_wf1 == avg_8_bert['f1_weighted']['mean']:
        best_config = "8-class + BERT"
    elif best_wf1 == avg_6_bert['f1_weighted']['mean']:
        best_config = "6-class + BERT"
    elif best_wf1 == avg_8_baseline['f1_weighted']['mean']:
        best_config = "8-class + Baseline"
    else:
        best_config = "6-class + Baseline"
    
    print(f"\n1. ✅ 最佳配置: {best_config} (W-F1 = {best_wf1:.4f})")
    
    # BERT improvement
    bert_improve_8 = avg_8_bert['f1_weighted']['mean'] - avg_8_baseline['f1_weighted']['mean']
    bert_improve_6 = avg_6_bert['f1_weighted']['mean'] - avg_6_baseline['f1_weighted']['mean']
    
    print(f"\n2. 🚀 BERT 性能提升:")
    print(f"   8-class: +{bert_improve_8:.4f} ({bert_improve_8/avg_8_baseline['f1_weighted']['mean']*100:+.1f}%)")
    print(f"   6-class: +{bert_improve_6:.4f} ({bert_improve_6/avg_6_baseline['f1_weighted']['mean']*100:+.1f}%)")
    
    # Class balance
    print(f"\n3. ⚖️ 类别平衡:")
    print(f"   8-class: {n_classes_8} 个类别 (21.5% ~ 4.7%)")
    print(f"   6-class: {n_classes_6} 个类别 (26.2% ~ 4.7%, LDA-mapped)")
    
    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print("="*80)
    
    if avg_6_bert['f1_weighted']['mean'] > avg_8_bert['f1_weighted']['mean']:
        print("✅ 推荐: 6-class (LDA-mapped) + DistilBERT")
        print(f"   - Weighted F1: {avg_6_bert['f1_weighted']['mean']:.4f}")
        print(f"   - Macro F1: {avg_6_bert['f1_macro']['mean']:.4f}")
        print("   - 类别平衡且业务可解释")
        print("   - matchmaking_issues, cheating, multiplayer_features, ...")
    else:
        print("✅ 推荐: 8-class + DistilBERT")
        print(f"   - Weighted F1: {avg_8_bert['f1_weighted']['mean']:.4f}")
        print(f"   - Macro F1: {avg_8_bert['f1_macro']['mean']:.4f}")
        print("   - 最平衡的类别分布")
    
    # Save results
    output_path = OUTPUT_DIR / 'sample_test_results_with_bert.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == '__main__':
    main()
