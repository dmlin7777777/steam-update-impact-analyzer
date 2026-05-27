"""
FPS Sentiment Training with BERT
=================================
专门针对 FPS 游戏数据的 sentiment 训练脚本

基于诊断结果的优化配置：
- Full dataset (176k samples) 预期 Baseline 82-83%, BERT 85-87%
- Class weights 解决类别不平衡
- TimeSeriesSplit (5折) 时序惩罚 <5%
- N-gram (1,3) trigram 扩展
- 完全排除泄露特征

数据路径：
- features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet

预计时间：5-6 小时（5折 × 1小时/折）
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
    precision_score, recall_score, precision_recall_curve, auc
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = Path(r'C:\Users\12932\Desktop\nus\BAP')
DATA_PATH = BASE_DIR / 'features' / 'fps' / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet'
OUTPUT_DIR = BASE_DIR / 'analysis_results' / 'train' / 'fps_sentiment_bert'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model Parameters
MAX_LENGTH = 256  # BERT sequence length
BATCH_SIZE = 16   # Batch size for BERT training
BERT_EPOCHS = 5   # BERT training epochs
LEARNING_RATE = 2e-5

# CV Strategy
N_SPLITS = 5  # TimeSeriesSplit folds

# Text Features
TEXT_COLUMN = 'review_content_clean'  # Cleaned text (lowercase, normalized)
TEXT_COLUMN_PROCESSED = 'review_content_processed'  # Processed text (stopwords removed, for TF-IDF)
TARGET_COLUMN = 'sentiment_label'  # 0=negative, 1=neutral, 2=positive

# N-gram Configuration
NGRAM_RANGE = (1, 3)  # Unigram + Bigram + Trigram
MAX_FEATURES = 10000

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    """Load FPS data and perform basic validation"""
    print("\n" + "="*80)
    print("LOADING FPS DATA")
    print("="*80)
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH)
    print(f"✓ Loaded {len(df):,} samples from {DATA_PATH.name}")
    
    # Validate required columns
    required_cols = [TEXT_COLUMN, TEXT_COLUMN_PROCESSED, TARGET_COLUMN, 'timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Time range
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
        min_time = pd.to_datetime(df['timestamp'].min(), unit='s')
        max_time = pd.to_datetime(df['timestamp'].max(), unit='s')
        time_span = (max_time - min_time).days
        print(f"✓ Time range: {min_time.date()} to {max_time.date()} ({time_span} days)")
    
    # Class distribution
    class_dist = df[TARGET_COLUMN].value_counts().sort_index()
    print(f"\n✓ Class distribution:")
    labels_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    for label, count in class_dist.items():
        pct = count / len(df) * 100
        print(f"  {labels_map[label]:8s} ({label}): {count:7,} ({pct:5.2f}%)")
    
    # Handle missing text in both columns
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')
    df[TEXT_COLUMN_PROCESSED] = df[TEXT_COLUMN_PROCESSED].fillna('')
    
    # Remove samples where either text column is empty
    valid_mask = (df[TEXT_COLUMN].str.strip() != '') & (df[TEXT_COLUMN_PROCESSED].str.strip() != '')
    df = df[valid_mask].reset_index(drop=True)
    print(f"✓ After removing empty reviews: {len(df):,} samples")
    
    return df

# ============================================================================
# Baseline Model (TF-IDF + Logistic Regression)
# ============================================================================

def train_baseline(X_train, y_train, X_test, y_test):
    """Train baseline TF-IDF + Logistic Regression
    
    Uses review_content_processed (stopwords removed) for better TF-IDF performance
    """
    
    # Vectorization
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    
    # Train Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
        solver='liblinear',
        multi_class='ovr'
    )
    
    model.fit(X_train_vec, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)  # Probability outputs
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    # Per-class metrics
    per_class_f1 = f1_score(y_test, y_pred, average=None)  # [neg, neu, pos]
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    
    # PR-AUC for each class (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    pr_auc_scores = []
    for i in range(3):
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        pr_auc_scores.append(auc(recall_curve, precision_curve))
    
    return {
        'accuracy': acc,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1.tolist(),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'pr_auc': pr_auc_scores,  # [neg_auc, neu_auc, pos_auc]
        'model': model,
        'vectorizer': vectorizer
    }

# ============================================================================
# BERT Dataset
# ============================================================================

class SentimentDataset(Dataset):
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
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ============================================================================
# BERT Training
# ============================================================================

def train_bert_epoch(model, dataloader, optimizer, class_weights, device):
    """Train BERT for one epoch with class weights"""
    model.train()
    total_loss = 0
    
    # Move class weights to device
    class_weights = class_weights.to(device)
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Apply class weights to loss
        loss = outputs.loss
        weights = class_weights[labels]
        weighted_loss = (loss * weights).mean()
        
        weighted_loss.backward()
        optimizer.step()
        
        total_loss += weighted_loss.item()
    
    return total_loss / len(dataloader)

def evaluate_bert(model, dataloader, device):
    """Evaluate BERT model"""
    model.eval()
    predictions = []
    true_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels.numpy())
            all_logits.extend(logits.cpu().numpy())
    
    # Convert logits to probabilities
    from scipy.special import softmax
    y_proba = softmax(np.array(all_logits), axis=1)
    
    # Metrics
    acc = accuracy_score(true_labels, predictions)
    weighted_f1 = f1_score(true_labels, predictions, average='weighted')
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    
    # Per-class metrics
    per_class_f1 = f1_score(true_labels, predictions, average=None)  # [neg, neu, pos]
    per_class_precision = precision_score(true_labels, predictions, average=None, zero_division=0)
    per_class_recall = recall_score(true_labels, predictions, average=None, zero_division=0)
    
    # PR-AUC for each class (one-vs-rest)
    y_test_bin = label_binarize(true_labels, classes=[0, 1, 2])
    pr_auc_scores = []
    for i in range(3):
        precision_curve, recall_curve, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        pr_auc_scores.append(auc(recall_curve, precision_curve))
    
    return {
        'accuracy': acc,
        'weighted_f1': weighted_f1,
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1.tolist(),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'pr_auc': pr_auc_scores,  # [neg_auc, neu_auc, pos_auc]
        'predictions': predictions,
        'true_labels': true_labels
    }

def train_bert(X_train, y_train, X_test, y_test):
    """Train BERT model with TimeSeriesSplit"""
    
    print("\n  Loading BERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Suppress the "newly initialized" warning (this is expected for classification head)
    import logging
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Create datasets
    train_dataset = SentimentDataset(
        X_train.values, y_train.values, tokenizer, MAX_LENGTH
    )
    test_dataset = SentimentDataset(
        X_test.values, y_test.values, tokenizer, MAX_LENGTH
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    )
    model.to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"  Training BERT for {BERT_EPOCHS} epochs...")
    for epoch in range(BERT_EPOCHS):
        avg_loss = train_bert_epoch(model, train_loader, optimizer, class_weights, DEVICE)
        print(f"    Epoch {epoch+1}/{BERT_EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Evaluation
    results = evaluate_bert(model, test_loader, DEVICE)
    results['model'] = model
    results['tokenizer'] = tokenizer
    
    return results

# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """Main training pipeline"""
    
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"FPS SENTIMENT TRAINING - STARTED AT {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Load data
    df = load_data()
    
    # Prepare features: different text columns for different models
    X_bert = df[TEXT_COLUMN]  # Clean text for BERT (preserves grammar)
    X_baseline = df[TEXT_COLUMN_PROCESSED]  # Processed text for TF-IDF (stopwords removed)
    y = df[TARGET_COLUMN]
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    baseline_results = []
    bert_results = []
    
    print(f"\n{'='*80}")
    print(f"CROSS-VALIDATION ({N_SPLITS} FOLDS)")
    print(f"{'='*80}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_bert), 1):
        print(f"\n{'─'*80}")
        print(f"FOLD {fold_idx}/{N_SPLITS}")
        print(f"{'─'*80}")
        
        # Split data: different text for different models
        X_bert_train, X_bert_test = X_bert.iloc[train_idx], X_bert.iloc[test_idx]
        X_baseline_train, X_baseline_test = X_baseline.iloc[train_idx], X_baseline.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Distribution info
        train_dist = y_train.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()
        dist_shift = np.abs(train_dist - test_dist).mean() * 100
        
        print(f"Train: {len(X_bert_train):,} samples [{train_dist[0]:.1%}, {train_dist[1]:.1%}, {train_dist[2]:.1%}]")
        print(f"Test:  {len(X_bert_test):,} samples [{test_dist[0]:.1%}, {test_dist[1]:.1%}, {test_dist[2]:.1%}]")
        print(f"Distribution shift: {dist_shift:.2f}%")
        
        # Baseline (uses processed text)
        print(f"\n[Fold {fold_idx}] Training Baseline (TF-IDF + LR on processed text)...")
        baseline_res = train_baseline(X_baseline_train, y_train, X_baseline_test, y_test)
        baseline_results.append({
            'fold': fold_idx,
            'accuracy': baseline_res['accuracy'],
            'weighted_f1': baseline_res['weighted_f1'],
            'macro_f1': baseline_res['macro_f1'],
            'neg_f1': baseline_res['per_class_f1'][0],
            'neu_f1': baseline_res['per_class_f1'][1],
            'pos_f1': baseline_res['per_class_f1'][2],
            'neg_precision': baseline_res['per_class_precision'][0],
            'neu_precision': baseline_res['per_class_precision'][1],
            'pos_precision': baseline_res['per_class_precision'][2],
            'neg_recall': baseline_res['per_class_recall'][0],
            'neu_recall': baseline_res['per_class_recall'][1],
            'pos_recall': baseline_res['per_class_recall'][2],
            'neg_pr_auc': baseline_res['pr_auc'][0],
            'neu_pr_auc': baseline_res['pr_auc'][1],
            'pos_pr_auc': baseline_res['pr_auc'][2]
        })
        print(f"  → Acc: {baseline_res['accuracy']:.4f}, W-F1: {baseline_res['weighted_f1']:.4f}, M-F1: {baseline_res['macro_f1']:.4f}")
        print(f"  → Per-class F1: Neg={baseline_res['per_class_f1'][0]:.4f}, Neu={baseline_res['per_class_f1'][1]:.4f}, Pos={baseline_res['per_class_f1'][2]:.4f}")
        print(f"  → PR-AUC: Neg={baseline_res['pr_auc'][0]:.4f}, Neu={baseline_res['pr_auc'][1]:.4f}, Pos={baseline_res['pr_auc'][2]:.4f}")
        
        # BERT (uses clean text)
        print(f"\n[Fold {fold_idx}] Training BERT (DistilBERT on clean text)...")
        bert_res = train_bert(X_bert_train, y_train, X_bert_test, y_test)
        bert_results.append({
            'fold': fold_idx,
            'accuracy': bert_res['accuracy'],
            'weighted_f1': bert_res['weighted_f1'],
            'macro_f1': bert_res['macro_f1'],
            'neg_f1': bert_res['per_class_f1'][0],
            'neu_f1': bert_res['per_class_f1'][1],
            'pos_f1': bert_res['per_class_f1'][2],
            'neg_precision': bert_res['per_class_precision'][0],
            'neu_precision': bert_res['per_class_precision'][1],
            'pos_precision': bert_res['per_class_precision'][2],
            'neg_recall': bert_res['per_class_recall'][0],
            'neu_recall': bert_res['per_class_recall'][1],
            'pos_recall': bert_res['per_class_recall'][2],
            'neg_pr_auc': bert_res['pr_auc'][0],
            'neu_pr_auc': bert_res['pr_auc'][1],
            'pos_pr_auc': bert_res['pr_auc'][2]
        })
        print(f"  → Acc: {bert_res['accuracy']:.4f}, W-F1: {bert_res['weighted_f1']:.4f}, M-F1: {bert_res['macro_f1']:.4f}")
        print(f"  → Per-class F1: Neg={bert_res['per_class_f1'][0]:.4f}, Neu={bert_res['per_class_f1'][1]:.4f}, Pos={bert_res['per_class_f1'][2]:.4f}")
        print(f"  → PR-AUC: Neg={bert_res['pr_auc'][0]:.4f}, Neu={bert_res['pr_auc'][1]:.4f}, Pos={bert_res['pr_auc'][2]:.4f}")
        
        # Save fold models (last fold only to save space)
        if fold_idx == N_SPLITS:
            print(f"\n[Fold {fold_idx}] Saving final models...")
            
            # Save Baseline
            import joblib
            baseline_model_path = OUTPUT_DIR / 'baseline_model_fps.joblib'
            baseline_vec_path = OUTPUT_DIR / 'baseline_vectorizer_fps.joblib'
            joblib.dump(baseline_res['model'], baseline_model_path)
            joblib.dump(baseline_res['vectorizer'], baseline_vec_path)
            print(f"  ✓ Baseline saved to {baseline_model_path.name}")
            
            # Save BERT
            bert_model_dir = OUTPUT_DIR / 'bert_model_fps'
            bert_res['model'].save_pretrained(bert_model_dir)
            bert_res['tokenizer'].save_pretrained(bert_model_dir)
            print(f"  ✓ BERT saved to {bert_model_dir.name}/")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    
    # Baseline summary
    baseline_df = pd.DataFrame(baseline_results)
    print(f"\nBaseline (TF-IDF + LR):")
    print(f"  Accuracy:    {baseline_df['accuracy'].mean():.4f} ± {baseline_df['accuracy'].std():.4f}")
    print(f"  Weighted F1: {baseline_df['weighted_f1'].mean():.4f} ± {baseline_df['weighted_f1'].std():.4f}")
    print(f"  Macro F1:    {baseline_df['macro_f1'].mean():.4f} ± {baseline_df['macro_f1'].std():.4f}")
    print(f"  Per-class F1 / Precision / Recall / PR-AUC:")
    print(f"    Negative:  {baseline_df['neg_f1'].mean():.4f} / {baseline_df['neg_precision'].mean():.4f} / {baseline_df['neg_recall'].mean():.4f} / {baseline_df['neg_pr_auc'].mean():.4f}")
    print(f"    Neutral:   {baseline_df['neu_f1'].mean():.4f} / {baseline_df['neu_precision'].mean():.4f} / {baseline_df['neu_recall'].mean():.4f} / {baseline_df['neu_pr_auc'].mean():.4f}")
    print(f"    Positive:  {baseline_df['pos_f1'].mean():.4f} / {baseline_df['pos_precision'].mean():.4f} / {baseline_df['pos_recall'].mean():.4f} / {baseline_df['pos_pr_auc'].mean():.4f}")
    
    # BERT summary
    bert_df = pd.DataFrame(bert_results)
    print(f"\nBERT (DistilBERT):")
    print(f"  Accuracy:    {bert_df['accuracy'].mean():.4f} ± {bert_df['accuracy'].std():.4f}")
    print(f"  Weighted F1: {bert_df['weighted_f1'].mean():.4f} ± {bert_df['weighted_f1'].std():.4f}")
    print(f"  Macro F1:    {bert_df['macro_f1'].mean():.4f} ± {bert_df['macro_f1'].std():.4f}")
    print(f"  Per-class F1 / Precision / Recall / PR-AUC:")
    print(f"    Negative:  {bert_df['neg_f1'].mean():.4f} / {bert_df['neg_precision'].mean():.4f} / {bert_df['neg_recall'].mean():.4f} / {bert_df['neg_pr_auc'].mean():.4f}")
    print(f"    Neutral:   {bert_df['neu_f1'].mean():.4f} / {bert_df['neu_precision'].mean():.4f} / {bert_df['neu_recall'].mean():.4f} / {bert_df['neu_pr_auc'].mean():.4f}")
    print(f"    Positive:  {bert_df['pos_f1'].mean():.4f} / {bert_df['pos_precision'].mean():.4f} / {bert_df['pos_recall'].mean():.4f} / {bert_df['pos_pr_auc'].mean():.4f}")
    
    # Improvement
    improvement = (bert_df['weighted_f1'].mean() - baseline_df['weighted_f1'].mean()) * 100
    print(f"\nBERT Improvement: {improvement:+.2f}% weighted F1")
    
    # Save metrics
    metrics = {
        'dataset': 'fps',
        'n_samples': len(df),
        'n_splits': N_SPLITS,
        'baseline': {
            'mean_accuracy': float(baseline_df['accuracy'].mean()),
            'std_accuracy': float(baseline_df['accuracy'].std()),
            'mean_weighted_f1': float(baseline_df['weighted_f1'].mean()),
            'std_weighted_f1': float(baseline_df['weighted_f1'].std()),
            'mean_macro_f1': float(baseline_df['macro_f1'].mean()),
            'std_macro_f1': float(baseline_df['macro_f1'].std()),
            'per_fold': baseline_results
        },
        'bert': {
            'mean_accuracy': float(bert_df['accuracy'].mean()),
            'std_accuracy': float(bert_df['accuracy'].std()),
            'mean_weighted_f1': float(bert_df['weighted_f1'].mean()),
            'std_weighted_f1': float(bert_df['weighted_f1'].std()),
            'mean_macro_f1': float(bert_df['macro_f1'].mean()),
            'std_macro_f1': float(bert_df['macro_f1'].std()),
            'per_fold': bert_results
        },
        'improvement_pct': float(improvement),
        'config': {
            'max_length': MAX_LENGTH,
            'batch_size': BATCH_SIZE,
            'bert_epochs': BERT_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'ngram_range': NGRAM_RANGE,
            'max_features': MAX_FEATURES
        },
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_path = OUTPUT_DIR / 'metrics_fps_sentiment.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Training time
    end_time = datetime.now()
    duration = end_time - start_time
    hours = duration.total_seconds() / 3600
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED AT {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {hours:.2f} hours")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
