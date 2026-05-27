"""
Compare TimeSeriesSplit vs StratifiedKFold to diagnose temporal split impact.

This script runs BOTH split strategies on the same data to show:
1. How much performance loss is due to temporal distribution shift
2. Whether the model can generalize to future time periods
3. If we should use temporal or random split for production

Quick validation mode: 5k samples, 2 epochs, single genre (fps)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

ROOT = Path(r"c:/Users/12932/Desktop/nus/BAP")
PARQUET = ROOT / r"features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet"

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# GPU settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Quick validation settings
MAX_LENGTH = 128
BERT_BATCH_SIZE = 32
BERT_EPOCHS = 2
BERT_LR = 2e-5
SAMPLE_SIZE = 10000  # Use 10k for better signal


class SentimentDataset(Dataset):
    """Dataset for BERT sentiment classification."""
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_bert_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    """Train BERT model for one epoch."""
    model.train()
    total_loss = 0
    
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        if class_weights is not None:
            weights = class_weights[labels]
            loss = (loss * weights).mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_bert(model, dataloader, device):
    """Evaluate BERT model."""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)


def run_baseline_cv(texts, y, cv_splitter, split_name):
    """Run Baseline (TF-IDF + LR) with given CV strategy."""
    print(f"\n[Baseline - {split_name}] Running 3-fold CV...")
    
    fold_f1s = []
    fold = 0
    
    for train_idx, test_idx in cv_splitter.split(texts if isinstance(texts, np.ndarray) else range(len(texts)), y):
        fold += 1
        
        if isinstance(texts, list):
            texts_train = [texts[i] for i in train_idx]
            texts_test = [texts[i] for i in test_idx]
        else:
            texts_train = texts[train_idx]
            texts_test = texts[test_idx]
        
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check temporal shift (if sorted by time)
        if hasattr(cv_splitter, 'n_splits'):  # TimeSeriesSplit
            train_classes = dict(zip(*np.unique(y_train, return_counts=True)))
            test_classes = dict(zip(*np.unique(y_test, return_counts=True)))
            print(f"  Fold {fold}: Train {train_classes} → Test {test_classes}")
        
        vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
        X_train_tfidf = vectorizer.fit_transform(texts_train)
        X_test_tfidf = vectorizer.transform(texts_test)
        
        clf = LogisticRegression(
            max_iter=200,
            solver='liblinear',
            class_weight='balanced',
            random_state=RANDOM_STATE,
            multi_class='ovr'
        )
        clf.fit(X_train_tfidf, y_train)
        pred = clf.predict(X_test_tfidf)
        
        f1_w = f1_score(y_test, pred, average='weighted')
        fold_f1s.append(f1_w)
        print(f"    Fold {fold} W-F1: {f1_w:.4f}")
    
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    print(f"  → {split_name} Baseline Mean W-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    return mean_f1, std_f1


def run_bert_cv(texts, y, cv_splitter, split_name, tokenizer):
    """Run BERT with given CV strategy."""
    print(f"\n[BERT - {split_name}] Running 3-fold CV...")
    
    fold_f1s = []
    fold = 0
    
    for train_idx, test_idx in cv_splitter.split(texts if isinstance(texts, np.ndarray) else range(len(texts)), y):
        fold += 1
        print(f"\n  Fold {fold}/{cv_splitter.n_splits}")
        
        if isinstance(texts, list):
            texts_train = [texts[i] for i in train_idx]
            texts_test = [texts[i] for i in test_idx]
        else:
            texts_train = texts[train_idx]
            texts_test = texts[test_idx]
        
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
        
        # Datasets
        train_dataset = SentimentDataset(texts_train, y_train, tokenizer, MAX_LENGTH)
        test_dataset = SentimentDataset(texts_test, y_test, tokenizer, MAX_LENGTH)
        
        train_loader = DataLoader(train_dataset, batch_size=BERT_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BERT_BATCH_SIZE, shuffle=False)
        
        # Model
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=3
        ).to(DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=BERT_LR, eps=1e-8)
        total_steps = len(train_loader) * BERT_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Train
        for epoch in range(BERT_EPOCHS):
            train_loss = train_bert_epoch(model, train_loader, optimizer, scheduler, DEVICE, class_weights)
            print(f"    Epoch {epoch+1}/{BERT_EPOCHS}: loss={train_loss:.4f}")
        
        # Evaluate
        pred_bert, true_bert = evaluate_bert(model, test_loader, DEVICE)
        f1_w = f1_score(true_bert, pred_bert, average='weighted')
        fold_f1s.append(f1_w)
        print(f"    Fold {fold} W-F1: {f1_w:.4f}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    print(f"  → {split_name} BERT Mean W-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    return mean_f1, std_f1


def main():
    print("="*80)
    print("TEMPORAL SPLIT IMPACT ANALYSIS")
    print("="*80)
    print(f"\nComparing TimeSeriesSplit vs StratifiedKFold")
    print(f"Sample size: {SAMPLE_SIZE:,}, BERT epochs: {BERT_EPOCHS}, Device: {DEVICE}\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(PARQUET)
    
    target = 'sentiment_label'
    mask = df[target].notna()
    df2 = df.loc[mask].copy()
    
    # Sort by timestamp BEFORE sampling (critical for TimeSeriesSplit)
    if 'timestamp' in df2.columns:
        df2 = df2.sort_values('timestamp').reset_index(drop=True)
    
    # Sample (take last N samples to get recent data)
    if len(df2) > SAMPLE_SIZE:
        df2 = df2.tail(SAMPLE_SIZE).reset_index(drop=True)
        print(f"Using last {SAMPLE_SIZE:,} samples (most recent data)")
    
    y = df2[target].astype(int).to_numpy()
    
    print(f"Total samples: {len(df2):,}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    if 'timestamp' in df2.columns:
        df2['datetime'] = pd.to_datetime(df2['timestamp'], unit='s')
        print(f"Time range: {df2['datetime'].min()} to {df2['datetime'].max()}")
    
    # Text
    text_col = 'review_content_processed' if 'review_content_processed' in df2.columns else 'review_content'
    texts = df2[text_col].fillna('').astype(str).tolist()
    
    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # CV strategies (use 3 folds for speed)
    tss = TimeSeriesSplit(n_splits=3)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    results = {}
    
    # ===== TimeSeriesSplit =====
    print("\n" + "="*80)
    print("STRATEGY 1: TimeSeriesSplit (Train on past, test on future)")
    print("="*80)
    
    results['tss_baseline_mean'], results['tss_baseline_std'] = run_baseline_cv(texts, y, tss, "TimeSeries")
    results['tss_bert_mean'], results['tss_bert_std'] = run_bert_cv(texts, y, tss, "TimeSeries", tokenizer)
    
    # ===== StratifiedKFold =====
    print("\n" + "="*80)
    print("STRATEGY 2: StratifiedKFold (Random split with balanced classes)")
    print("="*80)
    
    results['skf_baseline_mean'], results['skf_baseline_std'] = run_baseline_cv(texts, y, skf, "Stratified")
    results['skf_bert_mean'], results['skf_bert_std'] = run_bert_cv(texts, y, skf, "Stratified", tokenizer)
    
    # ===== Comparison =====
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n{'Strategy':<20} {'Model':<12} {'Mean W-F1':>12} {'Std':>8} {'vs Baseline':>12}")
    print("-" * 80)
    
    print(f"{'TimeSeriesSplit':<20} {'Baseline':<12} {results['tss_baseline_mean']:>12.4f} {results['tss_baseline_std']:>8.4f} {'-':>12}")
    print(f"{'TimeSeriesSplit':<20} {'BERT':<12} {results['tss_bert_mean']:>12.4f} {results['tss_bert_std']:>8.4f} {results['tss_bert_mean'] - results['tss_baseline_mean']:>+11.4f}")
    print("-" * 80)
    print(f"{'StratifiedKFold':<20} {'Baseline':<12} {results['skf_baseline_mean']:>12.4f} {results['skf_baseline_std']:>8.4f} {'-':>12}")
    print(f"{'StratifiedKFold':<20} {'BERT':<12} {results['skf_bert_mean']:>12.4f} {results['skf_bert_std']:>8.4f} {results['skf_bert_mean'] - results['skf_baseline_mean']:>+11.4f}")
    print("-" * 80)
    
    # Analysis
    temporal_penalty_baseline = results['skf_baseline_mean'] - results['tss_baseline_mean']
    temporal_penalty_bert = results['skf_bert_mean'] - results['tss_bert_mean']
    
    print(f"\n{'TEMPORAL PENALTY (Performance loss due to time-based split)':}")
    print(f"  Baseline: {temporal_penalty_baseline:+.4f} ({temporal_penalty_baseline/results['skf_baseline_mean']*100:+.1f}%)")
    print(f"  BERT:     {temporal_penalty_bert:+.4f} ({temporal_penalty_bert/results['skf_bert_mean']*100:+.1f}%)")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION & RECOMMENDATION")
    print(f"{'='*80}\n")
    
    if temporal_penalty_baseline > 0.10:
        print("⚠️  HIGH TEMPORAL PENALTY DETECTED (>10% drop)")
        print("\nThis means:")
        print("  - Your data has STRONG temporal distribution shift")
        print("  - Models trained on past data struggle with future data")
        print("  - This is the REAL production scenario")
        print("\nRecommendations:")
        print("  ✓ Use TimeSeriesSplit for evaluation (realistic)")
        print("  ✓ Retrain model periodically (e.g., monthly)")
        print("  ✓ Monitor performance drift in production")
        print("  ✓ Consider online learning or active learning")
    elif temporal_penalty_baseline > 0.05:
        print("⚠️  MODERATE TEMPORAL PENALTY (5-10% drop)")
        print("\nRecommendations:")
        print("  ✓ Use TimeSeriesSplit for conservative estimates")
        print("  ✓ Retrain quarterly or when performance drops")
        print("  ✓ Add temporal features (time since release, etc.)")
    else:
        print("✓ LOW TEMPORAL PENALTY (<5% drop)")
        print("\nYour data is relatively stable over time.")
        print("Both TimeSeriesSplit and StratifiedKFold are acceptable.")
    
    bert_benefit_tss = results['tss_bert_mean'] - results['tss_baseline_mean']
    bert_benefit_skf = results['skf_bert_mean'] - results['skf_baseline_mean']
    
    print(f"\n{'BERT vs Baseline:'}")
    if bert_benefit_tss > 0.05:
        print(f"  ✓ BERT provides {bert_benefit_tss:+.4f} improvement (TimeSeriesSplit)")
        print("  → BERT is worth the complexity")
    else:
        print(f"  ⚠️  BERT only provides {bert_benefit_tss:+.4f} improvement (TimeSeriesSplit)")
        print("  → Consider sticking with Baseline (simpler, faster)")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}\n")
    
    if temporal_penalty_baseline > 0.10:
        print("1. Run full 5-fold TimeSeriesSplit on all genres")
        print("2. Accept lower (but realistic) performance")
        print("3. Set up monitoring for temporal drift")
        print("4. Plan for regular retraining")
    else:
        print("1. You can use either split strategy")
        print("2. TimeSeriesSplit is more conservative (recommended)")
        print("3. Run full training: python scripts/train/train_sentiment_per_genre.py")


if __name__ == '__main__':
    main()
