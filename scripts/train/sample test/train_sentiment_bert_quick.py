"""
Quick validation version of BERT sentiment training - single fold, smaller sample.
Use this to verify the class weights fix works before running full training.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
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
print(f"Using device: {DEVICE}")

# BERT settings - REDUCED for quick validation
MAX_LENGTH = 128  # Reduced from 256
BERT_BATCH_SIZE = 64  # Increased for faster training
BERT_EPOCHS = 2  # Reduced from 5
BERT_LR = 2e-5
SAMPLE_SIZE = 5000  # Use only 5k samples for quick validation


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
    
    # Move class weights to device once
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Apply class weights if provided
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
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)


def quick_validation():
    print(f"Quick Validation Mode - Using {SAMPLE_SIZE} samples, {BERT_EPOCHS} epochs")
    print(f"This should complete in ~5-10 minutes\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(PARQUET)
    
    target = 'sentiment_label'
    mask = df[target].notna()
    df2 = df.loc[mask].copy()
    
    # Sample for quick validation
    if len(df2) > SAMPLE_SIZE:
        df2 = df2.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
        print(f"Sampled {SAMPLE_SIZE} rows for quick validation")
    
    y = df2[target].astype(int).to_numpy()
    
    # Sort by timestamp to preserve time series nature
    if 'timestamp' in df2.columns:
        df2 = df2.sort_values('timestamp').reset_index(drop=True)
        y = df2[target].astype(int).to_numpy()
    
    print(f"Total samples: {len(df2)}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Text column
    text_col = 'review_content_processed' if 'review_content_processed' in df2.columns else 'review_content'
    texts = df2[text_col].fillna('').astype(str).tolist()
    
    # Single train/test split (80/20)
    train_idx, test_idx = train_test_split(
        range(len(y)), 
        test_size=0.2, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    y_train, y_test = y[train_idx], y[test_idx]
    texts_train = [texts[i] for i in train_idx]
    texts_test = [texts[i] for i in test_idx]
    
    print(f"\nTrain size: {len(y_train)}, Test size: {len(y_test)}")
    print(f"Train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # --- Baseline: TF-IDF + LogisticRegression ---
    print("\n" + "="*60)
    print("[Baseline] Training TF-IDF + Logistic Regression...")
    print("="*60)
    
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(texts_train)
    X_test_tfidf = vectorizer.transform(texts_test)
    
    clf_base = LogisticRegression(
        max_iter=200,
        solver='liblinear',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        multi_class='ovr'
    )
    clf_base.fit(X_train_tfidf, y_train)
    pred_base = clf_base.predict(X_test_tfidf)
    
    acc_base = accuracy_score(y_test, pred_base)
    f1_w_base = f1_score(y_test, pred_base, average='weighted')
    f1_m_base = f1_score(y_test, pred_base, average='macro')
    
    print(f"\nBaseline Results:")
    print(f"  Accuracy:    {acc_base:.4f}")
    print(f"  Weighted F1: {f1_w_base:.4f}")
    print(f"  Macro F1:    {f1_m_base:.4f}")
    print("\nPer-class metrics:")
    print(classification_report(y_test, pred_base, target_names=['negative', 'neutral', 'positive']))
    
    # --- BERT: DistilBERT fine-tuning ---
    print("\n" + "="*60)
    print(f"[BERT] Training DistilBERT (device: {DEVICE})...")
    print("="*60)
    
    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
    print(f"Class weights: {dict(zip(np.unique(y_train), class_weights_np))}")
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = SentimentDataset(texts_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = SentimentDataset(texts_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BERT_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BERT_BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    ).to(DEVICE)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=BERT_LR, eps=1e-8)
    total_steps = len(train_loader) * BERT_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nTraining for {BERT_EPOCHS} epochs...")
    for epoch in range(BERT_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{BERT_EPOCHS}")
        train_loss = train_bert_epoch(model, train_loader, optimizer, scheduler, DEVICE, class_weights)
        print(f"Training loss: {train_loss:.4f}")
    
    # Evaluation
    print("\nEvaluating on test set...")
    pred_bert, true_bert = evaluate_bert(model, test_loader, DEVICE)
    
    acc_bert = accuracy_score(true_bert, pred_bert)
    f1_w_bert = f1_score(true_bert, pred_bert, average='weighted')
    f1_m_bert = f1_score(true_bert, pred_bert, average='macro')
    
    print(f"\nBERT Results:")
    print(f"  Accuracy:    {acc_bert:.4f}")
    print(f"  Weighted F1: {f1_w_bert:.4f}")
    print(f"  Macro F1:    {f1_m_bert:.4f}")
    print("\nPer-class metrics:")
    print(classification_report(true_bert, pred_bert, target_names=['negative', 'neutral', 'positive']))
    
    # Comparison
    print("\n" + "="*60)
    print("SUMMARY - Quick Validation Results")
    print("="*60)
    print(f"{'Model':<15} {'Accuracy':>10} {'W-F1':>10} {'M-F1':>10}")
    print("-" * 60)
    print(f"{'Baseline':<15} {acc_base:>10.4f} {f1_w_base:>10.4f} {f1_m_base:>10.4f}")
    print(f"{'BERT':<15} {acc_bert:>10.4f} {f1_w_bert:>10.4f} {f1_m_bert:>10.4f}")
    print("-" * 60)
    
    improvement = f1_w_bert - f1_w_base
    print(f"\nBERT W-F1 improvement: {improvement:+.4f} ({improvement/f1_w_base*100:+.1f}%)")
    
    if f1_m_bert > f1_m_base:
        print("✓ Class weights are working - BERT is handling all classes better")
    else:
        print("⚠ BERT still struggling with class balance")
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    if f1_w_bert > 0.45:
        print("✓ Results look promising! Ready for full 5-fold CV training")
        print("  Run: python scripts/train/train_sentiment_bert.py")
    else:
        print("⚠ Performance still low. Consider:")
        print("  1. Check label quality (VADER may be noisy)")
        print("  2. Try using 'voted_up' as alternative label")
        print("  3. Increase epochs to 8-10")
        print("  4. Use original 'review_content' instead of processed text")


if __name__ == '__main__':
    quick_validation()
