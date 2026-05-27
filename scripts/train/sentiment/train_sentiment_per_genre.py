"""
Per-genre sentiment training - trains separate models for fps/leisure/strategy.

Each genre gets:
- Independent model (BERT + Baseline)
- Genre-specific class weights
- Own metrics.json and model files
- Handles different temporal patterns and class distributions

Run this after temporal diagnosis shows high variance across genres.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import joblib
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Per-genre parquet files
GENRE_PARQUETS = {
    'fps': ROOT / r"features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet",
    'leisure': ROOT / r"features/leisure/gpu_optimized_features_leisure_exclflagged_enhanced_features_with_weaklabels.parquet",
    'strategy': ROOT / r"features/strategy/gpu_optimized_features_strategy_exclflagged_enhanced_features_with_weaklabels.parquet",
}

BASE_OUT_DIR = ROOT / r"analysis_results/train/per_genre_sentiment"
BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# GPU settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# BERT settings
MAX_LENGTH = 256
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 5
BERT_LR = 2e-5


def _make_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (str, type(None), bool, int, float)):
        return obj
    if isinstance(obj, np.generic):
        try:
            return obj.item()
        except Exception:
            return obj.tolist()
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, np.generic):
                try:
                    nk = k.item()
                except Exception:
                    nk = str(k)
            else:
                nk = k
            if not isinstance(nk, (str, int, float, bool, type(None))):
                nk = str(nk)
            out[nk] = _make_json_serializable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _make_json_serializable(obj.tolist())
    try:
        return obj.item()
    except Exception:
        try:
            return str(obj)
        except Exception:
            return None


class SentimentDataset(Dataset):
    """Dataset for BERT sentiment classification."""
    def __init__(self, texts, labels, tokenizer, max_length=256):
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
    """Evaluate BERT model and return predictions with probabilities."""
    model.eval()
    predictions = []
    true_labels = []
    all_logits = []
    
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
            all_logits.extend(logits.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels), np.array(all_logits)


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics including PR-AUC."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': float(acc),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(prec_weighted),
        'recall_weighted': float(rec_weighted),
        'f1_weighted': float(f1_weighted),
        'per_class_f1': per_class_f1.tolist(),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'confusion_matrix': cm.tolist(),
    }
    
    # Compute PR-AUC if probabilities are provided
    if y_proba is not None:
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        pr_auc_scores = []
        for i in range(3):
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            pr_auc_scores.append(float(auc(recall_curve, precision_curve)))
        metrics['pr_auc'] = pr_auc_scores  # [neg_auc, neu_auc, pos_auc]
    
    return metrics


def train_genre(genre: str, parquet_path: Path):
    """Train sentiment models for a single genre."""
    print(f"\n{'='*80}")
    print(f"TRAINING GENRE: {genre.upper()}")
    print(f"{'='*80}\n")
    
    # Create genre-specific output directory
    out_dir = BASE_OUT_DIR / genre
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_parquet(parquet_path)
    
    target = 'sentiment_label'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {genre} parquet.")
    
    # Drop NA targets
    mask = df[target].notna()
    df2 = df.loc[mask].copy()
    y = df2[target].astype(int).to_numpy()
    
    print(f"Total samples: {len(df2):,}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Sort by timestamp
    if 'timestamp' in df2.columns:
        df2 = df2.sort_values('timestamp').reset_index(drop=True)
        y = df2[target].astype(int).to_numpy()
        print(f"Time range: {df2['timestamp'].min()} - {df2['timestamp'].max()}")
    
    # Text column
    text_col = 'review_content_processed' if 'review_content_processed' in df2.columns else 'review_content'
    if text_col not in df2.columns:
        raise ValueError(f"No text column found in {genre}.")
    texts = df2[text_col].fillna('').astype(str).tolist()
    
    print(f"Using '{text_col}' for text features")
    
    # TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=5)
    
    baseline_metrics = []
    bert_metrics = []
    per_fold = []
    fold = 0
    
    # Initialize BERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    for train_idx, test_idx in tss.split(y):
        fold += 1
        print(f"\n{'-'*60}")
        print(f"Fold {fold}/5")
        print(f"{'-'*60}")
        
        y_train, y_test = y[train_idx], y[test_idx]
        texts_train = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]
        
        print(f"Train: {len(y_train):,} samples, Test: {len(y_test):,} samples")
        print(f"Train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        
        # --- Baseline: TF-IDF + LogisticRegression ---
        print("\n[Baseline] Training TF-IDF + Logistic Regression...")
        
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3)
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
        proba_base = clf_base.predict_proba(X_test_tfidf)
        base_res = compute_metrics(y_test, pred_base, proba_base)
        
        print(f"Baseline  - Acc: {base_res['accuracy']:.4f}, W-F1: {base_res['f1_weighted']:.4f}, M-F1: {base_res['f1_macro']:.4f}")
        if 'pr_auc' in base_res:
            print(f"            PR-AUC: Neg={base_res['pr_auc'][0]:.4f}, Neu={base_res['pr_auc'][1]:.4f}, Pos={base_res['pr_auc'][2]:.4f}")
        
        # --- BERT: DistilBERT fine-tuning ---
        print(f"\n[BERT] Training DistilBERT (device: {DEVICE})...")
        
        # Compute class weights
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
        print(f"Class weights: {dict(zip(np.unique(y_train), class_weights_np.round(3)))}")
        
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
        for epoch in range(BERT_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{BERT_EPOCHS}")
            train_loss = train_bert_epoch(model, train_loader, optimizer, scheduler, DEVICE, class_weights)
            print(f"Training loss: {train_loss:.4f}")
        
        # Evaluation
        pred_bert, true_bert, logits_bert = evaluate_bert(model, test_loader, DEVICE)
        
        # Convert logits to probabilities
        from scipy.special import softmax
        proba_bert = softmax(logits_bert, axis=1)
        
        bert_res = compute_metrics(true_bert, pred_bert, proba_bert)
        
        print(f"BERT      - Acc: {bert_res['accuracy']:.4f}, W-F1: {bert_res['f1_weighted']:.4f}, M-F1: {bert_res['f1_macro']:.4f}")
        if 'pr_auc' in bert_res:
            print(f"            PR-AUC: Neg={bert_res['pr_auc'][0]:.4f}, Neu={bert_res['pr_auc'][1]:.4f}, Pos={bert_res['pr_auc'][2]:.4f}")
        
        # Save fold results
        per_fold.append({
            'fold': fold,
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'class_counts_train': dict(zip(*np.unique(y_train, return_counts=True))),
            'class_counts_test': dict(zip(*np.unique(y_test, return_counts=True))),
            'baseline': base_res,
            'bert': bert_res,
        })
        
        baseline_metrics.append(base_res)
        bert_metrics.append(bert_res)
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Aggregate metrics
    def agg(mlist):
        out = {}
        keys = set().union(*[m.keys() for m in mlist])
        for k in keys:
            if k == 'confusion_matrix':
                continue
            vals = [m[k] for m in mlist if k in m and m[k] is not None]
            out[k] = float(np.mean(vals)) if vals else None
        return out
    
    metrics = {
        'genre': genre,
        'target': target,
        'n_rows': int(df.shape[0]),
        'n_samples_used': int(len(df2)),
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'bert_config': {
            'model': 'distilbert-base-uncased',
            'max_length': MAX_LENGTH,
            'batch_size': BERT_BATCH_SIZE,
            'epochs': BERT_EPOCHS,
            'learning_rate': BERT_LR,
            'device': str(DEVICE),
        },
        'baseline_cv': agg(baseline_metrics),
        'bert_cv': agg(bert_metrics),
        'per_fold': per_fold,
    }
    
    # Save metrics
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(_make_json_serializable(metrics), f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Summary for {genre.upper()} (CV Average)")
    print(f"{'='*60}")
    print(f"Baseline  - Acc: {metrics['baseline_cv']['accuracy']:.4f}, W-F1: {metrics['baseline_cv']['f1_weighted']:.4f}, M-F1: {metrics['baseline_cv']['f1_macro']:.4f}")
    if 'pr_auc' in metrics['baseline_cv'] and metrics['baseline_cv']['pr_auc']:
        pr_auc_base = metrics['baseline_cv']['pr_auc']
        print(f"            PR-AUC: Neg={pr_auc_base[0]:.4f}, Neu={pr_auc_base[1]:.4f}, Pos={pr_auc_base[2]:.4f}")
    print(f"BERT      - Acc: {metrics['bert_cv']['accuracy']:.4f}, W-F1: {metrics['bert_cv']['f1_weighted']:.4f}, M-F1: {metrics['bert_cv']['f1_macro']:.4f}")
    if 'pr_auc' in metrics['bert_cv'] and metrics['bert_cv']['pr_auc']:
        pr_auc_bert = metrics['bert_cv']['pr_auc']
        print(f"            PR-AUC: Neg={pr_auc_bert[0]:.4f}, Neu={pr_auc_bert[1]:.4f}, Pos={pr_auc_bert[2]:.4f}")
    print(f"\nMetrics saved to: {out_dir / 'metrics.json'}")
    
    # Train final models on full data
    print(f"\n{'-'*60}")
    print("Training final models on full dataset...")
    print(f"{'-'*60}")
    
    # Final baseline
    print("\n[Baseline] Training final model...")
    vectorizer_final = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=3)
    X_tfidf_full = vectorizer_final.fit_transform(texts)
    clf_base_final = LogisticRegression(
        max_iter=200,
        solver='liblinear',
        class_weight='balanced',
        random_state=RANDOM_STATE,
        multi_class='ovr'
    )
    clf_base_final.fit(X_tfidf_full, y)
    joblib.dump(
        {'vectorizer': vectorizer_final, 'model': clf_base_final}, 
        out_dir / f'baseline_model_{genre}.joblib'
    )
    print(f"✓ Baseline model saved to: {out_dir / f'baseline_model_{genre}.joblib'}")
    
    # Final BERT
    print(f"\n[BERT] Training final model (device: {DEVICE})...")
    
    # Compute class weights for full dataset
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_np_full = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_full = torch.tensor(class_weights_np_full, dtype=torch.float32)
    print(f"Class weights: {dict(zip(np.unique(y), class_weights_np_full.round(3)))}")
    
    full_dataset = SentimentDataset(texts, y, tokenizer, MAX_LENGTH)
    full_loader = DataLoader(full_dataset, batch_size=BERT_BATCH_SIZE, shuffle=True)
    
    model_final = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3
    ).to(DEVICE)
    
    optimizer_final = AdamW(model_final.parameters(), lr=BERT_LR, eps=1e-8)
    total_steps_final = len(full_loader) * BERT_EPOCHS
    scheduler_final = get_linear_schedule_with_warmup(
        optimizer_final,
        num_warmup_steps=int(0.1 * total_steps_final),
        num_training_steps=total_steps_final
    )
    
    for epoch in range(BERT_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{BERT_EPOCHS}")
        train_loss = train_bert_epoch(model_final, full_loader, optimizer_final, scheduler_final, DEVICE, class_weights_full)
        print(f"Training loss: {train_loss:.4f}")
    
    # Save BERT model
    bert_model_dir = out_dir / f'bert_model_{genre}'
    model_final.save_pretrained(bert_model_dir)
    tokenizer.save_pretrained(bert_model_dir)
    print(f"✓ BERT model saved to: {bert_model_dir}")
    
    print(f"\n✓ {genre.upper()} training complete!")
    return metrics


def main():
    print("="*80)
    print("PER-GENRE SENTIMENT TRAINING")
    print("="*80)
    print(f"\nTraining separate models for: {', '.join(GENRE_PARQUETS.keys())}")
    print(f"Output directory: {BASE_OUT_DIR}\n")
    
    all_metrics = {}
    
    for genre, parquet_path in GENRE_PARQUETS.items():
        if not parquet_path.exists():
            print(f"\n⚠️  Skipping {genre}: File not found - {parquet_path}")
            continue
        
        try:
            metrics = train_genre(genre, parquet_path)
            all_metrics[genre] = metrics
        except Exception as e:
            print(f"\n❌ Error training {genre}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined summary
    print(f"\n{'='*80}")
    print("ALL GENRES SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Genre':<12} {'Samples':>10} {'Baseline W-F1':>15} {'BERT W-F1':>12} {'Improvement':>12}")
    print("-" * 80)
    
    for genre, metrics in all_metrics.items():
        n_samples = metrics['n_samples_used']
        base_f1 = metrics['baseline_cv']['f1_weighted']
        bert_f1 = metrics['bert_cv']['f1_weighted']
        improvement = bert_f1 - base_f1
        
        print(f"{genre:<12} {n_samples:>10,} {base_f1:>15.4f} {bert_f1:>12.4f} {improvement:>+11.4f}")
    
    # Save combined report
    summary_path = BASE_OUT_DIR / 'all_genres_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(_make_json_serializable(all_metrics), f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ All genres trained!")
    print(f"✓ Summary saved to: {summary_path}")
    print(f"\nIndividual models and metrics saved to:")
    for genre in all_metrics.keys():
        print(f"  - {BASE_OUT_DIR / genre}")


if __name__ == '__main__':
    main()
