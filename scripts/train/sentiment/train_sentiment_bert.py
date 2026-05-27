from __future__ import annotations
import json
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
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

PARQUET = ROOT / r"features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet"
OUT_DIR = ROOT / r"analysis_results/train/ab_weaklabels/sentiment_bert"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# GPU settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# BERT settings
MAX_LENGTH = 256
BERT_BATCH_SIZE = 16
BERT_EPOCHS = 5  # Increased from 3 to allow better convergence
BERT_LR = 2e-5

# Try import LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    LGB_AVAILABLE = False


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


def select_feature_columns_sentiment(df: pd.DataFrame) -> list[str]:
    """Select numerical features excluding sentiment-related and ID columns."""
    drop_like = {
        'sentiment_label', 'risk_label', 'risk_label_weak', 'trend_alert_weak',
        'anomaly_label_weak', 'is_anomaly_weak', 'is_coordinated', 'is_coordinated_auto',
        'appid', 'SteamID', 'review_id', 'review_content', 'review_content_processed',
        'review_datetime', 'timestamp',
        # exclude all sentiment-derived features to avoid leakage
        'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
        'vader_compound_z', 'sentiment_x_reputation', 'sentiment_x_bug',
        'controversial_sentiment', 'recommendation_sentiment_mismatch',
        'sentiment_rolling_mean_24h', 'sentiment_rolling_std_24h',
        'sentiment_rolling_min_24h', 'sentiment_rolling_max_24h',
        'sentiment_rolling_mean_48h', 'sentiment_rolling_std_48h',
        'sentiment_rolling_min_48h', 'sentiment_rolling_max_48h',
        'sentiment_rolling_mean_72h', 'sentiment_rolling_std_72h',
        'sentiment_rolling_min_72h', 'sentiment_rolling_max_72h',
        'sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h',
        'sentiment_acceleration_24h', 'sentiment_acceleration_48h', 'sentiment_acceleration_72h',
        'sentiment_trend_direction', 'sentiment_volatility',
        'is_sentiment_local_peak', 'is_sentiment_local_trough',
        'avg_sentiment_in_24h', 'avg_sentiment_in_48h', 'avg_sentiment_in_72h',
        'sentiment_category', 'negative_but_helpful', 'positive_but_unhelpful',
    }
    nums = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in nums if c not in drop_like]
    return features


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


def compute_metrics(y_true, y_pred):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': float(acc),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(prec_weighted),
        'recall_weighted': float(rec_weighted),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
    }


def train_sentiment():
    print("Loading data...")
    df = pd.read_parquet(PARQUET)
    
    target = 'sentiment_label'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in parquet.")
    
    # Drop NA targets
    mask = df[target].notna()
    df2 = df.loc[mask].copy()
    y = df2[target].astype(int).to_numpy()
    
    print(f"Total samples: {len(df2)}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Device: {DEVICE}")
    
    # Sort by timestamp
    if 'timestamp' in df2.columns:
        df2 = df2.sort_values('timestamp').reset_index(drop=True)
    
    # Text column
    text_col = 'review_content_processed' if 'review_content_processed' in df2.columns else 'review_content'
    if text_col not in df2.columns:
        raise ValueError(f"No text column found.")
    texts = df2[text_col].fillna('').astype(str).tolist()
    
    # Numerical features for advanced
    Xcols = select_feature_columns_sentiment(df2)
    X_num = df2[Xcols].to_numpy()
    
    print(f"Using {len(Xcols)} numerical features for Advanced model")
    print(f"Using '{text_col}' for Baseline and BERT models")
    
    # TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=5)
    
    baseline_metrics = []
    advanced_metrics = []
    bert_metrics = []
    per_fold = []
    fold = 0
    
    # Initialize BERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    for train_idx, test_idx in tss.split(X_num):
        fold += 1
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")
        
        y_train, y_test = y[train_idx], y[test_idx]
        
        # --- Baseline: TF-IDF (trigram) + LogisticRegression ---
        print("\n[Baseline] Training TF-IDF + Logistic Regression...")
        texts_train = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]
        
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
        base_res = compute_metrics(y_test, pred_base)
        
        print(f"Baseline  - Acc: {base_res['accuracy']:.4f}, W-F1: {base_res['f1_weighted']:.4f}, M-F1: {base_res['f1_macro']:.4f}")
        
        # --- Advanced: Numerical features + LightGBM ---
        print("\n[Advanced] Training LightGBM with numerical features...")
        X_train_num, X_test_num = X_num[train_idx], X_num[test_idx]
        
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train_adv = scaler.fit_transform(imputer.fit_transform(X_train_num))
        X_test_adv = scaler.transform(imputer.transform(X_test_num))
        
        if LGB_AVAILABLE:
            clf_adv = lgb.LGBMClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                verbose=-1
            )
        else:
            clf_adv = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        
        clf_adv.fit(X_train_adv, y_train)
        pred_adv = clf_adv.predict(X_test_adv)
        adv_res = compute_metrics(y_test, pred_adv)
        
        print(f"Advanced  - Acc: {adv_res['accuracy']:.4f}, W-F1: {adv_res['f1_weighted']:.4f}, M-F1: {adv_res['f1_macro']:.4f}")
        
        # --- BERT: DistilBERT fine-tuning ---
        print(f"\n[BERT] Training DistilBERT (device: {DEVICE})...")
        
        # Compute class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        class_weights_np = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights_np, dtype=torch.float32)
        print(f"Class weights: {dict(zip(np.unique(y_train), class_weights_np))}")
        
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
        pred_bert, true_bert = evaluate_bert(model, test_loader, DEVICE)
        bert_res = compute_metrics(true_bert, pred_bert)
        
        print(f"BERT      - Acc: {bert_res['accuracy']:.4f}, W-F1: {bert_res['f1_weighted']:.4f}, M-F1: {bert_res['f1_macro']:.4f}")
        
        # Save fold results
        per_fold.append({
            'fold': fold,
            'n_train': int(len(train_idx)),
            'n_test': int(len(test_idx)),
            'class_counts_train': dict(zip(*np.unique(y_train, return_counts=True))),
            'class_counts_test': dict(zip(*np.unique(y_test, return_counts=True))),
            'baseline': base_res,
            'advanced': adv_res,
            'bert': bert_res,
        })
        
        baseline_metrics.append(base_res)
        advanced_metrics.append(adv_res)
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
        'target': target,
        'n_rows': int(df.shape[0]),
        'n_samples_used': int(len(df2)),
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'features_advanced': Xcols,
        'bert_config': {
            'model': 'distilbert-base-uncased',
            'max_length': MAX_LENGTH,
            'batch_size': BERT_BATCH_SIZE,
            'epochs': BERT_EPOCHS,
            'learning_rate': BERT_LR,
            'device': str(DEVICE),
        },
        'baseline_cv': agg(baseline_metrics),
        'advanced_cv': agg(advanced_metrics),
        'bert_cv': agg(bert_metrics),
        'per_fold': per_fold,
    }
    
    # Save metrics
    with open(OUT_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(_make_json_serializable(metrics), f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("Summary (CV Average)")
    print(f"{'='*60}")
    print(f"Baseline  - Acc: {metrics['baseline_cv']['accuracy']:.4f}, W-F1: {metrics['baseline_cv']['f1_weighted']:.4f}, M-F1: {metrics['baseline_cv']['f1_macro']:.4f}")
    print(f"Advanced  - Acc: {metrics['advanced_cv']['accuracy']:.4f}, W-F1: {metrics['advanced_cv']['f1_weighted']:.4f}, M-F1: {metrics['advanced_cv']['f1_macro']:.4f}")
    print(f"BERT      - Acc: {metrics['bert_cv']['accuracy']:.4f}, W-F1: {metrics['bert_cv']['f1_weighted']:.4f}, M-F1: {metrics['bert_cv']['f1_macro']:.4f}")
    print(f"\nMetrics saved to: {OUT_DIR / 'metrics.json'}")
    
    # Train final models on full data
    print("\n" + "="*60)
    print("Training final models on full dataset...")
    print("="*60)
    
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
    joblib.dump({'vectorizer': vectorizer_final, 'model': clf_base_final}, OUT_DIR / 'baseline_model_sentiment.joblib')
    print("✓ Baseline model saved")
    
    # Final advanced
    print("\n[Advanced] Training final model...")
    imputer_final = SimpleImputer(strategy='median')
    scaler_final = StandardScaler()
    X_adv_full = scaler_final.fit_transform(imputer_final.fit_transform(X_num))
    
    if LGB_AVAILABLE:
        clf_adv_final = lgb.LGBMClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            verbose=-1
        )
    else:
        clf_adv_final = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    
    clf_adv_final.fit(X_adv_full, y)
    joblib.dump({
        'imputer': imputer_final,
        'scaler': scaler_final,
        'model': clf_adv_final,
        'features': Xcols
    }, OUT_DIR / 'advanced_model_sentiment.joblib')
    print("✓ Advanced model saved")
    
    # Final BERT
    print(f"\n[BERT] Training final model (device: {DEVICE})...")
    
    # Compute class weights for full dataset
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_np_full = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_full = torch.tensor(class_weights_np_full, dtype=torch.float32)
    print(f"Class weights: {dict(zip(np.unique(y), class_weights_np_full))}")
    
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
    model_final.save_pretrained(OUT_DIR / 'bert_model')
    tokenizer.save_pretrained(OUT_DIR / 'bert_model')
    print("✓ BERT model saved")
    
    print(f"\nAll models saved to: {OUT_DIR}")
    print("\n✓ Training complete!")


if __name__ == '__main__':
    train_sentiment()
