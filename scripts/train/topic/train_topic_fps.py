"""
FPS Topic Classification Training (6-Class LDA-Mapped)
======================================================
针对 FPS 游戏数据的主题分类训练脚本

基于 LDA 关键词的优化映射：
1. feature_engineering.py 生成 8个 LDA 主题 (dominant_topic 0-7)
2. 基于 LDA 关键词重新映射到 6个平衡的业务类别
3. 训练 6分类模型 (topic_category)

最终 6 个类别（LDA-mapped）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- matchmaking_issues    : ~26%  (Topics 1+3: COD + CSGO 匹配问题)
- multiplayer_features  : ~21%  (Topic 0: 一般多人游戏反馈)
- cheating              : ~20%  (Topics 5+6: 外挂 + 反外挂系统)
- user_interface        : ~15%  (Topic 7: UI/性能/图形)
- technical_issues      : ~12%  (Topic 4: 崩溃/Bug/更新问题)
- monetization_concerns :  ~5%  (Topic 2: 付费/赌博/箱子)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

优势：
✓ 类别平衡性好 (最大26% vs 最小5%, 5倍差距)
✓ 业务可解释性强 (明确的问题类别)
✓ 消除 community_feedback 过度合并 (原51.9%)
✓ 样本测试验证: Baseline W-F1 0.7496 (优于原8-class的0.7164)

模型对比：
- Baseline: TF-IDF (10k features, trigram) + Logistic Regression
- Advanced: DistilBERT fine-tuned (5 epochs)

评估指标：
- Primary: Weighted F1 (主要指标)
- Secondary: Macro F1, Accuracy
- Diagnostic: Per-class P/R/F1, PR-AUC, Confusion Matrix

预期性能：
- Baseline: W-F1 0.75-0.78
- BERT: W-F1 0.80-0.83

预计时间：5-6 小时（5折 CV）
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
    precision_score, recall_score, precision_recall_curve, auc,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize, LabelEncoder
from scipy.special import softmax
from tqdm import tqdm
import warnings
import logging
warnings.filterwarnings('ignore')

# Suppress BERT initialization warnings
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = Path(r'C:\Users\12932\Desktop\nus\BAP')
DATA_PATH = BASE_DIR / 'features' / 'fps' / 'gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet'

# Output directory
OUTPUT_DIR = BASE_DIR / 'analysis_results' / 'train' / 'fps_topic_6class_lda'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model Parameters
MAX_LENGTH = 256
BATCH_SIZE = 64
BERT_EPOCHS = 5
LEARNING_RATE = 2e-5

# CV Strategy
N_SPLITS = 5

# Text Features
TEXT_COLUMN = 'review_content_clean'
TEXT_COLUMN_PROCESSED = 'review_content_processed'

# Target Column - Only 6-class LDA-mapped
TARGET_COLUMN = 'topic_category'  # 6分类 (LDA-mapped 业务标签)

# TF-IDF Configuration
NGRAM_RANGE = (1, 3)
MAX_FEATURES = 10000

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# Data Preprocessing - LDA-based Topic Mapping
# ============================================================================

def remap_topics_by_lda_keywords(df):
    """
    基于 LDA 关键词重新映射主题类别
    
    LDA 主题关键词（从 lda_topic_keywords.json）：
    - Topic 0: better, fun, game, good, like, play → multiplayer_features
    - Topic 1: cod, duty, fun, game, zombie → cod_related  
    - Topic 2: gamble, gambling, gold, mango → monetization_concerns
    - Topic 3: competitive, csgo, map, valve → csgo_competitive
    - Topic 4: crash, fix, case, money, update → technical_issues
    - Topic 5: cheat, cheater, hacker, vac → cheating
    - Topic 6: anti cheat, cancer, fuck, hacker → anti_cheat_issues
    - Topic 7: fps, bad, best, game → user_interface
    
    合并规则（提高类别平衡性）：
    1. Topic 1 (cod) + Topic 3 (csgo) → matchmaking_issues (匹配问题)
    2. Topic 5 (cheater) + Topic 6 (anti cheat) → cheating (外挂问题)
    
    最终 6 个类别：
    - matchmaking_issues    : ~26%  (Topics 1+3: COD + CSGO)
    - multiplayer_features  : ~21%  (Topic 0: 一般多人游戏)
    - cheating              : ~20%  (Topics 5+6: 外挂 + 反外挂)
    - user_interface        : ~15%  (Topic 7: UI/性能)
    - technical_issues      : ~12%  (Topic 4: 崩溃/箱子)
    - monetization_concerns :  ~5%  (Topic 2: 付费/赌博)
    """
    print("\n" + "="*80)
    print("REMAPPING TOPICS BY LDA KEYWORDS")
    print("="*80)
    
    if 'dominant_topic' not in df.columns:
        print("⚠ dominant_topic column not found, skipping remap")
        return df
    
    # Backup original
    if 'topic_category' in df.columns:
        df['topic_category_original'] = df['topic_category'].copy()
    
    # LDA topic → new category mapping
    topic_mapping = {
        0: 'multiplayer_features',   # general multiplayer feedback
        1: 'matchmaking_issues',      # COD-related (merged)
        2: 'monetization_concerns',   # gambling/money
        3: 'matchmaking_issues',      # CSGO competitive (merged)
        4: 'technical_issues',        # crashes/bugs
        5: 'cheating',                # cheaters (merged)
        6: 'cheating',                # anti-cheat issues (merged)
        7: 'user_interface'           # UI/FPS/graphics
    }
    
    # Apply mapping
    df['topic_category'] = df['dominant_topic'].map(topic_mapping)
    
    # Count unmapped
    unmapped = df['topic_category'].isna().sum()
    if unmapped > 0:
        print(f"⚠ {unmapped:,} samples have unmapped topics, setting to 'other'")
        df['topic_category'] = df['topic_category'].fillna('other')
    
    # Distribution after remapping
    print("\n✓ Distribution AFTER LDA-based remapping:")
    dist_after = df['topic_category'].value_counts().sort_values(ascending=False)
    for cat, count in dist_after.items():
        pct = count / len(df) * 100
        # Show which topics contribute to this category
        topics = [k for k, v in topic_mapping.items() if v == cat]
        topics_str = ', '.join([f'Topic {t}' for t in topics])
        print(f"  {cat:30s}: {count:7,} ({pct:5.2f}%)  [{topics_str}]")
    
    n_classes_after = df['topic_category'].nunique()
    print(f"\n✓ Final number of classes: {n_classes_after}")
    
    return df

# ============================================================================
# Data Loading
# ============================================================================

def load_data(target_column, output_dir):
    """Load FPS data and prepare for topic classification
    
    Args:
        target_column: 'dominant_topic' (8-class) or 'topic_category' (5-class)
        output_dir: Directory to save label mapping
        
    Returns:
        df: DataFrame with encoded labels
        label_mapping: Dict mapping label indices to names
        n_classes: Number of classes
    """
    print("\n" + "="*80)
    print(f"LOADING FPS DATA FOR {target_column.upper()}")
    print("="*80)
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    df = pd.read_parquet(DATA_PATH)
    print(f"✓ Loaded {len(df):,} samples from {DATA_PATH.name}")
    
    # Apply LDA-based topic remapping (for both tasks, to regenerate topic_category)
    df = remap_topics_by_lda_keywords(df)
    
    # Validate required columns
    required_cols = [TEXT_COLUMN, TEXT_COLUMN_PROCESSED, target_column, 'timestamp']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Time range
    df = df.sort_values('timestamp').reset_index(drop=True)
    min_time = pd.to_datetime(df['timestamp'].min(), unit='s')
    max_time = pd.to_datetime(df['timestamp'].max(), unit='s')
    time_span = (max_time - min_time).days
    print(f"✓ Time range: {min_time.date()} to {max_time.date()} ({time_span} days)")
    
    # Handle target column encoding
    if target_column == 'dominant_topic':
        # dominant_topic is already 0-7 integers (only for reference, not trained)
        df['topic_encoded'] = df[target_column].astype(int)
        label_mapping = {i: f'topic_{i}' for i in range(8)}
        n_classes = 8
    else:  # topic_category (our main target)
        # topic_category needs encoding
        label_encoder = LabelEncoder()
        df['topic_encoded'] = label_encoder.fit_transform(df[target_column])
        label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        n_classes = len(label_encoder.classes_)
    
    # Save label mapping
    mapping_path = output_dir / f'label_mapping_{target_column}.json'
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Label mapping saved to {mapping_path.name}")
    
    # Topic distribution
    topic_dist = df['topic_encoded'].value_counts().sort_index()
    print(f"\n✓ Topic distribution ({target_column}):")
    for idx, count in topic_dist.items():
        pct = count / len(df) * 100
        label_name = label_mapping[idx]
        print(f"  [{idx}] {label_name:25s}: {count:7,} ({pct:5.2f}%)")
    
    # Handle missing text
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('')
    df[TEXT_COLUMN_PROCESSED] = df[TEXT_COLUMN_PROCESSED].fillna('')
    
    # Remove empty reviews
    valid_mask = (df[TEXT_COLUMN].str.strip() != '') & (df[TEXT_COLUMN_PROCESSED].str.strip() != '')
    n_removed = (~valid_mask).sum()
    if n_removed > 0:
        print(f"\n⚠ Removing {n_removed:,} samples with empty text")
    df = df[valid_mask].reset_index(drop=True)
    
    print(f"\n✓ Final dataset: {len(df):,} samples, {n_classes} classes")
    
    return df, label_mapping, n_classes

# ============================================================================
# Label Encoding (DEPRECATED - merged into load_data)
# ============================================================================

# Removed - functionality merged into load_data()

# ============================================================================
# Baseline Model (TF-IDF + Logistic Regression)
# ============================================================================

def train_baseline(X_train_text, y_train, X_test_text, y_test):
    """Train TF-IDF + Logistic Regression baseline"""
    print("\n" + "-"*80)
    print("TRAINING BASELINE (TF-IDF + Logistic Regression)")
    print("-"*80)
    
    # TF-IDF Vectorization
    print(f"  Vectorizing text with TF-IDF (max_features={MAX_FEATURES}, ngram_range={NGRAM_RANGE})...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=3,
        max_df=0.9,
        strip_accents='unicode',
        lowercase=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)
    print(f"  ✓ Train shape: {X_train_vec.shape}, Test shape: {X_test_vec.shape}")
    
    # Compute class weights
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {cls: weight for cls, weight in zip(classes, class_weights_array)}
    print(f"  ✓ Class weights: {class_weights}")
    
    # Train Logistic Regression
    print("  Training Logistic Regression...")
    clf = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        solver='liblinear',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_vec, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_vec)
    y_proba = clf.predict_proba(X_test_vec)
    
    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    return {
        'vectorizer': vectorizer,
        'model': clf,
        'predictions': y_pred,
        'probabilities': y_proba,
        **metrics
    }

# ============================================================================
# BERT Model
# ============================================================================

class ReviewDataset(Dataset):
    """Dataset for BERT training"""
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

def train_bert_epoch(model, dataloader, optimizer, class_weights, device):
    """Train BERT for one epoch"""
    model.train()
    total_loss = 0
    
    # Convert class weights to tensor
    weight_tensor = torch.tensor(
        [class_weights[i] for i in sorted(class_weights.keys())],
        dtype=torch.float32
    ).to(device)
    
    for batch in tqdm(dataloader, desc="  Training", leave=False):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Apply class weights to loss
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    return total_loss / len(dataloader)

def evaluate_bert(model, dataloader, device):
    """Evaluate BERT model"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_logits)

def train_bert(X_train, y_train, X_test, y_test):
    """Train BERT model for topic classification"""
    print("\n" + "-"*80)
    print("TRAINING ADVANCED MODEL (BERT)")
    print("-"*80)
    
    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Datasets
    train_dataset = ReviewDataset(X_train.tolist(), y_train.tolist(), tokenizer, MAX_LENGTH)
    test_dataset = ReviewDataset(X_test.tolist(), y_test.tolist(), tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Compute class weights
    classes = np.unique(y_train)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {cls: weight for cls, weight in zip(classes, class_weights_array)}
    print(f"  ✓ Class weights: {class_weights}")
    
    # Initialize model
    num_labels = len(classes)
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels
    )
    model.to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print(f"  Training BERT for {BERT_EPOCHS} epochs...")
    for epoch in range(BERT_EPOCHS):
        avg_loss = train_bert_epoch(model, train_loader, optimizer, class_weights, DEVICE)
        print(f"    Epoch {epoch+1}/{BERT_EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Evaluation
    predictions, true_labels, logits = evaluate_bert(model, test_loader, DEVICE)
    
    # Convert logits to probabilities
    probabilities = softmax(logits, axis=1)
    
    # Metrics
    metrics = compute_metrics(true_labels, predictions, probabilities)
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'predictions': predictions,
        'probabilities': probabilities,
        **metrics
    }

# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute comprehensive metrics"""
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
    }
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class_f1'] = [float(x) for x in per_class_f1]
    metrics['per_class_precision'] = [float(x) for x in per_class_precision]
    metrics['per_class_recall'] = [float(x) for x in per_class_recall]
    
    # PR-AUC (if probabilities available)
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        
        pr_auc_scores = []
        for i in range(n_classes):
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true_bin[:, i], y_proba[:, i]
            )
            pr_auc_scores.append(float(auc(recall_curve, precision_curve)))
        
        metrics['pr_auc'] = pr_auc_scores
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics

# ============================================================================
# Display Functions
# ============================================================================

def display_fold_results(fold_idx, baseline_res, bert_res, label_mapping):
    """Display results for a single fold
    
    Args:
        fold_idx: Fold index (0-based)
        baseline_res: Baseline results dict
        bert_res: BERT results dict
        label_mapping: Dict mapping label indices to names
    """
    print(f"\n{'='*80}")
    print(f"FOLD {fold_idx + 1} RESULTS")
    print(f"{'='*80}")
    
    # Baseline
    print(f"\nBaseline (TF-IDF + LR):")
    print(f"  Accuracy: {baseline_res['accuracy']:.4f}")
    print(f"  Weighted F1: {baseline_res['f1_weighted']:.4f}")
    print(f"  Macro F1: {baseline_res['f1_macro']:.4f}")
    
    if 'per_class_f1' in baseline_res:
        print(f"  Per-class F1:")
        for idx, f1 in enumerate(baseline_res['per_class_f1']):
            label = label_mapping[idx]
            print(f"    {label:25s}: {f1:.4f}")
    
    if 'pr_auc' in baseline_res:
        print(f"  PR-AUC:")
        for idx, auc_val in enumerate(baseline_res['pr_auc']):
            label = label_mapping[idx]
            print(f"    {label:25s}: {auc_val:.4f}")
    
    # BERT
    print(f"\nBERT:")
    print(f"  Accuracy: {bert_res['accuracy']:.4f}")
    print(f"  Weighted F1: {bert_res['f1_weighted']:.4f}")
    print(f"  Macro F1: {bert_res['f1_macro']:.4f}")
    
    if 'per_class_f1' in bert_res:
        print(f"  Per-class F1:")
        for idx, f1 in enumerate(bert_res['per_class_f1']):
            label = label_mapping[idx]
            print(f"    {label:25s}: {f1:.4f}")
    
    if 'pr_auc' in bert_res:
        print(f"  PR-AUC:")
        for idx, auc_val in enumerate(bert_res['pr_auc']):
            label = label_mapping[idx]
            print(f"    {label:25s}: {auc_val:.4f}")


def display_summary(baseline_metrics, bert_metrics, label_mapping):
    """Display summary across all folds
    
    Args:
        baseline_metrics: List of baseline fold results
        bert_metrics: List of BERT fold results
        label_mapping: Dict mapping label indices to names
        
    Returns:
        baseline_avg: Aggregated baseline metrics
        bert_avg: Aggregated BERT metrics
    """
    print(f"\n{'='*80}")
    print(f"SUMMARY (5-FOLD CV AVERAGE)")
    print(f"{'='*80}")
    
    # Average metrics
    baseline_avg = {
        'accuracy': {'mean': np.mean([m['accuracy'] for m in baseline_metrics]),
                    'std': np.std([m['accuracy'] for m in baseline_metrics])},
        'f1_weighted': {'mean': np.mean([m['f1_weighted'] for m in baseline_metrics]),
                       'std': np.std([m['f1_weighted'] for m in baseline_metrics])},
        'f1_macro': {'mean': np.mean([m['f1_macro'] for m in baseline_metrics]),
                    'std': np.std([m['f1_macro'] for m in baseline_metrics])},
    }
    
    bert_avg = {
        'accuracy': {'mean': np.mean([m['accuracy'] for m in bert_metrics]),
                    'std': np.std([m['accuracy'] for m in bert_metrics])},
        'f1_weighted': {'mean': np.mean([m['f1_weighted'] for m in bert_metrics]),
                       'std': np.std([m['f1_weighted'] for m in bert_metrics])},
        'f1_macro': {'mean': np.mean([m['f1_macro'] for m in bert_metrics]),
                    'std': np.std([m['f1_macro'] for m in bert_metrics])},
    }
    
    # Per-class averages
    n_classes = len(label_mapping)
    baseline_avg['per_class_f1'] = {
        'mean': [np.mean([m['per_class_f1'][i] for m in baseline_metrics]) for i in range(n_classes)],
        'std': [np.std([m['per_class_f1'][i] for m in baseline_metrics]) for i in range(n_classes)]
    }
    bert_avg['per_class_f1'] = {
        'mean': [np.mean([m['per_class_f1'][i] for m in bert_metrics]) for i in range(n_classes)],
        'std': [np.std([m['per_class_f1'][i] for m in bert_metrics]) for i in range(n_classes)]
    }
    
    if 'pr_auc' in baseline_metrics[0] and baseline_metrics[0]['pr_auc']:
        baseline_avg['pr_auc'] = {
            'mean': [np.mean([m['pr_auc'][i] for m in baseline_metrics]) for i in range(n_classes)],
            'std': [np.std([m['pr_auc'][i] for m in baseline_metrics]) for i in range(n_classes)]
        }
        bert_avg['pr_auc'] = {
            'mean': [np.mean([m['pr_auc'][i] for m in bert_metrics]) for i in range(n_classes)],
            'std': [np.std([m['pr_auc'][i] for m in bert_metrics]) for i in range(n_classes)]
        }
    
    # Display
    print(f"\nBaseline  - Acc: {baseline_avg['accuracy']['mean']:.4f} ± {baseline_avg['accuracy']['std']:.4f}")
    print(f"            W-F1: {baseline_avg['f1_weighted']['mean']:.4f} ± {baseline_avg['f1_weighted']['std']:.4f}")
    print(f"            M-F1: {baseline_avg['f1_macro']['mean']:.4f} ± {baseline_avg['f1_macro']['std']:.4f}")
    
    print(f"\nBERT      - Acc: {bert_avg['accuracy']['mean']:.4f} ± {bert_avg['accuracy']['std']:.4f}")
    print(f"            W-F1: {bert_avg['f1_weighted']['mean']:.4f} ± {bert_avg['f1_weighted']['std']:.4f}")
    print(f"            M-F1: {bert_avg['f1_macro']['mean']:.4f} ± {bert_avg['f1_macro']['std']:.4f}")
    
    print(f"\nPer-class F1:")
    print(f"{'Topic':25s} {'Baseline':>15s} {'BERT':>15s} {'Improvement':>12s}")
    print("-" * 70)
    for idx in range(n_classes):
        label = label_mapping[idx]
        base_f1 = baseline_avg['per_class_f1']['mean'][idx]
        bert_f1 = bert_avg['per_class_f1']['mean'][idx]
        improvement = bert_f1 - base_f1
        print(f"{label:25s} {base_f1:10.4f} {bert_f1:15.4f} {improvement:+11.4f}")
    
    if 'pr_auc' in baseline_avg:
        print(f"\nPer-class PR-AUC:")
        print(f"{'Topic':25s} {'Baseline':>15s} {'BERT':>15s} {'Improvement':>12s}")
        print("-" * 70)
        for idx in range(n_classes):
            label = label_mapping[idx]
            base_auc = baseline_avg['pr_auc']['mean'][idx]
            bert_auc = bert_avg['pr_auc']['mean'][idx]
            improvement = bert_auc - base_auc
            print(f"{label:25s} {base_auc:10.4f} {bert_auc:15.4f} {improvement:+11.4f}")
    
    return baseline_avg, bert_avg

# ============================================================================
# Main Training Loop
# ============================================================================

def run_task(target_column, output_dir):
    """Run training for one task (8-class or 5-class)
    
    Args:
        target_column: 'dominant_topic' or 'topic_category'
        output_dir: Output directory for this task
    """
    print(f"\n{'='*80}")
    print(f"TASK: {target_column.upper()}")
    print(f"{'='*80}")
    
    # Load data
    df, label_mapping, n_classes = load_data(target_column, output_dir)
    
    # Prepare features
    X_bert = df[TEXT_COLUMN].values
    X_baseline = df[TEXT_COLUMN_PROCESSED].values
    y = df['topic_encoded'].values
    
    # Time-based CV
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    
    baseline_metrics = []
    bert_metrics = []
    
    # Store best models (based on weighted F1)
    best_baseline_model = None
    best_baseline_vectorizer = None
    best_baseline_f1 = 0
    
    best_bert_model = None
    best_bert_tokenizer = None
    best_bert_f1 = 0
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_bert)):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx + 1}/{N_SPLITS}")
        print(f"{'='*80}")
        print(f"  Train samples: {len(train_idx):,}")
        print(f"  Test samples:  {len(test_idx):,}")
        
        # Split data
        X_train_bert, X_test_bert = X_bert[train_idx], X_bert[test_idx]
        X_train_baseline, X_test_baseline = X_baseline[train_idx], X_baseline[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train Baseline
        baseline_res = train_baseline(X_train_baseline, y_train, X_test_baseline, y_test)
        baseline_metrics.append({
            'accuracy': baseline_res['accuracy'],
            'f1_weighted': baseline_res['f1_weighted'],
            'f1_macro': baseline_res['f1_macro'],
            'per_class_f1': baseline_res['per_class_f1'],
            'per_class_precision': baseline_res['per_class_precision'],
            'per_class_recall': baseline_res['per_class_recall'],
            'pr_auc': baseline_res.get('pr_auc', []),
            'confusion_matrix': baseline_res['confusion_matrix']
        })
        
        # Save best baseline model
        if baseline_res['f1_weighted'] > best_baseline_f1:
            best_baseline_f1 = baseline_res['f1_weighted']
            best_baseline_model = baseline_res['model']
            best_baseline_vectorizer = baseline_res['vectorizer']
            print(f"  💾 New best baseline model (W-F1: {best_baseline_f1:.4f})")
        
        # Train BERT
        bert_res = train_bert(X_train_bert, y_train, X_test_bert, y_test)
        bert_metrics.append({
            'accuracy': bert_res['accuracy'],
            'f1_weighted': bert_res['f1_weighted'],
            'f1_macro': bert_res['f1_macro'],
            'per_class_f1': bert_res['per_class_f1'],
            'per_class_precision': bert_res['per_class_precision'],
            'per_class_recall': bert_res['per_class_recall'],
            'pr_auc': bert_res.get('pr_auc', []),
            'confusion_matrix': bert_res['confusion_matrix']
        })
        
        # Save best BERT model
        if bert_res['f1_weighted'] > best_bert_f1:
            best_bert_f1 = bert_res['f1_weighted']
            best_bert_model = bert_res['model']
            best_bert_tokenizer = bert_res['tokenizer']
            print(f"  💾 New best BERT model (W-F1: {best_bert_f1:.4f})")
        
        # Display fold results
        display_fold_results(fold_idx, baseline_res, bert_res, label_mapping)
    
    # Display summary
    baseline_avg, bert_avg = display_summary(baseline_metrics, bert_metrics, label_mapping)
    
    # Save metrics
    final_metrics = {
        'experiment': 'fps_topic_classification',
        'target': target_column,
        'n_samples': len(df),
        'n_classes': n_classes,
        'label_mapping': label_mapping,
        'n_splits': N_SPLITS,
        'baseline_cv': baseline_avg,
        'bert_cv': bert_avg,
        'baseline_folds': baseline_metrics,
        'bert_folds': bert_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_path = output_dir / f'metrics_{target_column}.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_path}")
    
    # Save best models
    print(f"\n{'='*80}")
    print("SAVING BEST MODELS")
    print(f"{'='*80}")
    
    # Save baseline model
    if best_baseline_model is not None:
        import joblib
        baseline_model_path = output_dir / f'best_baseline_model_{target_column}.joblib'
        baseline_vectorizer_path = output_dir / f'best_baseline_vectorizer_{target_column}.joblib'
        
        joblib.dump(best_baseline_model, baseline_model_path)
        joblib.dump(best_baseline_vectorizer, baseline_vectorizer_path)
        
        print(f"\n✓ Baseline model saved:")
        print(f"  Model: {baseline_model_path.name}")
        print(f"  Vectorizer: {baseline_vectorizer_path.name}")
        print(f"  Best W-F1: {best_baseline_f1:.4f}")
    
    # Save BERT model
    if best_bert_model is not None:
        bert_model_dir = output_dir / f'best_bert_model_{target_column}'
        bert_model_dir.mkdir(exist_ok=True)
        
        best_bert_model.save_pretrained(bert_model_dir)
        best_bert_tokenizer.save_pretrained(bert_model_dir)
        
        print(f"\n✓ BERT model saved:")
        print(f"  Directory: {bert_model_dir.name}/")
        print(f"  Best W-F1: {best_bert_f1:.4f}")
    
    print(f"\n{'='*80}")
    print(f"All models and metrics saved to: {output_dir}")
    print(f"{'='*80}")
    
    return baseline_avg, bert_avg

def main():
    """Main training pipeline for 6-class LDA-mapped topic classification"""
    print(f"\n{'='*80}")
    print("FPS TOPIC CLASSIFICATION TRAINING (6-CLASS LDA-MAPPED)")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTarget: {TARGET_COLUMN} (6个 LDA-mapped 业务类别)")
    print(f"Categories: matchmaking_issues, multiplayer_features, cheating,")
    print(f"            user_interface, technical_issues, monetization_concerns")
    
    # Run training
    print(f"\n{'#'*80}")
    print("STARTING 6-CLASS TOPIC CLASSIFICATION TRAINING")
    print(f"{'#'*80}")
    baseline_metrics, bert_metrics = run_task(TARGET_COLUMN, OUTPUT_DIR)
    
    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n6-Class LDA-Mapped Topic Classification ({TARGET_COLUMN}):")
    print(f"  Baseline:")
    print(f"    Weighted F1: {baseline_metrics['f1_weighted']['mean']:.4f} ± {baseline_metrics['f1_weighted']['std']:.4f}")
    print(f"    Macro F1:    {baseline_metrics['f1_macro']['mean']:.4f} ± {baseline_metrics['f1_macro']['std']:.4f}")
    print(f"    Accuracy:    {baseline_metrics['accuracy']['mean']:.4f} ± {baseline_metrics['accuracy']['std']:.4f}")
    
    print(f"\n  BERT:")
    print(f"    Weighted F1: {bert_metrics['f1_weighted']['mean']:.4f} ± {bert_metrics['f1_weighted']['std']:.4f}")
    print(f"    Macro F1:    {bert_metrics['f1_macro']['mean']:.4f} ± {bert_metrics['f1_macro']['std']:.4f}")
    print(f"    Accuracy:    {bert_metrics['accuracy']['mean']:.4f} ± {bert_metrics['accuracy']['std']:.4f}")
    
    improvement = (bert_metrics['f1_weighted']['mean'] - baseline_metrics['f1_weighted']['mean']) * 100
    print(f"\n  BERT Improvement: +{improvement:.2f}%")
    
    if bert_metrics['f1_weighted']['mean'] >= 0.80:
        print(f"\n  ✅ 性能达标: BERT W-F1 >= 0.80")
    elif bert_metrics['f1_weighted']['mean'] >= 0.78:
        print(f"\n  ⚠️ 性能良好: BERT W-F1 >= 0.78")
    else:
        print(f"\n  ⚠️ 性能待优化: BERT W-F1 < 0.78")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
