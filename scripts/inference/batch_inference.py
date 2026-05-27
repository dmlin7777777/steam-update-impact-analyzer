"""
Batch Inference Engine for Game Review Analysis

批量推理引擎：对指定时间窗口的评论进行四个任务的预测
- 情感分类 (Sentiment)
- 主题识别 (Topic)
- 风险评分 (Risk)
- 趋势告警 (Trend Alert)

用法:
    python batch_inference.py --genre fps --start_date 2024-03-20 --window_hours 48
    python batch_inference.py --genre fps --input_file custom_reviews.parquet
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# BERT 相关导入
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN]  Warning: torch/transformers not available. BERT inference will be disabled.")
    print("   Install with: pip install torch transformers")

# 特征工程导入
try:
    from scripts.feature.feature_engineering import GPUOptimizedFeatureEngineer
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    print("[WARN]  Warning: Feature engineering module not available.")
    print("   Some features may not be computed for new data.")

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
# Models are stored under <repo>/model (singular) per user request
MODELS_DIR = BASE_DIR / 'model'
FEATURES_DIR = BASE_DIR / 'features'
OUTPUT_DIR = BASE_DIR / 'analysis_results' / 'inference'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class BatchInferenceEngine:
    """
    批量推理引擎
    """
    
    def __init__(self, genre='fps'):
        self.genre = genre
        self.models = {}
        self.vectorizers = {}
        self.feature_engineer = None
        
    def load_models(self):
        """
        加载训练好的模型（BERT 版本）
        
        注意：BERT 模型是 DistilBertForSequenceClassification，
        已包含分类头，可以直接进行端到端推理
        """
        print("\n" + "="*80)
        print("加载模型")
        print("="*80)
        
        if not TORCH_AVAILABLE:
            print("[WARN]  torch/transformers 不可用，请安装: pip install torch transformers")
            return
        
        import joblib
        
        # 1. 情感分类模型 - DistilBERT for Sequence Classification (3 classes)
        print("\n--- 情感分类模型 (DistilBERT) ---")
        sentiment_bert_path = MODELS_DIR / 'fps_sentiment_bert_model' / 'bert_model_fps'
        
        if sentiment_bert_path.exists():
            try:
                from transformers import AutoModelForSequenceClassification
                # 加载 tokenizer
                self.models['sentiment_tokenizer'] = AutoTokenizer.from_pretrained(str(sentiment_bert_path))
                # 加载完整的分类模型（包含分类头）
                self.models['sentiment_model'] = AutoModelForSequenceClassification.from_pretrained(str(sentiment_bert_path))
                print(f"[OK] 情感 DistilBERT 模型: {sentiment_bert_path}")
            except Exception as e:
                print(f"[FAIL] 加载情感 DistilBERT 失败: {e}")
                self.models['sentiment_model'] = None
        else:
            print(f"[WARN]  情感 DistilBERT 模型未找到: {sentiment_bert_path}")
        
        # 2. 主题识别模型 - DistilBERT for Sequence Classification (6 classes)
        print("\n--- 主题识别模型 (DistilBERT) ---")
        topic_bert_path = MODELS_DIR / 'fps_topic_6class_lda' / 'best_bert_model_topic_category'
        topic_labels_path = MODELS_DIR / 'fps_topic_6class_lda' / 'label_mapping_topic_category.json'
        
        if topic_bert_path.exists():
            try:
                from transformers import AutoModelForSequenceClassification
                # 加载 tokenizer
                self.models['topic_tokenizer'] = AutoTokenizer.from_pretrained(str(topic_bert_path))
                # 加载完整的分类模型（包含分类头）
                self.models['topic_model'] = AutoModelForSequenceClassification.from_pretrained(str(topic_bert_path))
                print(f"[OK] 主题 DistilBERT 模型: {topic_bert_path}")
                
                # 加载标签映射
                if topic_labels_path.exists():
                    with open(topic_labels_path, 'r', encoding='utf-8') as f:
                        self.models['topic_labels'] = json.load(f)
                    print(f"[OK] 主题标签映射: {topic_labels_path.name}")
            except Exception as e:
                print(f"[FAIL] 加载主题 DistilBERT 失败: {e}")
                self.models['topic_model'] = None
        else:
            print(f"[WARN]  主题 DistilBERT 模型未找到: {topic_bert_path}")
        
        # 3. 趋势告警模型（LightGBM）+ TF-IDF
        print("\n--- 趋势告警模型 (LightGBM) ---")
        trend_model_path = MODELS_DIR / 'trend' / 'final_model_oversample.joblib'
        trend_vec_path = MODELS_DIR / 'trend' / 'final_vectorizer_oversample.joblib'
        
        if trend_model_path.exists():
            self.models['trend'] = joblib.load(trend_model_path)
            print(f"[OK] 趋势告警模型: {trend_model_path.name}")
        else:
            print(f"[WARN]  趋势告警模型未找到: {trend_model_path}")
        
        if trend_vec_path.exists():
            self.vectorizers['trend'] = joblib.load(trend_vec_path)
            print(f"[OK] 趋势告警向量化器: {trend_vec_path.name}")
        else:
            print(f"[WARN]  趋势告警向量化器未找到: {trend_vec_path}")
        
        # 4. 风险评分（规则引擎）
        print("\n--- 风险评分配置 ---")
        risk_config_path = MODELS_DIR / 'risk_scoring_system' / 'risk_thresholds.json'
        if risk_config_path.exists():
            with open(risk_config_path, 'r', encoding='utf-8') as f:
                self.models['risk_config'] = json.load(f)
            print(f"[OK] 风险配置: {risk_config_path.name}")
        else:
            print(f"[WARN]  风险配置未找到: {risk_config_path}")
    
    def load_data(self, start_date=None, window_hours=48, input_file=None):
        """
        加载待推理数据
        
        Args:
            start_date: 起始日期 (YYYY-MM-DD)
            window_hours: 时间窗口（小时）
            input_file: 自定义输入文件
        """
        print("\n" + "="*80)
        print("加载数据")
        print("="*80)
        
        if input_file:
            # 从自定义文件加载
            print(f"\n[FILE] 读取自定义文件: {input_file}")
            # If a CSV is provided, make sure we convert / clean it to parquet first
            input_path = Path(input_file)
            if input_path.suffix.lower() == '.csv':
                # attempt to create a cleaned parquet next to the CSV
                parquet_path = input_path.with_name(input_path.stem + '_cleaned.parquet')
                if not parquet_path.exists():
                    print(f"[CONVERT] CSV detected; converting/cleaning to: {parquet_path}")
                    # Prefer using the user's DataCleaner if available
                    try:
                        from scripts.cleaning.review_data_cleaning_ import DataCleaner
                        print("使用 DataCleaner 进行清洗 (scripts.cleaning.review_data_cleaning_).")
                        cleaner = DataCleaner()
                        # Best-effort: try a few common method names
                        if hasattr(cleaner, 'clean_reviews_data'):
                            cleaner.clean_reviews_data(str(input_path))
                        elif hasattr(cleaner, 'clean_file'):
                            cleaner.clean_file(str(input_path), str(parquet_path))
                        else:
                            # Fallback to minimal cleaning below
                            raise AttributeError('DataCleaner has no known clean method')
                        # If DataCleaner wrote a known output, try to find it
                        if parquet_path.exists():
                            print(f"[OK] DataCleaner output found: {parquet_path}")
                        else:
                            # Try common filename used by the repository (best-effort)
                            alt = input_path.parent / (input_path.stem + '_cleaned.parquet')
                            if alt.exists():
                                parquet_path = alt
                                print(f"[OK] Found cleaned parquet at: {parquet_path}")
                            else:
                                print("[WARN]  DataCleaner did not produce expected parquet, falling back to lightweight cleaning.")
                                raise Exception('DataCleaner output missing')
                    except Exception:
                        # Lightweight fallback cleaning: safe, minimal operations
                        print("使用内置回退清洗器（轻量级）对 CSV 进行预处理...")
                        df_csv = pd.read_csv(input_path)
                        # Ensure numeric columns expected by downstream code
                        for col in ('played_hours', 'num_reviews'):
                            if col not in df_csv.columns:
                                df_csv[col] = 0
                        # Basic text normalization for required column
                        if 'review_content' in df_csv.columns:
                            df_csv['review_content'] = df_csv['review_content'].astype(str)
                            df_csv['review_content_processed'] = (
                                df_csv['review_content']
                                .str.replace(r'http\\S+', '', regex=True)
                                .str.replace(r'<[^>]+>', '', regex=True)
                                .str.strip()
                            )
                        else:
                            df_csv['review_content_processed'] = ''
                        # Parse time -> timestamp if possible
                        if 'time' in df_csv.columns:
                            try:
                                df_csv['timestamp'] = pd.to_datetime(df_csv['time'], errors='coerce')
                            except Exception:
                                df_csv['timestamp'] = pd.NaT
                        else:
                            df_csv['timestamp'] = pd.NaT
                        # Save parquet
                        df_csv.to_parquet(parquet_path, index=False)
                        print(f"[OK] 回退清洗完成并写入: {parquet_path}")

                # load the parquet that was produced (or already existed)
                df = pd.read_parquet(parquet_path)
            else:
                df = pd.read_parquet(input_file)
        else:
            # 从特征库加载
            feature_file = FEATURES_DIR / self.genre / f'gpu_optimized_features_{self.genre}_exclflagged_enhanced_features_with_weaklabels.parquet'
            
            if not feature_file.exists():
                raise FileNotFoundError(f"特征文件未找到: {feature_file}")
            
            print(f"\n[FILE] 读取特征文件: {feature_file.name}")
            df = pd.read_parquet(feature_file)
            
            # 时间窗口过滤
            if start_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = start_dt + timedelta(hours=window_hours)
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)]
                
                print(f"\n[TIME] 时间窗口:")
                print(f"   起始: {start_dt}")
                print(f"   结束: {end_dt}")
                print(f"   窗口: {window_hours} 小时")
        
        print(f"\n[OK] 加载样本数: {len(df):,}")
        
        # 检查必需列
        required_cols = ['review_content_processed', 'timestamp']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"缺少必需列: {missing}")
        
        # 检查并计算缺失的特征（如果数据来自新的CSV输入）
        df = self._ensure_features(df)
        
        return df
    
    def _ensure_features(self, df):
        """
        确保DataFrame包含推理所需的所有特征
        如果缺失关键特征，使用特征工程模块计算或映射现有列
        """
        # 特征名称映射：趋势分析模块使用不同的命名约定
        # 映射格式: {期望的列名: 实际可能存在的列名列表}
        feature_mapping = {
            'rolling_24h_sentiment_mean': ['sentiment_rolling_mean_24h', 'rolling_24h_sentiment_mean'],
            'rolling_24h_sentiment_std': ['sentiment_rolling_std_24h', 'rolling_24h_sentiment_std'],
            'rolling_48h_sentiment_mean': ['sentiment_rolling_mean_48h', 'rolling_48h_sentiment_mean'],
            'rolling_48h_sentiment_std': ['sentiment_rolling_std_48h', 'rolling_48h_sentiment_std'],
            'rolling_72h_sentiment_mean': ['sentiment_rolling_mean_72h', 'rolling_72h_sentiment_mean'],
            'rolling_72h_sentiment_std': ['sentiment_rolling_std_72h', 'rolling_72h_sentiment_std'],
            'review_length': ['text_length', 'review_length'],
            'review_count_1h': ['comment_rate_24h', 'review_count_1h', 'count_in_24h']
        }
        
        # 执行列名映射/重命名
        for target_col, source_candidates in feature_mapping.items():
            if target_col not in df.columns:
                # 查找第一个存在的候选列
                for source_col in source_candidates:
                    if source_col in df.columns:
                        df[target_col] = df[source_col]
                        print(f"   [MAP] {source_col} -> {target_col}")
                        break
        
        # 趋势模型需要的时间序列特征
        trend_features = [
            'rolling_24h_sentiment_mean', 'rolling_24h_sentiment_std',
            'rolling_48h_sentiment_mean', 'rolling_48h_sentiment_std',
            'rolling_72h_sentiment_mean', 'rolling_72h_sentiment_std',
            'prophet_yhat_normalized', 'prophet_yhat_upper_breach',
            'prophet_yhat_lower_breach', 'review_length', 'review_count_1h'
        ]
        
        # 可选的扩展特征（训练时可能使用）
        optional_trend_features = [
            'sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h',
            'comment_rate_change_24h'
        ]
        
        # 检查哪些特征缺失
        missing_features = [col for col in trend_features if col not in df.columns]
        
        if missing_features:
            print(f"\n[WARN]  检测到缺失特征: {len(missing_features)} 个")
            print(f"   缺失: {missing_features}")
            
            # Prophet 特征通常需要专门的时序建模，这里用合理的默认值
            prophet_features = ['prophet_yhat_normalized', 'prophet_yhat_upper_breach', 'prophet_yhat_lower_breach']
            prophet_missing = [f for f in prophet_features if f in missing_features]
            
            if prophet_missing:
                print(f"   [INFO] Prophet 特征缺失（需要时序建模），使用趋势特征近似")
                
                # 使用现有趋势特征近似 Prophet 预测
                # prophet_yhat_normalized: 归一化预测值 -> 使用情感滚动均值作为proxy
                if 'prophet_yhat_normalized' in prophet_missing:
                    if 'sentiment_rolling_mean_72h' in df.columns:
                        # 归一化到 [-1, 1]
                        df['prophet_yhat_normalized'] = df['sentiment_rolling_mean_72h'].fillna(0)
                    else:
                        df['prophet_yhat_normalized'] = 0.0
                
                # prophet_yhat_upper_breach: 超过预测上限 -> 使用情感加速度 + 波动性
                if 'prophet_yhat_upper_breach' in prophet_missing:
                    if 'sentiment_acceleration_72h' in df.columns and 'sentiment_volatility' in df.columns:
                        # 正加速度 + 高波动 = 突破上限
                        accel = df['sentiment_acceleration_72h'].fillna(0)
                        vol = df['sentiment_volatility'].fillna(0)
                        df['prophet_yhat_upper_breach'] = ((accel > 0.05) & (vol > 0.3)).astype(float)
                    else:
                        df['prophet_yhat_upper_breach'] = 0.0
                
                # prophet_yhat_lower_breach: 低于预测下限 -> 使用负向加速度 + 波动性
                if 'prophet_yhat_lower_breach' in prophet_missing:
                    if 'sentiment_acceleration_72h' in df.columns and 'sentiment_volatility' in df.columns:
                        # 负加速度 + 高波动 = 突破下限
                        accel = df['sentiment_acceleration_72h'].fillna(0)
                        vol = df['sentiment_volatility'].fillna(0)
                        df['prophet_yhat_lower_breach'] = ((accel < -0.05) & (vol > 0.3)).astype(float)
                    else:
                        df['prophet_yhat_lower_breach'] = 0.0
            
            # 检查是否还有其他缺失特征需要计算
            other_missing = [f for f in missing_features if f not in prophet_features]
            
            if not other_missing:
                print(f"   [OK] 所有必需特征已就绪（Prophet特征已填充默认值）")
                return df
            
            print(f"   将使用特征工程模块计算剩余特征...")
            
            if not FEATURE_ENGINEERING_AVAILABLE:
                print("   [FAIL] 特征工程模块不可用，将使用默认值填充")
                # 使用默认值填充
                for col in other_missing:
                    df[col] = 0.0
                return df
            
            # 初始化特征工程器（如果还没有）
            if self.feature_engineer is None:
                print("   初始化特征工程器...")
                self.feature_engineer = GPUOptimizedFeatureEngineer(
                    load_heavy_models=False,  # 不加载BERT等重模型以节省时间
                    force_cpu=False
                )
            
            try:
                # 计算基础文本特征
                print("   计算文本特征...")
                df = self.feature_engineer.extract_text_features(df)
                
                # 计算情感特征（用于rolling窗口）
                print("   计算情感特征...")
                df = self.feature_engineer.extract_sentiment_features(df)
                
                # 计算时间窗口特征（如果有timestamp）
                if 'timestamp' in df.columns and 'sentiment_score' in df.columns:
                    print("   计算时间窗口特征...")
                    df = self._compute_rolling_features(df)
                
                # 对于仍然缺失的特征，用0填充
                for col in trend_features:
                    if col not in df.columns:
                        df[col] = 0.0
                
                print(f"   [OK] 特征计算完成")
                
            except Exception as e:
                print(f"   [WARN]  特征计算失败: {e}")
                print(f"   使用默认值填充")
                for col in missing_features:
                    if col not in df.columns:
                        df[col] = 0.0
        
        return df
    
    def _compute_rolling_features(self, df):
        """
        计算滚动窗口特征（简化版本）
        """
        # 确保timestamp是datetime类型
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # 使用sentiment_score（VADER compound）作为情感分数
        if 'sentiment_compound' in df.columns:
            sentiment_col = 'sentiment_compound'
        elif 'sentiment_score' in df.columns:
            sentiment_col = 'sentiment_score'
        else:
            # 如果没有情感分数，创建一个默认的
            df['sentiment_score'] = 0.0
            sentiment_col = 'sentiment_score'
        
        # 计算滚动窗口统计（24h, 48h, 72h）
        for window_hours in [24, 48, 72]:
            window = f'{window_hours}h'
            
            # 使用rolling window按时间计算
            df[f'rolling_{window_hours}h_sentiment_mean'] = (
                df.set_index('timestamp')[sentiment_col]
                .rolling(f'{window_hours}h', min_periods=1)
                .mean()
                .reset_index(drop=True)
            )
            
            df[f'rolling_{window_hours}h_sentiment_std'] = (
                df.set_index('timestamp')[sentiment_col]
                .rolling(f'{window_hours}h', min_periods=1)
                .std()
                .reset_index(drop=True)
                .fillna(0)
            )
        
        # Prophet相关特征（简化：使用移动平均作为proxy）
        if 'review_length' not in df.columns:
            df['review_length'] = df['review_content_processed'].fillna('').str.len()
        
        # 计算每小时评论数（简化）
        df['review_count_1h'] = 1  # 简化处理，每条评论计为1
        
        # Prophet预测相关（简化为0，实际使用需要Prophet模型）
        df['prophet_yhat_normalized'] = 0.0
        df['prophet_yhat_upper_breach'] = 0.0
        df['prophet_yhat_lower_breach'] = 0.0
        
        return df
    
    def predict_sentiment(self, df):
        """
        情感分类推理（使用 DistilBERT）
        
        DistilBertForSequenceClassification 是完整的端到端模型
        可以直接进行推理，无需额外的分类器头
        """
        print("\n" + "="*80)
        print("情感分类推理（DistilBERT）")
        print("="*80)
        
        if self.models.get('sentiment_model') is None:
            print("[WARN]  情感 DistilBERT 模型未加载，跳过")
            df['sentiment_pred'] = 'unknown'
            df['sentiment_proba_negative'] = 0.0
            df['sentiment_proba_neutral'] = 0.0
            df['sentiment_proba_positive'] = 0.0
            return df
        
        texts = df['review_content_processed'].fillna('').values
        tokenizer = self.models['sentiment_tokenizer']
        model = self.models['sentiment_model']
        
        print(f"\n[PROCESS] 处理 {len(texts):,} 条评论...")
        
        # BERT 端到端推理（每个批次）
        batch_size = 32
        all_probas = []
        all_preds = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts.tolist(),
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                )
                
                # 移至设备
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 前向传播（DistilBertForSequenceClassification 直接输出 logits）
                outputs = model(**inputs)
                logits = outputs.logits
                
                # 转换为概率
                probas = torch.nn.functional.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_probas.append(probas.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        # 合并结果
        all_probas = np.vstack(all_probas)
        all_preds = np.concatenate(all_preds)
        
        print(f"[OK] DistilBERT 推理完成: {all_probas.shape}")
        
        # 映射到标签
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        df['sentiment_pred'] = [label_map.get(p, 'unknown') for p in all_preds]
        
        # 概率（3 列：negative, neutral, positive）
        df['sentiment_proba_negative'] = all_probas[:, 0]
        df['sentiment_proba_neutral'] = all_probas[:, 1]
        df['sentiment_proba_positive'] = all_probas[:, 2]
        
        print(f"\n[STATS] 情感分布:")
        print(df['sentiment_pred'].value_counts())
        print(f"\n[OK] 预测完成: {len(df):,} 样本")
        
        return df
    
    def predict_topic(self, df):
        """
        主题识别推理（使用 DistilBERT）
        
        DistilBertForSequenceClassification 是完整的端到端模型
        可以直接进行推理，无需额外的分类器头
        """
        print("\n" + "="*80)
        print("主题识别推理（DistilBERT）")
        print("="*80)
        
        if self.models.get('topic_model') is None:
            print("[WARN]  主题 DistilBERT 模型未加载，跳过")
            df['topic_pred'] = 'unknown'
            df['topic_proba'] = 0.0
            return df
        
        texts = df['review_content_processed'].fillna('').values
        tokenizer = self.models['topic_tokenizer']
        model = self.models['topic_model']
        label_map = self.models.get('topic_labels', {})
        
        print(f"\n[PROCESS] 处理 {len(texts):,} 条评论...")
        
        # BERT 端到端推理（每个批次）
        batch_size = 32
        all_probas = []
        all_preds = []
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = tokenizer(
                    batch_texts.tolist(),
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors='pt'
                )
                
                # 移至设备
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 前向传播（DistilBertForSequenceClassification 直接输出 logits）
                outputs = model(**inputs)
                logits = outputs.logits
                
                # 转换为概率
                probas = torch.nn.functional.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_probas.append(probas.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
        
        # 合并结果
        all_probas = np.vstack(all_probas)
        all_preds = np.concatenate(all_preds)
        
        print(f"[OK] DistilBERT 推理完成: {all_probas.shape}")
        
        # 映射到标签（根据保存的标签映射，key 为字符串）
        df['topic_pred'] = [label_map.get(str(p), f'class_{p}') for p in all_preds]
        df['topic_proba'] = all_probas.max(axis=1)
        
        print(f"\n[STATS] 主题分布:")
        print(df['topic_pred'].value_counts())
        print(f"\n[OK] 预测完成: {len(df):,} 样本")
        
        return df
    
    def predict_risk(self, df):
        """
        风险评分（规则引擎）
        """
        print("\n" + "="*80)
        print("风险评分")
        print("="*80)
        
        if 'risk_score' not in df.columns:
            print("[WARN]  risk_score列不存在，使用默认值")
            df['risk_score'] = 0.0
        
        # 使用已有的risk_score或重新计算
        risk_scores = df['risk_score'].values
        
        # 分级
        if self.models['risk_config']:
            thresholds = self.models['risk_config'].get('recommended_thresholds', {})
            critical_thresh = thresholds.get('critical', 3.0)
            high_thresh = thresholds.get('high', 2.0)
            medium_thresh = thresholds.get('medium', 1.0)
        else:
            critical_thresh = 3.0
            high_thresh = 2.0
            medium_thresh = 1.0
        
        df['risk_level'] = pd.cut(
            df['risk_score'],
            bins=[-np.inf, medium_thresh, high_thresh, critical_thresh, np.inf],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        print(f"\n[STATS] 风险分布:")
        print(df['risk_level'].value_counts())
        print(f"\n[OK] 评分完成: {len(df):,} 样本")
        
        return df
    
    def predict_trend_alert(self, df):
        """
        趋势告警推理（使用 LightGBM + TF-IDF）
        """
        print("\n" + "="*80)
        print("趋势告警推理（LightGBM）")
        print("="*80)
        
        if self.models.get('trend') is None:
            print("[WARN]  趋势模型未加载，跳过")
            df['trend_alert_pred'] = 0
            df['trend_alert_proba'] = 0.0
            return df
        
        if self.vectorizers.get('trend') is None:
            print("[WARN]  趋势向量化器未加载，跳过")
            df['trend_alert_pred'] = 0
            df['trend_alert_proba'] = 0.0
            return df
        
        texts = df['review_content_processed'].fillna('').values
        vectorizer = self.vectorizers['trend']
        model = self.models['trend']
        
        print(f"\n[PROCESS] TF-IDF 向量化 {len(texts):,} 条评论...")
        
        try:
            # TF-IDF 转换
            text_features = vectorizer.transform(texts)
            print(f"[OK] TF-IDF 完成: {text_features.shape}")
            
            # 趋势模型训练时使用的完整特征列表（按训练时的顺序）
            # 基础特征（必需）
            feature_cols = [
                'rolling_24h_sentiment_mean', 'rolling_24h_sentiment_std',
                'rolling_48h_sentiment_mean', 'rolling_48h_sentiment_std',
                'rolling_72h_sentiment_mean', 'rolling_72h_sentiment_std',
                'prophet_yhat_normalized', 'prophet_yhat_upper_breach',
                'prophet_yhat_lower_breach', 'review_length', 'review_count_1h'
            ]
            
            # 可能的扩展特征（训练时可能使用了这些）
            optional_features = [
                'sentiment_diff_24h', 'sentiment_diff_48h', 'sentiment_diff_72h',
                'comment_rate_change_24h'
            ]
            
            # 检查哪些特征存在，并按顺序添加
            available_features = []
            for col in feature_cols + optional_features:
                if col in df.columns:
                    available_features.append(col)
            
            if available_features:
                other_features = df[available_features].fillna(0).values
                # 组合 TF-IDF 和其他特征
                combined_features = np.hstack([text_features.toarray(), other_features])
                print(f"[OK] 组合特征: {combined_features.shape}")
                print(f"   TF-IDF: {text_features.shape[1]}, 其他特征: {len(available_features)}")
                
                # 检查特征数量是否匹配
                expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
                if expected_features and combined_features.shape[1] != expected_features:
                    print(f"[WARN]  特征数量不匹配: 当前={combined_features.shape[1]}, 期望={expected_features}")
                    print(f"   差异={expected_features - combined_features.shape[1]} 个特征")
                    
                    # 尝试添加缺失的特征（用0填充）
                    if combined_features.shape[1] < expected_features:
                        missing_count = expected_features - combined_features.shape[1]
                        print(f"   [FIX] 添加 {missing_count} 个零填充特征")
                        zero_features = np.zeros((len(df), missing_count))
                        combined_features = np.hstack([combined_features, zero_features])
                        print(f"   修复后特征维度: {combined_features.shape}")
            else:
                combined_features = text_features.toarray()
                print(f"[WARN]  未找到时间序列特征，仅使用 TF-IDF")
            
            # LightGBM 预测
            probas = model.predict_proba(combined_features)
            preds = model.predict(combined_features)
            
            # 保存预测结果
            df['trend_alert_pred'] = preds
            df['trend_alert_proba'] = probas[:, 1] if probas.shape[1] > 1 else probas[:, 0]
            
            print(f"\n[STATS] 趋势告警分布:")
            print(df['trend_alert_pred'].value_counts())
            print(f"   告警率: {df['trend_alert_pred'].mean():.2%}")
            print(f"\n[OK] 预测完成: {len(df):,} 样本")
            
        except Exception as e:
            print(f"[FAIL] 预测失败: {e}")
            df['trend_alert_pred'] = 0
            df['trend_alert_proba'] = 0.0
        
        return df
    
    def run_inference(self, df):
        """
        运行完整推理流程
        """
        print("\n" + "="*80)
        print("开始批量推理")
        print("="*80)
        
        # 1. 情感分类
        df = self.predict_sentiment(df)
        
        # 2. 主题识别
        df = self.predict_topic(df)
        
        # 3. 风险评分
        df = self.predict_risk(df)
        
        # 4. 趋势告警
        df = self.predict_trend_alert(df)
        
        return df
    
    def save_results(self, df, output_prefix='inference'):
        """
        保存推理结果
        """
        print("\n" + "="*80)
        print("保存结果")
        print("="*80)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 完整结果（parquet）
        output_file = OUTPUT_DIR / f'{output_prefix}_{self.genre}_{timestamp}.parquet'
        df.to_parquet(output_file, index=False)
        print(f"[OK] Parquet: {output_file}")
        
        # 2. 汇总统计（JSON）
        summary = {
            'metadata': {
                'genre': self.genre,
                'n_samples': len(df),
                'timestamp': timestamp,
                'inference_date': datetime.now().isoformat()
            },
            'sentiment_distribution': df['sentiment_pred'].value_counts().to_dict() if 'sentiment_pred' in df else {},
            'topic_distribution': df['topic_pred'].value_counts().to_dict() if 'topic_pred' in df else {},
            'risk_distribution': df['risk_level'].value_counts().to_dict() if 'risk_level' in df else {},
            'trend_alert_rate': float(df['trend_alert_pred'].mean()) if 'trend_alert_pred' in df else 0.0
        }
        
        summary_file = OUTPUT_DIR / f'{output_prefix}_{self.genre}_{timestamp}_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[OK] Summary: {summary_file}")
        
        # 3. 高风险样本（CSV）
        if 'risk_level' in df:
            high_risk = df[df['risk_level'].isin(['high', 'critical'])].copy()
            if len(high_risk) > 0:
                high_risk_file = OUTPUT_DIR / f'{output_prefix}_{self.genre}_{timestamp}_high_risk.csv'
                high_risk[['timestamp', 'review_content_processed', 'sentiment_pred', 
                          'topic_pred', 'risk_score', 'risk_level']].to_csv(
                    high_risk_file, index=False, encoding='utf-8-sig'
                )
                print(f"[OK] High Risk: {high_risk_file} ({len(high_risk):,} 样本)")
        
        return output_file


def main():
    # Non-interactive entrypoint: no CLI input. Use sensible defaults.
    genre = 'fps'
    start_date = '2025-10-22'
    window_hours = 72
    input_file = r'C:\Users\12932\Desktop\nus\BAP\test\fps\gpu_optimized_features_fps_inclflagged_enhanced_features_with_weaklabels.parquet'
    output_prefix = 'inference'

    print("="*80)
    print("批量推理引擎 (非交互模式)")
    print("="*80)
    print(f"类型: {genre}")

    # 初始化引擎
    engine = BatchInferenceEngine(genre=genre)

    # 加载模型
    engine.load_models()

    # 加载数据 (will use feature store if no input_file provided)
    df = engine.load_data(
        start_date=start_date,
        window_hours=window_hours,
        input_file=input_file
    )

    # 运行推理
    df = engine.run_inference(df)

    # 保存结果
    output_file = engine.save_results(df, output_prefix=output_prefix)

    print("\n" + "="*80)
    print("[OK] 批量推理完成")
    print("="*80)
    print(f"输出文件: {output_file}")


if __name__ == "__main__":
    main()
