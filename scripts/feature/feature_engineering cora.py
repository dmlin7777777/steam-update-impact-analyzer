import pandas as pd
import numpy as np
import re
import os
import difflib
import warnings
import logging
from typing import List, Dict, Tuple, Set, Optional
import glob
import time
import pickle
import joblib

# NLP and ML libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import textstat
import torch
from sentence_transformers import SentenceTransformer

# Optional imports for toxicity
try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False
    
try:
    from googleapiclient import discovery
    PERSPECTIVE_API_AVAILABLE = True
except ImportError:
    PERSPECTIVE_API_AVAILABLE = False

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUOptimizedFeatureEngineer:
    """
    GPU优化的游戏评论特征工程类
    提供全面的文本特征提取、情感分析、主题建模等功能
    """
    
    def __init__(self, perspective_api_key: Optional[str] = None, force_cpu: bool = False, load_heavy_models: bool = True, lda_vectorizer: str = 'count', serialized_models_dir: Optional[str] = None, serialized_models_prefix: Optional[str] = None, preloaded_topic_pca_path: Optional[str] = None, preloaded_embedding_kmeans_path: Optional[str] = None):
        self.perspective_api_key = perspective_api_key
        # 控制是否在初始化时加载大型/需要联网的模型（如 sentence-transformers, detoxify）
        self.load_heavy_models = load_heavy_models
        # LDA 向量器选择：'tfidf' 或 'count'
        self.lda_vectorizer = lda_vectorizer
        # optional directory & prefix for loading pre-saved topic_pca / embedding_kmeans
        self.serialized_models_dir = serialized_models_dir
        self.serialized_models_prefix = serialized_models_prefix
        # optional explicit full-paths to pre-saved models (takes precedence over dir+prefix)
        self.preloaded_topic_pca_path = preloaded_topic_pca_path
        self.preloaded_embedding_kmeans_path = preloaded_embedding_kmeans_path
        
        # 设置设备
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.setup_nltk()
        self.setup_models()
        # 默认的列名别名映射：canonical -> alternatives
        # 方便处理不同数据源中列名不一致的问题
        self.column_aliases: Dict[str, List[str]] = {
            'review_content_clean': ['review_content_clean', 'review_content', 'review_text', 'content', 'text', 'clean_text'],
            'review_content_processed': ['review_content_processed', 'review_content_clean', 'review_content', 'review_text', 'processed_text'],
            'review_date': ['review_date', 'date', 'created_at', 'timestamp'],
            'appid': ['appid', 'app_id', 'app', 'game_id'],
            'genre': ['genre'],
            'num_games_owned': ['num_games_owned', 'games_owned', 'owned_games'],
            'num_reviews': ['num_reviews', 'review_count'],
            'played_hours': ['played_hours', 'hours_played', 'playtime'],
            'playtime_last_two_weeks': ['playtime_last_two_weeks', 'playtime_2weeks'],
            'playtime_at_review': ['playtime_at_review', 'playtime_at_time_of_review'],
            'steam_purchase': ['steam_purchase', 'is_steam_purchase', 'purchased_on_steam'],
            'received_for_free': ['received_for_free', 'free', 'is_free'],
            # 互动计数（用于 engagement_score 等特征）
            'comment_count': ['comment_count', 'comments', 'num_comments', 'replies', 'reply_count']
        }
        
    def setup_nltk(self):
        """下载并设置NLTK所需的数据包"""
        nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']
        for package in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{package}' if package in ['punkt'] 
                             else f'corpora/{package}' if package in ['stopwords', 'wordnet']
                             else f'sentiment/{package}')
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package)
        
        self.english_stopwords = set(stopwords.words('english'))
        self.vader_analyzer = SentimentIntensityAnalyzer()
    
    def setup_models(self):
        """初始化机器学习模型"""
        logger.info("Initializing feature extraction models...")
        
        # TF-IDF向量化器
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # BERT模型（使用sentence-transformers，GPU优化）
        self.sentence_model = None
        if self.load_heavy_models:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                logger.info("BERT sentence transformer model loaded successfully on GPU")
            except Exception as e:
                logger.warning(f"Could not load BERT model: {e}")
                self.sentence_model = None
        
        # LDA主题模型
        self.lda_model = None
        # PCA for topic distributions
        self.topic_pca = None
        # embedding clustering model (MiniBatchKMeans)
        self.embedding_kmeans = None

        # 毒性检测模型（GPU优化）
        self.toxicity_model = None
        if self.load_heavy_models and DETOXIFY_AVAILABLE:
            try:
                # 优先在 GPU 上加载（如果可用），否则在 CPU 上加载
                if self.device.type == 'cuda':
                    self.toxicity_model = Detoxify('original', device=self.device)
                    logger.info("Detoxify toxicity model loaded successfully on GPU")
                else:
                    self.toxicity_model = Detoxify('original', device='cpu')
                    logger.info("Detoxify toxicity model loaded on CPU")
            except Exception as e:
                logger.warning(f"Could not load toxicity model: {e}")
                # 如果尝试 GPU 失败，尝试 CPU 回退
                try:
                    self.toxicity_model = Detoxify('original', device='cpu')
                    logger.info("Detoxify toxicity model loaded on CPU as fallback")
                except Exception as e2:
                    logger.warning(f"Could not load toxicity model at all: {e2}")
                    self.toxicity_model = None
        else:
            self.toxicity_model = None

        # Perspective API客户端
        self.perspective_client = None
        # 只有在明确提供了 API key 且不为空时才初始化
        if (self.load_heavy_models and 
            self.perspective_api_key and 
            self.perspective_api_key.strip() and 
            PERSPECTIVE_API_AVAILABLE):
            try:
                self.perspective_client = discovery.build(
                    "commentanalyzer",
                    "v1alpha1",
                    developerKey=self.perspective_api_key,
                    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                    static_discovery=False,
                )
                logger.info("Perspective API client initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize Perspective API: {e}")
                self.perspective_client = None
        else:
            if not PERSPECTIVE_API_AVAILABLE:
                logger.info("Perspective API library not available, skipping initialization")
            elif not self.perspective_api_key or not self.perspective_api_key.strip():
                logger.info("Perspective API key not provided, skipping initialization")
            self.perspective_client = None

        # 尝试自动加载已序列化的 PCA / KMeans 模型（如果提供了目录）
        try:
            self._auto_load_serialized_models()
        except Exception as e:
            logger.warning(f"Auto-loading serialized models failed: {e}")

    def _auto_load_serialized_models(self):
        """Attempt to load serialized topic_pca and embedding_kmeans from provided directory and prefix."""
        # If explicit full paths were provided, try those first
        if getattr(self, 'preloaded_topic_pca_path', None):
            path = self.preloaded_topic_pca_path
            logger.info(f"Attempting to load explicit topic_pca from {path}")
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.topic_pca = pickle.load(f)
                    logger.info(f"Loaded topic_pca from {path} (pickle)")
                    return
                except Exception as e:
                    try:
                        self.topic_pca = joblib.load(path)
                        logger.info(f"Loaded topic_pca from {path} (joblib)")
                        return
                    except Exception:
                        logger.warning(f"Failed to load explicit topic_pca from {path}: {e}")
            else:
                logger.info(f"Explicit topic_pca path does not exist: {path}")

        if getattr(self, 'preloaded_embedding_kmeans_path', None):
            path = self.preloaded_embedding_kmeans_path
            logger.info(f"Attempting to load explicit embedding_kmeans from {path}")
            if os.path.exists(path):
                try:
                    with open(path, 'rb') as f:
                        self.embedding_kmeans = pickle.load(f)
                    logger.info(f"Loaded embedding_kmeans from {path} (pickle)")
                    return
                except Exception as e:
                    try:
                        self.embedding_kmeans = joblib.load(path)
                        logger.info(f"Loaded embedding_kmeans from {path} (joblib)")
                        return
                    except Exception:
                        logger.warning(f"Failed to load explicit embedding_kmeans from {path}: {e}")
            else:
                logger.info(f"Explicit embedding_kmeans path does not exist: {path}")

        if not self.serialized_models_dir:
            return
        model_dir = self.serialized_models_dir
        prefix = self.serialized_models_prefix or ''

        # topic_pca file patterns
        pca_candidates = []
        if prefix:
            pca_candidates.append(os.path.join(model_dir, f"{prefix}_topic_pca.pkl"))
        pca_candidates.append(os.path.join(model_dir, "*topic_pca*.pkl"))

        # embedding kmeans patterns
        kmeans_candidates = []
        if prefix:
            kmeans_candidates.append(os.path.join(model_dir, f"{prefix}_embedding_kmeans.pkl"))
        kmeans_candidates.append(os.path.join(model_dir, "*embedding_kmeans*.pkl"))

        # try load first existing pca
        logger.info(f"Looking for topic_pca in: {pca_candidates}")
        for pat in pca_candidates:
            for path in glob.glob(pat):
                try:
                    with open(path, 'rb') as f:
                        self.topic_pca = pickle.load(f)
                    logger.info(f"Loaded topic_pca from {path} (pickle)")
                    return
                except Exception as e:
                    # try joblib as a fallback (some users save with joblib.dump)
                    try:
                        self.topic_pca = joblib.load(path)
                        logger.info(f"Loaded topic_pca from {path} (joblib)")
                        return
                    except Exception:
                        logger.warning(f"Failed to load topic_pca from {path}: {e}")

        # try load first existing kmeans
        logger.info(f"Looking for embedding_kmeans in: {kmeans_candidates}")
        for pat in kmeans_candidates:
            for path in glob.glob(pat):
                try:
                    with open(path, 'rb') as f:
                        self.embedding_kmeans = pickle.load(f)
                    logger.info(f"Loaded embedding_kmeans from {path} (pickle)")
                    return
                except Exception as e:
                    # try joblib as a fallback
                    try:
                        self.embedding_kmeans = joblib.load(path)
                        logger.info(f"Loaded embedding_kmeans from {path} (joblib)")
                        return
                    except Exception:
                        logger.warning(f"Failed to load embedding_kmeans from {path}: {e}")

        logger.info("No serialized topic_pca or embedding_kmeans found with provided patterns")

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        根据 self.column_aliases 将输入 DataFrame 标准化为包含一组 canonical 列名的副本。

        - 如果找到替代列名则重命名为 canonical
        - 如果未找到则创建空列（数值为 NaN 或适当的默认值），以避免 KeyError
        返回新的 DataFrame 副本（不修改原始 df）
        """
        df_copy = df.copy()
        for canonical, alternatives in self.column_aliases.items():
            # 如果已经存在 canonical 列则跳过
            if canonical in df_copy.columns:
                continue
            # 找到第一个存在的备用列并重命名
            found = None
            for alt in alternatives:
                if alt in df_copy.columns:
                    found = alt
                    break
            if found:
                df_copy[canonical] = df_copy[found]
            else:
                # 如果是用于数值计算的列，填充为 NaN 或 0 视情况而定
                if canonical in ['num_games_owned', 'num_reviews', 'played_hours', 'playtime_last_two_weeks', 'playtime_at_review']:
                    df_copy[canonical] = 0
                else:
                    df_copy[canonical] = np.nan

        # 类型与规范化处理
        # 1) 统一 review_date 为 pandas datetime（无时区）
        if 'review_date' in df_copy.columns:
            try:
                df_copy['review_date'] = pd.to_datetime(df_copy['review_date'], errors='coerce')
            except Exception:
                pass
        # 2) 统一 appid 为数值型（尽量整数）
        if 'appid' in df_copy.columns:
            try:
                df_copy['appid'] = pd.to_numeric(df_copy['appid'], errors='coerce')
            except Exception:
                pass

        return df_copy
    
    def extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取文本特征
        
        Args:
            df: 包含评论数据的DataFrame
            
        Returns:
            添加了文本特征的DataFrame
        """
        logger.info("Extracting text features...")
        df = df.copy()
        df = self._standardize_columns(df)
        
        # 基本文本特征
        df['text_length'] = df['review_content_clean'].str.len()
        df['word_count'] = df['review_content_clean'].str.split().str.len()
        df['sentence_count'] = df['review_content_clean'].apply(lambda x: len(sent_tokenize(str(x))))
        df['avg_word_length'] = df['review_content_clean'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0
        )
        
        # 标点符号和特殊字符特征
        df['exclamation_count'] = df['review_content_clean'].str.count('!')
        df['question_count'] = df['review_content_clean'].str.count(r'\?')
        df['uppercase_ratio'] = df['review_content_clean'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
        )
        df['punctuation_ratio'] = df['review_content_clean'].apply(
            lambda x: sum(1 for c in str(x) if c in '.,!?;:') / len(str(x)) if len(str(x)) > 0 else 0
        )
        
        # 可读性指数
        df['readability_score'] = df['review_content_clean'].apply(
            lambda x: textstat.flesch_reading_ease(str(x)) if str(x).strip() else 0
        )
        df['flesch_kincaid_grade'] = df['review_content_clean'].apply(
            lambda x: textstat.flesch_kincaid_grade(str(x)) if str(x).strip() else 0
        )
        
        # 是否包含问句
        df['has_question'] = (df['question_count'] > 0).astype(int)
        
        # 是否包含外部引用（链接、@mentions等）
        df['has_url'] = df['review_content_clean'].str.contains(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            regex=True, na=False
        ).astype(int)
        
        df['has_mention'] = df['review_content_clean'].str.contains(
            r'@\w+', regex=True, na=False
        ).astype(int)
        
        df['has_external_reference'] = ((df['has_url'] == 1) | (df['has_mention'] == 1)).astype(int)
        
        logger.info("Basic text features extracted successfully")
        return df
    
    def extract_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取情感特征
        
        Args:
            df: 包含评论数据的DataFrame
            
        Returns:
            添加了情感特征的DataFrame
        """
        logger.info("Extracting sentiment features using VADER...")
        df = df.copy()
        df = self._standardize_columns(df)
        
        # VADER情感分析
        sentiment_scores = df['review_content_clean'].apply(
            lambda x: self.vader_analyzer.polarity_scores(str(x))
        )
        
        df['vader_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        df['vader_positive'] = sentiment_scores.apply(lambda x: x['pos'])
        df['vader_negative'] = sentiment_scores.apply(lambda x: x['neg'])
        df['vader_neutral'] = sentiment_scores.apply(lambda x: x['neu'])
        
        # 情感分类
        df['sentiment_category'] = df['vader_compound'].apply(
            lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral'
        )
        # 数值化情感标签：0=negative,1=neutral,2=positive（供统一系统直接使用）
        mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
        df['sentiment_label'] = df['sentiment_category'].map(mapping)
        
        logger.info("Sentiment features extracted successfully")
        return df
    
    def extract_game_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取游戏特定特征
        
        Args:
            df: 包含评论数据的DataFrame
            
        Returns:
            添加了游戏特定特征的DataFrame
        """
        logger.info("Extracting game-specific features...")
        df = df.copy()
        df = self._standardize_columns(df)
        
        # Bug投诉相关词汇
        bug_keywords = [
            'crash', 'crashes', 'crashed', 'crashing',
            'glitch', 'glitches', 'glitchy', 'glitching',
            'bug', 'bugs', 'buggy', 'bugged',
            'error', 'errors', 'freeze', 'freezes', 'freezing',
            'lag', 'lagging', 'laggy', 'stuttering',
            'broken', 'fix', 'patch', 'update'
        ]
        
        bug_pattern = '|'.join([f'\\b{word}\\b' for word in bug_keywords])
        df['contains_bug_report'] = df['review_content_clean'].str.contains(
            bug_pattern, case=False, regex=True, na=False
        ).astype(int)
        
        # 平衡性抱怨相关词汇
        balance_keywords = [
            'op', 'overpowered', 'over powered', 'too strong',
            'nerf', 'nerfed', 'nerfs', 'nerfing',
            'buff', 'buffed', 'buffs', 'buffing',
            'unbalanced', 'imbalanced', 'unfair',
            'cheater', 'cheaters', 'cheating', 'hacker',
            'meta', 'broken', 'toxic'
        ]
        
        balance_pattern = '|'.join([f'\\b{word}\\b' for word in balance_keywords])
        df['contains_balance_complaint'] = df['review_content_clean'].str.contains(
            balance_pattern, case=False, regex=True, na=False
        ).astype(int)
        
        # 付费相关争议
        monetization_keywords = [
            'pay to win', 'p2w', 'paywall', 'microtransaction',
            'dlc', 'expensive', 'overpriced', 'greedy',
            'money grab', 'cash grab', 'scam'
        ]
        
        monetization_pattern = '|'.join([f'\\b{word}\\b' for word in monetization_keywords])
        df['contains_monetization_complaint'] = df['review_content_clean'].str.contains(
            monetization_pattern, case=False, regex=True, na=False
        ).astype(int)
        
        # 性能相关问题
        performance_keywords = [
            'fps', 'framerate', 'frame rate', 'performance',
            'optimization', 'optimized', 'slow', 'fast',
            'graphics', 'visual', 'quality'
        ]
        
        performance_pattern = '|'.join([f'\\b{word}\\b' for word in performance_keywords])
        df['mentions_performance'] = df['review_content_clean'].str.contains(
            performance_pattern, case=False, regex=True, na=False
        ).astype(int)
        
        logger.info("Game-specific features extracted successfully")
        return df
    
    def extract_toxicity_features(self, df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
        """
        提取毒性/攻击性特征（GPU优化）
        
        Args:
            df: 包含评论数据的DataFrame
            batch_size: 批处理大小，GPU可以使用更大的批次
            
        Returns:
            添加了毒性特征的DataFrame
        """
        logger.info("Extracting toxicity features...")
        df = df.copy()
        df = self._standardize_columns(df)
        
        if self.toxicity_model:
            try:
                # 使用GPU优化的批量处理
                if self.device.type == 'cuda':
                    batch_size = min(batch_size, 64)  # GPU可以处理更大的批次
                else:
                    batch_size = min(batch_size, 16)  # CPU使用较小的批次
                
                toxicity_scores = []
                
                with torch.no_grad():  # 节省GPU内存
                    for i in range(0, len(df), batch_size):
                        batch_texts = df['review_content_clean'].iloc[i:i+batch_size].fillna('').tolist()
                        
                        # 限制文本长度以避免GPU内存不足
                        batch_texts = [text[:512] if len(text) > 512 else text for text in batch_texts]
                        
                        try:
                            batch_scores = self.toxicity_model.predict(batch_texts)
                            toxicity_scores.extend(batch_scores['toxicity'])
                        except Exception as e:
                            logger.warning(f"Error in batch {i//batch_size}: {e}")
                            # 如果批次失败，使用默认值
                            toxicity_scores.extend([0.0] * len(batch_texts))
                        
                        # 清理GPU缓存
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        
                        if i % (batch_size * 10) == 0:
                            logger.info(f"Processed {i}/{len(df)} samples for toxicity detection")
                
                df['toxicity_score'] = toxicity_scores
                df['is_toxic'] = (df['toxicity_score'] > 0.5).astype(int)
                
                logger.info("Toxicity features extracted using Detoxify (GPU optimized)")
                
            except Exception as e:
                logger.warning(f"Error in GPU toxicity detection: {e}")
                # 回退到关键词方法
                self._extract_toxicity_keywords(df)
        else:
            # 基于关键词的简单毒性检测
            self._extract_toxicity_keywords(df)
        
        return df
    
    def _extract_toxicity_keywords(self, df: pd.DataFrame):
        """基于关键词的毒性检测回退方法"""
        toxic_keywords = [
            'hate', 'stupid', 'idiot', 'trash', 'garbage',
            'suck', 'sucks', 'terrible', 'awful', 'horrible'
        ]
        
        toxic_pattern = '|'.join([f'\\b{word}\\b' for word in toxic_keywords])
        df['contains_toxic_keywords'] = df['review_content_clean'].str.contains(
            toxic_pattern, case=False, regex=True, na=False
        ).astype(int)
        
        df['toxicity_score'] = df['contains_toxic_keywords'] * 0.7  # 简单评分
        df['is_toxic'] = df['contains_toxic_keywords']
        
        logger.info("Toxicity features extracted using keyword-based method")

    def extract_perspective_features(self, df: pd.DataFrame, batch_size: int = 16, sleep_between_calls: float = 0.1) -> pd.DataFrame:
        """
        使用 Google Perspective API 为每条评论提取内容分类评分（TOXICITY / INSULT / SEVERE_TOXICITY 等）。

        - 仅在 `self.perspective_client` 已初始化时才会尝试调用；否则直接返回原始 DataFrame
        - 简单的批量实现，遇到错误会记录并用 NaN 回退

        Args:
            df: 包含评论的 DataFrame（需包含 `review_content_clean`）
            batch_size: 每次调用 API 的批量大小（API 的请求体可以是单条或多次单条调用，这里逐条调用以保持兼容性）
            sleep_between_calls: 在连续调用之间短暂 sleep，用于减缓速率（秒）

        Returns:
            带有 perspective_* 列的 DataFrame
        """
        df_copy = df.copy()
        df_copy = self._standardize_columns(df_copy)

        # 检查客户端
        if not getattr(self, 'perspective_client', None):
            logger.info("Perspective client not available; skipping Perspective feature extraction")
            return df_copy

        # 要请求的属性
        requested_attributes = {
            'TOXICITY': {},
            'SEVERE_TOXICITY': {},
            'INSULT': {},
            'THREAT': {},
            'IDENTITY_ATTACK': {}
        }

        texts = df_copy['review_content_clean'].fillna('').astype(str).tolist()
        n = len(texts)

        # 预建列
        for attr in requested_attributes.keys():
            df_copy[f'perspective_{attr.lower()}'] = np.nan

        # 逐条发送请求（Perspective client 的批量接口并不总是可用，且 discovery client 易于出错）
        # 如果希望更高吞吐，可改为并发或使用并行化并处理速率限制
        for i, text in enumerate(texts):
            if not text:
                continue
            try:
                body = {
                    'comment': {'text': text},
                    'requestedAttributes': requested_attributes
                }
                # 使用 discovery client 发起请求
                resp = self.perspective_client.comments().analyze(body=body).execute()
                scores = resp.get('attributeScores', {})
                for attr in requested_attributes.keys():
                    val = np.nan
                    if attr in scores and 'summaryScore' in scores[attr]:
                        val = scores[attr]['summaryScore'].get('value', np.nan)
                    df_copy.at[df_copy.index[i], f'perspective_{attr.lower()}'] = val
            except Exception as e:
                logger.warning(f"Perspective API call failed for index {i}: {e}")
                # 保持 NaN
            # 简单速率控制
            time.sleep(sleep_between_calls)

        logger.info(f"Perspective features extracted for {n} samples (may contain NaN for failures)")
        return df_copy
    
    def extract_tfidf_features(self, df: pd.DataFrame, max_features: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        提取TF-IDF特征
        
        Args:
            df: 包含评论数据的DataFrame
            max_features: TF-IDF特征的最大数量
            
        Returns:
            Tuple[DataFrame, TF-IDF矩阵]
        """
        logger.info(f"Extracting TF-IDF features (max_features={max_features})...")
        df = df.copy()
        df = self._standardize_columns(df)
        # 使用处理后的文本进行TF-IDF
        texts = df['review_content_processed'].fillna('').tolist()
        
        # 重新配置TF-IDF向量化器
        # 根据样本量动态设置min_df和max_df，避免冲突
        n_samples = len(df)
        min_df = 1
        # 如果样本量很小，max_df设为1.0，否则0.95
        if n_samples < 20:
            max_df = 1.0
        else:
            max_df = 0.95
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # 获取特征名称
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # 为每个文档创建TF-IDF统计特征（稀疏安全实现，避免 toarray）
        n_features = tfidf_matrix.shape[1] if tfidf_matrix.shape[1] > 0 else 1

        # row-wise max: sparse max returns a 2D matrix -> convert
        try:
            row_max = tfidf_matrix.max(axis=1).toarray().ravel()
        except Exception:
            # fallback: convert small chunks (should rarely occur)
            row_max = np.array([x.max() if x.nnz else 0.0 for x in tfidf_matrix])

        # row-wise mean across all features (including zeros)
        row_sum = np.asarray(tfidf_matrix.sum(axis=1)).ravel()
        row_mean = row_sum / float(n_features)

        # row-wise std: sqrt(mean(x^2) - mean(x)^2)
        row_sq_sum = np.asarray(tfidf_matrix.power(2).sum(axis=1)).ravel()
        row_mean_sq = row_sq_sum / float(n_features)
        row_std = np.sqrt(np.maximum(0.0, row_mean_sq - row_mean ** 2))

        # non-zero count per row
        row_nonzero = tfidf_matrix.getnnz(axis=1)

        df_copy = df.copy()
        df_copy['tfidf_max'] = row_max
        df_copy['tfidf_mean'] = row_mean
        df_copy['tfidf_std'] = row_std
        df_copy['tfidf_nonzero_count'] = row_nonzero
        
        logger.info(f"TF-IDF features extracted: {tfidf_matrix.shape}")
        return df_copy, tfidf_matrix
    
    def extract_bert_features(self, df: pd.DataFrame, max_length: int = 256, batch_size: int = 32) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """
        提取BERT嵌入特征（GPU优化）
        
        Args:
            df: 包含评论数据的DataFrame
            max_length: 文本的最大长度（GPU优化，使用较短长度）
            batch_size: 批处理大小
            
        Returns:
            Tuple[DataFrame, BERT嵌入矩阵]
        """
        logger.info("Extracting BERT embedding features (GPU optimized)...")
        df = self._standardize_columns(df)

        if self.sentence_model is None:
            logger.warning("BERT model not available, skipping BERT features")
            return df, None
        
        try:
            # 使用清洗后的文本
            texts = df['review_content_clean'].fillna('').tolist()
            
            # 截断过长的文本以节省GPU内存
            texts = [text[:max_length] if len(text) > max_length else text for text in texts]
            
            # 根据设备调整批次大小
            if self.device.type == 'cuda':
                batch_size = min(batch_size, 64)  # GPU可以处理更大的批次
            else:
                batch_size = min(batch_size, 16)  # CPU使用较小的批次
            
            embeddings = []
            
            with torch.no_grad():  # 节省GPU内存
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    
                    try:
                        batch_embeddings = self.sentence_model.encode(
                            batch_texts, 
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        embeddings.append(batch_embeddings)
                    except Exception as e:
                        logger.warning(f"Error in BERT batch {i//batch_size}: {e}")
                        # 如果批次失败，使用零向量
                        embeddings.append(np.zeros((len(batch_texts), 384)))
                    
                    # 清理GPU缓存
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    if i % (batch_size * 10) == 0:
                        logger.info(f"Processed {i}/{len(texts)} samples for BERT embeddings")
            
            embeddings = np.vstack(embeddings)
            
            # 添加BERT嵌入统计特征到DataFrame
            df_copy = df.copy()
            df_copy['bert_embedding_norm'] = np.linalg.norm(embeddings, axis=1)
            df_copy['bert_embedding_mean'] = np.mean(embeddings, axis=1)
            df_copy['bert_embedding_std'] = np.std(embeddings, axis=1)

            # 生成 embedding 聚类特征（cluster id 和 distance to center）
            try:
                cluster_df = self._generate_embedding_clusters(embeddings)
                # cluster_df is dict with 'cluster_id' and 'cluster_dist'
                df_copy['embed_cluster_id'] = cluster_df['cluster_id']
                df_copy['embed_cluster_dist'] = cluster_df['cluster_dist']
            except Exception as e:
                logger.warning(f"Embedding clustering failed: {e}")
            
            logger.info(f"BERT embeddings extracted: {embeddings.shape}")
            return df_copy, embeddings
            
        except Exception as e:
            logger.error(f"Error extracting BERT features: {e}")
            return df, None
    
    def extract_topic_features(self, df: pd.DataFrame, n_topics: int = 10, vectorizer: str = 'tfidf') -> pd.DataFrame:
        """
        使用LDA提取主题特征
        
        Args:
            df: 包含评论数据的DataFrame
            n_topics: 主题数量
            
        Returns:
            添加了主题特征的DataFrame
        """
        logger.info(f"Extracting topic features using LDA (n_topics={n_topics}, vectorizer={vectorizer})...")
        df = self._standardize_columns(df)
        
        # 释放 GPU 内存（如果之前使用过）
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("Cleared GPU cache before LDA")
        # 根据 vectorizer 选择文本表示：'tfidf' 或 'count'
        if vectorizer == 'tfidf':
            _, tfidf_matrix = self.extract_tfidf_features(df, max_features=2000)
            lda_input = tfidf_matrix
        elif vectorizer == 'count':
            # 使用 CountVectorizer 更适合 LDA（LDA 是基于计数的生成模型）
            texts = df['review_content_processed'].fillna('').tolist()
            count_vectorizer = CountVectorizer(
                max_features=2000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            lda_input = count_vectorizer.fit_transform(texts)
            # 更新 self.tfidf_vectorizer 名称并非必要；保留 tfidf_vectorizer 不变
        else:
            raise ValueError(f"Unsupported vectorizer type for LDA: {vectorizer}")
        
        # 训练LDA模型
        # 注意：在 Windows 上使用 n_jobs=1 避免序列化问题（特别是与 PyTorch 共存时）
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100,
            learning_method='batch',
            n_jobs=1  # Windows 上使用单进程避免序列化错误
        )

        topic_distributions = self.lda_model.fit_transform(lda_input)
        logger.info("LDA model trained successfully")
        
        # 添加主题特征
        df_copy = df.copy()
        
        # 为每个主题添加概率分布
        for i in range(n_topics):
            df_copy[f'topic_{i}_prob'] = topic_distributions[:, i]
        
        # 主导主题
        df_copy['dominant_topic'] = np.argmax(topic_distributions, axis=1)
        df_copy['dominant_topic_prob'] = np.max(topic_distributions, axis=1)
        
        # 主题多样性（熵）
        df_copy['topic_entropy'] = -np.sum(
            topic_distributions * np.log(topic_distributions + 1e-10), axis=1
        )

        # Topic PCA: 将主题分布降维为连续特征（默认 K = min(5, n_topics)）。
        # 持久化行为：如果 self.topic_pca 为空则 fit（通常在训练阶段），否则复用已有的 PCA（推理阶段）。
        try:
            k = min(5, n_topics)
            if self.topic_pca is None:
                # 在训练时拟合 PCA 并持久化到 self.topic_pca
                self.topic_pca = PCA(n_components=k)
                topic_pca_feats = self.topic_pca.fit_transform(topic_distributions)
                logger.info(f"Fitted topic_pca with n_components={k}")
            else:
                # 复用已存在的 PCA（可能来自训练时保存/加载）
                # 如果已拟合 PCA 的 n_components 与期望值不一致，则记录信息但仍使用已存在的模型
                existing_k = getattr(self.topic_pca, 'n_components_', getattr(self.topic_pca, 'n_components', None))
                if existing_k is not None and existing_k != k:
                    logger.info(f"Using existing topic_pca with n_components={existing_k} (requested {k})")
                topic_pca_feats = self.topic_pca.transform(topic_distributions)

            for j in range(topic_pca_feats.shape[1]):
                df_copy[f'topic_pca_{j}'] = topic_pca_feats[:, j]
        except Exception as e:
            logger.warning(f"Topic PCA failed: {e}")

        # top-2 topic interaction: top2 index pair and prob product
        try:
            # 获取 top2 索引（按行）
            top2_idx = np.argpartition(-topic_distributions, 2, axis=1)[:, :2]
            # 确保顺序 (小->大) 或按概率大小排序
            top_order = np.argsort(-topic_distributions[np.arange(topic_distributions.shape[0])[:, None], top2_idx], axis=1)
            top_sorted = np.take_along_axis(top2_idx, top_order, axis=1)
            top1 = top_sorted[:, 0]
            top2 = top_sorted[:, 1]
            prod = topic_distributions[np.arange(topic_distributions.shape[0]), top1] * topic_distributions[np.arange(topic_distributions.shape[0]), top2]
            df_copy['topic_top2_pair'] = [f"t{min(a,b)}_t{max(a,b)}" for a, b in zip(top1, top2)]
            df_copy['topic_top2_prob_product'] = prod
        except Exception as e:
            logger.warning(f"Top2 topic interaction creation failed: {e}")
        
        # 预定义主题标签
        topic_labels = [
            'gameplay_mechanics',
            'technical_issues',
            'story_content',
            'graphics_performance',
            'multiplayer_features',
            'monetization_concerns',
            'character_balance',
            'user_interface',
            'community_feedback',
            'general_opinion',
            'cheating'
        ]

        # 自动生成主题名称（高频关键词）
        topic_names = []
        n_top_words = 5  # 可调整关键词数量
        if hasattr(self.lda_model, 'components_'):
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
                topic_names.append(', '.join(top_words))
        else:
            topic_names = [f'topic_{i}' for i in range(n_topics)]

        # 自动映射：用关键词与预定义标签做简单相似度匹配
        mapped_labels = []
        for topic_keywords in topic_names:
            # 计算每个预定义标签与关键词的相似度，选最接近的
            best_label = difflib.get_close_matches(topic_keywords, topic_labels, n=1, cutoff=0.0)
            if best_label:
                mapped_labels.append(best_label[0])
            else:
                mapped_labels.append(topic_keywords)

        # 如果主题数量超过预定义标签，则多余主题用关键词
        if len(mapped_labels) < n_topics:
            mapped_labels += topic_names[len(mapped_labels):]

        df_copy['topic_category'] = df_copy['dominant_topic'].map(
            {i: mapped_labels[i] for i in range(n_topics)}
        )
        
        logger.info("Topic features extracted successfully")
        return df_copy

    def load_reviews_from_combined_dir(self, combined_dir: str, max_rows_per_file: Optional[int] = None) -> pd.DataFrame:
        """
        从按类型合并的评论目录加载所有评论文件，并在每条记录上添加 `genre` 字段（从文件名推断）。

        支持 .csv 和 .xlsx 文件。
        """
        all_dfs = []
        for fname in os.listdir(combined_dir):
            path = os.path.join(combined_dir, fname)
            if not os.path.isfile(path):
                continue
            lower = fname.lower()
            try:
                if lower.endswith('.csv'):
                    df = pd.read_csv(path, encoding='utf-8', nrows=max_rows_per_file)
                elif lower.endswith(('.xls', '.xlsx')):
                    # 通过 engine='openpyxl' 读取；对于非常大的 excel 文件，可限制 nrows
                    if max_rows_per_file is not None:
                        df = pd.read_excel(path, engine='openpyxl', nrows=max_rows_per_file)
                    else:
                        df = pd.read_excel(path, engine='openpyxl')
                else:
                    logger.debug(f"Skipping unsupported file type: {path}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to read reviews file {path}: {e}")
                continue

            # 推断 genre：尝试从文件名如 combined_fps_reviews.xlsx 提取 fps
            m = re.search(r'combined[_\-]?(.+?)[_\-]?reviews', fname, flags=re.IGNORECASE)
            if m:
                genre = m.group(1).lower()
            else:
                # 备选：文件名前缀
                genre = os.path.splitext(fname)[0]

            df = df.copy()
            df['genre'] = genre

            # 尝试统一列名（最少需要 review_date 或 review_time 和 review content）
            # 不强制，但尽可能解析日期列
            if 'review_date' in df.columns:
                df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
            else:
                for c in ['date', 'created_at', 'timestamp']:
                    if c in df.columns:
                        df['review_date'] = pd.to_datetime(df[c], errors='coerce')
                        break

            all_dfs.append(df)

        if not all_dfs:
            logger.warning(f"No review files loaded from {combined_dir}")
            return pd.DataFrame()

        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} reviews from {combined_dir}")
        return combined

    def load_news_from_dir(self, news_dir: str) -> pd.DataFrame:
        """
        从新闻/更新目录加载所有新闻文件，返回标准化的 DataFrame，包含 ['appid', 'news_date']。

        文件名样式支持 cleaned_{appid}_news.xlsx，如果文件本身包含 appid 列则优先使用。
        """
        all_rows = []
        for fname in os.listdir(news_dir):
            path = os.path.join(news_dir, fname)
            if not os.path.isfile(path):
                continue
            lower = fname.lower()
            try:
                if lower.endswith('.csv'):
                    df = pd.read_csv(path, encoding='utf-8')
                elif lower.endswith(('.xls', '.xlsx')):
                    df = pd.read_excel(path, engine='openpyxl')
                else:
                    logger.debug(f"Skipping unsupported news file type: {path}")
                    continue
            except Exception as e:
                logger.warning(f"Failed to read news file {path}: {e}")
                continue

            # 提取 appid：优先使用列，否则从文件名 cleaned_{appid}_news.* 提取数字
            if 'appid' in df.columns:
                df['appid'] = pd.to_numeric(df['appid'], errors='coerce')
            else:
                m = re.search(r'cleaned[_\-]?(\d+)[_\-]?news', fname, flags=re.IGNORECASE)
                if m:
                    df['appid'] = int(m.group(1))
                else:
                    df['appid'] = np.nan

            # 解析可能的日期列
            date_col = None
            for c in ['post_date', 'date', 'published_at', 'publish_date', 'news_date', 'created_at']:
                if c in df.columns:
                    date_col = c
                    break

            if date_col:
                df['news_date'] = pd.to_datetime(df[date_col], errors='coerce')
            else:
                # 尝试查找任何 datetime-like 列
                for c in df.columns:
                    if 'date' in c.lower() or 'time' in c.lower():
                        try:
                            df['news_date'] = pd.to_datetime(df[c], errors='coerce')
                            date_col = c
                            break
                        except Exception:
                            continue

            # 仅保留有解析时间的行
            df = df.loc[df['news_date'].notna(), ['appid', 'news_date']].copy()
            all_rows.append(df)

        if not all_rows:
            logger.warning(f"No news files loaded from {news_dir}")
            return pd.DataFrame(columns=['appid', 'news_date'])

        news_combined = pd.concat(all_rows, ignore_index=True)
        # 强制 appid 为整数/字符串相同类型
        news_combined['appid'] = pd.to_numeric(news_combined['appid'], errors='coerce')
        news_combined = news_combined.dropna(subset=['appid']).copy()
        news_combined['appid'] = news_combined['appid'].astype(int)
        logger.info(f"Loaded {len(news_combined)} news rows from {news_dir}")
        return news_combined

    def compute_within_update_window(self, comments_df: pd.DataFrame, news_df: pd.DataFrame, hours: int = 48) -> pd.DataFrame:
        """
        计算评论是否发生在对应游戏最近一次更新后的指定小时窗口内（默认 48 小时）。

        该方法会调用 `align_comments_to_latest_patch` 来确保没有未来信息泄露。
        返回添加了 'hours_since_latest_news' 和 f'within_{hours}h_update' 列的 DataFrame。
        """
        comments = self.align_comments_to_latest_patch(comments_df, news_df, appid_col='appid', comment_date_col='review_date')
        col_name = f'within_{hours}h_update'
        comments[col_name] = comments['hours_since_latest_news'].between(0, hours).astype(int)
        logger.info(f"Computed {col_name} for {len(comments)} comments")
        return comments

    def aggregate_comments_by_news_windows(self, news_df: pd.DataFrame, comments_df: pd.DataFrame,
                                           appid_col: str = 'appid', news_date_col: str = 'news_date',
                                           comment_date_col: str = 'review_date',
                                           sentiment_col: str = 'vader_compound',
                                           windows: List[int] = [24, 48, 72]) -> pd.DataFrame:
        """
    为每个 news（按 appid + news_date）计算 windows（小时）内的评论统计（count, avg_sent, neg_ratio, pos_ratio）。

        返回 news_level DataFrame，包含原始 news_date 与为每个 window 计算的统计。
        """
        news = news_df.copy()
        comments = comments_df.copy()
        news[news_date_col] = pd.to_datetime(news[news_date_col], errors='coerce')
        comments[comment_date_col] = pd.to_datetime(comments[comment_date_col], errors='coerce')

        rows = []

        # 按 appid 逐组处理
        for appid, news_grp in news.groupby(appid_col):
            news_grp = news_grp.sort_values(news_date_col).reset_index(drop=True)
            comm_grp = comments[comments[appid_col] == appid].sort_values(comment_date_col)
            if comm_grp.empty:
                for _, nrow in news_grp.iterrows():
                    out = {appid_col: appid, news_date_col: nrow[news_date_col]}
                    for w in windows:
                        out.update({f'count_in_{w}h': 0,
                                    f'avg_sent_in_{w}h': np.nan,
                                    f'neg_ratio_in_{w}h': 0.0,
                                    f'pos_ratio_in_{w}h': 0.0})
                    rows.append(out)
                continue

            comm_times = comm_grp[comment_date_col].values.astype('datetime64[ns]')
            sentiments = comm_grp.get(sentiment_col, pd.Series(np.nan, index=comm_grp.index)).to_numpy(dtype=float)

            for _, nrow in news_grp.iterrows():
                nd = np.datetime64(nrow[news_date_col])
                out = {appid_col: appid, news_date_col: nrow[news_date_col]}
                l_idx = np.searchsorted(comm_times, nd, side='right')
                for w in windows:
                    upper = nd + np.timedelta64(int(w * 3600), 's')
                    r_idx = np.searchsorted(comm_times, upper, side='right')
                    if r_idx <= l_idx:
                        out[f'count_in_{w}h'] = 0
                        out[f'avg_sent_in_{w}h'] = np.nan
                        out[f'neg_ratio_in_{w}h'] = 0.0
                        out[f'pos_ratio_in_{w}h'] = 0.0
                    else:
                        seg = sentiments[l_idx:r_idx]
                        valid = ~np.isnan(seg)
                        cnt = int((r_idx - l_idx))
                        avg_sent = float(np.nanmean(seg)) if valid.sum() else np.nan
                        neg_ratio = float(((seg <= -0.05) & valid).sum() / cnt) if cnt else 0.0
                        pos_ratio = float(((seg >= 0.05) & valid).sum() / cnt) if cnt else 0.0
                        out[f'count_in_{w}h'] = cnt
                        out[f'avg_sent_in_{w}h'] = avg_sent
                        out[f'neg_ratio_in_{w}h'] = neg_ratio
                        out[f'pos_ratio_in_{w}h'] = pos_ratio
                rows.append(out)

        result = pd.DataFrame(rows)
        return result

    def process_and_save_news_level(self, reviews_dir: str, news_dir: str, output_path: str,
                                    max_rows_per_file: Optional[int] = None,
                                    windows: List[int] = [24, 48, 72]):
        """
        便利方法：从目录加载 reviews/news（respectively 使用 max_rows_per_file 限制），计算 news-level 聚合并保存为 parquet。
        """
        reviews = self.load_reviews_from_combined_dir(reviews_dir, max_rows_per_file=max_rows_per_file)
        news = self.load_news_from_dir(news_dir)

        if reviews.empty or news.empty:
            logger.warning('Reviews or news empty; aborting news-level aggregation')
            return None

        news_level = self.aggregate_comments_by_news_windows(news, reviews, windows=windows)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            news_level.to_parquet(output_path, index=False)
            logger.info(f'News-level aggregation saved to {output_path} (rows: {len(news_level)})')
        except Exception as e:
            # 不再回退到 CSV；向用户提示缺失 parquet 引擎或其他写入问题
            msg = (
                f"Failed to write parquet to {output_path}: {e}.\n"
                "Ensure a parquet engine is installed (pyarrow or fastparquet).\n"
                "Install with: pip install pyarrow"
            )
            logger.error(msg)
            raise RuntimeError(msg)
        return news_level

    def align_comments_to_latest_patch(self, comments_df: pd.DataFrame, news_df: pd.DataFrame,
                                       appid_col: str = 'appid', comment_date_col: str = 'review_date') -> pd.DataFrame:
        """
        为每条评论查找紧邻其之前（或最近）的更新/新闻时间，并计算时间差（小时）。

        这会避免数据泄露：对每条评论只使用其评论时间之前已发生的新闻时间。

        Args:
            comments_df: 含合并评论的 DataFrame（按 genre 合并）
            news_df: 标准化后含 ['appid','news_date'] 的 DataFrame
            appid_col: comments_df 中表示游戏 id 的列名
            comment_date_col: 评论时间列，需为可解析的 datetime

        Returns:
            comments_df 的拷贝，包含两列新字段：
              - 'latest_news_date': 评论时间之前最近的 news_date (或 NaT)
              - 'hours_since_latest_news': 评论距离该 news 的小时数 (>=0)
        """
        comments = comments_df.copy()
        # 确保评论时间为 datetime
        comments[comment_date_col] = pd.to_datetime(comments[comment_date_col], errors='coerce')

        # 如果 news_df 为空则快速返回
        if news_df is None or news_df.empty:
            comments['latest_news_date'] = pd.NaT
            comments['hours_since_latest_news'] = np.nan
            return comments

        # 为性能：将 news 按 appid 分组并排序
        news_sorted = news_df.sort_values(['appid', 'news_date']).copy()

        # 准备结果列
        latest_dates = []

        # 逐行查找——对大数据集可考虑更高效的向量化或 merge_asof
        try:
            # 使用 pandas.merge_asof 来做按时间的向前合并（需排序）
            comments_sorted = comments.sort_values([appid_col, comment_date_col]).copy()
            # merge_asof 只支持合并在同一列名上，这里先按 appid 分组做拼接
            merged = pd.merge_asof(
                comments_sorted,
                news_sorted.rename(columns={'news_date': 'latest_news_date'}),
                left_on=comment_date_col,
                right_on='latest_news_date',
                by=appid_col,
                direction='backward',
                allow_exact_matches=True
            )
            # merged 现在包含 latest_news_date
            merged['hours_since_latest_news'] = (
                (merged[comment_date_col] - merged['latest_news_date']).dt.total_seconds() / 3600
            )
            # 恢复原始顺序
            merged = merged.reindex(comments.index)
            # 填充回结果
            comments = merged
        except Exception:
            # 退回到按-appid 迭代方法（更慢但健壮）
            news_grouped = {k: g['news_date'].sort_values().values for k, g in news_sorted.groupby('appid')}
            for _, row in comments.iterrows():
                app = row.get(appid_col)
                cdate = row.get(comment_date_col)
                if pd.isna(cdate) or app not in news_grouped:
                    latest_dates.append(pd.NaT)
                    continue
                # 找到小于等于 cdate 的最大 news_date
                candidates = news_grouped.get(app, [])
                prior = candidates[candidates <= np.datetime64(cdate)] if len(candidates) else []
                if len(prior):
                    latest_dates.append(pd.Timestamp(prior[-1]))
                else:
                    latest_dates.append(pd.NaT)

            comments['latest_news_date'] = latest_dates
            comments['hours_since_latest_news'] = (
                (comments[comment_date_col] - comments['latest_news_date']).dt.total_seconds() / 3600
            )

        # 确保非负数值（注：如果找不到 prior 更新，hours 为 NaN）
        comments['hours_since_latest_news'] = comments['hours_since_latest_news'].where(
            comments['hours_since_latest_news'] >= 0, np.nan
        )

        return comments

    def aggregate_time_window_features(self, comments_df: pd.DataFrame, windows: List[int] = [24, 48, 72],
                                       time_col: str = 'hours_since_latest_news', sentiment_col: str = 'vader_compound') -> pd.DataFrame:
        """
        基于相对于最近一次更新的小时数，在多个时间窗口内聚合评论/情绪统计特征。

        Args:
            comments_df: 包含每条评论相对于最近更新小时数的 DataFrame（需包含 time_col）
            windows: 窗口大小（小时），会生成每个窗口的聚合特征
            time_col: 表示 "距最新更新的小时数" 的列名
            sentiment_col: 用于聚合情绪的列名

        Returns:
            DataFrame，每条评论复制并附加窗口级别的聚合特征（例如 rate_in_24h, neg_ratio_24h 等）
        """
        df = comments_df.copy()
        # 确保存在 group keys：按 genre 或 appid 聚合通常更有用，尝试识别
        group_keys = []
        for k in ['genre', 'app_id', 'appid', 'game_id']:
            if k in df.columns:
                group_keys.append(k)
                break
        # 如果没有分组键则做全局统计
        group_cols = group_keys if group_keys else None

        for w in windows:
            mask = df[time_col].notna() & (df[time_col] <= w)
            # 计算窗口内每组的计数/率/情感汇总
            if group_cols:
                agg = (
                    df[mask].groupby(group_cols)
                    .agg(
                        **{
                            f'count_in_{w}h': ('review_content_clean', 'count'),
                            f'avg_sentiment_in_{w}h': (sentiment_col, 'mean'),
                            f'neg_ratio_in_{w}h': (sentiment_col, lambda x: (x <= -0.05).sum() / len(x) if len(x) else 0),
                            f'pos_ratio_in_{w}h': (sentiment_col, lambda x: (x >= 0.05).sum() / len(x) if len(x) else 0),
                        }
                    )
                    .reset_index()
                )
                df = df.merge(agg, on=group_cols, how='left')
            else:
                subset = df[mask]
                count = len(subset)
                avg_sent = subset[sentiment_col].mean() if count else 0
                neg_ratio = (subset[sentiment_col] <= -0.05).sum() / count if count else 0
                pos_ratio = (subset[sentiment_col] >= 0.05).sum() / count if count else 0
                df[f'count_in_{w}h'] = count
                df[f'avg_sentiment_in_{w}h'] = avg_sent
                df[f'neg_ratio_in_{w}h'] = neg_ratio
                df[f'pos_ratio_in_{w}h'] = pos_ratio

        # 填充缺失值
        df.fillna({c: 0 for c in df.columns if c.startswith('count_in_') or c.startswith('avg_sentiment_in_') or c.startswith('neg_ratio_in_') or c.startswith('pos_ratio_in_')}, inplace=True)

        return df
    
    def extract_author_features(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        提取作者相关特征
        
        Args:
            df: 包含评论数据的DataFrame
            
        Returns:
            添加了作者特征的DataFrame
        """
        if verbose:
            logger.info("Extracting author-related features...")
        df_copy = df.copy()
        df_copy = self._standardize_columns(df_copy)
        # 强制相关列为数值类型，非数值填充为0
        numeric_cols = ['num_games_owned', 'num_reviews', 'played_hours', 'playtime_last_two_weeks', 'playtime_at_review']
        for col in numeric_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)

        # 现有的作者统计特征
        df_copy['games_per_review_ratio'] = df_copy['num_games_owned'] / (df_copy['num_reviews'] + 1)
        df_copy['hours_per_game_ratio'] = df_copy['played_hours'] / (df_copy['num_games_owned'] + 1)

        # 作者活跃度特征
        df_copy['is_active_reviewer'] = (df_copy['num_reviews'] > 10).astype(int)
        df_copy['is_hardcore_gamer'] = (df_copy['played_hours'] > 100).astype(int)
        df_copy['is_casual_gamer'] = (df_copy['played_hours'] <= 10).astype(int)

        # 最近游戏活跃度
        df_copy['recent_activity_ratio'] = (
            df_copy['playtime_last_two_weeks'] / (df_copy['playtime_at_review'] + 1)
        )

        # 购买行为特征
        df_copy['is_steam_purchaser'] = df_copy['steam_purchase'].astype(int)
        df_copy['received_free'] = df_copy['received_for_free'].astype(int)

        # 历史声望估算（整合评论互动指标）
        reputation_components = [
            0.2 * np.log1p(df_copy['num_games_owned']),
            0.3 * np.log1p(df_copy['num_reviews']),
            0.2 * np.log1p(df_copy['played_hours'])
        ]
        
        # 如果有评论互动数据，加入声望计算
        if 'votes_up' in df_copy.columns:
            reputation_components.append(0.2 * np.log1p(df_copy['votes_up']))
        if 'weighted_vote_score' in df_copy.columns:
            reputation_components.append(0.1 * df_copy['weighted_vote_score'])
        
        df_copy['author_reputation_score'] = sum(reputation_components)
        
        # 评论者质量分类（基于互动指标）
        if 'votes_up' in df_copy.columns:
            # 高质量评论者：经常获得高赞
            df_copy['is_quality_reviewer'] = (df_copy['votes_up'] >= 10).astype(int)
            
            # 影响力评论者：总赞数高
            if 'num_reviews' in df_copy.columns:
                avg_votes_per_review = df_copy['votes_up'] / (df_copy['num_reviews'] + 1)
                df_copy['avg_votes_per_review'] = avg_votes_per_review
                df_copy['is_influential_reviewer'] = (avg_votes_per_review >= 5).astype(int)
        
        # 标准化声望分数
        if df_copy['author_reputation_score'].std() > 0:
            df_copy['author_reputation_score'] = (
                (df_copy['author_reputation_score'] - df_copy['author_reputation_score'].mean()) /
                df_copy['author_reputation_score'].std()
            )

        if verbose:
            logger.info("Author features extracted successfully")
        return df_copy

    def extract_sentiment_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-appid or per-genre sentiment group statistics and interaction features."""
        df_copy = df.copy()
        # choose group key
        group_key = None
        for k in ['appid', 'app_id', 'genre']:
            if k in df_copy.columns:
                group_key = k
                break

        if group_key is None:
            # no grouping available, return unchanged
            return df_copy

        # ensure vader_compound exists
        if 'vader_compound' not in df_copy.columns:
            return df_copy

        # compute group mean/std
        df_copy['_group_sent_mean'] = df_copy.groupby(group_key)['vader_compound'].transform('mean')
        df_copy['_group_sent_std'] = df_copy.groupby(group_key)['vader_compound'].transform('std').fillna(0.0)
        # z-score
        df_copy['vader_compound_z'] = (df_copy['vader_compound'] - df_copy['_group_sent_mean']) / (df_copy['_group_sent_std'] + 1e-6)

        # interaction features
        if 'author_reputation_score' in df_copy.columns:
            df_copy['sentiment_x_reputation'] = df_copy['vader_compound'] * df_copy['author_reputation_score']

        # interaction with bug reports
        if 'contains_bug_report' in df_copy.columns:
            df_copy['sentiment_x_bug'] = df_copy['vader_compound'] * df_copy['contains_bug_report']

        # drop temp cols
        df_copy.drop(columns=['_group_sent_mean', '_group_sent_std'], inplace=True, errors='ignore')

        return df_copy

    def extract_review_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        提取评论互动与影响力特征
        
        Args:
            df: 包含评论数据的DataFrame
            
        Returns:
            添加了评论互动特征的DataFrame
            
        Features:
            - voted_up: 是否推荐（布尔值，可作为强标签）
            - votes_up: 有用票数（反映评论质量）
            - votes_funny: 有趣票数（反映评论风格）
            - comment_count: 回复数量（反映讨论热度）
            - weighted_vote_score: Steam 权重评分（综合质量）
        """
        logger.info("Extracting review engagement features...")
        df_copy = df.copy()
        df_copy = self._standardize_columns(df_copy)
        
        # 确保数值列为正确类型
        engagement_cols = ['votes_up', 'votes_funny', 'comment_count', 'weighted_vote_score']
        for col in engagement_cols:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
        
        # 1. 推荐状态（可作为情感标签）
        if 'voted_up' in df_copy.columns:
            df_copy['is_recommended'] = df_copy['voted_up'].astype(int)
        else:
            df_copy['is_recommended'] = 0
        
        # 2. 评论质量指标（有用投票）
        if 'votes_up' in df_copy.columns:
            df_copy['votes_up_log'] = np.log1p(df_copy['votes_up'])
            df_copy['is_helpful_review'] = (df_copy['votes_up'] >= 5).astype(int)  # 至少5票认为有用
            df_copy['is_highly_helpful'] = (df_copy['votes_up'] >= 20).astype(int)  # 高质量评论
        else:
            df_copy['votes_up_log'] = 0
            df_copy['is_helpful_review'] = 0
            df_copy['is_highly_helpful'] = 0
        
        # 3. 评论风格指标（有趣投票）
        if 'votes_funny' in df_copy.columns:
            df_copy['votes_funny_log'] = np.log1p(df_copy['votes_funny'])
            df_copy['is_funny_review'] = (df_copy['votes_funny'] >= 3).astype(int)
            
            # 计算有趣/有用比率（反映评论风格：严肃 vs 幽默）
            if 'votes_up' in df_copy.columns:
                total_votes = df_copy['votes_up'] + df_copy['votes_funny'] + 1
                df_copy['funny_ratio'] = df_copy['votes_funny'] / total_votes
                df_copy['serious_ratio'] = df_copy['votes_up'] / total_votes
        else:
            df_copy['votes_funny_log'] = 0
            df_copy['is_funny_review'] = 0
            df_copy['funny_ratio'] = 0
            df_copy['serious_ratio'] = 0
        
        # 4. 讨论热度（回复数量）
        if 'comment_count' in df_copy.columns:
            df_copy['comment_count_log'] = np.log1p(df_copy['comment_count'])
            df_copy['is_controversial'] = (df_copy['comment_count'] >= 10).astype(int)  # 引发大量讨论
            df_copy['is_highly_discussed'] = (df_copy['comment_count'] >= 30).astype(int)
        else:
            df_copy['comment_count_log'] = 0
            df_copy['is_controversial'] = 0
            df_copy['is_highly_discussed'] = 0
        
        # 5. Steam 权重评分（综合质量）
        if 'weighted_vote_score' in df_copy.columns:
            df_copy['weighted_score_normalized'] = df_copy['weighted_vote_score']
            df_copy['is_high_quality'] = (df_copy['weighted_vote_score'] >= 0.7).astype(int)
        else:
            df_copy['weighted_score_normalized'] = 0
            df_copy['is_high_quality'] = 0
        
        # 6. 综合影响力评分
        if all(col in df_copy.columns for col in ['votes_up', 'votes_funny', 'comment_count']):
            df_copy['engagement_score'] = (
                0.5 * df_copy['votes_up_log'] +
                0.2 * df_copy['votes_funny_log'] +
                0.3 * df_copy['comment_count_log']
            )
            
            # 标准化影响力评分
            if df_copy['engagement_score'].std() > 0:
                df_copy['engagement_score_normalized'] = (
                    (df_copy['engagement_score'] - df_copy['engagement_score'].mean()) /
                    df_copy['engagement_score'].std()
                )
            else:
                df_copy['engagement_score_normalized'] = 0
        else:
            df_copy['engagement_score'] = 0
            df_copy['engagement_score_normalized'] = 0
        
        # 7. 评论影响力等级（分类特征）
        if 'engagement_score' in df_copy.columns:
            conditions = [
                df_copy['engagement_score'] <= df_copy['engagement_score'].quantile(0.25),
                df_copy['engagement_score'] <= df_copy['engagement_score'].quantile(0.50),
                df_copy['engagement_score'] <= df_copy['engagement_score'].quantile(0.75),
                df_copy['engagement_score'] > df_copy['engagement_score'].quantile(0.75)
            ]
            choices = ['low_engagement', 'medium_engagement', 'high_engagement', 'viral']
            df_copy['engagement_level'] = np.select(conditions, choices, default='low_engagement')
        
        # 8. 评论与情感的交互特征
        if 'vader_compound' in df_copy.columns:
            # 负面评论但高赞（可能是建设性批评）
            if 'votes_up' in df_copy.columns:
                df_copy['negative_but_helpful'] = (
                    (df_copy['vader_compound'] < -0.3) & (df_copy['votes_up'] >= 10)
                ).astype(int)
                
                # 正面评论但低赞（可能是刷评或无价值内容）
                df_copy['positive_but_unhelpful'] = (
                    (df_copy['vader_compound'] > 0.3) & (df_copy['votes_up'] <= 1)
                ).astype(int)
            
            # 争议性评论（高讨论+极端情感）
            if 'comment_count' in df_copy.columns:
                df_copy['controversial_sentiment'] = (
                    (df_copy['comment_count'] >= 10) & 
                    (np.abs(df_copy['vader_compound']) >= 0.5)
                ).astype(int)
        
        # 9. 推荐与情感一致性
        if 'voted_up' in df_copy.columns and 'vader_compound' in df_copy.columns:
            # 推荐但情感负面（矛盾）
            df_copy['recommendation_sentiment_mismatch'] = (
                ((df_copy['voted_up'] == 1) & (df_copy['vader_compound'] < -0.1)) |
                ((df_copy['voted_up'] == 0) & (df_copy['vader_compound'] > 0.1))
            ).astype(int)
        else:
            df_copy['recommendation_sentiment_mismatch'] = 0
        
        logger.info("Review engagement features extracted successfully")
        return df_copy

    def extract_syntax_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract POS proportion features and simple negation/intensifier counts using spaCy if available.

        Falls back to simple heuristics if spaCy isn't installed.
        """
        df_copy = df.copy()
        texts = df_copy.get('review_content_clean', df_copy.get('review_content', pd.Series([''] * len(df_copy))))

        try:
            import spacy
            # load model if available; do not attempt automatic download here
            nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
            docs = list(nlp.pipe(texts.astype(str).tolist(), batch_size=256))

            props = {'prop_noun': [], 'prop_verb': [], 'prop_adj': [], 'prop_adv': [], 'negation_count': [], 'intensifier_count': []}
            negations = set(['not', 'no', 'never', 'none', 'nobody', 'neither', 'cannot', "can't"]) 
            intensifiers = set(['very', 'extremely', 'really', 'so', 'too', 'incredibly'])

            for doc in docs:
                n_tokens = len(doc)
                if n_tokens == 0:
                    props['prop_noun'].append(0.0)
                    props['prop_verb'].append(0.0)
                    props['prop_adj'].append(0.0)
                    props['prop_adv'].append(0.0)
                    props['negation_count'].append(0)
                    props['intensifier_count'].append(0)
                    continue

                noun = sum(1 for t in doc if t.pos_ == 'NOUN')
                verb = sum(1 for t in doc if t.pos_ == 'VERB')
                adj = sum(1 for t in doc if t.pos_ == 'ADJ')
                adv = sum(1 for t in doc if t.pos_ == 'ADV')
                tokens_low = [t.text.lower() for t in doc]
                neg_count = sum(1 for tok in tokens_low if tok in negations)
                int_count = sum(1 for tok in tokens_low if tok in intensifiers)

                props['prop_noun'].append(noun / n_tokens)
                props['prop_verb'].append(verb / n_tokens)
                props['prop_adj'].append(adj / n_tokens)
                props['prop_adv'].append(adv / n_tokens)
                props['negation_count'].append(neg_count)
                props['intensifier_count'].append(int_count)

            for k, v in props.items():
                df_copy[k] = v

        except Exception as e:
            # fallback simple heuristic token scanning
            logger.warning(f"spaCy not available or failed; using fallback syntax heuristics: {e}")
            negations = set(['not', 'no', 'never', 'none', 'nobody', 'neither', 'cannot', "can't"]) 
            intensifiers = set(['very', 'extremely', 'really', 'so', 'too', 'incredibly'])
            props = {'prop_noun': [], 'prop_verb': [], 'prop_adj': [], 'prop_adv': [], 'negation_count': [], 'intensifier_count': []}
            for text in texts.astype(str):
                toks = text.split()
                n = len(toks)
                low = [t.lower().strip(".,!?;:()\"'") for t in toks]
                neg_count = sum(1 for t in low if t in negations)
                int_count = sum(1 for t in low if t in intensifiers)
                # fallback proportions are zeros (can't POS tag without spacy)
                props['prop_noun'].append(0.0)
                props['prop_verb'].append(0.0)
                props['prop_adj'].append(0.0)
                props['prop_adv'].append(0.0)
                props['negation_count'].append(neg_count)
                props['intensifier_count'].append(int_count)

            for k, v in props.items():
                df_copy[k] = v

        return df_copy
    
    def create_comprehensive_features(
        self, 
        df: pd.DataFrame, 
        news_df: Optional[pd.DataFrame] = None,
        use_news_events: bool = False,
        use_perspective: bool = False,
    event_windows: List[int] = [24, 48, 72],
        include_bert: bool = True,
        include_topics: bool = True,
        n_topics: int = 8,
        tfidf_features: int = 500,
        max_bert_length: int = 256,
        bert_batch_size: int = 32,
        toxicity_batch_size: int = 32,
        include_flagged: bool = False
    ) -> Dict[str, any]:
        """
        创建全面的特征集（GPU优化）
        
        Args:
            df: 输入的DataFrame
            （已移除 patch_release_date 参数；仅支持基于 news/events 的时间对齐）
            include_bert: 是否包含BERT嵌入
            include_topics: 是否包含主题特征
            n_topics: LDA主题数量
            tfidf_features: TF-IDF特征数量
            max_bert_length: BERT最大文本长度
            bert_batch_size: BERT批处理大小
            toxicity_batch_size: 毒性检测批处理大小
            
        Returns:
            包含所有特征和矩阵的字典
        """
        logger.info("Starting comprehensive feature extraction (GPU optimized)...")
        # 如果用户选择不包含被标记的垃圾/机器人评论，则在一开始进行过滤
        df = df.copy()
        if not include_flagged and 'is_removed_cleaning' in df.columns:
            before = len(df)
            df = df[df['is_removed_cleaning'].fillna(0).astype(int) == 0]
            logger.info(f"Excluded {before - len(df)} flagged rows (is_removed_cleaning==1) before feature extraction")
        # patch_release_date removed: temporal features are event-centered
        # 1. 基本文本特征
        df_features = self.extract_text_features(df)
        # 2. 情感特征
        df_features = self.extract_sentiment_features(df_features)
        # 3. 游戏特定特征
        df_features = self.extract_game_specific_features(df_features)
        # 4. 毒性特征（GPU优化）
        df_features = self.extract_toxicity_features(df_features, toxicity_batch_size)
        # 5. 时间特征：如果指定 use_news_events 且提供 news_df，则使用事件中心（news/update）对齐与窗口聚合
        if use_news_events and news_df is not None:
            # 对评论对齐到其最近的已发生 news/update，避免未来泄露
            df_features = self.align_comments_to_latest_patch(df_features, news_df, appid_col='appid', comment_date_col='review_date')
            # 以 hours_since_latest_news 为时间列，聚合多个时间窗口
            df_features = self.aggregate_time_window_features(df_features, windows=event_windows, time_col='hours_since_latest_news', sentiment_col='vader_compound')
        else:
            # 强制只使用事件中心时间特征；在未提供 news_df 时抛出错误以提示用户
            raise ValueError("Temporal features are now event-centered. Provide 'news_df' and set use_news_events=True.")
        # 6. 评论互动特征（新增：votes_up, votes_funny, comment_count, weighted_vote_score）
        df_features = self.extract_review_engagement_features(df_features)
        # 6.1 作者特征（整合了评论互动指标）
        df_features = self.extract_author_features(df_features, verbose=False)
        # 6.5 sentiment group stats & interactions (per appid or genre)
        df_features = self.extract_sentiment_interaction_features(df_features)
        # 6.6 syntax features (POS proportions, negation, intensifiers)
        df_features = self.extract_syntax_features(df_features)
        # 6.7 Perspective API features (可选)
        if use_perspective:
            try:
                df_features = self.extract_perspective_features(df_features)
            except Exception as e:
                logger.warning(f"extract_perspective_features failed: {e}")
        # 7. TF-IDF特征
        df_features, tfidf_matrix = self.extract_tfidf_features(df_features, tfidf_features)
        # 8. BERT嵌入（GPU优化，可选）
        bert_embeddings = None
        if include_bert:
            df_features, bert_embeddings = self.extract_bert_features(
                df_features, max_bert_length, bert_batch_size
            )
        # 9. 主题特征（可选）
        if include_topics:
            # 支持为 LDA 选择向量化器（'tfidf' 或 'count'）
            df_features = self.extract_topic_features(df_features, n_topics, vectorizer=getattr(self, 'lda_vectorizer', 'count'))
        # 创建结果字典
        results = {
            'dataframe': df_features,
            'tfidf_matrix': tfidf_matrix,
            'tfidf_feature_names': self.tfidf_vectorizer.get_feature_names_out(),
            'bert_embeddings': bert_embeddings,
            'feature_summary': self._create_feature_summary(df_features)
        }
        # 清理GPU缓存
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        logger.info("Comprehensive feature extraction completed successfully")
        return results
    
    def _create_feature_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """创建特征摘要 - 使用精确的特征名匹配而非关键词搜索"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 定义每个类别的精确特征名（基础特征 + 模式匹配）
        # 文本特征 - 来自 extract_text_features
        text_feature_names = {
            'text_length', 'word_count', 'sentence_count', 'avg_word_length',
            'exclamation_count', 'question_count', 'uppercase_ratio', 'punctuation_ratio',
            'readability_score', 'flesch_kincaid_grade', 'has_question',
            'has_url', 'has_mention', 'has_external_reference'
        }
        
        # 情感特征 - 来自 extract_sentiment_features
        sentiment_feature_names = {
            'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
            'sentiment_category'
        }
        
        # 游戏特定特征 - 来自 extract_game_specific_features
        game_feature_names = {
            'contains_bug_report', 'contains_balance_complaint',
            'contains_monetization_complaint', 'mentions_performance'
        }
        
        # 毒性特征 - 来自 extract_toxicity_features
        toxicity_feature_names = {
            'toxicity_score', 'is_toxic', 'contains_toxic_keywords'
        }
        
        # 时间特征 - 来自 align_comments_to_latest_patch 和 aggregate_time_window_features
        temporal_feature_names = {
            'hours_since_latest_news', 'aligned_patch_idx'
        }
    # 动态时间窗口特征（24h, 48h, 72h 等）
        temporal_window_patterns = ['count_in_', 'avg_sentiment_in_', 'neg_ratio_in_', 'pos_ratio_in_']
        
        # 评论互动特征 - 来自 extract_review_engagement_features
        engagement_feature_names = {
            'is_recommended', 'votes_up_log', 'is_helpful_review', 'is_highly_helpful',
            'votes_funny_log', 'is_funny_review', 'funny_ratio', 'serious_ratio',
            'comment_count_log', 'is_controversial', 'is_highly_discussed',
            'weighted_score_normalized', 'is_high_quality',
            'engagement_score', 'engagement_score_normalized', 'engagement_level',
            'negative_but_helpful', 'positive_but_unhelpful', 'controversial_sentiment',
            'recommendation_sentiment_mismatch'
        }
        
        # 作者特征 - 来自 extract_author_features（已整合评论互动指标）
        author_feature_names = {
            'games_per_review_ratio', 'hours_per_game_ratio',
            'is_active_reviewer', 'is_hardcore_gamer', 'is_casual_gamer',
            'recent_activity_ratio', 'is_steam_purchaser', 'received_free',
            'author_reputation_score', 'is_quality_reviewer', 'avg_votes_per_review',
            'is_influential_reviewer'
        }

        # 语法特征 - 来自 extract_syntax_features
        syntax_feature_names = {
            'prop_noun', 'prop_verb', 'prop_adj', 'prop_adv',
            'negation_count', 'intensifier_count'
        }
        
        # TF-IDF特征 - 来自 extract_tfidf_features
        tfidf_feature_names = {
            'tfidf_max', 'tfidf_mean', 'tfidf_std', 'tfidf_nonzero_count'
        }
        
        # BERT特征 - 来自 extract_bert_features
        bert_feature_names = {
            'bert_embedding_norm', 'bert_embedding_mean', 'bert_embedding_std'
        }
        # include embedding cluster features if present
        bert_feature_names.update({'embed_cluster_id', 'embed_cluster_dist'})
        
        # 主题特征模式 - 来自 extract_topic_features（动态生成）
        # topic_0_prob, topic_1_prob, ..., dominant_topic, dominant_topic_prob, topic_entropy, topic_category
        
        # 分类所有列
        all_columns = set(df.columns)
        categorized = {
            'text_features': [],
            'sentiment_features': [],
            'game_features': [],
            'toxicity_features': [],
            'temporal_features': [],
            'engagement_features': [],  # 新增
            'author_features': [],
            'topic_features': [],
            'tfidf_features': [],
            'bert_features': [],
            'syntax_features': [],
            'other_features': []
        }
        
        classified = set()
        
        # 精确匹配基础特征
        for col in all_columns:
            if col in text_feature_names:
                categorized['text_features'].append(col)
                classified.add(col)
            elif col in sentiment_feature_names:
                categorized['sentiment_features'].append(col)
                classified.add(col)
            elif col in game_feature_names:
                categorized['game_features'].append(col)
                classified.add(col)
            elif col in toxicity_feature_names:
                categorized['toxicity_features'].append(col)
                classified.add(col)
            elif col in temporal_feature_names:
                categorized['temporal_features'].append(col)
                classified.add(col)
            elif col in engagement_feature_names:
                categorized['engagement_features'].append(col)
                classified.add(col)
            elif col in author_feature_names:
                categorized['author_features'].append(col)
                classified.add(col)
            elif col in syntax_feature_names:
                categorized['syntax_features'].append(col)
                classified.add(col)
            elif col in tfidf_feature_names:
                categorized['tfidf_features'].append(col)
                classified.add(col)
            elif col in bert_feature_names:
                categorized['bert_features'].append(col)
                classified.add(col)
            elif col in syntax_feature_names:
                categorized['text_features'].append(col)
                classified.add(col)
        
        # 模式匹配动态生成的特征
        for col in all_columns - classified:
            # 时间窗口特征（优先级最高，避免与情感特征混淆）
            if any(col.startswith(pattern) for pattern in temporal_window_patterns):
                categorized['temporal_features'].append(col)
                classified.add(col)
            # 主题特征
            elif col.startswith('topic_') or col in ['dominant_topic', 'dominant_topic_prob', 'topic_entropy', 'topic_category']:
                categorized['topic_features'].append(col)
                classified.add(col)
        
        # 剩余未分类的特征
        for col in all_columns - classified:
            categorized['other_features'].append(col)
        
        # 对每个类别排序
        for category in categorized:
            categorized[category].sort()
        
        summary = {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'numeric_features': len(numeric_columns),
            'device_used': str(self.device),
            'feature_categories': categorized
        }
        
        return summary

    def _generate_embedding_clusters(self, embeddings: np.ndarray, n_clusters: int = 16, sample_size: int = 20000) -> Dict[str, np.ndarray]:
        """
        使用 MiniBatchKMeans 生成 embedding 聚类（fit 或 reuse self.embedding_kmeans），返回 cluster ids 和到中心的距离。

        - 如果 self.embedding_kmeans 已存在则复用（predict）
        - 否则在 sample embeddings 上 fit 并持久化
        """
        if embeddings is None or len(embeddings) == 0:
            return {'cluster_id': np.array([]), 'cluster_dist': np.array([])}

        # 如果已有模型则直接 predict
        if self.embedding_kmeans is not None:
            labels = self.embedding_kmeans.predict(embeddings)
            centers = self.embedding_kmeans.cluster_centers_
        else:
            # 采样用于 fit
            n_samples = min(sample_size, embeddings.shape[0])
            if n_samples < embeddings.shape[0]:
                rng = np.random.default_rng(42)
                idx = rng.choice(embeddings.shape[0], size=n_samples, replace=False)
                sample = embeddings[idx]
            else:
                sample = embeddings

            mbk = MiniBatchKMeans(n_clusters=min(n_clusters, max(2, embeddings.shape[0] // 10)), batch_size=1024, random_state=42)
            mbk.fit(sample)
            self.embedding_kmeans = mbk
            labels = mbk.predict(embeddings)
            centers = mbk.cluster_centers_

        # 计算到中心的距离
        dists = np.linalg.norm(embeddings - centers[labels], axis=1)

        return {'cluster_id': labels, 'cluster_dist': dists}

    def save_topic_pca(self, path: str):
        """Serialize the fitted topic_pca to disk"""
        if self.topic_pca is None:
            logger.warning("No topic_pca to save")
            return
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.topic_pca, f)
            logger.info(f"Saved topic_pca to {path}")
        except Exception as e:
            logger.warning(f"Failed to save topic_pca: {e}")

    def load_topic_pca(self, path: str):
        """Load a serialized topic_pca from disk into self.topic_pca"""
        try:
            with open(path, 'rb') as f:
                self.topic_pca = pickle.load(f)
            logger.info(f"Loaded topic_pca from {path}")
        except Exception as e:
            logger.warning(f"Failed to load topic_pca: {e}")
    
    def save_features(self, results: Dict[str, any], output_dir: str, file_prefix: str = "features", include_flagged: Optional[bool] = None):
        """
        保存特征到文件
        
        Args:
            results: create_comprehensive_features的返回结果
            output_dir: 输出目录
            file_prefix: 文件前缀
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果提供 include_flagged，则在文件名前缀加入标志，便于区分
        if include_flagged is True:
            flag_suffix = '_inclflagged'
        elif include_flagged is False:
            flag_suffix = '_exclflagged'
        else:
            flag_suffix = ''

        # 保存主要DataFrame 为 Parquet（不再输出 Excel/CSV）
        df_path = os.path.join(output_dir, f"{file_prefix}{flag_suffix}_dataframe.parquet")
        try:
            results['dataframe'].to_parquet(df_path, index=False)
            logger.info(f"Features DataFrame saved to: {df_path}")
        except Exception as e:
            msg = (
                f"Failed to write features DataFrame to parquet at {df_path}: {e}.\n"
                "Ensure a parquet engine is installed (pyarrow or fastparquet).\n"
                "Install with: pip install pyarrow"
            )
            logger.error(msg)
            raise RuntimeError(msg)
        
        # 保存TF-IDF矩阵
        if results['tfidf_matrix'] is not None:
            tfidf_path = os.path.join(output_dir, f"{file_prefix}{flag_suffix}_tfidf_matrix.npz")
            from scipy.sparse import save_npz
            save_npz(tfidf_path, results['tfidf_matrix'])
            
            # 保存特征名称
            feature_names_path = os.path.join(output_dir, f"{file_prefix}{flag_suffix}_tfidf_features.txt")
            with open(feature_names_path, 'w', encoding='utf-8') as f:
                for feature in results['tfidf_feature_names']:
                    f.write(f"{feature}\n")
            logger.info(f"TF-IDF matrix saved to: {tfidf_path}")
        
        # 保存BERT嵌入
        if results['bert_embeddings'] is not None:
            bert_path = os.path.join(output_dir, f"{file_prefix}{flag_suffix}_bert_embeddings.npy")
            np.save(bert_path, results['bert_embeddings'])
            logger.info(f"BERT embeddings saved to: {bert_path}")
        
        # 保存特征摘要
        summary_path = os.path.join(output_dir, f"{file_prefix}{flag_suffix}_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            summary = results['feature_summary']
            f.write("=== GPU-Optimized Feature Engineering Summary ===\n\n")
            f.write(f"Device Used: {summary['device_used']}\n")
            f.write(f"Total Samples: {summary['total_samples']}\n")
            f.write(f"Total Features: {summary['total_features']}\n")
            if include_flagged is not None:
                f.write(f"Included flagged comments: {include_flagged}\n")
            f.write(f"Numeric Features: {summary['numeric_features']}\n\n")
            
            for category, features in summary['feature_categories'].items():
                f.write(f"{category.upper()} ({len(features)} features):\n")
                for feature in features:
                    f.write(f"  - {feature}\n")
                f.write("\n")
        
        logger.info(f"Feature summary saved to: {summary_path}")

        # 保存 topic_pca 和 embedding_kmeans 模型（如果存在）
        try:
            if self.topic_pca is not None:
                pca_path = os.path.join(output_dir, f"{file_prefix}{flag_suffix}_topic_pca.pkl")
                with open(pca_path, 'wb') as f:
                    pickle.dump(self.topic_pca, f)
                logger.info(f"Saved topic PCA to: {pca_path}")
        except Exception as e:
            logger.warning(f"Failed to save topic_pca: {e}")

        try:
            if self.embedding_kmeans is not None:
                kmeans_path = os.path.join(output_dir, f"{file_prefix}{flag_suffix}_embedding_kmeans.pkl")
                with open(kmeans_path, 'wb') as f:
                    pickle.dump(self.embedding_kmeans, f)
                logger.info(f"Saved embedding kmeans to: {kmeans_path}")
        except Exception as e:
            logger.warning(f"Failed to save embedding_kmeans: {e}")


def main():
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        force_cpu = False
    else:
        logger.info("No GPU detected, using CPU")
        force_cpu = True

    feature_engineer = GPUOptimizedFeatureEngineer()

    FLAG = True
    reviews_dir = r"D:\BAP\data_label\combined"
    news_dir = r"D:\BAP\data_nolabel\cleaned\news"
    output_base = r"D:\BAP\features"

    df_news = feature_engineer.load_news_from_dir(news_dir)

    for fname in os.listdir(reviews_dir):
        path = os.path.join(reviews_dir, fname)
        if not os.path.isfile(path):
            continue
        lower = fname.lower()
        try:
            if lower.endswith('.csv'):
                df_file = pd.read_csv(path, encoding='utf-8')
            elif lower.endswith(('.xls', '.xlsx')):
                df_file = pd.read_excel(path, engine='openpyxl')
            else:
                logger.debug(f"Skipping unsupported file type: {path}")
                continue
        except Exception as e:
            logger.warning(f"Failed to read reviews file {path}: {e}")
            continue

        m = re.search(r'combined[_\-]?(.+?)[_\-]?reviews', fname, flags=re.IGNORECASE)
        if m:
            genre = m.group(1).lower()
        else:
            genre = os.path.splitext(fname)[0]

        if df_file.empty:
            logger.info(f"File {fname} produced empty dataframe; skipping")
            continue

        sample_size = min(2000, len(df_file)) if force_cpu else min(1000000, len(df_file))
        df_sample = df_file.sample(n=sample_size, random_state=42) if len(df_file) > sample_size else df_file.copy()

        logger.info(f"Processing file {fname} (genre={genre}) with {len(df_sample)} samples")

        try:
            results = feature_engineer.create_comprehensive_features(
                df_sample,
                news_df=df_news,
                use_news_events=True,
                use_perspective=False,
                event_windows=[24, 48, 72],
                include_bert=True,
                include_topics=True,
                n_topics=8,
                tfidf_features=4200,
                max_bert_length=256,
                bert_batch_size=64 if not force_cpu else 16,
                toxicity_batch_size=64 if not force_cpu else 16,
                include_flagged=FLAG
            )
        except Exception as e:
            logger.exception(f"Feature extraction failed for {fname}: {e}")
            continue

        out_dir = os.path.join(output_base, genre)
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"gpu_optimized_features_{genre}"
        feature_engineer.save_features(results, out_dir, prefix, include_flagged=FLAG)

    logger.info("Completed processing all files in combined directory")


if __name__ == "__main__":
    main()