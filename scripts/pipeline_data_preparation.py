"""
完整数据准备管道
End-to-End Data Preparation Pipeline

集成所有数据处理步骤:
1. 数据清洗 (DataCleaner)
2. 基础特征工程 (GPUOptimizedFeatureEngineer)  
3. 趋势与异常特征 (TrendFeatureExtractor, AnomalyDetector)
4. 弱标签生成 (generate_weak_labels)

使用方法:
    python pipeline_data_preparation.py --input reviews.csv --genre fps
    
或者在代码中调用:
    from scripts.pipeline_data_preparation import DataPreparationPipeline
    
    pipeline = DataPreparationPipeline(genre='fps')
    results = pipeline.process_file('path/to/reviews.csv')
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入项目模块
try:
    from scripts.cleaning.review_data_cleaning_ import DataCleaner
    CLEANER_AVAILABLE = True
except ImportError:
    CLEANER_AVAILABLE = False
    print("[WARN] DataCleaner not available, will skip advanced cleaning")

try:
    from scripts.feature.feature_engineering import GPUOptimizedFeatureEngineer
    FEATURE_ENG_AVAILABLE = True
except ImportError:
    FEATURE_ENG_AVAILABLE = False
    print("[WARN] GPUOptimizedFeatureEngineer not available, will skip feature engineering")

try:
    from scripts.feature.trend_anomaly_analysis import (
        TrendFeatureExtractor, 
        AnomalyDetector,
        ComprehensiveAnalysisSystem
    )
    TREND_AVAILABLE = True
except ImportError:
    TREND_AVAILABLE = False
    print("[WARN] Trend analysis module not available, will skip trend features")

try:
    from scripts.train.weaklabeling import generate_weak_labels
    WEAKLABEL_AVAILABLE = True
except ImportError:
    WEAKLABEL_AVAILABLE = False
    print("[WARN] Weak labeling module not available, will skip label generation")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataPreparationPipeline:
    """
    完整数据准备管道
    
    处理流程:
    1. 加载原始数据
    2. 数据清洗 (去重、垃圾过滤、文本清理)
    3. 基础特征提取 (文本统计、情感、主题、毒性等)
    4. 趋势特征提取 (时间窗口、滚动统计、变化率)
    5. 异常检测 (统计、ML、协同)
    6. 弱标签生成 (情感、主题、风险)
    7. 保存结果
    """
    
    def __init__(
        self, 
        genre: str = 'fps',
        output_dir: Optional[str] = None,
        include_flagged: bool = False,
        use_gpu: bool = True,
        anomaly_method: str = 'isolation_forest',
        windows: List[int] = [24, 48, 72]
    ):
        """
        Args:
            genre: 游戏类型 (fps, strategy, leisure等)
            output_dir: 输出目录，默认为 features/{genre}/
            include_flagged: 是否在特征中包含被标记为垃圾的评论
            use_gpu: 是否使用GPU加速
            anomaly_method: 异常检测方法 ('statistical', 'isolation_forest', 'elliptic_envelope')
            windows: 时间窗口列表（小时）
        """
        self.genre = genre
        self.include_flagged = include_flagged
        self.windows = windows
        
        # 设置输出目录
        if output_dir is None:
            base_dir = Path(__file__).parent.parent
            self.output_dir = base_dir / 'features' / genre
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化各个模块
        self._init_modules(use_gpu, anomaly_method)
        
        logger.info(f"Pipeline initialized for genre={genre}, output_dir={self.output_dir}")
    
    def _init_modules(self, use_gpu: bool, anomaly_method: str):
        """初始化所有处理模块"""
        logger.info("Initializing pipeline modules...")
        
        # 1. 数据清洗器
        if CLEANER_AVAILABLE:
            self.cleaner = DataCleaner()
            logger.info("[OK] DataCleaner initialized")
        else:
            self.cleaner = None
            logger.warning("[WARN] DataCleaner not available")
        
        # 2. 特征工程器
        if FEATURE_ENG_AVAILABLE:
            self.feature_engineer = GPUOptimizedFeatureEngineer(
                force_cpu=not use_gpu,
                load_heavy_models=True,
                lda_vectorizer='tfidf'
            )
            logger.info("[OK] FeatureEngineer initialized")
        else:
            self.feature_engineer = None
            logger.warning("[WARN] FeatureEngineer not available")
        
        # 3. 趋势特征提取器
        if TREND_AVAILABLE:
            self.trend_extractor = TrendFeatureExtractor(windows=self.windows)
            self.anomaly_detector = AnomalyDetector(
                method=anomaly_method,
                sensitivity='medium'
            )
            logger.info("[OK] TrendAnalysis modules initialized")
        else:
            self.trend_extractor = None
            self.anomaly_detector = None
            logger.warning("[WARN] TrendAnalysis not available")
    
    def load_data(self, input_path: str) -> pd.DataFrame:
        """
        加载原始数据
        
        支持格式: CSV, Excel, Parquet
        """
        logger.info(f"Loading data from: {input_path}")
        
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # 根据文件扩展名选择读取方法
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_path)
        elif input_path.suffix.lower() == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        logger.info(f"[OK] Loaded {len(df):,} rows, {len(df.columns)} columns")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗步骤
        
        - 去重
        - 垃圾过滤
        - 文本规范化
        - 时间解析
        """
        logger.info("Step 1: Data Cleaning")
        
        if self.cleaner is None:
            logger.warning("Cleaner not available, performing minimal cleaning")
            # 最小清洗
            df = df.copy()
            if 'review_content' in df.columns:
                df['review_content_clean'] = df['review_content'].astype(str).str.strip()
            if 'time' in df.columns or 'review_date' in df.columns:
                time_col = 'time' if 'time' in df.columns else 'review_date'
                df['review_date'] = pd.to_datetime(df[time_col], errors='coerce')
            # 添加缺失的数值列
            for col in ['played_hours', 'num_reviews', 'num_games_owned']:
                if col not in df.columns:
                    df[col] = 0
            return df
        
        # 使用DataCleaner的完整清洗流程
        original_count = len(df)
        
        # 标准化列名
        df = self._standardize_columns(df)
        
        # 去重
        df = df.drop_duplicates(subset=['review_content'], keep='first')
        logger.info(f"   Deduplication: {original_count} -> {len(df)} rows")
        
        # 标记垃圾和机器人评论
        df['is_spam'] = df['review_content_clean'].apply(
            lambda x: self.cleaner.is_spam_content(str(x))
        )
        df['is_bot'] = df.apply(self.cleaner.is_bot_review, axis=1)
        df['is_removed_cleaning'] = (df['is_spam'] | df['is_bot']).astype(int)
        
        spam_count = df['is_removed_cleaning'].sum()
        logger.info(f"   Flagged {spam_count} spam/bot reviews")
        
        # 根据include_flagged决定是否过滤
        if not self.include_flagged:
            df = df[df['is_removed_cleaning'] == 0].copy()
            logger.info(f"   Filtered to {len(df)} clean reviews")
        
        # 文本处理
        if 'review_content_clean' in df.columns:
            df['review_content_processed'] = df['review_content_clean'].fillna('')
        
        # 时间解析
        if 'review_date' in df.columns:
            df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        
        logger.info(f"[OK] Cleaning complete: {len(df)} rows retained")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> Dict:
        """
        基础特征提取
        
        包括:
        - 文本统计特征
        - 情感特征 (VADER)
        - 游戏相关特征 (bug, balance等)
        - 毒性特征
        - TF-IDF特征
        - BERT嵌入 (可选)
        - LDA主题
        """
        logger.info("Step 2: Feature Engineering")
        
        if self.feature_engineer is None:
            logger.warning("FeatureEngineer not available, skipping")
            return {'dataframe': df}
        
        try:
            # 使用综合特征提取
            results = self.feature_engineer.create_comprehensive_features(
                df,
                include_bert=True,
                include_topics=True,
                n_topics=6,
                tfidf_features=500,
                include_flagged=self.include_flagged
            )
            
            df_features = results['dataframe']
            logger.info(f"[OK] Features extracted: {len(df_features.columns)} total columns")
            
            return results
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}", exc_info=True)
            return {'dataframe': df}
    
    def extract_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        趋势与时序特征提取
        
        包括:
        - 滚动窗口统计 (24h, 48h, 72h)
        - 情感趋势
        - 评论量变化
        - 波动性指标
        """
        logger.info("Step 3: Trend Feature Extraction")
        
        if self.trend_extractor is None:
            logger.warning("TrendExtractor not available, skipping")
            return df
        
        try:
            # 确保有必需的列
            if 'review_date' not in df.columns:
                logger.warning("Missing 'review_date' column, skipping trend extraction")
                return df
            
            if 'vader_compound' not in df.columns:
                logger.warning("Missing 'vader_compound' column, using default sentiment")
                df['vader_compound'] = 0.0
            
            # 按appid分组提取趋势（如果有多个游戏）
            if 'appid' in df.columns and df['appid'].nunique() > 1:
                trend_frames = []
                for appid, group in df.groupby('appid'):
                    group_trend = self.trend_extractor.extract(
                        group.sort_values('review_date'),
                        appid=appid
                    )
                    trend_frames.append(group_trend)
                df_trend = pd.concat(trend_frames, ignore_index=True)
            else:
                df_trend = self.trend_extractor.extract(
                    df.sort_values('review_date')
                )
            
            logger.info(f"[OK] Trend features extracted: {len(df_trend.columns)} columns")
            return df_trend
            
        except Exception as e:
            logger.error(f"Trend extraction failed: {e}", exc_info=True)
            return df
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        异常检测
        
        包括:
        - 统计异常
        - ML异常 (Isolation Forest)
        - 组合异常分数
        - 协同刷评检测
        """
        logger.info("Step 4: Anomaly Detection")
        
        if self.anomaly_detector is None:
            logger.warning("AnomalyDetector not available, skipping")
            return df
        
        try:
            # 组合检测（统计 + ML）
            df_anomaly = self.anomaly_detector.detect_combined(df)
            
            anomaly_count = df_anomaly['is_ensemble_anomaly'].sum()
            logger.info(f"   Detected {anomaly_count} anomalies (ensemble)")
            
            # 协同检测（可选，比较耗时）
            if len(df) < 10000:  # 只在数据量不太大时运行
                try:
                    df_coord, coord_groups = self.anomaly_detector.detect_coordinated(df_anomaly)
                    df_anomaly['is_coordinated'] = df_coord.get('is_coordinated', 0)
                    logger.info(f"   Detected {len(coord_groups)} coordinated groups")
                except Exception as e:
                    logger.warning(f"Coordinated detection failed: {e}")
                    df_anomaly['is_coordinated'] = 0
            else:
                df_anomaly['is_coordinated'] = 0
            
            logger.info(f"[OK] Anomaly detection complete")
            return df_anomaly
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}", exc_info=True)
            return df
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        生成弱标签
        
        包括:
        - 情感标签 (基于VADER)
        - 主题标签 (基于dominant_topic)
        - 风险标签 (基于多维度评分)
        - 趋势告警标签
        """
        logger.info("Step 5: Weak Label Generation")
        
        if not WEAKLABEL_AVAILABLE:
            logger.warning("WeakLabeling not available, skipping")
            return df
        
        try:
            df_labeled = generate_weak_labels(df)
            
            # 统计标签分布
            if 'risk_label_weak' in df_labeled.columns:
                risk_dist = df_labeled['risk_label_weak'].value_counts()
                logger.info(f"   Risk labels: {risk_dist.to_dict()}")
            
            if 'risk_level' in df_labeled.columns:
                level_dist = df_labeled['risk_level'].value_counts()
                logger.info(f"   Risk levels: {level_dist.to_dict()}")
            
            logger.info(f"[OK] Labels generated")
            return df_labeled
            
        except Exception as e:
            logger.error(f"Label generation failed: {e}", exc_info=True)
            return df
    
    def save_results(
        self, 
        df: pd.DataFrame, 
        results: Dict,
        prefix: str = None
    ) -> Dict[str, str]:
        """
        保存处理结果
        
        保存内容:
        - DataFrame (Parquet + CSV)
        - TF-IDF矩阵 (NPZ)
        - BERT嵌入 (NPY)
        - 特征摘要 (JSON)
        - 处理报告 (JSON)
        """
        logger.info("Step 6: Saving Results")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if prefix is None:
            flagged_suffix = 'inclflagged' if self.include_flagged else 'exclflagged'
            prefix = f"{self.genre}_{flagged_suffix}_{timestamp}"
        
        output_paths = {}
        
        # 1. 主数据框（Parquet优先）
        parquet_path = self.output_dir / f"{prefix}_dataframe.parquet"
        try:
            df.to_parquet(parquet_path, index=False)
            output_paths['dataframe_parquet'] = str(parquet_path)
            logger.info(f"   Saved Parquet: {parquet_path.name}")
        except Exception as e:
            logger.warning(f"Parquet save failed: {e}, trying CSV")
            csv_path = self.output_dir / f"{prefix}_dataframe.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            output_paths['dataframe_csv'] = str(csv_path)
            logger.info(f"   Saved CSV: {csv_path.name}")
        
        # 2. TF-IDF矩阵（如果有）
        if 'tfidf_matrix' in results and results['tfidf_matrix'] is not None:
            try:
                import scipy.sparse
                tfidf_path = self.output_dir / f"{prefix}_tfidf_matrix.npz"
                scipy.sparse.save_npz(tfidf_path, results['tfidf_matrix'])
                output_paths['tfidf_matrix'] = str(tfidf_path)
                logger.info(f"   Saved TF-IDF: {tfidf_path.name}")
            except Exception as e:
                logger.warning(f"TF-IDF save failed: {e}")
        
        # 3. BERT嵌入（如果有）
        if 'bert_embeddings' in results and results['bert_embeddings'] is not None:
            try:
                bert_path = self.output_dir / f"{prefix}_bert_embeddings.npy"
                np.save(bert_path, results['bert_embeddings'])
                output_paths['bert_embeddings'] = str(bert_path)
                logger.info(f"   Saved BERT: {bert_path.name}")
            except Exception as e:
                logger.warning(f"BERT save failed: {e}")
        
        # 4. 特征摘要
        if 'feature_summary' in results:
            try:
                import json
                summary_path = self.output_dir / f"{prefix}_feature_summary.json"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(results['feature_summary'], f, indent=2, ensure_ascii=False)
                output_paths['feature_summary'] = str(summary_path)
                logger.info(f"   Saved summary: {summary_path.name}")
            except Exception as e:
                logger.warning(f"Summary save failed: {e}")
        
        # 5. 处理报告
        report = self._generate_report(df, results)
        report_path = self.output_dir / f"{prefix}_processing_report.json"
        try:
            import json
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            output_paths['report'] = str(report_path)
            logger.info(f"   Saved report: {report_path.name}")
        except Exception as e:
            logger.warning(f"Report save failed: {e}")
        
        logger.info(f"[OK] All results saved to: {self.output_dir}")
        return output_paths
    
    def _generate_report(self, df: pd.DataFrame, results: Dict) -> Dict:
        """生成处理报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'genre': self.genre,
            'include_flagged': self.include_flagged,
            'windows': self.windows,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
        }
        
        # 添加各类统计
        if 'is_removed_cleaning' in df.columns:
            report['flagged_reviews'] = int(df['is_removed_cleaning'].sum())
        
        if 'risk_level' in df.columns:
            report['risk_distribution'] = df['risk_level'].value_counts().to_dict()
        
        if 'is_ensemble_anomaly' in df.columns:
            report['anomaly_count'] = int(df['is_ensemble_anomaly'].sum())
        
        if 'anomaly_severity' in df.columns:
            report['severity_distribution'] = df['anomaly_severity'].value_counts().to_dict()
        
        if 'dominant_topic' in df.columns:
            report['topic_distribution'] = df['dominant_topic'].value_counts().head(10).to_dict()
        
        return report
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        column_mapping = {
            'review_content': 'review_content_clean',
            'review_text': 'review_content_clean',
            'content': 'review_content_clean',
            'time': 'review_date',
            'date': 'review_date',
            'created_at': 'review_date',
            'timestamp': 'review_date',
        }
        
        df = df.rename(columns=column_mapping)
        
        # 确保关键列存在
        if 'review_content_clean' not in df.columns and 'review_content' in df.columns:
            df['review_content_clean'] = df['review_content']
        
        return df
    
    def process_file(self, input_path: str, save: bool = True) -> Dict:
        """
        处理单个文件的完整流程
        
        Args:
            input_path: 输入文件路径
            save: 是否保存结果
            
        Returns:
            包含处理结果的字典
        """
        logger.info("="*80)
        logger.info(f"Starting Data Preparation Pipeline")
        logger.info(f"Input: {input_path}")
        logger.info(f"Genre: {self.genre}")
        logger.info(f"Include Flagged: {self.include_flagged}")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        try:
            # 1. 加载数据
            df = self.load_data(input_path)
            
            # 2. 清洗
            df = self.clean_data(df)
            
            # 3. 基础特征
            results = self.extract_features(df)
            df = results['dataframe']
            
            # 4. 趋势特征
            df = self.extract_trend_features(df)
            
            # 5. 异常检测
            df = self.detect_anomalies(df)
            
            # 6. 弱标签
            df = self.generate_labels(df)
            
            # 更新结果
            results['dataframe'] = df
            
            # 7. 保存
            if save:
                output_paths = self.save_results(df, results)
                results['output_paths'] = output_paths
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            logger.info("="*80)
            logger.info(f"Pipeline completed successfully in {elapsed:.1f}s")
            logger.info(f"Final shape: {df.shape}")
            logger.info("="*80)
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='Data Preparation Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--genre', type=str, default='fps', help='Game genre')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--include_flagged', action='store_true', help='Include flagged reviews')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--anomaly_method', type=str, default='isolation_forest',
                       choices=['statistical', 'isolation_forest', 'elliptic_envelope'])
    parser.add_argument('--windows', type=int, nargs='+', default=[24, 48, 72],
                       help='Time windows in hours')
    
    args = parser.parse_args()
    
    # 创建管道
    pipeline = DataPreparationPipeline(
        genre=args.genre,
        output_dir=args.output_dir,
        include_flagged=args.include_flagged,
        use_gpu=not args.no_gpu,
        anomaly_method=args.anomaly_method,
        windows=args.windows
    )
    
    # 处理文件
    results = pipeline.process_file(args.input)
    
    print("\n" + "="*80)
    print("Processing Complete!")
    print("="*80)
    if 'output_paths' in results:
        print("\nOutput files:")
        for key, path in results['output_paths'].items():
            print(f"  {key}: {path}")
    print("="*80)


if __name__ == '__main__':
    main()
