"""
趋势与异常分析模块
用于游戏评论的时序趋势预测和异常检测

功能：
1. 时序趋势特征提取
2. 趋势预测模型（情感走势、评论量预测）
3. 多种异常检测方法（统计、ML、协同）
4. 综合分析系统集成
5. 可视化与监控
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
import os
import glob

# ML 库
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# 图分析（协同检测）
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available, coordinated behavior detection will be limited")

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrendFeatureExtractor:
    """时序趋势特征提取器"""
    
    def __init__(self, windows: List[int] = [24, 48, 72]):
        """
        Args:
            windows: 时间窗口列表（小时）
        """
        self.windows = windows
        
    def extract(self, df: pd.DataFrame, appid: Optional[str] = None, log_level: str = 'info') -> pd.DataFrame:
        """
        提取时序趋势特征
        
        Args:
            df: 包含时间戳的评论数据（需按 review_date 排序）
            
        Returns:
            添加了趋势特征的 DataFrame
        """
        logger.info("提取时序趋势特征...")
        df = df.sort_values('review_date').copy()
        
        # 时间特征
        df['review_datetime'] = pd.to_datetime(df['review_date'])
        df = df.set_index('review_datetime')
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # 滚动窗口特征（基于时间窗口，而非行数）
        for w in self.windows:
            window_str = f"{w}H"

            # 情感统计（time-based rolling）
            df[f'sentiment_rolling_mean_{w}h'] = (
                df['vader_compound'].rolling(window=window_str, min_periods=1)
                .mean()
            )

            df[f'sentiment_rolling_std_{w}h'] = (
                df['vader_compound'].rolling(window=window_str, min_periods=1)
                .std().fillna(0)
            )

            df[f'sentiment_rolling_min_{w}h'] = (
                df['vader_compound'].rolling(window=window_str, min_periods=1)
                .min()
            )

            df[f'sentiment_rolling_max_{w}h'] = (
                df['vader_compound'].rolling(window=window_str, min_periods=1)
                .max()
            )

            # 负面比率趋势
            df[f'neg_ratio_rolling_{w}h'] = (
                (df['vader_compound'] < -0.05).astype(int).rolling(window=window_str, min_periods=1).mean()
            )

            # 正面比率趋势
            df[f'pos_ratio_rolling_{w}h'] = (
                (df['vader_compound'] > 0.05).astype(int).rolling(window=window_str, min_periods=1).mean()
            )

            # 评论量趋势（基于时间窗口计数）
            # We compute counts per time-window by using rolling on a ones-series
            ones = pd.Series(1, index=df.index)
            df[f'comment_rate_{w}h'] = ones.rolling(window=window_str, min_periods=1).sum()

            # 互动指标趋势
            if 'engagement_score' in df.columns:
                df[f'engagement_rolling_{w}h'] = (
                    df['engagement_score'].rolling(window=window_str, min_periods=1).mean()
                )

            # 毒性趋势
            if 'toxicity_score' in df.columns:
                df[f'toxicity_rolling_{w}h'] = (
                    df['toxicity_score'].rolling(window=window_str, min_periods=1).mean()
                )
        
        # 差分特征（变化率）
        # Since we have a datetime index, diff() will compute differences in the sequence order
        # which aligns with time-based rolling results.
        df['sentiment_diff_24h'] = df['sentiment_rolling_mean_24h'].diff()
        df['sentiment_diff_48h'] = df['sentiment_rolling_mean_48h'].diff()
        df['sentiment_diff_72h'] = df['sentiment_rolling_mean_72h'].diff()

        # 加速度（二阶差分）
        df['sentiment_acceleration_24h'] = df['sentiment_diff_24h'].diff()
        df['sentiment_acceleration_48h'] = df['sentiment_diff_48h'].diff()
        # 添加 72h 的二阶差分以保持尺度对称性
        df['sentiment_acceleration_72h'] = df['sentiment_diff_72h'].diff()

        # 评论量变化率（pct change）
        df['comment_rate_change_24h'] = df['comment_rate_24h'].pct_change()
        # 补上 48h 的变化率以对称化尺度
        df['comment_rate_change_48h'] = df['comment_rate_48h'].pct_change()
        df['comment_rate_change_72h'] = df['comment_rate_72h'].pct_change()

        # 峰值与谷值检测
        df['is_sentiment_local_peak'] = (
            (df['sentiment_rolling_mean_24h'] > df['sentiment_rolling_mean_24h'].shift(1)) &
            (df['sentiment_rolling_mean_24h'] > df['sentiment_rolling_mean_24h'].shift(-1))
        ).astype(int)

        df['is_sentiment_local_trough'] = (
            (df['sentiment_rolling_mean_24h'] < df['sentiment_rolling_mean_24h'].shift(1)) &
            (df['sentiment_rolling_mean_24h'] < df['sentiment_rolling_mean_24h'].shift(-1))
        ).astype(int)

        # 趋势方向
        df['sentiment_trend_direction'] = np.sign(df['sentiment_diff_72h'])

        # 波动性
        df['sentiment_volatility'] = df['sentiment_rolling_std_72h'] / (
            np.abs(df['sentiment_rolling_mean_72h']) + 0.01
        )

        # Reset index so returned DF uses a default integer index and keeps review_datetime as a column
        try:
            df = df.reset_index()
        except Exception:
            # fallback: if index reset fails, leave as-is
            pass

        n_new = len([c for c in df.columns if 'rolling' in c or 'diff' in c])
        ctx = f" (appid={appid})" if appid is not None else ""
        msg = f"趋势特征提取完成{ctx}，新增 {n_new} 个特征，rows={len(df)}"
        if str(log_level).lower() == 'debug':
            logger.debug(msg)
        else:
            logger.info(msg)

        return df


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, method: str = 'isolation_forest', sensitivity: str = 'medium'):
        """
        Args:
            method: 'statistical', 'isolation_forest', 'elliptic_envelope'
            sensitivity: 'low' (3σ), 'medium' (2.5σ), 'high' (2σ)
        """
        self.method = method
        self.sensitivity = sensitivity
        self.sigma_thresholds = {'low': 3.0, 'medium': 2.5, 'high': 2.0}
        self.model = None
        self.feature_cols = None
        
    def detect_statistical(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于统计的异常检测"""
        logger.info("执行统计异常检测...")
        sigma = self.sigma_thresholds[self.sensitivity]
        
        df = df.copy()
        
        # 1. 情感异常
        sentiment_mean = df['vader_compound'].mean()
        sentiment_std = df['vader_compound'].std()
        df['is_sentiment_anomaly'] = (
            np.abs(df['vader_compound'] - sentiment_mean) > sigma * sentiment_std
        ).astype(int)
        
        # 2. 评论量异常
        if 'hourly_count' not in df.columns:
            df['hourly_count'] = df.groupby(
                pd.to_datetime(df['review_date']).dt.floor('H')
            )['review_date'].transform('count')
        
        count_mean = df['hourly_count'].mean()
        count_std = df['hourly_count'].std()
        df['is_volume_anomaly'] = (
            df['hourly_count'] > count_mean + sigma * count_std
        ).astype(int)
        
        # 3. 互动异常
        if 'votes_up' in df.columns and 'text_length' in df.columns:
            df['votes_per_char'] = df['votes_up'] / (df['text_length'] + 1)
            votes_ratio_mean = df['votes_per_char'].mean()
            votes_ratio_std = df['votes_per_char'].std()
            df['is_vote_anomaly'] = (
                df['votes_per_char'] > votes_ratio_mean + sigma * votes_ratio_std
            ).astype(int)
        
        # 4. 主题异常
        if 'dominant_topic' in df.columns:
            topic_freq = df['dominant_topic'].value_counts(normalize=True)
            df['topic_rarity'] = df['dominant_topic'].map(topic_freq)
            df['is_rare_topic'] = (df['topic_rarity'] < 0.05).astype(int)
        
        # 5. 趋势异常（突然的变化）
        if 'sentiment_diff_24h' in df.columns:
            diff_mean = df['sentiment_diff_24h'].mean()
            diff_std = df['sentiment_diff_24h'].std()
            df['is_trend_anomaly'] = (
                np.abs(df['sentiment_diff_24h'] - diff_mean) > sigma * diff_std
            ).astype(int)
        
        # 综合异常分数
        anomaly_cols = [c for c in df.columns if c.startswith('is_') and 'anomaly' in c]
        df['statistical_anomaly_score'] = df[anomaly_cols].sum(axis=1)
        df['is_statistical_anomaly'] = (df['statistical_anomaly_score'] >= 2).astype(int)
        
        logger.info(f"检测到 {df['is_statistical_anomaly'].sum()} 条统计异常")
        
        return df
    
    def detect_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于机器学习的异常检测"""
        logger.info(f"执行 {self.method} 异常检测...")
        
        # 特征选择
        self.feature_cols = [
            'vader_compound', 'sentiment_rolling_std_24h', 'sentiment_diff_24h',
            'toxicity_score', 'text_length', 'comment_rate_24h',
            'neg_ratio_rolling_24h', 'sentiment_volatility'
        ]
        
        if 'engagement_score' in df.columns:
            self.feature_cols.append('engagement_score')
        if 'votes_up_log' in df.columns:
            self.feature_cols.extend(['votes_up_log', 'comment_count_log'])
        
        # 过滤存在的特征
        self.feature_cols = [f for f in self.feature_cols if f in df.columns]
        
        # 更稳健的预处理：中位数填充 + 标准化
        X_raw = df[self.feature_cols].values
        try:
            # 创建/重用 SimpleImputer 和 StandardScaler
            if not hasattr(self, 'imputer') or self.imputer is None:
                self.imputer = SimpleImputer(strategy='median')
            if not hasattr(self, 'scaler') or self.scaler is None:
                self.scaler = StandardScaler()

            X_imputed = self.imputer.fit_transform(X_raw)
            # 如果所有列都是常数，StandardScaler 会产生 0 variance；guard it
            try:
                X = self.scaler.fit_transform(X_imputed)
            except Exception:
                # fallback: if scaling fails, use imputed values directly
                X = X_imputed
        except Exception:
            # 最后回退到旧行为（用 0 填充）以保证稳定运行
            logger.warning("Imputer/Scaler failed, falling back to fillna(0)")
            X = df[self.feature_cols].fillna(0).values
        
        # 选择模型
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=0.03,
                random_state=42,
                n_estimators=100,
                n_jobs=-1
            )
        elif self.method == 'elliptic_envelope':
            self.model = EllipticEnvelope(
                contamination=0.03,
                random_state=42
            )
        
        # 训练与预测
        predictions = self.model.fit_predict(X)
        df['ml_anomaly'] = (predictions == -1).astype(int)
        df['ml_anomaly_score'] = -self.model.score_samples(X)
        
        logger.info(f"检测到 {df['ml_anomaly'].sum()} 条 ML 异常")
        
        return df
    
    def detect_coordinated(self, df: pd.DataFrame, 
                          time_window_hours: int = 24, 
                          similarity_threshold: float = 0.85) -> Tuple[pd.DataFrame, List[Dict]]:
        """检测协同刷评行为"""
        if not NETWORKX_AVAILABLE:
            logger.warning("NetworkX 未安装，跳过协同检测")
            df['is_coordinated'] = 0
            return df, []
        
        logger.info("执行协同行为检测...")
        df = df.sort_values('review_date').copy()
        
        # 时间窗口分组
        df['time_window'] = pd.to_datetime(df['review_date']).dt.floor(f'{time_window_hours}H')
        
        coordinated_groups = []
        
        for window, group in df.groupby('time_window'):
            if len(group) < 10:
                continue
            
            # 计算文本相似度
            texts = group['review_content_clean'].fillna('').tolist()
            
            try:
                vectorizer = TfidfVectorizer(max_features=500)
                tfidf = vectorizer.fit_transform(texts)
                similarity_matrix = cosine_similarity(tfidf)
                
                # 找高相似度对
                high_sim_pairs = []
                for i in range(len(similarity_matrix)):
                    for j in range(i+1, len(similarity_matrix)):
                        if similarity_matrix[i, j] > similarity_threshold:
                            high_sim_pairs.append((i, j, similarity_matrix[i, j]))
                
                # 构建图，找连通分量
                if high_sim_pairs:
                    G = nx.Graph()
                    G.add_weighted_edges_from(high_sim_pairs)
                    
                    for component in nx.connected_components(G):
                        if len(component) >= 10:
                            coordinated_groups.append({
                                'window': window,
                                'indices': [group.index[i] for i in component],
                                'size': len(component),
                                'avg_similarity': np.mean([
                                    similarity_matrix[i, j] 
                                    for i in component for j in component if i < j
                                ])
                            })
            except:
                continue
        
        # 标记协同行为
        df['is_coordinated'] = 0
        for group in coordinated_groups:
            df.loc[group['indices'], 'is_coordinated'] = 1
        
        logger.info(f"检测到 {len(coordinated_groups)} 个可疑刷评团伙")
        
        return df, coordinated_groups
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行异常检测"""
        if self.method == 'statistical':
            return self.detect_statistical(df)
        else:
            return self.detect_ml(df)

    def detect_combined(self, df: pd.DataFrame, ml_weight: float = 0.5, q_medium: float = 0.85, q_high: float = 0.98) -> pd.DataFrame:
        """同时运行统计与 ML 异常检测，并合并成一个 ensemble 分数与标签。

        Args:
            df: 输入 DataFrame（不修改原始 df）
            ml_weight: ML 部分在 ensemble 中的权重（0-1），统计的权重为 1-ml_weight

        Returns:
            df_out: 包含统计/ML单独结果与组合分数/标签的 DataFrame
        """
        logger.info("执行组合（统计 + ML）异常检测...")

        # 先运行统计检测（得到 is_statistical_anomaly, statistical_anomaly_score）
        df_stat = self.detect_statistical(df.copy())

        # 再运行 ML 检测（得到 ml_anomaly, ml_anomaly_score）
        df_ml = self.detect_ml(df.copy())

        # 合并结果（以索引对齐）
        df_out = df.copy()
        # Ensure indices align
        df_out = df_out.reset_index(drop=True)
        df_stat = df_stat.reset_index(drop=True)
        df_ml = df_ml.reset_index(drop=True)

        # Attach per-method outputs
        df_out['is_statistical_anomaly'] = df_stat.get('is_statistical_anomaly', pd.Series([0]*len(df_out))).values
        df_out['statistical_anomaly_score'] = df_stat.get('statistical_anomaly_score', pd.Series([0]*len(df_out))).values

        df_out['ml_anomaly'] = df_ml.get('ml_anomaly', pd.Series([0]*len(df_out))).values
        df_out['ml_anomaly_score'] = df_ml.get('ml_anomaly_score', pd.Series([0.0]*len(df_out))).values

        # 归一化分数到 [0,1]：先做 z-score（(x-mean)/std），再通过高斯累积分布函数映射到 [0,1]
        # 这种方式在没有标注数据时更稳健（相对排名信息），并且对极端值更温和。
        def _gauss_map(arr):
            x = np.array(arr, dtype=float)
            x = np.nan_to_num(x, nan=0.0)
            mu = np.nanmean(x)
            sigma = np.nanstd(x)
            if sigma == 0 or np.isnan(sigma):
                # constant series -> map to zeros
                return np.zeros_like(x)
            z = (x - mu) / sigma
            # try using scipy if available for norm.cdf
            try:
                from scipy.stats import norm
                mapped = norm.cdf(z)
            except Exception:
                # fallback using numpy's erf: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
                mapped = 0.5 * (1.0 + np.erf(z / np.sqrt(2.0)))
            mapped[np.isnan(mapped)] = 0.0
            return mapped

        stat_norm = _gauss_map(df_out['statistical_anomaly_score'].fillna(0).values)
        ml_norm = _gauss_map(df_out['ml_anomaly_score'].fillna(0).values)

        ensemble_score = (1 - ml_weight) * stat_norm + ml_weight * ml_norm
        df_out['ensemble_anomaly_score'] = ensemble_score

        # 决策阈值：ensemble_score 的 95th 百分位为阈值（可调整/暴露为参数）
        try:
            thresh = np.nanpercentile(ensemble_score, 95)
        except Exception:
            thresh = 0.5

        df_out['is_ensemble_anomaly'] = (ensemble_score >= thresh).astype(int)

        logger.info(f"组合检测：ensemble 异常数 = {int(df_out['is_ensemble_anomaly'].sum())}")
        # 校准 & 分段（low/medium/high）以便统一告警策略
        try:
            df_out = self._calibrate_and_bucket(df_out, score_col='ensemble_anomaly_score', q_medium=q_medium, q_high=q_high)
        except Exception:
            logger.warning("Failed to calibrate/bucket ensemble scores, skipping severity bucketing")

        return df_out

    def _calibrate_and_bucket(self, df: pd.DataFrame, score_col: str = 'ensemble_anomaly_score', q_medium: float = 0.8, q_high: float = 0.95) -> pd.DataFrame:
        """基于分位数对异常分数做简单校准与分段。

        Adds:
            - `anomaly_severity_score` (0-1 normalized score)
            - `anomaly_severity` categorical column with values {'low','medium','high'}

        Bucketing rule (default):
            - low: score < 0.8 quantile
            - medium: 0.8 <= score < 0.95
            - high: score >= 0.95

        这些阈值是经验值，可根据业务调整或暴露为函数参数。
        """
        df = df.copy()
        if score_col not in df.columns:
            df['anomaly_severity_score'] = 0.0
            df['anomaly_severity'] = 'low'
            return df

        scores = np.array(df[score_col].fillna(0).astype(float))
        # 简单线性缩放到 0-1（基于最小/最大），同时 guard 常量序列
        min_s = np.nanmin(scores)
        max_s = np.nanmax(scores)
        if np.isfinite(min_s) and np.isfinite(max_s) and max_s > min_s:
            norm = (scores - min_s) / (max_s - min_s)
        else:
            norm = np.zeros_like(scores)

        df['anomaly_severity_score'] = norm

        # 分段（使用分位数）
        try:
            q80 = np.nanpercentile(norm, int(q_medium * 100))
            q95 = np.nanpercentile(norm, int(q_high * 100))
        except Exception:
            q80, q95 = 0.5, 0.8

        severity = np.array(['low'] * len(norm), dtype=object)
        severity[(norm >= q80) & (norm < q95)] = 'medium'
        severity[norm >= q95] = 'high'
        df['anomaly_severity'] = severity

        logger.info(f"Anomaly severity distribution: low={int((df['anomaly_severity']=='low').sum())}, medium={int((df['anomaly_severity']=='medium').sum())}, high={int((df['anomaly_severity']=='high').sum())}")
        return df


class ComprehensiveAnalysisSystem:
    """综合分析系统：分类 + 趋势 + 异常"""
    
    def __init__(self):
        self.trend_extractor = TrendFeatureExtractor()
        self.anomaly_detector = AnomalyDetector(method='isolation_forest')
        self.sentiment_model = None
        self.risk_model = None
        
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """提取所有特征"""
        logger.info("提取综合特征...")
        
        # 1. 趋势特征
        df = self.trend_extractor.extract(df)
        # 返回带有趋势特征的表
        return df
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """分析新评论"""
        logger.info("分析新评论...")
        
        # 提取特征
        df_features = self.extract_all_features(df)
        
        results = {}
        
        # 1. 组合异常检测（统计 + ML）
        df_combined = self.anomaly_detector.detect_combined(df_features)
        results['statistical_anomaly'] = df_combined.get('is_statistical_anomaly', pd.Series([0]*len(df_combined))).values
        results['statistical_score'] = df_combined.get('statistical_anomaly_score', pd.Series([0.0]*len(df_combined))).values
        results['ml_anomaly'] = df_combined.get('ml_anomaly', pd.Series([0]*len(df_combined))).values
        results['ml_score'] = df_combined.get('ml_anomaly_score', pd.Series([0.0]*len(df_combined))).values
        results['ensemble_score'] = df_combined.get('ensemble_anomaly_score', pd.Series([0.0]*len(df_combined))).values
        results['ensemble_anomaly'] = df_combined.get('is_ensemble_anomaly', pd.Series([0]*len(df_combined))).values

        # severity outputs (if available)
        results['anomaly_severity_score'] = df_combined.get('anomaly_severity_score', pd.Series([0.0]*len(df_combined))).values
        results['anomaly_severity'] = df_combined.get('anomaly_severity', pd.Series(['low']*len(df_combined))).values

        # 2. 协同检测
        df_coord, coord_groups = self.anomaly_detector.detect_coordinated(df_features)
        # Merge coordinated flag into the combined detection dataframe so callers can inspect it there
        try:
            df_combined['is_coordinated'] = df_coord.get('is_coordinated', pd.Series(0, index=df_coord.index)).values
        except Exception:
            # fallback: ensure column exists
            df_combined['is_coordinated'] = 0
        results['coordinated'] = df_combined['is_coordinated'].values
        results['coordinated_groups'] = coord_groups

        # 3. 综合告警（基于 ensemble 异常与协同行为）
        results['critical_alert'] = (
            (results['ensemble_anomaly'] == 1) |
            (results['coordinated'] == 1)
        ).astype(int)
        
        results['dataframe'] = df_combined

        return results
    
    def generate_report(self, results: Dict) -> Dict:
        """生成分析报告"""
        # Build a report using available keys from the combined analysis results
        df = results.get('dataframe')
        total_comments = len(df) if df is not None else 0
        ensemble_score = results.get('ensemble_score', None)
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_comments': int(total_comments),
            'anomalies': int(np.sum(results.get('ensemble_anomaly', []))) ,
            'coordinated_behavior': int(np.sum(results.get('coordinated', []))) ,
            'coordinated_groups': int(len(results.get('coordinated_groups', []))),
            'critical_alerts': int(np.sum(results.get('critical_alert', []))) ,
            'avg_ensemble_score': float(np.nanmean(ensemble_score)) if ensemble_score is not None else None,
            'max_ensemble_score': float(np.nanmax(ensemble_score)) if ensemble_score is not None else None,
        }
        
        return report


def visualize_analysis(df: pd.DataFrame, results: Dict, save_path: str = 'analysis_report.png'):
    """可视化分析结果"""
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    
    df_plot = df.copy()
    df_plot['hour'] = pd.to_datetime(df_plot['review_date']).dt.floor('H')
    
    # 1. 情感趋势
    ax1 = axes[0, 0]
    hourly_sentiment = df_plot.groupby('hour')['vader_compound'].mean()
    ax1.plot(hourly_sentiment.index, hourly_sentiment.values, linewidth=2, label='平均情感')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('情感趋势（按小时）', fontsize=14, fontweight='bold')
    ax1.set_ylabel('平均情感分数')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 评论量与趋势告警
    ax2 = axes[0, 1]
    hourly_count = df_plot.groupby('hour').size()
    ax2.bar(hourly_count.index, hourly_count.values, alpha=0.6, label='评论量')
    
    alert_hours = df_plot[results['trend_alert'] == 1].groupby('hour').size()
    if len(alert_hours) > 0:
        ax2.scatter(alert_hours.index, alert_hours.values * 2, 
                   color='red', s=100, marker='^', label='趋势告警', zorder=5)
    
    ax2.set_title('评论量与趋势告警', fontsize=14, fontweight='bold')
    ax2.set_ylabel('评论数量')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 异常分数分布
    ax3 = axes[1, 0]
    ax3.hist(results['anomaly_score'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    threshold = np.percentile(results['anomaly_score'], 95)
    ax3.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'95th percentile')
    ax3.set_title('异常分数分布', fontsize=14, fontweight='bold')
    ax3.set_xlabel('异常分数')
    ax3.set_ylabel('频数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 情感分布（正常 vs 异常）
    ax4 = axes[1, 1]
    normal_sentiment = df_plot[results['anomaly'] == 0]['vader_compound']
    anomaly_sentiment = df_plot[results['anomaly'] == 1]['vader_compound']
    
    ax4.hist(normal_sentiment, bins=30, alpha=0.5, label='正常评论', color='blue')
    ax4.hist(anomaly_sentiment, bins=30, alpha=0.5, label='异常评论', color='red')
    ax4.set_title('情感分布对比', fontsize=14, fontweight='bold')
    ax4.set_xlabel('情感分数')
    ax4.set_ylabel('频数')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 综合告警时间线
    ax5 = axes[2, 0]
    df_plot['critical'] = results['critical_alert']
    critical_timeline = df_plot.groupby('hour')['critical'].sum()
    
    ax5.fill_between(critical_timeline.index, critical_timeline.values, 
                     alpha=0.3, color='red', label='告警密度')
    ax5.plot(critical_timeline.index, critical_timeline.values, 
            color='darkred', linewidth=2, marker='o')
    ax5.set_title('综合告警时间线', fontsize=14, fontweight='bold')
    ax5.set_ylabel('告警数量')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 特征重要性（如果有趋势模型）
    ax6 = axes[2, 1]
    try:
        importance_df = results.get('feature_importance')
        if importance_df is not None:
            top_features = importance_df.head(10)
            ax6.barh(top_features['feature'], top_features['importance'], color='steelblue')
            ax6.set_title('Top 10 重要特征', fontsize=14, fontweight='bold')
            ax6.set_xlabel('重要性')
        else:
            ax6.text(0.5, 0.5, '特征重要性数据不可用', 
                    ha='center', va='center', fontsize=12)
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
    except:
        ax6.text(0.5, 0.5, '特征重要性计算失败', 
                ha='center', va='center', fontsize=12)
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"可视化结果已保存至 {save_path}")
    
    return fig


# 便捷函数
def quick_analysis(df: pd.DataFrame, train: bool = True) -> Tuple[ComprehensiveAnalysisSystem, Dict]:
    """
    快速分析接口
    
    Args:
        df: 评论数据
        train: 是否训练模型（首次使用设为 True）
        
    Returns:
        system: 分析系统实例
        results: 分析结果
    """
    system = ComprehensiveAnalysisSystem()
    results = system.analyze(df)
    report = system.generate_report(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("分析报告")
    logger.info("=" * 60)
    for key, value in report.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 60)
    
    return system, results


def load_feature_dataframe_from_dir(output_dir: str, file_prefix: str, include_flagged: Optional[bool] = None) -> Tuple[pd.DataFrame, Optional[bool]]:
    """
    从指定的 features 输出目录加载由 feature_engineering 生成的 DataFrame 文件。

    参数:
        output_dir: features 文件所在目录
        file_prefix: 保存时使用的前缀（例如 gpu_optimized_features_action）
        include_flagged: 如果为 True 则只查找包含被标记评论的文件（_inclflagged），
                        如果为 False 则只查找不包含被标记评论的文件（_exclflagged），
                        如果为 None 则会自动搜索可用的文件并按优先级选择（inclflagged > exclflagged > 任意匹配）

    返回: (df, used_flag)
        df: 加载的 DataFrame
        used_flag: 如果能确定文件是 inclflagged 或 exclflagged，则返回布尔值；否则返回 None
    """
    os.makedirs(output_dir, exist_ok=True)

    # candidates may exist as parquet copies; prefer parquet if available
    pattern_incl_parquet = os.path.join(output_dir, f"{file_prefix}_inclflagged_dataframe.parquet")
    pattern_excl_parquet = os.path.join(output_dir, f"{file_prefix}_exclflagged_dataframe.parquet")
    pattern_any_parquet = os.path.join(output_dir, f"{file_prefix}*_dataframe.parquet")

    pattern_incl = os.path.join(output_dir, f"{file_prefix}_inclflagged_dataframe.xlsx")
    pattern_excl = os.path.join(output_dir, f"{file_prefix}_exclflagged_dataframe.xlsx")
    pattern_any = os.path.join(output_dir, f"{file_prefix}*_dataframe.xlsx")

    # 优先按照 include_flagged 指定的值查找
    # If parquet exists, prefer it (faster and avoids Excel parsing)
    if include_flagged is True and os.path.exists(pattern_incl_parquet):
        logger.info(f"Loading included-flagged features from Parquet: {pattern_incl_parquet}")
        return pd.read_parquet(pattern_incl_parquet), True
    if include_flagged is False and os.path.exists(pattern_excl_parquet):
        logger.info(f"Loading excluded-flagged features from Parquet: {pattern_excl_parquet}")
        return pd.read_parquet(pattern_excl_parquet), False

    # Next, if parquet not found, check Excel files (and create parquet copy after reading)
    if include_flagged is True and os.path.exists(pattern_incl):
        logger.info(f"Loading included-flagged features from Excel: {pattern_incl}")
        df = pd.read_excel(pattern_incl)
        # attempt to write parquet copy for future fast loads
        try:
            pq_path = os.path.splitext(pattern_incl)[0] + '.parquet'
            df.to_parquet(pq_path, index=False)
            logger.info(f"Wrote Parquet copy to {pq_path}")
        except Exception as e:
            logger.warning(f"Could not write parquet copy for {pattern_incl}: {e}")
        return df, True
    if include_flagged is False and os.path.exists(pattern_excl):
        logger.info(f"Loading excluded-flagged features from Excel: {pattern_excl}")
        df = pd.read_excel(pattern_excl)
        try:
            pq_path = os.path.splitext(pattern_excl)[0] + '.parquet'
            df.to_parquet(pq_path, index=False)
            logger.info(f"Wrote Parquet copy to {pq_path}")
        except Exception as e:
            logger.warning(f"Could not write parquet copy for {pattern_excl}: {e}")
        return df, False

    # 未指定或指定文件不存在，搜索候选文件
    # Prefer any parquet candidates first
    parquet_candidates = sorted(glob.glob(pattern_any_parquet))
    if parquet_candidates:
        logger.info(f"Auto-loading Parquet features from: {parquet_candidates[0]}")
        return pd.read_parquet(parquet_candidates[0]), None

    # Fallback: search Excel candidates and create parquet copy after reading
    candidates = sorted(glob.glob(pattern_any))
    if not candidates:
        raise FileNotFoundError(f"No feature dataframe found for prefix '{file_prefix}' in {output_dir}")

    # Prefer inclflagged/exclflagged variants
    for p in candidates:
        if p.endswith('_inclflagged_dataframe.xlsx'):
            logger.info(f"Auto-loading included-flagged features from: {p}")
            df = pd.read_excel(p)
            try:
                pq_path = os.path.splitext(p)[0] + '.parquet'
                df.to_parquet(pq_path, index=False)
                logger.info(f"Wrote Parquet copy to {pq_path}")
            except Exception as e:
                logger.warning(f"Could not write parquet copy for {p}: {e}")
            return df, True
    for p in candidates:
        if p.endswith('_exclflagged_dataframe.xlsx'):
            logger.info(f"Auto-loading excluded-flagged features from: {p}")
            df = pd.read_excel(p)
            try:
                pq_path = os.path.splitext(p)[0] + '.parquet'
                df.to_parquet(pq_path, index=False)
                logger.info(f"Wrote Parquet copy to {pq_path}")
            except Exception as e:
                logger.warning(f"Could not write parquet copy for {p}: {e}")
            return df, False

    # 回退：加载第一个匹配 Excel 文件
    logger.info(f"Auto-loading features from: {candidates[0]}")
    df = pd.read_excel(candidates[0])
    try:
        pq_path = os.path.splitext(candidates[0])[0] + '.parquet'
        df.to_parquet(pq_path, index=False)
        logger.info(f"Wrote Parquet copy to {pq_path}")
    except Exception as e:
        logger.warning(f"Could not write parquet copy for {candidates[0]}: {e}")
    return df, None


def quick_analysis_from_features_dir(output_dir: str, file_prefix: str, include_flagged: Optional[bool] = None, train: bool = True) -> Tuple[ComprehensiveAnalysisSystem, Dict, Optional[bool]]:
    """
    从保存的 features 文件直接加载并运行快速分析的便捷方法。

    返回: (system, results, used_flagged)
    """
    df, used_flagged = load_feature_dataframe_from_dir(output_dir, file_prefix, include_flagged=include_flagged)
    system, results = quick_analysis(df, train=train)
    return system, results, used_flagged


def process_features_to_full(output_dir: str, file_prefix: str, include_flagged: Optional[bool] = None, windows: Optional[List[int]] = None, save_path: Optional[str] = None, run_coordinated: bool = True, q_medium: float = 0.8, q_high: float = 0.95) -> Tuple[pd.DataFrame, Optional[bool]]:
    """
    从 FE 输出目录加载基础特征（由 feature_engineering.save_features 生成），
    对每个 appid 运行趋势提取与异常检测，生成完整的特征表并保存为 excel。

    Args:
        output_dir: features 输出目录（包含由 FE 保存的 *_dataframe.csv）
        file_prefix: 文件名前缀
        include_flagged: 指定是否加载包含被标记行的文件；None 将自动选择（incl > excl > any）
        windows: 可选的趋势窗口（小时列表），默认使用 [24,48,72]
        save_path: 可选，保存完整特征表的路径；若 None，则保存在 output_dir 下，使用 file_prefix 与 include_flagged 生成文件名

    Returns:
        df_full: 增强后的 DataFrame
        used_flagged: bool|None 表示实际使用的 flagged 状态
    """
    # Load FE dataframe
    df, used_flagged = load_feature_dataframe_from_dir(output_dir, file_prefix, include_flagged=include_flagged)

    # Ensure windows
    if windows is None:
        windows = [24, 48, 72]

    trend_extractor = TrendFeatureExtractor(windows=windows)
    detector = AnomalyDetector(method='isolation_forest', sensitivity='medium')

    # Guard appid
    if 'appid' not in df.columns:
        raise ValueError("'appid' column required in FE dataframe for grouped trend extraction")

    trend_frames = []
    for app_id, grp in df.groupby('appid', dropna=False):
        grp_sorted = grp.sort_values('review_date').reset_index(drop=True)
        tr = trend_extractor.extract(grp_sorted, appid=app_id)
        trend_frames.append(tr)

    df_trend = pd.concat(trend_frames, ignore_index=True)

    # Run anomaly detection (combined statistical + ML) to get ensemble score and severity buckets
    df_out = detector.detect_combined(df_trend, q_medium=q_medium, q_high=q_high)

    # Run coordinated detection if requested (defaults to True)
    coordinated_groups = []
    if run_coordinated:
        try:
            df_coord, coordinated_groups = detector.detect_coordinated(df_trend)
            # merge coordinated flag into output (preserve existing index alignment)
            if 'is_coordinated' in df_coord.columns:
                df_out['is_coordinated'] = df_coord['is_coordinated'].values
            else:
                df_out['is_coordinated'] = 0
        except Exception as e:
            logger.warning(f"Coordinated detection failed: {e}")
            df_out['is_coordinated'] = 0
    else:
        df_out['is_coordinated'] = 0

    # Save result - prefer parquet as the primary artifact
    suffix = '_inclflagged' if used_flagged is True else ('_exclflagged' if used_flagged is False else '')
    if save_path is None:
        parquet_path = os.path.join(output_dir, f"{file_prefix}{suffix}_enhanced_features.parquet")
        save_path = parquet_path
    else:
        # if user provided an .xlsx path, also write parquet copy alongside
        if save_path.lower().endswith('.xlsx'):
            parquet_path = os.path.splitext(save_path)[0] + '.parquet'
        elif save_path.lower().endswith('.parquet'):
            parquet_path = save_path
        else:
            # default to parquet with provided base
            parquet_path = os.path.splitext(save_path)[0] + '.parquet'

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    # write parquet primary
    try:
        df_out.to_parquet(parquet_path, index=False)
        logger.info(f"Saved enhanced features to Parquet: {parquet_path}")
    except Exception as e:
        logger.warning(f"Failed to write Parquet enhanced features: {e}")

    # write Excel if user explicitly requested an xlsx save_path
    if save_path.lower().endswith('.xlsx'):
        try:
            df_out.to_excel(save_path, index=False)
            logger.info(f"Also saved enhanced features to Excel: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to write Excel enhanced features: {e}")

    # If coordinated groups detected, save a small JSON report next to the CSV
    if coordinated_groups:
        try:
            import json
            report_path = os.path.splitext(save_path)[0] + '_coordinated_groups.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(coordinated_groups, f, default=str, ensure_ascii=False, indent=2)
            logger.info(f"Saved coordinated groups report to {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save coordinated groups JSON: {e}")

    return df_out, used_flagged


def main():
    """
    Main entry: 自动发现指定 features 目录下由 FE 生成的 base dataframe 前缀，
    并为每个前缀生成两类增强特征：excluded-flagged 和 included-flagged（若对应文件存在）。

    运行该脚本将保存增强特征为 Excel 文件到同一 features 目录下，文件名格式为
    {prefix}_exclflagged_enhanced_features.xlsx 和 {prefix}_inclflagged_enhanced_features.xlsx
    """
    # Default features root: use the user's workspace absolute features directory and search recursively
    features_root = r"C:\Users\12932\Desktop\nus\BAP\test\fps"
    features_root = os.path.abspath(features_root)

    logger.info(f"Recursively searching for FE dataframe files under: {features_root}")
    # Search for both Excel and Parquet dataframe artifacts
    pattern = os.path.join(features_root, "**", "*_dataframe.*")
    all_candidates = sorted(glob.glob(pattern, recursive=True))
    # filter to xlsx/xls/parquet
    candidates = [p for p in all_candidates if p.lower().endswith(('.xlsx', '.xls', '.parquet'))]
    if not candidates:
        logger.error(f"No FE dataframe files (.xlsx/.parquet) found under {features_root}. Ensure FE artifacts exist.")
        return

    logger.info(f"Found {len(candidates)} FE dataframe files (xlsx/parquet); grouping by folder+prefix and processing each")

    # Group unique (folder, prefix) so we process each prefix once regardless of extension
    seen = set()
    suffixes = [ '_inclflagged_dataframe.xlsx', '_inclflagged_dataframe.parquet',
                 '_exclflagged_dataframe.xlsx', '_exclflagged_dataframe.parquet',
                 '_dataframe.xlsx', '_dataframe.parquet', '_dataframe.xls']

    for candidate in candidates:
        folder = os.path.dirname(candidate)
        base = os.path.basename(candidate)
        prefix = None
        for suf in suffixes:
            if base.endswith(suf):
                prefix = base[:-len(suf)]
                break
        if prefix is None:
            # try a more relaxed split: remove last two extensions parts
            prefix = base.rsplit('_dataframe', 1)[0]

        key = (folder, prefix)
        if key in seen:
            continue
        seen.add(key)

        # Build candidate paths for explicit variants (either parquet or excel)
        path_incl_parquet = os.path.join(folder, f"{prefix}_inclflagged_dataframe.parquet")
        path_excl_parquet = os.path.join(folder, f"{prefix}_exclflagged_dataframe.parquet")
        path_any_parquet = os.path.join(folder, f"{prefix}_dataframe.parquet")

        path_incl_xlsx = os.path.join(folder, f"{prefix}_inclflagged_dataframe.xlsx")
        path_excl_xlsx = os.path.join(folder, f"{prefix}_exclflagged_dataframe.xlsx")
        path_any_xlsx = os.path.join(folder, f"{prefix}_dataframe.xlsx")

        processed_any = False

        # Prefer parquet variants if present (process_features_to_full will itself prefer parquet when loading)
        try:
            if os.path.exists(path_excl_parquet) or os.path.exists(path_excl_xlsx):
                chosen = path_excl_parquet if os.path.exists(path_excl_parquet) else path_excl_xlsx
                logger.info(f"Processing file={chosen} (include_flagged=False)")
                df_out, used_flagged = process_features_to_full(folder, prefix, include_flagged=False)
                logger.info(f"Saved enhanced features for {prefix} (include_flagged=False) rows={len(df_out)}")
                processed_any = True

            if os.path.exists(path_incl_parquet) or os.path.exists(path_incl_xlsx):
                chosen = path_incl_parquet if os.path.exists(path_incl_parquet) else path_incl_xlsx
                logger.info(f"Processing file={chosen} (include_flagged=True)")
                df_out, used_flagged = process_features_to_full(folder, prefix, include_flagged=True)
                logger.info(f"Saved enhanced features for {prefix} (include_flagged=True) rows={len(df_out)}")
                processed_any = True

            # If neither explicit variant exists, but any dataframe exists (parquet preferred), process generic
            if not processed_any:
                if os.path.exists(path_any_parquet) or os.path.exists(path_any_xlsx):
                    chosen = path_any_parquet if os.path.exists(path_any_parquet) else path_any_xlsx
                    logger.info(f"Processing file={chosen} (generic)")
                    df_out, used_flagged = process_features_to_full(folder, prefix, include_flagged=None)
                    logger.info(f"Saved enhanced features for {prefix} (include_flagged={used_flagged}) rows={len(df_out)}")
                    processed_any = True
        except Exception:
            logger.exception(f"Failed processing prefix={prefix} in folder={folder}")

        if not processed_any:
            logger.warning(f"No suitable dataframe variant found for prefix={prefix} in {folder}; skipped")


if __name__ == '__main__':
    main()
