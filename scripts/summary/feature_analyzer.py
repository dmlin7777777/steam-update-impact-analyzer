import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    """
    特征分析和模型训练类
    用于分析提取的特征并构建预测模型
    """
    
    def __init__(self, features_path: str):
        """
        初始化分析器
        
        Args:
            features_path: 特征CSV文件路径
        """
        self.features_df = pd.read_csv(features_path, encoding='utf-8')
        self.scaler = StandardScaler()
        print(f"Loaded {len(self.features_df)} samples with {len(self.features_df.columns)} features")
        # 自动检测主题概率列
        self.topic_prob_cols = [col for col in self.features_df.columns if col.startswith('topic_') and col.endswith('_prob')]
        
    def analyze_feature_distributions(self):
        """分析特征分布"""
        print("\n=== Feature Distribution Analysis ===")
        
        # 数值特征统计
        numeric_features = self.features_df.select_dtypes(include=[np.number]).columns
        print(f"\nNumeric features: {len(numeric_features)}")
        
        # 显示基本统计信息
        print("\nBasic Statistics for Key Features:")
        key_features = [
            'text_length', 'word_count', 'readability_score',
            'vader_compound', 'toxicity_score', 'author_reputation_score'
        ]
        
        for feature in key_features:
            if feature in self.features_df.columns:
                print(f"{feature}: mean={self.features_df[feature].mean():.3f}, "
                      f"std={self.features_df[feature].std():.3f}")
    
    def analyze_sentiment_patterns(self):
        """分析情感模式"""
        print("\n=== Sentiment Pattern Analysis ===")
        
        # 情感分布
        sentiment_dist = self.features_df['sentiment_category'].value_counts()
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_dist.items():
            print(f"  {sentiment}: {count} ({count/len(self.features_df)*100:.1f}%)")
        
        # 情感与其他特征的关系
        print("\nSentiment vs Text Characteristics:")
        sentiment_stats = self.features_df.groupby('sentiment_category')[
            ['text_length', 'word_count', 'readability_score', 'toxicity_score']
        ].mean()
        print(sentiment_stats.round(3))
    
    def analyze_game_feedback_patterns(self):
        """分析游戏反馈模式"""
        print("\n=== Game Feedback Pattern Analysis ===")
        
        # 游戏相关投诉分布
        feedback_features = [
            'contains_bug_report',
            'contains_balance_complaint', 
            'contains_monetization_complaint',
            'mentions_performance'
        ]
        
        print("Game Feedback Categories:")
        for feature in feedback_features:
            if feature in self.features_df.columns:
                count = self.features_df[feature].sum()
                percentage = (count / len(self.features_df)) * 100
                print(f"  {feature}: {count} ({percentage:.1f}%)")
        
        # 投诉与评分的关系
        if 'voted_up' in self.features_df.columns:
            print("\nFeedback vs Review Rating (voted_up):")
            for feature in feedback_features:
                if feature in self.features_df.columns:
                    complaint_reviews = self.features_df[self.features_df[feature] == 1]
                    if len(complaint_reviews) > 0:
                        positive_rate = complaint_reviews['voted_up'].mean()
                        print(f"  {feature}: {positive_rate:.3f} positive rating")
    
    def analyze_author_characteristics(self):
        """分析作者特征"""
        print("\n=== Author Characteristics Analysis ===")
        
        author_features = [
            'is_active_reviewer', 'is_hardcore_gamer', 'is_casual_gamer',
            'author_reputation_score'
        ]
        
        print("Author Categories:")
        for feature in author_features:
            if feature in self.features_df.columns:
                if feature == 'author_reputation_score':
                    print(f"  {feature}: mean={self.features_df[feature].mean():.3f}")
                else:
                    count = self.features_df[feature].sum()
                    percentage = (count / len(self.features_df)) * 100
                    print(f"  {feature}: {count} ({percentage:.1f}%)")
        
        # 作者类型与评论质量的关系
        if 'voted_up' in self.features_df.columns:
            print("\nAuthor Type vs Review Quality:")
            for feature in ['is_active_reviewer', 'is_hardcore_gamer', 'is_casual_gamer']:
                if feature in self.features_df.columns:
                    author_group = self.features_df[self.features_df[feature] == 1]
                    if len(author_group) > 0:
                        positive_rate = author_group['voted_up'].mean()
                        avg_text_length = author_group['text_length'].mean()
                        print(f"  {feature}: {positive_rate:.3f} positive rate, "
                              f"{avg_text_length:.0f} avg text length")
    
    def analyze_topic_distribution(self):
        """分析主题分布"""
        print("\n=== Topic Distribution Analysis ===")
        
        if 'topic_category' in self.features_df.columns:
            topic_dist = self.features_df['topic_category'].value_counts()
            print("Topic Distribution:")
            for topic, count in topic_dist.items():
                print(f"  {topic}: {count} ({count/len(self.features_df)*100:.1f}%)")

            # 打印每个主题编号对应的关键词（从 topic_category 和概率列自动提取）
            print("\nLDA主题编号与关键词映射：")
            if self.topic_prob_cols:
                # 取每个主题编号的代表关键词
                topic_map = {}
                for idx, col in enumerate(self.topic_prob_cols):
                    # 找出该主题概率最大的评论，取其 topic_category
                    max_idx = self.features_df[col].idxmax()
                    topic_map[f"topic_{idx}"] = self.features_df.loc[max_idx, 'topic_category']
                for k, v in topic_map.items():
                    print(f"  {k}: {v}")
            else:
                print("  未检测到主题概率列，无法自动映射。")

            # 主题与情感的关系
            if 'sentiment_category' in self.features_df.columns:
                print("\nTopic vs Sentiment Cross-tabulation:")
                crosstab = pd.crosstab(
                    self.features_df['topic_category'], 
                    self.features_df['sentiment_category'],
                    normalize='index'
                )
                print(crosstab.round(3))
    
    def build_review_quality_predictor(self):
        """构建评论质量预测模型"""
        print("\n=== Building Review Quality Prediction Model ===")
        
        if 'voted_up' not in self.features_df.columns:
            print("No 'voted_up' target variable found")
            return None
        
        # 选择特征
        feature_columns = [
            'text_length', 'word_count', 'readability_score',
            'vader_compound', 'vader_positive', 'vader_negative',
            'contains_bug_report', 'contains_balance_complaint',
            'contains_monetization_complaint', 'toxicity_score',
            'author_reputation_score', 'played_hours'
        ]
        
        # 只使用存在的特征
        available_features = [col for col in feature_columns if col in self.features_df.columns]
        
        if len(available_features) == 0:
            print("No suitable features found for modeling")
            return None
        
        # 准备数据
        X = self.features_df[available_features].fillna(0)
        y = self.features_df['voted_up']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练模型
        print(f"\nTraining models with {len(available_features)} features...")
        print(f"Features used: {available_features}")
        
        # 随机森林
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # 逻辑回归
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        
        # 评估模型
        print("\n--- Random Forest Results ---")
        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        print(f"Accuracy: {rf_model.score(X_test, y_test):.3f}")
        print(f"AUC-ROC: {roc_auc_score(y_test, rf_prob):.3f}")
        
        print("\n--- Logistic Regression Results ---")
        lr_pred = lr_model.predict(X_test_scaled)
        lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
        print(f"Accuracy: {lr_model.score(X_test_scaled, y_test):.3f}")
        print(f"AUC-ROC: {roc_auc_score(y_test, lr_prob):.3f}")
        
        # 特征重要性
        print("\n--- Feature Importance (Random Forest) ---")
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        return {
            'rf_model': rf_model,
            'lr_model': lr_model,
            'scaler': self.scaler,
            'features': available_features,
            'feature_importance': feature_importance
        }
    
    def build_toxicity_detector(self):
        """构建毒性检测模型"""
        print("\n=== Building Toxicity Detection Model ===")
        
        if 'is_toxic' not in self.features_df.columns:
            print("No 'is_toxic' target variable found")
            return None
        
        # 检查毒性标签分布
        toxic_dist = self.features_df['is_toxic'].value_counts()
        print(f"Toxicity distribution: {toxic_dist.to_dict()}")
        
        if toxic_dist.min() < 10:  # 如果毒性样本太少
            print("Insufficient toxic samples for model training")
            return None
        
        # 选择特征
        feature_columns = [
            'text_length', 'word_count', 'uppercase_ratio', 'punctuation_ratio',
            'exclamation_count', 'contains_toxic_keywords',
            'contains_aggressive_keywords', 'contains_profanity',
            'vader_negative', 'vader_compound'
        ]
        
        # 只使用存在的特征
        available_features = [col for col in feature_columns if col in self.features_df.columns]
        
        if len(available_features) == 0:
            print("No suitable features found for toxicity modeling")
            return None
        
        # 准备数据
        X = self.features_df[available_features].fillna(0)
        y = self.features_df['is_toxic']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练随机森林模型
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf_model.fit(X_train, y_train)
        
        # 评估模型
        rf_pred = rf_model.predict(X_test)
        rf_prob = rf_model.predict_proba(X_test)[:, 1]
        
        print(f"\nToxicity Detection Results:")
        print(f"Accuracy: {rf_model.score(X_test, y_test):.3f}")
        print(f"AUC-ROC: {roc_auc_score(y_test, rf_prob):.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, rf_pred, target_names=['Not Toxic', 'Toxic']))
        
        return {
            'model': rf_model,
            'features': available_features
        }
    
    def generate_insights_report(self, output_path: str):
        """生成洞察报告"""
        print(f"\n=== Generating Insights Report ===")
        
        insights = []
        
        # 数据概况
        insights.append("=== GAME REVIEW FEATURE ANALYSIS REPORT ===\n")
        insights.append(f"Total Reviews Analyzed: {len(self.features_df)}\n")
        insights.append(f"Total Features Extracted: {len(self.features_df.columns)}\n\n")
        
        # 情感分析洞察
        if 'sentiment_category' in self.features_df.columns:
            sentiment_dist = self.features_df['sentiment_category'].value_counts(normalize=True)
            insights.append("=== SENTIMENT ANALYSIS ===\n")
            for sentiment, ratio in sentiment_dist.items():
                insights.append(f"{sentiment.capitalize()} Reviews: {ratio:.1%}\n")
            insights.append("\n")
        
        # 游戏反馈洞察
        feedback_features = [
            ('contains_bug_report', 'Bug Reports'),
            ('contains_balance_complaint', 'Balance Complaints'),
            ('contains_monetization_complaint', 'Monetization Issues'),
            ('mentions_performance', 'Performance Mentions')
        ]
        
        insights.append("=== GAME FEEDBACK CATEGORIES ===\n")
        for feature, label in feedback_features:
            if feature in self.features_df.columns:
                ratio = self.features_df[feature].mean()
                insights.append(f"{label}: {ratio:.1%} of reviews\n")
        insights.append("\n")
        
        # 毒性分析洞察
        if 'is_toxic' in self.features_df.columns:
            toxic_ratio = self.features_df['is_toxic'].mean()
            insights.append("=== TOXICITY ANALYSIS ===\n")
            insights.append(f"Toxic Reviews: {toxic_ratio:.1%}\n")
            insights.append(f"Clean Reviews: {1-toxic_ratio:.1%}\n\n")
        
        # 作者特征洞察
        author_features = [
            ('is_active_reviewer', 'Active Reviewers'),
            ('is_hardcore_gamer', 'Hardcore Gamers'),
            ('is_casual_gamer', 'Casual Gamers')
        ]
        
        insights.append("=== AUTHOR CHARACTERISTICS ===\n")
        for feature, label in author_features:
            if feature in self.features_df.columns:
                ratio = self.features_df[feature].mean()
                insights.append(f"{label}: {ratio:.1%}\n")
        insights.append("\n")
        
        # 文本质量洞察
        if all(col in self.features_df.columns for col in ['text_length', 'readability_score']):
            avg_length = self.features_df['text_length'].mean()
            avg_readability = self.features_df['readability_score'].mean()
            insights.append("=== TEXT QUALITY METRICS ===\n")
            insights.append(f"Average Review Length: {avg_length:.0f} characters\n")
            insights.append(f"Average Readability Score: {avg_readability:.1f}\n\n")
        
        # 主题分析洞察
        if 'topic_category' in self.features_df.columns:
            topic_dist = self.features_df['topic_category'].value_counts(normalize=True)
            insights.append("=== TOPIC ANALYSIS ===\n")
            insights.append("Most Common Topics:\n")
            for topic, ratio in topic_dist.head(5).items():
                insights.append(f"  {topic}: {ratio:.1%}\n")
            insights.append("\n")
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(insights)
        
        print(f"Insights report saved to: {output_path}")
    
    def run_complete_analysis(self):
        """运行完整分析"""
        print("=== STARTING COMPLETE FEATURE ANALYSIS ===")
        
        # 基础分析
        self.analyze_feature_distributions()
        self.analyze_sentiment_patterns()
        self.analyze_game_feedback_patterns()
        self.analyze_author_characteristics()
        self.analyze_topic_distribution()
        
        # 模型构建
        review_model = self.build_review_quality_predictor()
        toxicity_model = self.build_toxicity_detector()
        
        # 生成报告
        report_path = "features/analysis_insights_report.txt"
        self.generate_insights_report(report_path)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Check the generated report for detailed insights!")


def main():
    """主函数示例"""
    # 初始化分析器
    # Prefer an explicit features file; otherwise search user's features directory for any FE artifact
    default_features_dir = r"c:\Users\12932\Desktop\nus\BAP\features"
    candidate = os.path.join(default_features_dir, "gpu_optimized_features_dataframe.csv")
    if os.path.exists(candidate):
        features_path = candidate
    else:
        # search recursively for first *_dataframe.csv under features folder
        import glob
        pattern = os.path.join(default_features_dir, "**", "*_dataframe.csv")
        found = glob.glob(pattern, recursive=True)
        if found:
            features_path = found[0]
        else:
            raise FileNotFoundError(f"No feature dataframe found under {default_features_dir}")

    analyzer = FeatureAnalyzer(features_path)
    
    # 运行完整分析
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()