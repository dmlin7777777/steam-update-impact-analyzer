"""
Update Monitoring Dashboard

更新监控仪表盘：针对游戏更新后48小时的评论数据生成交互式监控面板

功能:
- 情感趋势图（每小时）
- 主题分布饼图
- 风险时间线
- 高风险评论 Top 20

用法:
    python update_monitoring_dashboard.py --update_date 2024-03-20
    python update_monitoring_dashboard.py --update_date 2024-03-20 --window_hours 72
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
INFERENCE_DIR = BASE_DIR / 'analysis_results' / 'inference'
OUTPUT_DIR = BASE_DIR / 'analysis_results' / 'dashboards'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class UpdateMonitoringDashboard:
    """
    更新监控仪表盘生成器
    """
    
    def __init__(self, update_date, window_hours=48, genre='fps'):
        self.update_date = pd.to_datetime(update_date)
        self.window_hours = window_hours
        self.genre = genre
        self.end_date = self.update_date + timedelta(hours=window_hours)
        
    def load_inference_results(self, inference_file=None):
        """
        加载推理结果
        """
        print("\n" + "="*80)
        print("加载推理结果")
        print("="*80)
        
        if inference_file:
            df = pd.read_parquet(inference_file)
            print(f"✅ 加载自定义文件: {inference_file}")
        else:
            # 查找最新的推理结果
            pattern = f'inference_{self.genre}_*.parquet'
            files = list(INFERENCE_DIR.glob(pattern))
            
            if not files:
                raise FileNotFoundError(f"未找到推理结果: {pattern}")
            
            # 选择最新文件
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            df = pd.read_parquet(latest_file)
            print(f"✅ 加载最新结果: {latest_file.name}")
        
        # 时间过滤
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[(df['timestamp'] >= self.update_date) & (df['timestamp'] < self.end_date)]
        
        print(f"\n📊 数据统计:")
        print(f"   时间范围: {self.update_date} → {self.end_date}")
        print(f"   样本数: {len(df):,}")
        
        return df
    
    def plot_sentiment_trend(self, df):
        """
        情感趋势图（每小时）
        """
        print("\n生成情感趋势图...")
        
        # 按小时聚合
        df['hour'] = df['timestamp'].dt.floor('H')
        
        if 'sentiment_pred' in df.columns:
            hourly = df.groupby(['hour', 'sentiment_pred']).size().unstack(fill_value=0)
            
            # 转为百分比
            hourly_pct = hourly.div(hourly.sum(axis=1), axis=0) * 100
            
            fig = go.Figure()
            
            colors = {
                'positive': '#2ecc71',
                'neutral': '#95a5a6',
                'negative': '#e74c3c',
                'unknown': '#bdc3c7'
            }
            
            for sentiment in hourly_pct.columns:
                fig.add_trace(go.Scatter(
                    x=hourly_pct.index,
                    y=hourly_pct[sentiment],
                    name=sentiment.capitalize(),
                    mode='lines+markers',
                    line=dict(color=colors.get(sentiment, '#3498db'), width=2),
                    marker=dict(size=6),
                    stackgroup='one',
                    groupnorm='percent'
                ))
            
            fig.update_layout(
                title=f'Sentiment Trend - {self.update_date.strftime("%Y-%m-%d")} After update {self.window_hours}h',
                xaxis_title='Time',
                yaxis_title='Percent (%)',
                hovermode='x unified',
                height=500,
                template='plotly_white'
            )
            
            return fig
        else:
            print("⚠️  sentiment_pred列不存在，跳过")
            return None
    
    def plot_topic_distribution(self, df):
        """
        主题分布饼图
        """
        print("生成主题分布图...")
        
        if 'topic_pred' in df.columns:
            topic_counts = df['topic_pred'].value_counts()
            
            # 主题中文映射
            topic_labels = {
                'matchmaking_issues': '匹配问题',
                'cheating': '作弊举报',
                'monetization_concerns': '付费抱怨',
                'user_interface': '界面/UI',
                'technical_issues': '技术问题',
                'multiplayer_features': '多人模式',
                'unknown': '未知'
            }
            
            labels = [topic_labels.get(t, t) for t in topic_counts.index]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=topic_counts.values,
                hole=0.3,
                textinfo='label+percent',
                textposition='outside',
                marker=dict(
                    colors=px.colors.qualitative.Set3,
                    line=dict(color='white', width=2)
                )
            )])
            
            fig.update_layout(
                title=f'Topic Distribution - 总计 {len(df):,} 条评论',
                height=500,
                template='plotly_white',
                showlegend=True
            )
            
            return fig
        else:
            print("⚠️  topic_pred列不存在，跳过")
            return None
    
    def plot_risk_timeline(self, df):
        """
        风险时间线
        """
        print("生成风险时间线...")
        
        if 'risk_score' in df.columns:
            # 按小时聚合风险
            df['hour'] = df['timestamp'].dt.floor('H')
            
            hourly_risk = df.groupby('hour').agg({
                'risk_score': ['mean', 'max', 'count']
            }).reset_index()
            hourly_risk.columns = ['hour', 'avg_risk', 'max_risk', 'count']
            
            # 计算高风险比例
            if 'risk_level' in df.columns:
                # 按小时统计高风险数量
                high_risk_by_hour = df[df['risk_level'].isin(['high', 'critical'])].groupby('hour').size().to_dict()
                # 计算每小时的高风险比例
                hourly_risk['high_risk_pct'] = hourly_risk['hour'].map(
                    lambda h: (high_risk_by_hour.get(h, 0) / hourly_risk[hourly_risk['hour'] == h]['count'].values[0] * 100) 
                    if h in high_risk_by_hour else 0
                )
            else:
                hourly_risk['high_risk_pct'] = 0.0
            
            # 创建双Y轴图
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 平均风险分数
            fig.add_trace(
                go.Scatter(
                    x=hourly_risk['hour'],
                    y=hourly_risk['avg_risk'],
                    name='Avg Risk Score',
                    mode='lines+markers',
                    line=dict(color='#3498db', width=2),
                    marker=dict(size=6)
                ),
                secondary_y=False
            )
            
            # 最大风险分数
            fig.add_trace(
                go.Scatter(
                    x=hourly_risk['hour'],
                    y=hourly_risk['max_risk'],
                    name='Max Risk Score',
                    mode='lines',
                    line=dict(color='#e74c3c', width=2, dash='dash')
                ),
                secondary_y=False
            )
            
            # 高风险比例
            fig.add_trace(
                go.Bar(
                    x=hourly_risk['hour'],
                    y=hourly_risk['high_risk_pct'],
                    name='High Risk Percentage (%)',
                    marker=dict(color='#f39c12', opacity=0.5)
                ),
                secondary_y=True
            )
            
            # 更新布局
            fig.update_xaxes(title_text="Time")
            fig.update_yaxes(title_text="Risk Score", secondary_y=False)
            fig.update_yaxes(title_text="High Risk Percentage (%)", secondary_y=True)
            
            fig.update_layout(
                title=f'Risk Timeline',
                hovermode='x unified',
                height=500,
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            return fig
        else:
            print("⚠️  risk_score列不存在，跳过")
            return None
    
    def get_top_high_risk_reviews(self, df, top_n=20):
        """
        获取高风险评论 Top N
        """
        print(f"提取Top {top_n} 高风险评论...")
        
        if 'risk_score' not in df.columns:
            print("⚠️  risk_score列不存在，跳过")
            return None
        
        # 按风险分数排序
        top_risk = df.nlargest(top_n, 'risk_score').copy()
        
        # 选择关键列
        columns_to_show = [
            'timestamp', 'risk_score', 'risk_level',
            'sentiment_pred', 'topic_pred', 
            'review_content_processed'
        ]
        
        available_cols = [col for col in columns_to_show if col in top_risk.columns]
        top_risk = top_risk[available_cols]
        
        # 格式化时间
        top_risk['timestamp'] = top_risk['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        
        return top_risk
    
    def generate_dashboard(self, df):
        """
        生成完整仪表盘
        """
        print("\n" + "="*80)
        print("生成监控仪表盘")
        print("="*80)
        
        # 创建子图
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Sentiment Trend',
                'Topic Distribution',
                'Risk Timeline',
                'Key Statistics'
            ),
            specs=[
                [{"type": "scatter", "colspan": 2}, None],
                [{"type": "pie"}, {"type": "scatter"}],
                [{"type": "table", "colspan": 2}, None]
            ],
            row_heights=[0.3, 0.35, 0.35],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        # 1. 情感趋势（简化版）
        if 'sentiment_pred' in df.columns:
            df['hour'] = df['timestamp'].dt.floor('H')
            hourly = df.groupby(['hour', 'sentiment_pred']).size().unstack(fill_value=0)
            
            for sentiment in hourly.columns:
                fig.add_trace(
                    go.Scatter(
                        x=hourly.index,
                        y=hourly[sentiment],
                        name=sentiment,
                        mode='lines+markers',
                        stackgroup='one'
                    ),
                    row=1, col=1
                )
        
        # 2. 主题分布
        if 'topic_pred' in df.columns:
            topic_counts = df['topic_pred'].value_counts()
            fig.add_trace(
                go.Pie(labels=topic_counts.index, values=topic_counts.values, hole=0.3),
                row=2, col=1
            )
        
        # 3. 风险时间线
        if 'risk_score' in df.columns:
            df['hour'] = df['timestamp'].dt.floor('H')
            hourly_risk = df.groupby('hour')['risk_score'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=hourly_risk.index,
                    y=hourly_risk.values,
                    name='Avg Risk Score',
                    mode='lines+markers',
                    line=dict(color='#e74c3c', width=2)
                ),
                row=2, col=2
            )
        
        # 4. 关键统计表格
        stats_data = {
            'Index': [
                'Sample Count',
                'Negative Percent',
                'High Risk Percent',
                'Trend Alert Count',
                'Average Risk Score'
            ],
            'Value': [
                f"{len(df):,}",
                f"{(df['sentiment_pred'] == 'negative').mean()*100:.1f}%" if 'sentiment_pred' in df else 'N/A',
                f"{df['risk_level'].isin(['high', 'critical']).mean()*100:.1f}%" if 'risk_level' in df else 'N/A',
                f"{(df['trend_alert_pred'] == 1).sum():,}" if 'trend_alert_pred' in df else 'N/A',
                f"{df['risk_score'].mean():.2f}" if 'risk_score' in df else 'N/A'
            ]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(stats_data.keys()), fill_color='#3498db', font=dict(color='white', size=14)),
                cells=dict(values=list(stats_data.values()), fill_color='#ecf0f1', font=dict(size=12))
            ),
            row=3, col=1
        )
        
        # 更新布局
        fig.update_layout(
            title_text=f"Game Update Dashboard - {self.update_date.strftime('%Y-%m-%d')} ({self.window_hours}h)",
            showlegend=True,
            height=1200,
            template='plotly_white'
        )
        
        return fig
    
    def save_dashboard(self, output_prefix='dashboard'):
        """
        生成并保存完整仪表盘
        """
        # 加载数据
        df = self.load_inference_results()
        
        # 生成各个图表
        sentiment_fig = self.plot_sentiment_trend(df)
        topic_fig = self.plot_topic_distribution(df)
        risk_fig = self.plot_risk_timeline(df)
        top_risk = self.get_top_high_risk_reviews(df)
        
        # 生成完整仪表盘
        dashboard_fig = self.generate_dashboard(df)
        
        # 保存
        date_str = self.update_date.strftime('%Y%m%d')
        
        # 1. HTML仪表盘
        html_file = OUTPUT_DIR / f'{output_prefix}_{self.genre}_{date_str}_{self.window_hours}h.html'
        dashboard_fig.write_html(html_file)
        print(f"\n✅ 仪表盘: {html_file}")
        
        # 2. 高风险评论CSV
        if top_risk is not None:
            csv_file = OUTPUT_DIR / f'{output_prefix}_{self.genre}_{date_str}_top_risk.csv'
            top_risk.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"✅ 高风险列表: {csv_file}")
        
        # 3. 统计摘要JSON
        summary = {
            'update_date': self.update_date.strftime('%Y-%m-%d'),
            'window_hours': self.window_hours,
            'genre': self.genre,
            'total_reviews': len(df),
            'sentiment': df['sentiment_pred'].value_counts().to_dict() if 'sentiment_pred' in df else {},
            'topic': df['topic_pred'].value_counts().to_dict() if 'topic_pred' in df else {},
            'risk_stats': {
                'mean': float(df['risk_score'].mean()) if 'risk_score' in df else 0,
                'max': float(df['risk_score'].max()) if 'risk_score' in df else 0,
                'high_risk_count': int(df['risk_level'].isin(['high', 'critical']).sum()) if 'risk_level' in df else 0,
                'high_risk_pct': float(df['risk_level'].isin(['high', 'critical']).mean()*100) if 'risk_level' in df else 0
            },
            'trend_alert_count': int((df['trend_alert_pred'] == 1).sum()) if 'trend_alert_pred' in df else 0
        }
        
        json_file = OUTPUT_DIR / f'{output_prefix}_{self.genre}_{date_str}_summary.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ 统计摘要: {json_file}")
        
        return html_file


def main():
    parser = argparse.ArgumentParser(description='更新监控仪表盘')
    parser.add_argument('--update_date', type=str, required=True, help='更新日期 (YYYY-MM-DD)')
    parser.add_argument('--window_hours', type=int, default=48, help='监控窗口（小时）')
    parser.add_argument('--genre', type=str, default='fps', help='游戏类型')
    parser.add_argument('--inference_file', type=str, help='推理结果文件（可选）')
    parser.add_argument('--output_prefix', type=str, default='dashboard', help='输出文件前缀')
    
    args = parser.parse_args()
    
    print("="*80)
    print("更新监控仪表盘生成器")
    print("="*80)
    print(f"更新日期: {args.update_date}")
    print(f"监控窗口: {args.window_hours} 小时")
    print(f"游戏类型: {args.genre}")
    
    # 生成仪表盘
    dashboard = UpdateMonitoringDashboard(
        update_date=args.update_date,
        window_hours=args.window_hours,
        genre=args.genre
    )
    
    output_file = dashboard.save_dashboard(output_prefix=args.output_prefix)
    
    print("\n" + "="*80)
    print("✅ 仪表盘生成完成")
    print("="*80)
    print(f"输出文件: {output_file}")
    print(f"\n在浏览器中打开查看: file:///{output_file}")


if __name__ == "__main__":
    main()
