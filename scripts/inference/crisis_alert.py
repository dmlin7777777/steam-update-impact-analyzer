"""
Crisis Alert System

危机告警系统：基于推理结果检测潜在的舆情危机

告警规则:
1. 负面情感激增: 负面评论 > 60%
2. 高风险集中: Critical风险 > 5%
3. 付费争议: 付费主题 + 负面情感 > 30%
4. 评论速率异常: 评论量 > 基线3倍
5. 综合告警: 趋势告警 + 高风险评论

用法:
    python crisis_alert.py --update_date 2024-03-20
    python crisis_alert.py --update_date 2024-03-20 --window_hours 72 --baseline_days 7
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
INFERENCE_DIR = BASE_DIR / 'analysis_results' / 'inference'
FEATURES_DIR = BASE_DIR / 'features'
OUTPUT_DIR = BASE_DIR / 'analysis_results' / 'alerts'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class CrisisAlertSystem:
    """
    危机告警系统
    """
    
    def __init__(self, update_date, window_hours=48, baseline_days=7, genre='fps'):
        self.update_date = pd.to_datetime(update_date)
        self.window_hours = window_hours
        self.baseline_days = baseline_days
        self.genre = genre
        
        self.end_date = self.update_date + timedelta(hours=window_hours)
        self.baseline_start = self.update_date - timedelta(days=baseline_days)
        
        # 告警阈值配置
        self.thresholds = {
            'negative_sentiment_pct': 0.60,     # 负面情感占比
            'critical_risk_pct': 0.05,          # 高危风险占比
            'monetization_negative_pct': 0.30,  # 付费+负面占比
            'comment_rate_multiplier': 3.0,     # 评论速率倍数
            'trend_alert_threshold': 10         # 趋势告警最小数量
        }
        
        self.alerts = []
    
    def load_data(self):
        """
        加载推理结果和基线数据
        """
        print("\n" + "="*80)
        print("加载数据")
        print("="*80)
        
        # 1. 加载推理结果（更新后窗口）
        pattern = f'inference_{self.genre}_*.parquet'
        files = list(INFERENCE_DIR.glob(pattern))
        
        if files:
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            df_inference = pd.read_parquet(latest_file)
            df_inference['timestamp'] = pd.to_datetime(df_inference['timestamp'])
            
            # 过滤更新后窗口
            self.df_update = df_inference[
                (df_inference['timestamp'] >= self.update_date) & 
                (df_inference['timestamp'] < self.end_date)
            ].copy()
            
            print(f"✅ 更新后数据: {len(self.df_update):,} 条")
        else:
            print(f"⚠️  未找到推理结果，使用原始特征")
            self.df_update = pd.DataFrame()
        
        # 2. 加载基线数据（更新前N天）
        feature_file = FEATURES_DIR / self.genre / f'gpu_optimized_features_{self.genre}_exclflagged_enhanced_features_with_weaklabels.parquet'
        
        if feature_file.exists():
            df_all = pd.read_parquet(feature_file)
            df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
            
            # 过滤基线窗口
            self.df_baseline = df_all[
                (df_all['timestamp'] >= self.baseline_start) & 
                (df_all['timestamp'] < self.update_date)
            ].copy()
            
            print(f"✅ 基线数据: {len(self.df_baseline):,} 条 ({self.baseline_days}天)")
        else:
            print(f"⚠️  特征文件未找到: {feature_file}")
            self.df_baseline = pd.DataFrame()
        
        return self.df_update, self.df_baseline
    
    def check_negative_sentiment_surge(self):
        """
        规则1: 负面情感激增
        """
        if 'sentiment_pred' not in self.df_update.columns:
            return None
        
        # 计算负面比例
        negative_count = (self.df_update['sentiment_pred'] == 'negative').sum()
        total_count = len(self.df_update)
        negative_pct = negative_count / total_count if total_count > 0 else 0
        
        # 基线对比
        if 'sentiment_label_weak' in self.df_baseline.columns:
            baseline_negative_pct = (self.df_baseline['sentiment_label_weak'] == 'negative').mean()
        else:
            baseline_negative_pct = 0.3  # 假设基线
        
        is_alert = negative_pct > self.thresholds['negative_sentiment_pct']
        
        alert = {
            'rule_id': 'R1',
            'rule_name': '负面情感激增',
            'is_triggered': is_alert,
            'severity': 'HIGH' if is_alert else 'LOW',
            'metrics': {
                'current_negative_pct': round(negative_pct * 100, 2),
                'baseline_negative_pct': round(baseline_negative_pct * 100, 2),
                'threshold_pct': round(self.thresholds['negative_sentiment_pct'] * 100, 2),
                'negative_count': int(negative_count),
                'total_count': int(total_count)
            },
            'description': f"负面评论占比 {negative_pct*100:.1f}% {'超过' if is_alert else '未超过'}阈值 {self.thresholds['negative_sentiment_pct']*100:.0f}%"
        }
        
        if is_alert:
            self.alerts.append(alert)
        
        print(f"{'🚨' if is_alert else '✅'} [R1] 负面情感: {negative_pct*100:.1f}% (阈值: {self.thresholds['negative_sentiment_pct']*100:.0f}%)")
        
        return alert
    
    def check_critical_risk_concentration(self):
        """
        规则2: 高风险集中
        """
        if 'risk_level' not in self.df_update.columns:
            return None
        
        # 计算Critical风险比例
        critical_count = (self.df_update['risk_level'] == 'critical').sum()
        total_count = len(self.df_update)
        critical_pct = critical_count / total_count if total_count > 0 else 0
        
        # 基线对比
        if 'risk_level' in self.df_baseline.columns:
            baseline_critical_pct = (self.df_baseline['risk_level'] == 'critical').mean()
        else:
            baseline_critical_pct = 0.02  # 假设基线
        
        is_alert = critical_pct > self.thresholds['critical_risk_pct']
        
        alert = {
            'rule_id': 'R2',
            'rule_name': '高风险集中',
            'is_triggered': is_alert,
            'severity': 'CRITICAL' if is_alert else 'LOW',
            'metrics': {
                'current_critical_pct': round(critical_pct * 100, 2),
                'baseline_critical_pct': round(baseline_critical_pct * 100, 2),
                'threshold_pct': round(self.thresholds['critical_risk_pct'] * 100, 2),
                'critical_count': int(critical_count),
                'total_count': int(total_count)
            },
            'description': f"Critical风险占比 {critical_pct*100:.2f}% {'超过' if is_alert else '未超过'}阈值 {self.thresholds['critical_risk_pct']*100:.0f}%"
        }
        
        if is_alert:
            self.alerts.append(alert)
        
        print(f"{'🚨' if is_alert else '✅'} [R2] 高风险集中: {critical_pct*100:.2f}% (阈值: {self.thresholds['critical_risk_pct']*100:.0f}%)")
        
        return alert
    
    def check_monetization_controversy(self):
        """
        规则3: 付费争议
        """
        if 'topic_pred' not in self.df_update.columns or 'sentiment_pred' not in self.df_update.columns:
            return None
        
        # 付费主题 + 负面情感
        monetization_negative = self.df_update[
            (self.df_update['topic_pred'] == 'monetization_concerns') & 
            (self.df_update['sentiment_pred'] == 'negative')
        ]
        
        monetization_count = len(monetization_negative)
        total_count = len(self.df_update)
        monetization_pct = monetization_count / total_count if total_count > 0 else 0
        
        is_alert = monetization_pct > self.thresholds['monetization_negative_pct']
        
        alert = {
            'rule_id': 'R3',
            'rule_name': '付费争议',
            'is_triggered': is_alert,
            'severity': 'MEDIUM' if is_alert else 'LOW',
            'metrics': {
                'monetization_negative_pct': round(monetization_pct * 100, 2),
                'threshold_pct': round(self.thresholds['monetization_negative_pct'] * 100, 2),
                'monetization_negative_count': int(monetization_count),
                'total_count': int(total_count)
            },
            'description': f"付费负面评论占比 {monetization_pct*100:.1f}% {'超过' if is_alert else '未超过'}阈值 {self.thresholds['monetization_negative_pct']*100:.0f}%"
        }
        
        if is_alert:
            self.alerts.append(alert)
        
        print(f"{'🚨' if is_alert else '✅'} [R3] 付费争议: {monetization_pct*100:.1f}% (阈值: {self.thresholds['monetization_negative_pct']*100:.0f}%)")
        
        return alert
    
    def check_comment_rate_anomaly(self):
        """
        规则4: 评论速率异常
        """
        # 计算更新后平均每日评论量
        update_hours = (self.end_date - self.update_date).total_seconds() / 3600
        update_daily_rate = len(self.df_update) / (update_hours / 24) if update_hours > 0 else 0
        
        # 基线平均每日评论量
        if len(self.df_baseline) > 0:
            baseline_daily_rate = len(self.df_baseline) / self.baseline_days
        else:
            baseline_daily_rate = 100  # 假设基线
        
        # 速率倍数
        rate_multiplier = update_daily_rate / baseline_daily_rate if baseline_daily_rate > 0 else 0
        
        is_alert = rate_multiplier > self.thresholds['comment_rate_multiplier']
        
        alert = {
            'rule_id': 'R4',
            'rule_name': '评论速率异常',
            'is_triggered': is_alert,
            'severity': 'MEDIUM' if is_alert else 'LOW',
            'metrics': {
                'update_daily_rate': round(update_daily_rate, 1),
                'baseline_daily_rate': round(baseline_daily_rate, 1),
                'rate_multiplier': round(rate_multiplier, 2),
                'threshold_multiplier': self.thresholds['comment_rate_multiplier']
            },
            'description': f"评论速率 {rate_multiplier:.1f}x 基线 {'超过' if is_alert else '未超过'}阈值 {self.thresholds['comment_rate_multiplier']:.0f}x"
        }
        
        if is_alert:
            self.alerts.append(alert)
        
        print(f"{'🚨' if is_alert else '✅'} [R4] 评论速率: {rate_multiplier:.1f}x (阈值: {self.thresholds['comment_rate_multiplier']:.0f}x)")
        
        return alert
    
    def check_trend_risk_combo(self):
        """
        规则5: 综合告警（趋势 + 风险）
        """
        if 'trend_alert_pred' not in self.df_update.columns or 'risk_level' not in self.df_update.columns:
            return None
        
        # 趋势告警 + 高风险
        combo_alerts = self.df_update[
            (self.df_update['trend_alert_pred'] == 1) & 
            (self.df_update['risk_level'].isin(['high', 'critical']))
        ]
        
        combo_count = len(combo_alerts)
        is_alert = combo_count >= self.thresholds['trend_alert_threshold']
        
        alert = {
            'rule_id': 'R5',
            'rule_name': '综合告警（趋势+风险）',
            'is_triggered': is_alert,
            'severity': 'CRITICAL' if is_alert else 'LOW',
            'metrics': {
                'combo_alert_count': int(combo_count),
                'threshold_count': int(self.thresholds['trend_alert_threshold']),
                'total_trend_alerts': int((self.df_update['trend_alert_pred'] == 1).sum()),
                'total_high_risk': int(self.df_update['risk_level'].isin(['high', 'critical']).sum())
            },
            'description': f"趋势+风险组合告警 {combo_count} 条 {'超过' if is_alert else '未超过'}阈值 {self.thresholds['trend_alert_threshold']} 条"
        }
        
        if is_alert:
            self.alerts.append(alert)
        
        print(f"{'🚨' if is_alert else '✅'} [R5] 综合告警: {combo_count} 条 (阈值: {self.thresholds['trend_alert_threshold']} 条)")
        
        return alert
    
    def run_all_checks(self):
        """
        运行所有告警规则
        """
        print("\n" + "="*80)
        print("运行告警规则检测")
        print("="*80)
        
        self.alerts = []  # 重置告警列表
        
        results = {
            'R1': self.check_negative_sentiment_surge(),
            'R2': self.check_critical_risk_concentration(),
            'R3': self.check_monetization_controversy(),
            'R4': self.check_comment_rate_anomaly(),
            'R5': self.check_trend_risk_combo()
        }
        
        return results
    
    def generate_alert_report(self):
        """
        生成告警报告
        """
        print("\n" + "="*80)
        print("生成告警报告")
        print("="*80)
        
        # 告警统计
        n_critical = sum(1 for a in self.alerts if a['severity'] == 'CRITICAL')
        n_high = sum(1 for a in self.alerts if a['severity'] == 'HIGH')
        n_medium = sum(1 for a in self.alerts if a['severity'] == 'MEDIUM')
        
        # 总体风险等级
        if n_critical > 0:
            overall_risk = 'CRITICAL'
        elif n_high > 0:
            overall_risk = 'HIGH'
        elif n_medium > 0:
            overall_risk = 'MEDIUM'
        else:
            overall_risk = 'LOW'
        
        report = {
            'metadata': {
                'update_date': self.update_date.strftime('%Y-%m-%d'),
                'window_hours': self.window_hours,
                'baseline_days': self.baseline_days,
                'genre': self.genre,
                'report_time': datetime.now().isoformat()
            },
            'summary': {
                'overall_risk_level': overall_risk,
                'total_alerts': len(self.alerts),
                'critical_alerts': n_critical,
                'high_alerts': n_high,
                'medium_alerts': n_medium
            },
            'data_stats': {
                'update_window_reviews': len(self.df_update),
                'baseline_reviews': len(self.df_baseline)
            },
            'thresholds': self.thresholds,
            'alerts': self.alerts
        }
        
        # 保存JSON
        date_str = self.update_date.strftime('%Y%m%d')
        json_file = OUTPUT_DIR / f'crisis_alert_{self.genre}_{date_str}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 告警报告: {json_file}")
        
        # 生成Markdown摘要
        md_content = self._generate_markdown_summary(report)
        md_file = OUTPUT_DIR / f'crisis_alert_{self.genre}_{date_str}.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"✅ 摘要报告: {md_file}")
        
        return report
    
    def _generate_markdown_summary(self, report):
        """
        生成Markdown格式摘要
        """
        md = f"""# 🚨 危机告警报告

**更新日期**: {report['metadata']['update_date']}  
**监控窗口**: {report['metadata']['window_hours']} 小时  
**基线周期**: {report['metadata']['baseline_days']} 天  
**游戏类型**: {report['metadata']['genre']}  

---

## 📊 总体评估

**风险等级**: {'🔴 ' if report['summary']['overall_risk_level'] == 'CRITICAL' else '🟠 ' if report['summary']['overall_risk_level'] == 'HIGH' else '🟡 ' if report['summary']['overall_risk_level'] == 'MEDIUM' else '🟢 '}**{report['summary']['overall_risk_level']}**

**告警统计**:
- 总告警数: {report['summary']['total_alerts']}
- Critical: {report['summary']['critical_alerts']}
- High: {report['summary']['high_alerts']}
- Medium: {report['summary']['medium_alerts']}

**数据统计**:
- 更新后评论: {report['data_stats']['update_window_reviews']:,} 条
- 基线评论: {report['data_stats']['baseline_reviews']:,} 条

---

## 🔔 触发的告警

"""
        
        if report['alerts']:
            for i, alert in enumerate(report['alerts'], 1):
                severity_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
                md += f"""
### {i}. {severity_emoji.get(alert['severity'], '⚪')} [{alert['rule_id']}] {alert['rule_name']}

**级别**: {alert['severity']}  
**描述**: {alert['description']}

**指标**:
"""
                for key, value in alert['metrics'].items():
                    md += f"- {key}: {value}\n"
                
                md += "\n"
        else:
            md += "✅ 未触发任何告警，系统运行正常。\n\n"
        
        md += f"""
---

## 📋 告警阈值配置

| 规则 | 阈值 |
|------|------|
| 负面情感占比 | {report['thresholds']['negative_sentiment_pct']*100:.0f}% |
| Critical风险占比 | {report['thresholds']['critical_risk_pct']*100:.0f}% |
| 付费负面占比 | {report['thresholds']['monetization_negative_pct']*100:.0f}% |
| 评论速率倍数 | {report['thresholds']['comment_rate_multiplier']:.0f}x |
| 趋势告警阈值 | {report['thresholds']['trend_alert_threshold']} 条 |

---

*报告生成时间: {report['metadata']['report_time']}*
"""
        
        return md


def main():
    parser = argparse.ArgumentParser(description='危机告警系统')
    parser.add_argument('--update_date', type=str, required=True, help='更新日期 (YYYY-MM-DD)')
    parser.add_argument('--window_hours', type=int, default=48, help='监控窗口（小时）')
    parser.add_argument('--baseline_days', type=int, default=7, help='基线周期（天）')
    parser.add_argument('--genre', type=str, default='fps', help='游戏类型')
    
    args = parser.parse_args()
    
    print("="*80)
    print("危机告警系统")
    print("="*80)
    print(f"更新日期: {args.update_date}")
    print(f"监控窗口: {args.window_hours} 小时")
    print(f"基线周期: {args.baseline_days} 天")
    
    # 初始化系统
    alert_system = CrisisAlertSystem(
        update_date=args.update_date,
        window_hours=args.window_hours,
        baseline_days=args.baseline_days,
        genre=args.genre
    )
    
    # 加载数据
    alert_system.load_data()
    
    # 运行告警检测
    results = alert_system.run_all_checks()
    
    # 生成报告
    report = alert_system.generate_alert_report()
    
    # 打印总结
    print("\n" + "="*80)
    print("告警总结")
    print("="*80)
    print(f"风险等级: {report['summary']['overall_risk_level']}")
    print(f"触发告警: {report['summary']['total_alerts']} 条")
    
    if report['summary']['total_alerts'] > 0:
        print("\n⚠️  建议立即采取行动！")
    else:
        print("\n✅ 系统运行正常")


if __name__ == "__main__":
    main()
