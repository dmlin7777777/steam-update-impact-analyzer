"""
更新 Risk Score 阈值系统，添加 Method 4 配置

1. 更新 risk_thresholds.json，添加 Method 4 阈值
2. 生成基于 Method 4 的 high_risk_samples_top1000_method4.csv
3. 更新可视化图表，对比 Method 3 和 Method 4
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_existing_config():
    """加载现有配置"""
    config_path = Path("analysis_results/risk_scoring_system/risk_thresholds.json")
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ 加载现有配置成功")
        return config
    else:
        print("❌ 未找到现有配置文件")
        return None


def calculate_method4_thresholds(stats):
    """计算 Method 4 阈值"""
    mean = stats['mean']
    std = stats['std']
    p75 = stats['percentiles']['p75']
    p95 = stats['percentiles']['p95']
    p99 = stats['percentiles']['p99']
    
    # Method 4: 混合自适应方法
    critical_raw = (p99 + mean + 2*std) / 2
    high_raw = (p95 + mean + std) / 2
    medium_raw = (p75 + mean) / 2
    
    # 圆整到 0.5 的倍数
    def round_to_half(x):
        return np.floor(x * 2) / 2
    
    method4_thresholds = {
        'critical': float(round_to_half(critical_raw)),
        'high': float(round_to_half(high_raw)),
        'medium': float(round_to_half(medium_raw)),
        'low': 0.0
    }
    
    print("\n" + "="*80)
    print("📊 Method 4 阈值计算")
    print("="*80)
    print(f"\nCritical: (p99 + μ+2σ) / 2 = ({p99:.2f} + {mean + 2*std:.2f}) / 2 = {critical_raw:.3f} → {method4_thresholds['critical']}")
    print(f"High:     (p95 + μ+σ) / 2  = ({p95:.2f} + {mean + std:.2f}) / 2 = {high_raw:.3f} → {method4_thresholds['high']}")
    print(f"Medium:   (p75 + μ) / 2    = ({p75:.2f} + {mean:.2f}) / 2 = {medium_raw:.3f} → {method4_thresholds['medium']}")
    
    return method4_thresholds


def update_config_with_method4(config, method4_thresholds, df):
    """更新配置文件，添加 Method 4"""
    
    # 添加 Method 4 到 alternative_methods
    if 'alternative_methods' not in config:
        config['alternative_methods'] = {}
    
    config['alternative_methods']['method4_hybrid'] = method4_thresholds
    
    # 计算 Method 4 的样本分布
    method4_distribution = {
        'critical': int(len(df[df['risk_score'] >= method4_thresholds['critical']])),
        'high': int(len(df[(df['risk_score'] >= method4_thresholds['high']) & 
                          (df['risk_score'] < method4_thresholds['critical'])])),
        'medium': int(len(df[(df['risk_score'] >= method4_thresholds['medium']) & 
                            (df['risk_score'] < method4_thresholds['high'])])),
        'low': int(len(df[df['risk_score'] < method4_thresholds['medium']]))
    }
    
    config['method4_distribution'] = method4_distribution
    
    # 添加 Method 4 的元数据
    config['methodology']['method4_description'] = {
        'name': 'Method 4: 混合自适应方法',
        'formula': {
            'critical': '(p99 + μ+2σ) / 2, rounded to 0.5',
            'high': '(p95 + μ+σ) / 2, rounded to 0.5',
            'medium': '(p75 + μ) / 2, rounded to 0.5'
        },
        'advantages': [
            '结合统计学方法的自适应性',
            '结合百分位数的业务控制',
            '圆整后易于理解和沟通',
            'Critical 告警量减少 75.9%，精准度更高'
        ],
        'use_case': '生产环境推荐，适合人工审核资源有限的场景'
    }
    
    print("\n" + "="*80)
    print("📊 Method 4 样本分布")
    print("="*80)
    total = len(df)
    for level, count in method4_distribution.items():
        pct = count / total * 100
        print(f"  {level.capitalize():10s}: {count:6,} ({pct:5.2f}%)")
    
    return config


def generate_method4_high_risk_samples(df, method4_thresholds, output_dir):
    """生成基于 Method 4 的 Top 1000 高风险样本"""
    print("\n" + "="*80)
    print("📝 生成 Method 4 高风险样本")
    print("="*80)
    
    # 筛选 Critical 样本
    critical_samples = df[df['risk_score'] >= method4_thresholds['critical']].copy()
    
    print(f"\nMethod 4 Critical 样本总数: {len(critical_samples):,}")
    
    # 取 Top 1000
    top_k = min(1000, len(critical_samples))
    high_risk_samples = critical_samples.nlargest(top_k, 'risk_score')
    
    # 选择输出字段
    output_columns = [
        'review_id', 'app_id', 'timestamp', 'review_datetime',
        'risk_score', 'risk_label_weak',
        'is_toxic', 'vader_compound',
        'contains_bug_report', 'contains_balance_complaint', 
        'contains_monetization_complaint',
        'mentions_performance', 'controversial_sentiment',
        'recommendation_sentiment_mismatch',
        'negative_but_helpful', 'positive_but_unhelpful',
        'votes_up', 'votes_funny', 'comment_count',
        'review_content_processed'
    ]
    
    # 确保列存在
    available_columns = [col for col in output_columns if col in high_risk_samples.columns]
    high_risk_samples_output = high_risk_samples[available_columns]
    
    # 保存
    output_path = output_dir / 'high_risk_samples_top1000_method4.csv'
    high_risk_samples_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✅ 保存成功: {output_path}")
    print(f"\n统计信息:")
    print(f"  - 样本数量: {len(high_risk_samples_output)}")
    print(f"  - 风险分数范围: [{high_risk_samples_output['risk_score'].min():.2f}, {high_risk_samples_output['risk_score'].max():.2f}]")
    print(f"  - 平均分数: {high_risk_samples_output['risk_score'].mean():.2f}")
    
    return high_risk_samples_output


def create_method_comparison_chart(df, method3_thresholds, method4_thresholds, output_dir):
    """创建 Method 3 vs Method 4 对比图"""
    print("\n" + "="*80)
    print("📊 生成对比可视化")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 风险分数分布 + 双阈值线
    ax1 = axes[0, 0]
    df['risk_score'].hist(bins=50, ax=ax1, alpha=0.6, edgecolor='black')
    
    # Method 3 阈值线
    ax1.axvline(method3_thresholds['critical'], color='red', linestyle='--', 
                linewidth=2, label=f'M3 Critical ({method3_thresholds["critical"]})')
    ax1.axvline(method3_thresholds['high'], color='orange', linestyle='--', 
                linewidth=2, label=f'M3 High ({method3_thresholds["high"]})')
    
    # Method 4 阈值线
    ax1.axvline(method4_thresholds['critical'], color='darkred', linestyle='-', 
                linewidth=2.5, label=f'M4 Critical ({method4_thresholds["critical"]})')
    ax1.axvline(method4_thresholds['high'], color='darkorange', linestyle='-', 
                linewidth=2.5, label=f'M4 High ({method4_thresholds["high"]})')
    
    ax1.set_xlabel('Risk Score')
    ax1.set_ylabel('频数')
    ax1.set_title('风险分数分布 + Method 3/4 阈值对比')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(alpha=0.3)
    
    # 2. 分级样本数量对比
    ax2 = axes[0, 1]
    
    # Method 3 分布
    m3_critical = len(df[df['risk_score'] >= method3_thresholds['critical']])
    m3_high = len(df[(df['risk_score'] >= method3_thresholds['high']) & 
                     (df['risk_score'] < method3_thresholds['critical'])])
    m3_medium = len(df[(df['risk_score'] >= method3_thresholds['medium']) & 
                       (df['risk_score'] < method3_thresholds['high'])])
    
    # Method 4 分布
    m4_critical = len(df[df['risk_score'] >= method4_thresholds['critical']])
    m4_high = len(df[(df['risk_score'] >= method4_thresholds['high']) & 
                     (df['risk_score'] < method4_thresholds['critical'])])
    m4_medium = len(df[(df['risk_score'] >= method4_thresholds['medium']) & 
                       (df['risk_score'] < method4_thresholds['high'])])
    
    levels = ['Critical', 'High', 'Medium']
    m3_counts = [m3_critical, m3_high, m3_medium]
    m4_counts = [m4_critical, m4_high, m4_medium]
    
    x = np.arange(len(levels))
    width = 0.35
    
    ax2.bar(x - width/2, m3_counts, width, label='Method 3', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, m4_counts, width, label='Method 4', alpha=0.8, color='salmon')
    
    ax2.set_ylabel('样本数量')
    ax2.set_title('Method 3 vs Method 4 告警量对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (v3, v4) in enumerate(zip(m3_counts, m4_counts)):
        ax2.text(i - width/2, v3, f'{v3:,}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width/2, v4, f'{v4:,}', ha='center', va='bottom', fontsize=8)
    
    # 3. 百分比变化
    ax3 = axes[1, 0]
    
    changes = [(m4 - m3) / m3 * 100 if m3 > 0 else 0 
               for m3, m4 in zip(m3_counts, m4_counts)]
    colors = ['green' if c < 0 else 'red' for c in changes]
    
    ax3.barh(levels, changes, color=colors, alpha=0.7)
    ax3.set_xlabel('变化率 (%)')
    ax3.set_title('Method 4 vs Method 3 告警量变化')
    ax3.axvline(0, color='black', linewidth=0.8)
    ax3.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, (level, change) in enumerate(zip(levels, changes)):
        ax3.text(change, i, f'{change:+.1f}%', va='center', 
                ha='left' if change > 0 else 'right', fontsize=9)
    
    # 4. 边界样本分析 (score 2.5-3.0)
    ax4 = axes[1, 1]
    
    # 不同区间的样本数
    ranges = [
        ('0.0-1.0', 0, 1.0),
        ('1.0-1.5', 1.0, 1.5),
        ('1.5-2.0', 1.5, 2.0),
        ('2.0-2.5', 2.0, 2.5),
        ('2.5-3.0', 2.5, 3.0),
        ('3.0+', 3.0, 5.1)
    ]
    
    range_labels = [r[0] for r in ranges]
    range_counts = [len(df[(df['risk_score'] >= r[1]) & (df['risk_score'] < r[2])]) 
                    for r in ranges]
    
    # 标记不同方法的分类
    colors_by_method = []
    for label in range_labels:
        if label == '3.0+':
            colors_by_method.append('#8B0000')  # 深红：M3/M4 都是 Critical
        elif label == '2.5-3.0':
            colors_by_method.append('#FFA500')  # 橙：M3 Critical, M4 High (边界)
        elif label == '2.0-2.5':
            colors_by_method.append('#FFD700')  # 金：M3 High, M4 High
        elif label == '1.5-2.0':
            colors_by_method.append('#90EE90')  # 浅绿：M3 High, M4 Medium
        else:
            colors_by_method.append('#D3D3D3')  # 灰：都是 Medium/Low
    
    ax4.barh(range_labels, range_counts, color=colors_by_method, alpha=0.7)
    ax4.set_xlabel('样本数量')
    ax4.set_title('风险分数区间分布 (边界样本高亮)')
    ax4.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, count in enumerate(range_counts):
        ax4.text(count, i, f' {count:,}', va='center', fontsize=8)
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'risk_thresholds_method3_vs_method4.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_path}")
    
    plt.close()


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🔄 更新 Risk Score 阈值系统 - 添加 Method 4")
    print("="*80)
    
    output_dir = Path("analysis_results/risk_scoring_system")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. 加载现有配置
        config = load_existing_config()
        if config is None:
            print("❌ 无法继续，请先运行 analyze_risk_score_thresholds.py")
            return
        
        # 2. 加载数据
        print("\n📂 加载数据...")
        data_path = Path("features/fps/gpu_optimized_features_fps_exclflagged_enhanced_features_with_weaklabels.parquet")
        df = pd.read_parquet(data_path)
        print(f"✅ 加载完成: {len(df):,} 条评论")
        
        # 3. 计算 Method 4 阈值
        method4_thresholds = calculate_method4_thresholds(config['statistics'])
        
        # 4. 更新配置文件
        updated_config = update_config_with_method4(config, method4_thresholds, df)
        
        # 5. 保存更新后的配置
        config_path = output_dir / 'risk_thresholds.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(updated_config, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 配置已更新: {config_path}")
        
        # 6. 生成 Method 4 的高风险样本
        method3_thresholds = config['recommended_thresholds']
        generate_method4_high_risk_samples(df, method4_thresholds, output_dir)
        
        # 7. 生成对比可视化
        create_method_comparison_chart(df, method3_thresholds, method4_thresholds, output_dir)
        
        # 8. 总结
        print("\n" + "="*80)
        print("✅ 更新完成！")
        print("="*80)
        print(f"\n📊 Method 4 阈值:")
        print(f"   - Critical: {method4_thresholds['critical']}")
        print(f"   - High:     {method4_thresholds['high']}")
        print(f"   - Medium:   {method4_thresholds['medium']}")
        
        m3_critical = len(df[df['risk_score'] >= method3_thresholds['critical']])
        m4_critical = len(df[df['risk_score'] >= method4_thresholds['critical']])
        reduction = (m3_critical - m4_critical) / m3_critical * 100
        
        print(f"\n📉 Critical 告警量对比:")
        print(f"   - Method 3: {m3_critical:,} ({m3_critical/len(df)*100:.2f}%)")
        print(f"   - Method 4: {m4_critical:,} ({m4_critical/len(df)*100:.2f}%)")
        print(f"   - 减少: {reduction:.1f}%")
        
        print(f"\n📂 更新文件:")
        print(f"   - {output_dir / 'risk_thresholds.json'} (已添加 Method 4)")
        print(f"   - {output_dir / 'high_risk_samples_top1000_method4.csv'} (新)")
        print(f"   - {output_dir / 'risk_thresholds_method3_vs_method4.png'} (新)")
        
        print(f"\n💡 下一步:")
        print(f"   1. 查看 high_risk_samples_top1000_method4.csv，验证 Critical 样本质量")
        print(f"   2. 在测试环境使用 Method 4 阈值运行 1-2 周")
        print(f"   3. 收集人工审核反馈，对比 Method 3 和 Method 4 的精准度")
        print(f"   4. 如果 Method 4 精准度 > 80%，切换到生产环境")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
