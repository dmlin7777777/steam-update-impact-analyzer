"""
推理系统快速测试

测试3个核心脚本是否能正常运行
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试所有导入"""
    print("\n" + "="*80)
    print("测试导入")
    print("="*80)
    
    try:
        import batch_inference
        print("✅ batch_inference.py")
    except Exception as e:
        print(f"❌ batch_inference.py: {e}")
        return False
    
    try:
        import update_monitoring_dashboard
        print("✅ update_monitoring_dashboard.py")
    except Exception as e:
        print(f"❌ update_monitoring_dashboard.py: {e}")
        return False
    
    try:
        import crisis_alert
        print("✅ crisis_alert.py")
    except Exception as e:
        print(f"❌ crisis_alert.py: {e}")
        return False
    
    return True


def create_test_data():
    """创建测试数据"""
    print("\n" + "="*80)
    print("创建测试数据")
    print("="*80)
    
    # 生成模拟数据
    n_samples = 1000
    start_date = datetime(2024, 3, 20)
    
    data = {
        'timestamp': [start_date + timedelta(hours=i/100) for i in range(n_samples)],
        'review_content_processed': [f'test review {i}' for i in range(n_samples)],
        'vader_compound': np.random.randn(n_samples) * 0.3,
        'sentiment_label_weak': np.random.choice(['positive', 'neutral', 'negative'], n_samples),
        'topic_label_weak': np.random.choice(['technical_issues', 'matchmaking_issues', 'cheating'], n_samples),
        'risk_score': np.random.gamma(2, 0.5, n_samples),
        'risk_level': np.random.choice(['low', 'medium', 'high', 'critical'], n_samples, p=[0.7, 0.2, 0.08, 0.02]),
        'trend_alert_weak': np.random.choice([0, 1], n_samples, p=[0.92, 0.08])
    }
    
    df = pd.DataFrame(data)
    
    # 保存到临时文件
    test_dir = Path(__file__).parent.parent.parent / 'features' / 'test'
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / 'test_features.parquet'
    df.to_parquet(test_file, index=False)
    
    print(f"✅ 测试数据已创建: {test_file}")
    print(f"   样本数: {len(df):,}")
    print(f"   时间范围: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    return test_file


def test_batch_inference():
    """测试批量推理"""
    print("\n" + "="*80)
    print("测试批量推理")
    print("="*80)
    
    try:
        from batch_inference import BatchInferenceEngine
        
        # 初始化引擎
        engine = BatchInferenceEngine(genre='test')
        print("✅ 引擎初始化成功")
        
        # 加载模型（会显示警告，正常）
        engine.load_models()
        print("✅ 模型加载完成（可能显示警告）")
        
        return True
    except Exception as e:
        print(f"❌ 批量推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dashboard():
    """测试仪表盘"""
    print("\n" + "="*80)
    print("测试仪表盘")
    print("="*80)
    
    try:
        from update_monitoring_dashboard import UpdateMonitoringDashboard
        
        # 初始化仪表盘
        dashboard = UpdateMonitoringDashboard(
            update_date='2024-03-20',
            window_hours=48,
            genre='test'
        )
        print("✅ 仪表盘初始化成功")
        
        return True
    except Exception as e:
        print(f"❌ 仪表盘测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alerts():
    """测试告警系统"""
    print("\n" + "="*80)
    print("测试告警系统")
    print("="*80)
    
    try:
        from crisis_alert import CrisisAlertSystem
        
        # 初始化告警系统
        alert_system = CrisisAlertSystem(
            update_date='2024-03-20',
            window_hours=48,
            baseline_days=7,
            genre='test'
        )
        print("✅ 告警系统初始化成功")
        
        return True
    except Exception as e:
        print(f"❌ 告警系统测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("推理系统快速测试")
    print("="*80)
    
    # 1. 测试导入
    if not test_imports():
        print("\n❌ 导入测试失败，请检查依赖")
        return
    
    # 2. 创建测试数据
    test_file = create_test_data()
    
    # 3. 测试各个组件
    results = {
        'batch_inference': test_batch_inference(),
        'dashboard': test_dashboard(),
        'alerts': test_alerts()
    }
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    for component, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status} - {component}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！推理系统准备就绪。")
        print("\n下一步:")
        print("1. 等待Sentiment和Topic模型训练完成")
        print("2. 适配BERT模型推理（修改batch_inference.py）")
        print("3. 训练Trend Alert模型")
        print("4. 运行完整推理流程")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
