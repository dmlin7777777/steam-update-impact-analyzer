"""
快速测试脚本：验证模型加载是否正常
"""

import sys
from pathlib import Path

# 添加脚本目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from batch_inference import BatchInferenceEngine

def test_model_loading():
    """
    测试模型加载
    """
    print("\n" + "="*80)
    print("测试模型加载")
    print("="*80)
    
    engine = BatchInferenceEngine(genre='fps')
    engine.load_models()
    
    print("\n" + "="*80)
    print("加载结果汇总")
    print("="*80)
    
    # 检查各个模型
    checks = {
        'BERT Tokenizer (Sentiment)': 'sentiment_tokenizer' in engine.models,
        'BERT Model (Sentiment)': 'sentiment_bert' in engine.models,
        'Classifier (Sentiment)': 'sentiment_clf' in engine.models,
        'BERT Tokenizer (Topic)': 'topic_tokenizer' in engine.models,
        'BERT Model (Topic)': 'topic_bert' in engine.models,
        'Classifier (Topic)': 'topic_clf' in engine.models,
        'Label Mapping (Topic)': 'topic_labels' in engine.models,
        'LightGBM Model (Trend)': 'trend' in engine.models,
        'TF-IDF Vectorizer (Trend)': 'trend' in engine.vectorizers,
        'Risk Config': 'risk_config' in engine.models,
    }
    
    all_ok = True
    for name, is_loaded in checks.items():
        status = "✅" if is_loaded else "❌"
        print(f"{status} {name}")
        if not is_loaded:
            all_ok = False
    
    print("\n" + "="*80)
    if all_ok:
        print("✅ 所有模型加载成功！")
        return 0
    else:
        print("❌ 有些模型加载失败，请检查模型路径")
        return 1

if __name__ == "__main__":
    exit_code = test_model_loading()
    sys.exit(exit_code)
