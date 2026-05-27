"""
改进的 LDA 主题映射逻辑
基于关键词规则的精准映射，替代 difflib.get_close_matches

使用方法:
1. 在 feature_engineering.py 中替换现有的映射逻辑
2. 或作为独立函数用于重新映射已有数据
"""

import numpy as np
from typing import List, Dict, Tuple

def map_topic_by_keywords_rules(
    topic_id: int, 
    topic_keywords: List[str], 
    verbose: bool = False
) -> str:
    """
    基于关键词规则精准映射 LDA 主题到业务类别
    
    Args:
        topic_id: LDA 主题 ID (0-7)
        topic_keywords: 主题的 top-N 关键词列表
        verbose: 是否打印映射原因
        
    Returns:
        业务类别标签
    """
    keywords_lower = [kw.lower() for kw in topic_keywords]
    keywords_str = ' '.join(keywords_lower)
    
    # 规则 1: 外挂问题 (cheating)
    cheating_keywords = {'cheat', 'cheater', 'cheating', 'hacker', 'aimbot', 'wallhack'}
    anti_cheat_keywords = {'anti', 'vac', 'ban', 'report'}
    
    cheat_count = sum(1 for kw in keywords_lower if any(ck in kw for ck in cheating_keywords))
    anti_count = sum(1 for kw in keywords_lower if any(ak in kw for ak in anti_cheat_keywords))
    
    # 外挂关键词 >= 2 且无反作弊关键词 → cheating
    if cheat_count >= 2 and anti_count == 0:
        if verbose:
            print(f"  Topic {topic_id} → cheating (cheat keywords: {cheat_count})")
        return 'cheating'
    
    # 反作弊关键词 >= 1 且有外挂关键词 → anti_cheat_issues
    if anti_count >= 1 and cheat_count >= 1:
        if verbose:
            print(f"  Topic {topic_id} → anti_cheat_issues (anti: {anti_count}, cheat: {cheat_count})")
        return 'anti_cheat_issues'
    
    # 规则 2: 付费问题 (monetization_concerns)
    monetization_keywords = {'gamble', 'gambling', 'pay', 'money', 'case', 'loot', 'microtransaction', 
                            'dlc', 'expensive', 'gold', 'skin', 'box'}
    money_count = sum(1 for kw in keywords_lower if any(mk in kw for mk in monetization_keywords))
    
    if money_count >= 2:
        if verbose:
            print(f"  Topic {topic_id} → monetization_concerns (money keywords: {money_count})")
        return 'monetization_concerns'
    
    # 规则 3: 技术问题 (technical_issues)
    technical_keywords = {'crash', 'bug', 'glitch', 'freeze', 'lag', 'error', 'fix', 'broken', 'performance'}
    tech_count = sum(1 for kw in keywords_lower if any(tk in kw for tk in technical_keywords))
    
    if tech_count >= 2:
        if verbose:
            print(f"  Topic {topic_id} → technical_issues (tech keywords: {tech_count})")
        return 'technical_issues'
    
    # 规则 4: 特定游戏反馈
    if 'cod' in keywords_str or 'duty' in keywords_str:
        if verbose:
            print(f"  Topic {topic_id} → cod_related")
        return 'cod_related'
    
    if 'csgo' in keywords_str or ('competitive' in keywords_str and 'map' in keywords_str):
        if verbose:
            print(f"  Topic {topic_id} → csgo_competitive")
        return 'csgo_competitive'
    
    # 规则 5: 多人功能 (multiplayer_features)
    multiplayer_keywords = {'multiplayer', 'team', 'squad', 'party', 'coop', 'pvp', 'match', 'server'}
    mp_count = sum(1 for kw in keywords_lower if any(mk in kw for mk in multiplayer_keywords))
    
    if mp_count >= 1:
        if verbose:
            print(f"  Topic {topic_id} → multiplayer_features (mp keywords: {mp_count})")
        return 'multiplayer_features'
    
    # 规则 6: UI/性能 (user_interface)
    ui_keywords = {'fps', 'ui', 'interface', 'menu', 'hud', 'graphics', 'visual', 'control', 'setting'}
    ui_count = sum(1 for kw in keywords_lower if any(uk in kw for uk in ui_keywords))
    
    if ui_count >= 1:
        if verbose:
            print(f"  Topic {topic_id} → user_interface (ui keywords: {ui_count})")
        return 'user_interface'
    
    # 规则 7: 游戏机制 (gameplay_mechanics)
    gameplay_keywords = {'balance', 'weapon', 'character', 'ability', 'skill', 'mechanic', 'gameplay'}
    gp_count = sum(1 for kw in keywords_lower if any(gk in kw for gk in gameplay_keywords))
    
    if gp_count >= 1:
        if verbose:
            print(f"  Topic {topic_id} → gameplay_mechanics (gameplay keywords: {gp_count})")
        return 'gameplay_mechanics'
    
    # 默认: 通用反馈 (general_feedback)
    if verbose:
        print(f"  Topic {topic_id} → general_feedback (no specific pattern matched)")
    return 'general_feedback'


def get_improved_topic_mapping() -> Dict[int, Tuple[str, str]]:
    """
    返回基于实际数据分析的改进映射
    
    Returns:
        {topic_id: (category, reason)}
    """
    # 基于 lda_topic_keywords.json 的实际关键词
    actual_keywords = {
        0: ['better', 'dont', 'fun', 'game', 'good', 'got', 'hate', 'life', 'like', 'love'],
        1: ['cod', 'duty', 'fun', 'game', 'good', 'great', 'hour', 'like', 'make', 'play'],
        2: ['friend', 'fun', 'fun game', 'gamble', 'gamble gamble', 'gambling', 'game', 'gold', 'gold gold', 'mango'],
        3: ['competitive', 'csgo', 'experience', 'feel', 'game', 'gameplay', 'good', 'like', 'make', 'map'],
        4: ['best', 'best game', 'case', 'cool', 'crash', 'fix', 'game', 'good', 'like', 'money'],
        5: ['cheat', 'cheater', 'cheater cheater', 'cheating', 'game', 'game cheater', 'good', 'hacker', 'lot', 'match'],
        6: ['anti', 'anti cheat', 'cancer', 'cheat', 'dementia', 'dementia dementia', 'fix', 'fuck', 'fucking', 'game'],
        7: ['bad', 'best', 'cheater', 'fps', 'fun', 'game', 'game good', 'good', 'good game', 'like']
    }
    
    improved_mapping = {}
    
    print("="*80)
    print("使用改进的规则映射 LDA 主题")
    print("="*80)
    
    for topic_id, keywords in actual_keywords.items():
        category = map_topic_by_keywords_rules(topic_id, keywords, verbose=True)
        
        # 手动微调（基于语义理解）
        if topic_id == 0:
            category = 'multiplayer_features'
            reason = '通用正面反馈，偏向多人体验 (fun, like, love)'
        elif topic_id == 1:
            category = 'cod_related'
            reason = '明确提及 COD/Duty'
        elif topic_id == 2:
            category = 'monetization_concerns'
            reason = 'gamble 重复强调，明确付费不满'
        elif topic_id == 3:
            category = 'csgo_competitive'
            reason = '明确提及 CSGO + competitive + map'
        elif topic_id == 4:
            category = 'technical_issues'
            reason = 'crash + fix 主导，技术问题优先'
        elif topic_id == 5:
            category = 'cheating'
            reason = 'cheat/cheater/hacker 极度明确'
        elif topic_id == 6:
            category = 'anti_cheat_issues'
            reason = 'anti cheat + 极端负面情绪'
        elif topic_id == 7:
            category = 'user_interface'
            reason = 'fps 关键词 + UI 性能'
        
        improved_mapping[topic_id] = (category, reason)
        print(f"  ✓ Topic {topic_id} → {category:25s} ({reason})")
    
    return improved_mapping


def apply_improved_mapping_to_dataframe(df, topic_col='dominant_topic'):
    """
    将改进的映射应用到 DataFrame
    
    Args:
        df: DataFrame with dominant_topic column
        topic_col: 主题列名
        
    Returns:
        df with updated topic_category_improved column
    """
    import pandas as pd
    
    improved_mapping = get_improved_topic_mapping()
    category_map = {tid: cat for tid, (cat, _) in improved_mapping.items()}
    
    df['topic_category_improved'] = df[topic_col].map(category_map)
    
    print("\n改进后的类别分布:")
    dist = df['topic_category_improved'].value_counts()
    for cat, count in dist.items():
        pct = count / len(df) * 100
        print(f"  {cat:25s}: {count:7,} ({pct:5.2f}%)")
    
    return df


# ============================================================================
# 集成到 feature_engineering.py 的代码片段
# ============================================================================

def get_integration_code():
    """
    返回可以直接集成到 feature_engineering.py 的代码
    """
    code = '''
# 替换 feature_engineering.py 中的 topic_category 映射逻辑
# 在 extract_topic_features() 函数的末尾

# ============ 旧代码 (删除) ============
# topic_labels = [...]
# topic_names = []
# mapped_labels = []
# for topic_keywords in topic_names:
#     best_label = difflib.get_close_matches(topic_keywords, topic_labels, n=1, cutoff=0.0)
#     ...

# ============ 新代码 (替换) ============
def map_topic_by_keywords(topic_id, lda_model, tfidf_vectorizer, n_top_words=10):
    """基于 LDA 主题的关键词映射到业务类别"""
    if hasattr(lda_model, 'components_'):
        feature_names = tfidf_vectorizer.get_feature_names_out()
        topic = lda_model.components_[topic_id]
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words-1:-1]]
        keywords_str = ' '.join(top_words).lower()
        
        # 规则 1: 外挂问题
        if sum(kw in keywords_str for kw in ['cheat', 'cheater', 'hacker']) >= 2 and 'anti' not in keywords_str:
            return 'cheating'
        
        # 规则 2: 反作弊系统
        if 'anti' in keywords_str and any(kw in keywords_str for kw in ['cheat', 'vac']):
            return 'anti_cheat_issues'
        
        # 规则 3: 付费问题
        if sum(kw in keywords_str for kw in ['gamble', 'gambling', 'money', 'gold', 'case']) >= 2:
            return 'monetization_concerns'
        
        # 规则 4: 技术问题
        if sum(kw in keywords_str for kw in ['crash', 'bug', 'fix', 'lag', 'error']) >= 2:
            return 'technical_issues'
        
        # 规则 5: 特定游戏
        if 'cod' in keywords_str or 'duty' in keywords_str:
            return 'cod_related'
        if 'csgo' in keywords_str:
            return 'csgo_competitive'
        
        # 规则 6: 多人功能
        if any(kw in keywords_str for kw in ['multiplayer', 'team', 'match', 'server']):
            return 'multiplayer_features'
        
        # 规则 7: UI/性能
        if any(kw in keywords_str for kw in ['fps', 'ui', 'interface', 'graphics']):
            return 'user_interface'
        
        # 规则 8: 游戏机制
        if any(kw in keywords_str for kw in ['balance', 'weapon', 'gameplay', 'mechanic']):
            return 'gameplay_mechanics'
    
    return 'general_feedback'

# 应用映射
category_mapping = {i: map_topic_by_keywords(i, self.lda_model, self.tfidf_vectorizer) 
                   for i in range(n_topics)}
df_copy['topic_category'] = df_copy['dominant_topic'].map(category_mapping)
'''
    return code


if __name__ == '__main__':
    # 示例：获取改进的映射
    improved_mapping = get_improved_topic_mapping()
    
    print("\n" + "="*80)
    print("改进后的类别分布预测")
    print("="*80)
    
    # 基于实际数据统计
    actual_counts = {
        0: 38014,
        1: 27067,
        2: 8285,
        3: 19172,
        4: 21888,
        5: 23603,
        6: 11938,
        7: 26881
    }
    
    category_counts = {}
    for topic_id, count in actual_counts.items():
        category = improved_mapping[topic_id][0]
        category_counts[category] = category_counts.get(category, 0) + count
    
    total = sum(category_counts.values())
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {cat:25s}: {count:7,} ({pct:5.2f}%)")
    
    print("\n" + "="*80)
    print("集成代码 (复制到 feature_engineering.py)")
    print("="*80)
    print(get_integration_code())
