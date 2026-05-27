"""
分析当前 LDA 主题映射的问题并提出改进方案
"""

import json
from pathlib import Path

BASE_DIR = Path(r'C:\Users\12932\Desktop\nus\BAP')
KEYWORDS_PATH = BASE_DIR / 'analysis_results' / 'data_summary' / 'lda_topic_keywords.json'

# Load current mapping
with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*80)
print("当前映射分析")
print("="*80)

print("\n【问题 1】difflib.get_close_matches 映射效果差")
print("-" * 80)
print("当前代码使用字符串相似度匹配 LDA 关键词 → 预定义标签：")
print("  best_label = difflib.get_close_matches(topic_keywords, topic_labels)")
print("\n实际映射结果：")

for detail in data['topic_details']:
    topic_id = detail['topic_id']
    category = detail['category']
    keywords = detail['top_keywords'][:5]
    
    print(f"\nTopic {topic_id} → {category}")
    print(f"  关键词: {', '.join(keywords)}")
    
    # 分析映射合理性
    if topic_id == 0:
        print(f"  ✓ 映射合理: 'multiplayer_features'")
        print(f"    - 关键词包含: better, fun, game (多人体验相关)")
    elif topic_id == 1:
        print(f"  ⚠ 映射模糊: 'community_feedback' (太宽泛)")
        print(f"    - 关键词明确指向: COD (Call of Duty)")
        print(f"    - 建议: 创建 'specific_game_feedback' 或保留 'cod_related'")
    elif topic_id == 2:
        print(f"  ✓ 映射合理: 'monetization_concerns'")
        print(f"    - 关键词明确: gamble, gambling, gold")
    elif topic_id == 3:
        print(f"  ⚠ 映射模糊: 'community_feedback' (太宽泛)")
        print(f"    - 关键词明确指向: CSGO, competitive, map")
        print(f"    - 建议: 创建 'competitive_gameplay' 或 'csgo_related'")
    elif topic_id == 4:
        print(f"  ⚠ 映射模糊: 'community_feedback' (太宽泛)")
        print(f"    - 关键词混合: crash (技术), case (付费)")
        print(f"    - 建议: 'technical_and_monetization' 或细分")
    elif topic_id == 5:
        print(f"  ⚠ 映射错误: 应该映射到 'cheating' 而非 'community_feedback'")
        print(f"    - 关键词极度明确: cheat, cheater, hacker")
        print(f"    - 这是预定义标签中的 'cheating' 类别！")
    elif topic_id == 6:
        print(f"  ⚠ 映射模糊: 'gameplay_mechanics'")
        print(f"    - 关键词明确: anti cheat, cancer, fuck (反作弊系统)")
        print(f"    - 建议: 映射到 'anti_cheat_issues' 或 'cheating'")
    elif topic_id == 7:
        print(f"  ✓ 映射合理: 'user_interface'")
        print(f"    - 关键词包含: fps, bad, best (性能/UI)")

print("\n" + "="*80)
print("【问题 2】community_feedback 过度合并")
print("-" * 80)
community_topics = [d for d in data['topic_details'] if d['category'] == 'community_feedback']
print(f"有 {len(community_topics)} 个主题被映射到 'community_feedback':")
for detail in community_topics:
    print(f"  Topic {detail['topic_id']}: {', '.join(detail['top_keywords'][:3])}")

total_community = sum(d['count'] for d in community_topics)
print(f"\n合计样本: {total_community:,} ({total_community/data['n_samples']*100:.1f}%)")
print("❌ 问题: 占比过高 (51.9%)，导致类别严重不平衡")

print("\n" + "="*80)
print("【改进方案】")
print("-" * 80)

print("\n方案 1: 基于关键词规则的精准映射 (推荐)")
print("-" * 40)
improved_mapping = {
    0: ('multiplayer_features', '保持，关键词符合'),
    1: ('cod_related', '新增，明确 COD 反馈'),
    2: ('monetization_concerns', '保持，关键词明确'),
    3: ('csgo_competitive', '新增，明确 CSGO 竞技'),
    4: ('technical_issues', '修改，crash/fix 指向技术问题'),
    5: ('cheating', '修正！应使用预定义的 cheating 标签'),
    6: ('anti_cheat_issues', '新增，明确反外挂系统问题'),
    7: ('user_interface', '保持，fps/UI 相关')
}

print("\n改进后映射:")
for topic_id, (new_label, reason) in improved_mapping.items():
    old_label = data['topic_details'][topic_id]['category']
    change = "✓ 保持" if new_label == old_label else f"❌ {old_label} → {new_label}"
    print(f"  Topic {topic_id}: {change:50s} ({reason})")

print("\n改进后类别分布:")
new_categories = {}
for topic_id, (new_label, _) in improved_mapping.items():
    count = data['topic_details'][topic_id]['count']
    new_categories[new_label] = new_categories.get(new_label, 0) + count

for label, count in sorted(new_categories.items(), key=lambda x: -x[1]):
    pct = count / data['n_samples'] * 100
    print(f"  {label:25s}: {count:7,} ({pct:5.2f}%)")

print("\n优势:")
print("  ✓ 消除 community_feedback 过度合并 (51.9% → 0%)")
print("  ✓ 明确区分 外挂问题 vs 反外挂系统")
print("  ✓ 区分不同游戏反馈 (COD vs CSGO)")
print("  ✓ 类别更平衡 (最大 21.5% vs 最小 4.7%)")

print("\n方案 2: 基于关键词 + 语义相似度的智能映射")
print("-" * 40)
print("步骤:")
print("  1. 提取 LDA 主题的 top-10 关键词")
print("  2. 为每个预定义标签定义关键词集合:")
print("     - cheating: [cheat, cheater, hacker, aimbot]")
print("     - anti_cheat: [anti, vac, ban, report]")
print("     - monetization: [gamble, pay, money, case, loot]")
print("  3. 计算关键词重叠度 (Jaccard 或 cosine similarity)")
print("  4. 选择重叠度最高的标签")

print("\n优势:")
print("  ✓ 自动化程度高，减少人工规则")
print("  ✓ 可扩展到新游戏/新主题")

print("\n缺点:")
print("  ❌ 需要维护预定义关键词集合")
print("  ❌ 仍可能误判边界情况")

print("\n方案 3: 直接使用 8 分类，不映射到 5 分类")
print("-" * 40)
print("优势:")
print("  ✓ 保留 LDA 的细粒度信息")
print("  ✓ 避免映射错误")
print("  ✓ 类别更平衡")

print("\n缺点:")
print("  ❌ 业务可解释性弱 (topic_0 vs topic_1)")
print("  ❌ 需要手动标注每个主题的含义")

print("\n" + "="*80)
print("【推荐方案】")
print("="*80)
print("\n✅ 短期 (本次训练): 使用方案 3 (8 分类)")
print("   - 直接使用 dominant_topic (0-7)")
print("   - 在报告中补充每个主题的关键词解释")
print("   - 避免 community_feedback 过度合并问题")

print("\n✅ 长期 (生产部署): 使用方案 1 (精准映射)")
print("   - 修改 feature_engineering.py 的映射逻辑")
print("   - 基于关键词规则精准映射:")
print("     ```python")
print("     def map_topic_to_category(topic_id, keywords):")
print("         if 'cheat' in keywords or 'hacker' in keywords:")
print("             return 'cheating'")
print("         elif 'anti' in keywords and 'cheat' in keywords:")
print("             return 'anti_cheat_issues'")
print("         elif 'gamble' in keywords or 'gambling' in keywords:")
print("             return 'monetization_concerns'")
print("         # ... 更多规则")
print("     ```")

print("\n" + "="*80)
print("下一步行动")
print("="*80)
print("1. ✅ 已完成: 提取 LDA 主题关键词")
print("2. 📝 当前: 分析映射问题，提出改进方案")
print("3. 🏃 建议: 先用 8 分类训练 (train_topic_fps.py Task A)")
print("4. 📊 观察: 对比 8-class vs 5-class 的性能差异")
print("5. 🔧 长期: 如果需要 5 分类，实现方案 1 的精准映射")
