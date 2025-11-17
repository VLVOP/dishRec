import json 
import random 
import os
from typing import List, Dict

from .similarity_calc import load_structured_recipes, calculate_recipe_similarity

# --- 文件路径定义 --- 
RECIPES_JSON_PATH = "src/data/structured_recipes.json"
OUTPUT_TRAIN_PATH = "src/data/sft_train.jsonl"
OUTPUT_VAL_PATH = "src/data/sft_val.jsonl"

# --- 参数配置 ---
NUM_HISTORY_ITEMS = 3       # 模拟用户历史中已尝试的菜谱数量 N
NUM_RECOMMEND_TARGETS_MAX = 5   # 推荐目标菜谱数量最大值 M 
VAL_RATIO = 0.1             # 验证集比例

# --- 荤菜判断的硬编码列表 
MEAT_CATEGORIES = ["meat_dish", "aquatic"]
VEGETABLE_CATEGORIES = ["vegetable_dish", "soup", "staple"]

# 常用错别字词典
TYPO_MAP = {
    '鱼': '渔', '肉': '内', '排': '湃', '鸡': '饥', '菜': '柴', 
    '汤': '烫', '虾': '霞', '酱': '浆', '油': '尤'
}

# 数量强度词典
COUNT_INTENSITY = {
    3: ["少量", "少一些", "稍微给我点"],
    4: ["几个", "一些", "中等数量的"],
    5: ["很多", "多一些", "尽可能多地"],
}

def introduce_typo(text: str, rate: float = 0.4) -> str:
    """
    以一定概率在字符串中引入错别字
    """

    if random.random() < rate:
        # 遍历错字表，随机应用一个错别字
        for original, typo in TYPO_MAP.items():
            if original in text and random.random() < 0.5: # 50% 概率应用这个特定的错字
                # 只替换第一个出现的错字
                text = text.replace(original, typo, 1) 
                break
    return text

def get_popular_data(recipes: List[Dict], top_n: int = 5) -> tuple[List[str], List[str]]:
    """
    统计最常用的分类和食材
    """
    all_ingredients = {}
    all_categories = {}
    for recipe in recipes :
        all_categories[recipe.get('category')] = all_categories.get(recipe.get('category'), 0) + 1
        for ing in recipe.get('ingredients', []):
            # 去除空或是太泛的食材
            if ing and len(ing) > 1:
                all_ingredients[ing] = all_ingredients.get(ing, 0) + 1

    popular_categories = [c for c, count in sorted(all_categories.items(), key=lambda item: item[1], reverse=True) if c]
    popular_ingredients = [i for i, count in sorted(all_ingredients.items(), key=lambda item: item[1], reverse=True) if i]

    return popular_categories[: top_n], popular_ingredients[: top_n]

def create_sft_sample(user_history: List[str], target_recommendations: List[str], constraint: Dict = None) -> Dict:
    """
    将用户饮食历史和推荐目标格式化为 LLM SFT 所需的 Prompt-Response 结构
    同时加入复杂指令
    """

    output_M = len(target_recommendations)
    intensity_word = random.choice(COUNT_INTENSITY.get(output_M, ["一些"]))
    
    # 随机决定是否对历史记录引入错别字
    input_history = [introduce_typo(name) for name in user_history]

    # 基础指令
    base_instructions = [
        f"根据我之前做的这些菜，给我推荐{intensity_word}新的。",
        f"看看我喜欢这些菜，帮我找{intensity_word}类似的。",
        f"请为我生成{intensity_word}基于我历史口味的菜谱推荐。",
    ]

    instruction = random.choice(base_instructions)

    input_text = f"我的菜谱历史是：{', '.join(input_history)}。"

    # --- 引入约束和更复杂的指令 ---
    if constraint :
        c_type = constraint.get("type")
        c_value = constraint.get("value")

        # --- 泛化约束（荤素）和硬约束（类别/包含/排除）---

        if c_type == "non_veg":
            instruction = random.choice([
                f"我最近想吃肉，请多给我推荐{intensity_word}**荤菜**。",
                f"排除素食，给我{intensity_word}纯肉食的推荐。",
            ])
        elif c_type == "veg":
            instruction = random.choice([
                f"请给我推荐{intensity_word}新菜谱，**只推荐素菜**，不要肉类。",
                f"我希望是{intensity_word}健康的素食选择。",
            ])
        elif c_type == "category":
            instruction = random.choice([
                f"给我推荐{intensity_word}{c_value}类的相似菜谱。",
                f"我今天只考虑{c_value}菜，请多给我{intensity_word}。",
            ])
        elif c_type == "include":
            instruction = random.choice([
                f"我希望新菜里有 {c_value}，请推荐{intensity_word}。",
                f"给我{intensity_word}菜，要求里面**必须包含 '{c_value}'**。",
            ])
        elif c_type == "exclude":
            instruction = random.choice([
                f"我不喜欢 {c_value}，推荐时请排除，给我{intensity_word}选择。",
                f"给我{intensity_word}菜，请确保里面**不含 '{c_value}'**。",
            ])

    # 最终指令：不限定数量，只限定格式
    final_instruction = instruction + " **请只输出菜谱名称，用逗号分隔。**"

    output_text = ', '.join(target_recommendations)

    return {
        "instruction": final_instruction,
        "input": input_text,
        "output": output_text
    }

def generate_sft_data():
    """
    加载菜谱数据，模拟用户行为，并生成 SFT 训练集和验证集。
    """

    if not os.path.exists(RECIPES_JSON_PATH):
        print(f"错误：菜谱数据文件 {RECIPES_JSON_PATH} 不存在。请先运行 data_extractor.py")
        return 
    
    recipes = load_structured_recipes(RECIPES_JSON_PATH)

    if not recipes:
        print(f"错误：未能加载到任何菜谱数据，请检查 {RECIPES_JSON_PATH} 内容。")
        return
    
    print(f"--- 开始生成 SFT 训练数据（基于{len(recipes)} 个菜谱）---")

    popular_categories, popular_ingredients = get_popular_data(recipes, top_n=5)
    name_to_recipe = {r['name']: r for r in recipes}

    all_sft_samples = []

    for target_recipe in recipes:

        # 随机选择场景类型
        scenario_type = random.choice([
            "similarity", "similarity", "similarity",
            "category", "include", "exclude", "non_veg", "veg"
        ]) # 3 : 7
        constraint = None

        # 确定本次场景需要生成的推荐数量
        # 随机抽取 3，4，或 5 个，训练模型根据指令中的强度词进行推理
        current_M = random.choice([3, 4, 5])

        # 构造用户历史集合 (H)
        history_candidates = []
        for other_recipe in recipes:
            if other_recipe['name'] != target_recipe['name']:
                sim = calculate_recipe_similarity(target_recipe, other_recipe)  # 计算分数
                history_candidates.append((sim, other_recipe['name']))          # 将分数和对应名称作为一个元组返回
        history_candidates.sort(key=lambda x: x[0], reverse=True)
        user_history_names = [name for sim, name in history_candidates[:NUM_HISTORY_ITEMS]]

        # 构造待推荐候选集
        available_recipes = [r for r in recipes
                             if r['name'] not in user_history_names and r['name'] != target_recipe['name']]
        
        # 根据场景应用约束过滤 available_recipes
        filtered_candidates = available_recipes

        if scenario_type == "non_veg":
            constraint = {"type": "non_veg", "value": "荤菜"}
            filtered_candidates = [
                r for r in available_recipes
                if r.get('category') in MEAT_CATEGORIES
            ]

        elif scenario_type == "veg":
            constraint = {"type": "veg", "value": "素菜"}
            filtered_candidates = [
                r for r in available_recipes
                if r.get('category') in VEGETABLE_CATEGORIES and r.get('category') not in MEAT_CATEGORIES
            ]

        elif scenario_type == "category" and popular_categories:
            c_value = random.choice(popular_categories)
            constraint = {"type": "category", "value": c_value}
            filtered_candidates = [r for r in available_recipes if r.get('category') == c_value]

        elif scenario_type == "include" and popular_ingredients:
            c_value = random.choice(popular_ingredients)
            constraint = {"type": "include", "value": c_value}
            filtered_candidates = [r for r in available_recipes if c_value in r.get('ingredients', [])]

        elif scenario_type == "exclude" and popular_ingredients:
            c_value = random.choice(popular_ingredients)
            constraint = {"type": "exclude", "value": c_value}
            filtered_candidates = [r for r in available_recipes if c_value not in r.get('ingredients', [])]

        # 计算并选取推荐目标
        recommendation_scores = []
        if filtered_candidates and user_history_names:
            for candidate_recipe in filtered_candidates:
                avg_sim = sum(calculate_recipe_similarity(candidate_recipe, name_to_recipe[h_name])
                                for h_name in user_history_names) / NUM_HISTORY_ITEMS
                recommendation_scores.append((avg_sim, candidate_recipe['name']))

            # 选取得分最高的 current_M 个作为推荐目标
            recommendation_scores.sort(key=lambda x: x[0], reverse=True)
            target_recommendations = [name for sim, name in recommendation_scores[: current_M]]

        # 生成 SFT 样本
        if (len(user_history_names) == NUM_HISTORY_ITEMS and
            len(target_recommendations) == current_M):

            sample = create_sft_sample(user_history_names, target_recommendations, constraint)
            all_sft_samples.append(sample)

    if not all_sft_samples:
        print("警告：未能生成任何有效的 SFT 样本，请检查参数和数据。")
        return 
    
    # 划分训练集和验证集
    random.shuffle(all_sft_samples)
    val_size = int(len(all_sft_samples) * VAL_RATIO)
    train_data = all_sft_samples[val_size: ]
    val_data = all_sft_samples[: val_size]

    # 保存结果
    os.makedirs(os.path.dirname(OUTPUT_TRAIN_PATH), exist_ok=True)

    with open(OUTPUT_TRAIN_PATH, 'w', encoding='utf-8') as f:
        for sample in train_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
    with open(OUTPUT_VAL_PATH, 'w', encoding='utf-8') as f:
        for sample in val_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\n--- SFT 数据生成完毕！---")
    print(f"总样本数: {len(all_sft_samples)}")
    print(f"训练集样本数: {len(train_data)} ({OUTPUT_TRAIN_PATH})")
    print(f"验证集样本数: {len(val_data)} ({OUTPUT_VAL_PATH})")

if __name__ == "__main__":
    generate_sft_data()