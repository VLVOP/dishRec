import json
import numpy as np

def load_structured_recipes(json_path):
    """
    加载菜谱数据json文件
    """
    try:
        with open(json_path, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading recipes from {json_path}: {e}")
        return []
    
def jaccard_similarity(set1, set2):
    """
    计算 Jaccard 相似度
    """
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def calculate_recipe_similarity(recipe_a: dict, recipe_b: dict) -> float:
    """
    计算两个菜谱的整体相似度
    主要基于食材颗粒度上的重叠，辅以分类匹配
    """

    # 食材相似度 (Jaccard)
    ingredients_a = set(recipe_a.get('ingredients', []))
    ingredients_b = set(recipe_b.get('ingredients', []))
    ing_sim = jaccard_similarity(ingredients_a, ingredients_b)

    # 分类相似度 (Category)
    cat_sim = 1.0 if recipe_a.get('category') == recipe_b.get('category') else 0.0

    # 综合相似度
    # 权重设定：食材重叠更重要（0.7）, 分类一致性次之（0.3）
    combined_sim = 0.7 * ing_sim + 0.3 * cat_sim

    return combined_sim

if __name__ == "__main__":
    """
    TEST
    """
    RECIPES_JSON_PATH = "src/data/structured_recipes.json"
    recipes = load_structured_recipes(RECIPES_JSON_PATH)

    if len(recipes) >= 2:
        sim = calculate_recipe_similarity(recipes[0], recipes[1])
        print(f"菜谱 '{recipes[0]['name']}' 和 '{recipes[1]['name']}' 的相似度为：{sim: .4f}")
    elif recipes :
        print("---数据不足，至少需要两个菜谱才能进行相似度测试---")
    else :
        print("---未加载到任何菜谱数据---")