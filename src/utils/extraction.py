import os
import re
import json

RECIPES_ROOT_DIR = "src/data/dishes"
OUTPUT_JSON_PATH = "src/data/structured_recipes.json"


def extract_recipe_data(filepath, category):
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None
    
    data = {
        "recipe_id": os.path.splitext(os.path.basename(filepath))[0],
        "name": os.path.splitext(os.path.basename(filepath))[0].strip(),
        "category": category,
        "ingredients": [],
        "steps": [],
        "description": ""
    }
    
    # 提取描述
    description_match = re.search(r'^(.*?)(?=^##\s*)', content, re.MULTILINE | re.DOTALL)
    if description_match:
        desc_text = description_match.group(1).strip()
        desc_text = re.sub(r'#.*', '', desc_text)
        desc_text = re.sub(r'\*\*', '', desc_text)
        desc_text = re.sub(r'[\r\n]{2,}', '\n', desc_text).strip()
        data["description"] = desc_text.split('\n')[0].strip()

    # 提取步骤和做法以及用料食材等关键信息
    sections = re.findall(r'^(##\s*(.+?))\s*\n([\s\S]*?)(?=\n##\s*|\Z)', content, re.MULTILINE)

    for header_tag, header_name, section_content in sections:
        header_name_lower = header_name.lower()
        
        # 匹配食材标题
        if any(keyword in header_name_lower for keyword in ['食材', '用料', '原料', '计算']):
            ingredients_list = re.findall(r'^\s*[\*-]\s*(.+)', section_content, re.MULTILINE)
            data["ingredients"].extend([
                re.split(r'\(|（|:', item)[0].strip()
                for item in ingredients_list
                if not any(keyword in item for keyword in ['工具', '模具']) and item.strip()
            ])
            
        # 匹配做法标题
        elif any(keyword in header_name_lower for keyword in ['做法', '步骤', '操作']):
            # 尝试匹配带列表符号的行
            steps_list_marked = re.findall(r'^\s*(\d+\.|\*|\-)\s*(.+)', section_content, re.MULTILINE)
            
            if steps_list_marked:
                data["steps"] = [re.sub(r'\*\*', '', item[1]).strip() for item in steps_list_marked]
            else:
                # 匹配纯文本行（针对红烧鲤鱼等没有列表符号的格式）
                pure_text_lines = [
                    line.strip()
                    for line in section_content.split('\n')
                    if line.strip() and not line.strip().startswith('注：') and not line.strip().startswith('note:')
                ]
                data["steps"] = [re.sub(r'\*\*', '', line).strip() for line in pure_text_lines if line.strip()]

    if data["ingredients"]:
        data["ingredients"] = list(set(data["ingredients"]))

    # 修正键名：将 'step' 统一为 'steps'
    if 'step' in data:
        data['steps'] = data.pop('step')
        
    if not data["ingredients"] or not data["steps"]:
        return None
    
    return data

def process_all_dishes():
    print(f"--- 开始处理菜谱文件：{RECIPES_ROOT_DIR} ---")
    all_recipes = []
    processed_categories = set()
    
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    
    if not os.path.exists(RECIPES_ROOT_DIR):
        print(f"错误：菜谱根目录 {RECIPES_ROOT_DIR} 不存在。请检查路径是否存在。")
        return None

    # 使用 os.walk 递归遍历目录树（解决嵌套文件夹问题）
    for root, dirs, files in os.walk(RECIPES_ROOT_DIR):
        relative_path = os.path.relpath(root, RECIPES_ROOT_DIR)
        category = None
        
        if relative_path != '.':
            category = relative_path.split(os.path.sep)[0]
            if category not in processed_categories:
                print(f"处理类别: {category}...")
                processed_categories.add(category)
        
        if category:
            for filename in files:
                if filename.endswith(".md"):
                    filepath = os.path.join(root, filename)
                    recipe = extract_recipe_data(filepath, category)
                    
                    if recipe:
                        all_recipes.append(recipe)

    print(f"--- 处理完毕。共提取 {len(all_recipes)} 个有效菜谱。 ---")
    
    if all_recipes:
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(all_recipes, f, ensure_ascii=False, indent=4)
        print(f"--- 文件已保存至 {OUTPUT_JSON_PATH} ---")
        
    return all_recipes


if __name__ == "__main__":
    process_all_dishes()