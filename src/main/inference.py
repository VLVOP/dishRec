import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- 配置 ---
BASE_MODEL_NAME = "Qwen/Qwen1.5-1.8B"
ADAPTER_PATH = "sft_qlora_dish_recommender/final_adapter"  # LoRA 适配器路径

def load_model_and_tokenizer():
    """加载基础模型和 LoRA 适配器"""
    print("--- 正在加载模型和分词器 ---")
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载 LoRA 适配器
    print(f"--- 正在加载 LoRA 适配器: {ADAPTER_PATH} ---")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    
    print("--- 模型加载完成！---\n")
    return model, tokenizer

def generate_response(model, tokenizer, instruction, history="", max_length=512):
    """生成模型回复"""
    # 构建输入文本（与训练时的格式一致）
    prompt = f"### 用户指令:\n{instruction}\n\n### 历史菜谱:\n{history}\n\n### 模型推荐:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解码输出
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取模型推荐部分（去掉 prompt）
    response = full_output.split("### 模型推荐:\n")[-1].strip()
    
    return response

def interactive_chat():
    """交互式对话"""
    model, tokenizer = load_model_and_tokenizer()
    
    print("=" * 60)
    print("🍽️  菜谱推荐助手已启动！")
    print("=" * 60)
    print("输入你的需求，我会为你推荐菜谱！")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清空历史记录")
    print("=" * 60)
    print()
    
    history = ""
    
    while True:
        # 获取用户输入
        user_input = input("👤 你: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("\n👋 再见！")
            break
        
        if user_input.lower() in ['clear', '清空']:
            history = ""
            print("\n✅ 历史记录已清空\n")
            continue
        
        # 生成回复
        print("\n🤖 助手正在思考...\n")
        response = generate_response(model, tokenizer, user_input, history)
        
        print(f"🤖 助手: {response}\n")
        print("-" * 60)
        print()
        
        # 更新历史（可选：如果想让模型记住之前的对话）
        # history += f"{user_input}\n{response}\n"

def single_query(instruction, history=""):
    """单次查询模式"""
    model, tokenizer = load_model_and_tokenizer()
    response = generate_response(model, tokenizer, instruction, history)
    print(f"指令: {instruction}")
    print(f"历史: {history}")
    print(f"推荐: {response}")
    return response

if __name__ == "__main__":
    interactive_chat()