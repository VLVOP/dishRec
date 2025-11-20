import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- 配置常量 ---
MODEL_NAME = "Qwen/Qwen1.5-1.8B"
OUTPUT_DIR = "sft_qlora_dish_recommender"
TRAIN_DATA_PATH = "src/data/sft_train.jsonl"
VAL_DATA_PATH = "src/data/sft_val.jsonl"
MAX_SEQ_LENGTH = 512

# --- 数据处理函数 ---
def formatting_prompts_func(example):
    """
    将 JSONL 中的 instruction/input/output 格式化为模型可训练的对话格式
    注意: 在 trl 0.25.1 中，formatting_func 接收单个样本，不是批次
    """
    # 对话模板
    text = f"### 用户指令:\n{example['instruction']}\n\n"
    text += f"### 历史菜谱:\n{example['input']}\n\n"
    text += f"### 模型推荐:\n{example['output']}"
    
    # EOS标记
    return text + "<|endoftext|>"

def train_sft():
    # --- 1. BitsAndBytes 配置 (4-bit 量化) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # --- 2. 加载 Model 和 Tokenizer ---
    print(f"--- 正在加载模型和分词器: {MODEL_NAME} ---")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 3. LoRA 配置 ---
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # --- 4. 加载数据 ---
    print("--- 正在加载和处理数据集 ---")
    raw_datasets = load_dataset(
        "json", 
        data_files={
            "train": TRAIN_DATA_PATH, 
            "validation": VAL_DATA_PATH
        }
    )

    # --- 5. 设置训练参数 (使用 SFTConfig) ---
    print("--- 正在设置训练参数 ---")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,  # 降低 batch size
        per_device_eval_batch_size=1,   # 降低 batch size
        gradient_accumulation_steps=8,  # 增加累积步数保持有效 batch size
        optim="paged_adamw_32bit",
        
        max_steps=50,
        save_steps=50,
        logging_steps=10,
        
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=3,
        weight_decay=0.01,
        
        eval_strategy="steps",
        eval_steps=50,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        
        # SFT 特定参数
        max_length=MAX_SEQ_LENGTH,  # 新版本使用 max_length
        packing=False,
        dataloader_pin_memory=False,  # 禁用 pin_memory 以消除警告
    )

    # --- 6. 实例化 SFT Trainer 并开始训练 ---
    print("--- 实例化 SFT Trainer 并开始训练 ---")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["validation"],
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
    )

    print("--- Trainer 初始化完成，开始训练... ---")
    print(f"训练样本数: {len(raw_datasets['train'])}")
    print(f"验证样本数: {len(raw_datasets['validation'])}")
    print(f"总训练步数: {training_args.max_steps}")
    print("--- 第一步可能需要较长时间进行编译，请耐心等待... ---")
    
    trainer.train()

    # --- 7. 保存模型 ---
    print(f"--- 训练完成，正在保存 LoRA 适配器至 {OUTPUT_DIR}/final_adapter ---")
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    
    print("--- 训练完成！---")

if __name__ == "__main__":
    logging.set_verbosity_warning()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_sft()