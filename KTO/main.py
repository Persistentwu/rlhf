import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorWithPadding
)
from trl import KTOConfig, KTOTrainer
import swanlab
import json
import os
from copy import deepcopy
from peft import LoraConfig, get_peft_model
import random
from datasets import Dataset as HFDataset  # 导入Hugging Face的Dataset类

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# ==========================================
# 1. 配置与初始化
# ==========================================
api_key = os.environ.get("SWANLAB_api")
if api_key:
    swanlab.login(api_key=api_key, save=True)
    swanlab.config.update({
        "model": "Qwen_KTO/Qwen3-1.7B",
    })
    swanlab.init()

class Config:
    model_name = "../output_sft/checkpoint-1086"  # SFT后的基础模型
    data_path = "qwen_processed_samples.json"
    output_dir = "../kto_models"

    device = "cuda:0"
    
    learning_rate = 5e-7
    batch_size = 4  # KTO通常需要更小的batch size
    gradient_accumulation_steps = 2
    
    # KTO 特定参数
    max_length = 512
    max_completion_length = 128  # KTO生成回答的长度
    temperature = 0.7
    top_p = 0.9
    repetition_penalty = 1.1
    
    # KTO 核心参数
    beta = 1.0  # KL惩罚系数，控制与参考模型的偏离程度
    desirable_weight = 1.0  # 正面样本的权重
    undesirable_weight = 1.0  # 负面样本的权重，表示对损失的厌恶程度更高
    
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.2
    num_train_epochs = 4
    max_grad_norm = 0.5
    
    # LoRA配置（可选，用于高效训练）
    use_lora = True
    lora_r = 4
    lora_alpha = 16
    lora_dropout = 0.2

# ==========================================
# 2. 数据处理
# ==========================================
def load_kto_dataset(data_path, tokenizer, shuffle=True):
    # 加载原始数据
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)  # 直接加载整个JSON数组
    
    # 转换为KTO格式
    kto_data = []
    for item in raw_data:
        # KTO数据格式: {"question": "...", "answer": "...", "label": 0/1}
        question = item["question"]
        answer = item["answer"]
        label = item["label"]  # 0表示负面样本（不想要的），1表示正面样本（想要的）
        
        # 构建prompt
        messages = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        kto_data.append({
            "prompt": prompt_text,
            "completion": answer,
            "label": label,  # KTO直接使用这个标签
        })
    
    # 创建Hugging Face Dataset
    dataset = HFDataset.from_list(kto_data)
    
    # 如果需要打乱数据
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    
    return dataset

# ==========================================
# 3. 主训练流程
# ==========================================
def main():
    config = Config()
    
    torch.cuda.empty_cache()
    
    print(f"Available devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print(f"Loading model on {config.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": config.device}
    )
    
    # 可选：使用LoRA进行高效训练
    if config.use_lora:
        print("Applying LoRA to the model...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 准备数据集
    dataset = load_kto_dataset(config.data_path, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    # 统计正负样本数量
    positive_count = sum(1 for item in dataset if item["label"] == 1)
    negative_count = sum(1 for item in dataset if item["label"] == 0)
    print(f"Positive samples: {positive_count}, Negative samples: {negative_count}")
    
    total_steps = (len(dataset) // (config.batch_size * config.gradient_accumulation_steps)) * config.num_train_epochs
    # KTO 配置
    kto_config = KTOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=int(total_steps * config.warmup_ratio),
        max_grad_norm=config.max_grad_norm,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        save_strategy="steps",
        remove_unused_columns=False,
        max_length=config.max_length,
        beta=config.beta,  # KL惩罚系数
        desirable_weight=config.desirable_weight,  # 正面样本权重
        undesirable_weight=config.undesirable_weight,  # 负面样本权重
        report_to=["swanlab"],
        gradient_checkpointing=True,
        bf16=True,
        dataloader_drop_last=False,  # 避免丢弃最后一批数据
    )
    
    print("Initializing KTO Trainer...")
    print(f"KTO配置: beta={config.beta}, desirable_weight={config.desirable_weight}, undesirable_weight={config.undesirable_weight}")
    
    # 创建KTO Trainer - 注意：KTO不需要reward_funcs参数
    kto_trainer = KTOTrainer(
        model=model,
        args=kto_config,
        train_dataset=dataset,
        processing_class=tokenizer,  # 使用processing_class而不是tokenizer
    )
    
    # 执行训练
    print("Starting KTO training...")
    kto_trainer.train()
    
    # 保存模型
    # print("Training Complete. Saving models...")
    # kto_trainer.save_model(os.path.join(config.output_dir, "final_kto_model"))
    # tokenizer.save_pretrained(os.path.join(config.output_dir, "final_kto_model"))
    
    # print(f"Model saved to {os.path.join(config.output_dir, 'final_kto_model')}")

if __name__ == "__main__":
    main()
