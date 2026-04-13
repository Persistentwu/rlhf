import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorWithPadding
)
from trl import DPOConfig, DPOTrainer
import swanlab
import json
import os
from copy import deepcopy
from peft import LoraConfig, get_peft_model
import random
from datasets import Dataset as HFDataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# ==========================================
# 1. 配置与初始化
# ==========================================
api_key = os.environ.get("SWANLAB_api")
if api_key:
    swanlab.login(api_key=api_key, save=True)
    swanlab.config.update({
        "model": "Qwen_DPO/Qwen3-1.7B",
    })
    swanlab.init()

class Config:
    model_name = "../output_sft/checkpoint-1086"  # SFT后的基础模型
    data_path = "qwen_dpo_samples.json"  # DPO格式的数据
    output_dir = "../dpo_models"

    device = "cuda:0"
    
    learning_rate = 1e-6
    batch_size = 2  # DPO通常需要更小的batch size，因为每个样本包含两份completion
    gradient_accumulation_steps = 4
    
    # DPO 特定参数
    max_length = 512
    max_prompt_length = 256  # prompt的最大长度
    max_completion_length = 128  # completion的最大长度
    
    # DPO 核心参数
    beta = 0.05  # KL惩罚系数，控制与参考模型的偏离程度
    loss_type = "sigmoid"  # 可选: "sigmoid", "hinge", "ipo", "exo_pair", "nca_pair", "robust", "bco_pair", "sppo_hard", "aot", "apo_zero", "apo_down", "discopop"
    
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.2
    num_train_epochs = 5
    max_grad_norm = 0.5
    
    # LoRA配置（可选，用于高效训练）
    use_lora = True
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    
    # 参考模型配置
    ref_model_name = None  # 如果为None，则使用初始模型作为参考模型

# ==========================================
# 2. 数据处理 - DPO格式
# ==========================================
def load_dpo_dataset(data_path, tokenizer, shuffle=True):
    """
    加载DPO格式的数据集
    期望的数据格式: 
    [
        {
            "question": "电影《卢旺达饭店》的导演是谁？",
            "chosen": "电影《卢旺达饭店》的导演是特瑞·乔治（Terry George）。他是一位爱尔兰导演和编剧，这部电影是他执导的代表作之一。",
            "rejected": "电影《卢旺达饭店》的导演是史蒂文·斯皮尔伯格。他是一位美国著名导演，以拍摄《辛德勒的名单》等历史题材影片而闻名。"
        },
        ...
    ]
    """
    # 加载原始数据
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 转换为DPO格式
    dpo_data = []
    for item in raw_data:
        question = item["question"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # 构建prompt（对话格式）
        prompt_messages = [{"role": "user", "content": question}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 构建chosen和rejected的完整对话
        chosen_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": chosen}
        ]
        rejected_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": rejected}
        ]
        
        chosen_text = tokenizer.apply_chat_template(
            chosen_messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        dpo_data.append({
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        })
    
    # 创建Hugging Face Dataset
    dataset = HFDataset.from_list(dpo_data)
    
    # 如果需要打乱数据
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    
    return dataset

def load_dpo_dataset_conversational(data_path, tokenizer, shuffle=True):
    """
    加载对话格式的DPO数据集（推荐格式）
    这种格式更高效，因为不需要重复处理prompt部分
    """
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    dpo_data = []
    for item in raw_data:
        question = item["question"]
        chosen = item["chosen"]
        rejected = item["rejected"]
        
        # 使用对话格式（更高效）
        dpo_data.append({
            "prompt": [{"role": "user", "content": question}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    
    dataset = HFDataset.from_list(dpo_data)
    
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
    tokenizer.padding_side = "left"  # DPO需要使用left padding
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    print(f"Loading model on {config.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": config.device}
    )
    
    # 可选：加载参考模型（如果不指定，DPOTrainer会自动创建）
    ref_model = None
    if config.ref_model_name:
        print(f"Loading reference model: {config.ref_model_name}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.ref_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": config.device}
        )
    
    # 准备LoRA配置
    peft_config = None
    if config.use_lora:
        print("Preparing LoRA configuration...")
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        # 注意：不要在这里应用LoRA，让DPOTrainer来处理
    
    # 准备数据集 - 使用对话格式（推荐）
    dataset = load_dpo_dataset_conversational(config.data_path, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    # 打印数据样例
    print("\nSample data:")
    sample = dataset[0]
    print(f"Prompt: {sample['prompt']}")
    print(f"Chosen: {sample['chosen']}")
    print(f"Rejected: {sample['rejected']}")
    
    total_steps = (len(dataset) // (config.batch_size * config.gradient_accumulation_steps)) * config.num_train_epochs
    
    # DPO 配置
    dpo_config = DPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=int(total_steps * config.warmup_ratio),
        max_grad_norm=config.max_grad_norm,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        save_strategy="steps",
        remove_unused_columns=False,
        
        # DPO特定参数
        max_length=config.max_length,
        beta=config.beta,
        loss_type=config.loss_type,
        
        # 其他训练参数
        report_to=["swanlab"],
        gradient_checkpointing=True,
        bf16=True,
        dataloader_drop_last=False,
        
        # 模型初始化参数
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    )
    
    
    # 创建DPO Trainer
    print("Initializing DPO Trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # 如果为None，Trainer会自动创建
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,  # 传递peft_config而不是已经应用LoRA的模型
    )
    
    # 执行训练
    print("Starting DPO training...")
    dpo_trainer.train()
    
    # 保存最终模型
    # print("Training complete. Saving final model...")
    # final_model_path = os.path.join(config.output_dir, "final_dpo_model")
    # dpo_trainer.save_model(final_model_path)
    # tokenizer.save_pretrained(final_model_path)
    # print(f"Model saved to {final_model_path}")



if __name__ == "__main__":
    main()