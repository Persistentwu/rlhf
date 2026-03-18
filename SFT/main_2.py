import json
import torch
import os
import swanlab
from functools import partial
from datasets import Dataset
from modelscope import snapshot_download
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)

# 配置参数
PROMPT = "你是一个电影知识回答专业助手，提供流畅自然的多轮对话"
MAX_LENGTH = 1024
MODEL_PATH = "../model_1.7"  # 请确保路径正确
TRAIN_DATA_PATH = "./film/sft_train.json"
TEST_DATA_PATH = "./film/sft_test.json"

def convert_feature(sample, tokenizer, max_length=MAX_LENGTH, system_prompt=PROMPT):
    """
    核心逻辑：将整场多轮对话拼接为一个序列，仅对 assistant 的 content 部分计算 Loss
    """
    input_ids = []
    labels = []
    
    # 1. 编码 System Prompt
    # 格式: <|im_start|>system\n{PROMPT}<|im_end|>\n
    system_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    system_ids = tokenizer.encode(system_text, add_special_tokens=False)
    input_ids.extend(system_ids)
    labels.extend([-100] * len(system_ids)) # System 部分不计入 Loss

    # 2. 迭代处理多轮对话
    conversations = sample.get("conversations", [])
    for message in conversations:
        role = message.get("role", "")
        content = message.get("content", "")
        if not content:
            continue
            
        if role == "user":
            # 格式: <|im_start|>user\n{content}<|im_end|>\n
            user_text = f"<|im_start|>user\n{content}<|im_end|>\n"
            user_ids = tokenizer.encode(user_text, add_special_tokens=False)
            input_ids.extend(user_ids)
            labels.extend([-100] * len(user_ids)) # User 部分不计入 Loss
            
        elif role == "assistant":
            # Assistant 需拆分为：Header(不学) + Content(学) + End(学，用于学习停止)
            header_text = "<|im_start|>assistant\n"
            content_text = f"{content}<|im_end|>\n"
            
            header_ids = tokenizer.encode(header_text, add_special_tokens=False)
            content_ids = tokenizer.encode(content_text, add_special_tokens=False)
            
            input_ids.extend(header_ids + content_ids)
            # 重点：Header 部分设为 -100，Content 部分保留原始 ID
            labels.extend([-100] * len(header_ids) + content_ids)

    # 3. 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

def load_json_lines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

if __name__ == "__main__":
    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型 (使用 bfloat16 节省显存并保持精度)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    model.enable_input_require_grads() 

    # 数据准备
    # 注意：请确保你的 json 文件中每一行是一个完整的 "conversations" 列表
    train_data = load_json_lines(TRAIN_DATA_PATH)
    test_data = load_json_lines(TEST_DATA_PATH)

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    preprocess_func = partial(convert_feature, tokenizer=tokenizer)

    train_dataset = train_dataset.map(
        preprocess_func,
        remove_columns=train_dataset.column_names,
        num_proc=4
    )
    test_dataset = test_dataset.map(
        preprocess_func,
        remove_columns=test_dataset.column_names,
        num_proc=4
    )

    # 训练参数配置
    training_args = TrainingArguments(
        output_dir="../output_sft/qwen_sft",
        per_device_train_batch_size=1,  # Qwen3-0.6B 很小，可以适当调大
        gradient_accumulation_steps=4, 
        num_train_epochs=2,             # 多轮对话建议训练 3 轮左右以充分收敛
        learning_rate=3e-5,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        bf16=True,                      # 强力推荐使用 bf16
        gradient_checkpointing=True,
        save_strategy='no',
        report_to=["swanlab"],
        run_name="qwen3-1.7B"
    )

    # 使用 DataCollator 处理 Padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100 # 确保 Padding 部分不计算 Loss
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    # 开始训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model("../output_sft/qwen_sft")
    tokenizer.save_pretrained("../output_sft/qwen_sft")
    
    swanlab.finish()