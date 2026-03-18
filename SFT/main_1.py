'''
使用qwen message消息格式
'''

import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from functools import partial

# [可选] 登录 SwanLab
api_key = os.environ.get("SWANLAB_api")

PROMPT = "你是一个电影知识回答专业助手，提供流畅自然的多轮对话"
MAX_LENGTH = 1024

swanlab.config.update({
    "model": "Qwen/Qwen3-0.6B",
    "system_prompt": PROMPT,
    "data_max_length": MAX_LENGTH,
})

def convert_feature(sample, tokenizer, max_length=MAX_LENGTH, system_prompt=PROMPT):
    """
    将多轮对话转换为模型输入
    
    关键设计：
    1. 只对assistant的回复内容计算损失。
    2. 将user的问题、system提示以及assistant的角色标记设为-100，不计算梯度。
    3. 严格按照Qwen Chat格式拼接。
    
    对话格式示例：
    <|im_start|>system\n{system_prompt}<|im_end|>\n
    <|im_start|>user\n{user_content}<|im_end|>\n
    <|im_start|>assistant\n{assistant_content}<|im_end|>\n
    """
    input_ids = []
    labels = []
    
    # 1. 添加 System Prompt
    # 格式: <|im_start|>system\n{PROMPT}<|im_end|>\n
    system_tokens = tokenizer(
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n",
        add_special_tokens=False
    )["input_ids"]
    input_ids.extend(system_tokens)
    # System Prompt 不计算损失
    labels.extend([-100] * len(system_tokens))
    
    # 2. 处理多轮对话
    conversations = sample.get("conversations", [])
    for message in conversations:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if not content:
            continue
            
        if role == "user":
            # User 输入
            # 格式: <|im_start|>user\n{content}<|im_end|>\n
            user_tokens = tokenizer(
                f"<|im_start|>user\n{content}<|im_end|>\n",
                add_special_tokens=False
            )["input_ids"]
            input_ids.extend(user_tokens)
            # User 输入不计算损失
            labels.extend([-100] * len(user_tokens))
            
        elif role == "assistant":
            # Assistant 回复
            # 格式: <|im_start|>assistant\n{content}<|im_end|>\n
            # 注意：我们需要把回复拆分为 header 和 content，以便只对 content 计算 loss
            
            # 获取完整 tokens (包含 header)
            full_response_tokens = tokenizer(
                f"<|im_start|>assistant\n{content}<|im_end|>\n",
                add_special_tokens=False
            )["input_ids"]
            
            # 获取纯内容 tokens (不包含 header)
            # 这里假设 tokenizer 对 "<|im_start|>assistant\n" 的编码是固定的
            # 为了更稳健，我们单独 tokenize content，然后计算 header 长度
            content_tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
            
            # Header 长度 = 总长度 - 内容长度 - 结束符长度(<|im_end|>\n)
            # 但更简单的做法是：先追加完整 tokens，然后把 header 部分的 labels 设为 -100
            
            # 计算部分：header 是 "<|im_start|>assistant\n"
            # 我们可以手动 tokenize header 来获取精确长度
            header_text = "<|im_start|>assistant\n"
            header_tokens = tokenizer(header_text, add_special_tokens=False)["input_ids"]
            
            # 拼接 input_ids
            input_ids.extend(full_response_tokens)
            
            # 拼接 labels
            # Header 部分 mask 掉
            labels.extend([-100] * len(header_tokens))
            # Content 部分保留计算 loss
            labels.extend(content_tokens)
            # 结束符部分 (<|im_end|>\n) mask 掉
            # 结束符长度 = len(full_response_tokens) - len(header_tokens) - len(content_tokens)
            eos_len = len(full_response_tokens) - len(header_tokens) - len(content_tokens)
            labels.extend([-100] * eos_len)

    # 3. 截断处理
    # 如果超过最大长度，进行截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        
    # 4. 生成 Attention Mask
    attention_mask = [1] * len(input_ids)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

# 从JSON Lines文件加载数据
def load_json_lines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    # 模型路径
    model_path = "../models" 
    
    # 数据路径
    train_dataset_path = "./data/sft_train.json"
    test_dataset_path = "./data/sft_test.json"

    # 加载 Tokenizer
    # use_fast=False 对于 Qwen 来说通常更稳定
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    
    # 关键修复：显式设置 pad_token
    # Qwen 等 LLM 通常没有单独的 pad_token，通常使用 eos_token 作为 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, # 建议与训练时的精度保持一致
        trust_remote_code=True
    )

    model.enable_input_require_grads() # 开启梯度检查点时需要
    
    # 加载数据
    train_data = load_json_lines(train_dataset_path)
    test_data = load_json_lines(test_dataset_path)

    # 创建Dataset对象
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    # 使用 partial 固定 tokenizer 和其他参数
    preprocess_func = partial(convert_feature, tokenizer=tokenizer)

    # 应用预处理
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

    training_args = TrainingArguments(
        output_dir="../output_sft/qwen_sft",
        per_device_train_batch_size=1,  # 根据显存调整
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8, # 增加梯度累积以模拟大 batch size
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=20,
        save_steps=400,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        report_to=["swanlab"],
        run_name="qwen3-sft-medical",
        gradient_checkpointing=True, # 开启梯度检查点以节省显存
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True
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
    
    # 训练结束
    swanlab.finish()
