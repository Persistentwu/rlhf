import json
import torch
import os
import random
from functools import partial
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# 配置参数
MAX_LENGTH = 1024
MODEL_PATH = "../model_1.7"
TRAIN_DATA_PATH = "./data/train.jsonl"
TEST_DATA_PATH = "./data/test.jsonl"

def build_chat_samples(conversations):
    """
    把一整段长对话 → 拆成多条训练样本（每一步都学）
    例如：
    U1 → A1 → U2 → A2 → U3 → A3
    会生成 3 条训练数据：
    1. [U1] → A1
    2. [U1, A1, U2] → A2
    3. [U1, A1, U2, A2, U3] → A3
    """
    samples = []
    history = []
    for msg in conversations:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            history.append({"role": "user", "content": content})
        elif role == "assistant":
            # 到 assistant 时，用当前历史 + 这条 assistant 作为一条训练样本
            train_chat = history + [{"role": "assistant", "content": content}]
            samples.append(train_chat)
            history.append({"role": "assistant", "content": content})
    
    return samples

def convert_feature(chat_list, tokenizer, max_length=MAX_LENGTH):
    """
    使用官方 apply_chat_template 构造训练序列
    修正版：正确计算 labels 位置
    """
    # 1. 分离输入和目标
    # chat_list 的最后一条是 assistant，前面的是 history
    input_messages = chat_list[:-1]  # 历史对话（不含当前 assistant）
    target_message = chat_list[-1]   # 当前 assistant 回答
    
    # 2. 处理输入部分（历史 + 当前问题）
    # add_generation_prompt=True 会添加 "<|im_start|>assistant\n" 前缀
    input_text = tokenizer.apply_chat_template(
        input_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. 处理目标部分（当前 assistant 回答）
    # 注意：目标部分需要加上 eos_token，让模型学会在回答结束时停止
    target_text = target_message["content"] + tokenizer.eos_token
    
    # 4. 分别 Tokenize
    tokenized_input = tokenizer(
        input_text, 
        add_special_tokens=False, 
        return_tensors="pt"
    )
    tokenized_target = tokenizer(
        target_text, 
        add_special_tokens=False, 
        return_tensors="pt"
    )
    
    input_ids = tokenized_input["input_ids"][0]
    target_ids = tokenized_target["input_ids"][0]
    
    input_len = len(input_ids)
    target_len = len(target_ids)
    
    # 5. 拼接 Input IDs
    full_input_ids = torch.cat([input_ids, target_ids], dim=-1)
    
    # 6. 截断处理（如果超过最大长度）
    if len(full_input_ids) > max_length:
        # 计算可以保留多少输入部分
        max_input_len = max_length - target_len
        
        if max_input_len <= 0:
            # 目标本身太长，只能截断目标
            full_input_ids = full_input_ids[-max_length:]
            input_len = 0
            target_len = len(full_input_ids)
        else:
            # 截断输入部分，保留尾部
            full_input_ids = torch.cat([input_ids[-max_input_len:], target_ids], dim=-1)
            input_len = max_input_len
    
    # 7. 构造 Labels
    # 初始化为 -100（忽略 loss）
    labels = torch.full_like(full_input_ids, -100)
    # 只对目标部分计算 loss
    if input_len < len(labels):
        labels[input_len:] = full_input_ids[input_len:]
    
    # 8. 构造 Attention Mask（全为 1）
    attention_mask = torch.ones_like(full_input_ids)
    
    return {
        "input_ids": full_input_ids.tolist(),
        "labels": labels.tolist(),
        "attention_mask": attention_mask.tolist()
    }

def load_json_lines(file_path):
    """加载 JSONL 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def prepare_all_samples(file_path):
    """从 JSONL 文件加载并拆分成训练样本"""
    raw_convs = load_json_lines(file_path)
    all_chats = []
    for item in raw_convs:
        convs = item["conversations"]
        chat_samples = build_chat_samples(convs)
        all_chats.extend(chat_samples)
    return all_chats

if __name__ == "__main__":
    # 加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.enable_input_require_grads()
    
    # 准备数据
    print("正在加载训练数据...")
    train_chats = prepare_all_samples(TRAIN_DATA_PATH)
    test_chats = prepare_all_samples(TEST_DATA_PATH)
    
    print(f"训练样本数: {len(train_chats)}")
    print(f"测试样本数: {len(test_chats)}")
    
    # 创建 Dataset
    train_dataset = Dataset.from_list([{"chat": c} for c in train_chats])
    test_dataset = Dataset.from_list([{"chat": c} for c in test_chats])
    
    # 预处理
    print("正在预处理数据...")
    def preprocess(sample):
        return convert_feature(sample["chat"], tokenizer=tokenizer)
    
    train_dataset = train_dataset.map(
        preprocess,
        remove_columns=["chat"],
        num_proc=4
    )
    test_dataset = test_dataset.map(
        preprocess,
        remove_columns=["chat"],
        num_proc=4
    )
    
    # 打乱训练数据
    train_dataset = train_dataset.shuffle(seed=42)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="../output_sft",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        logging_steps=20,
        save_steps=400,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to=["swanlab"],
        run_name="qwen3-multi-chat-sft",
        load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # save_optimizer_state=False
    )
    
    # Data Collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
        
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    trainer.save_model("../output_sft/qwen_sft_final")
    tokenizer.save_pretrained("../output_sft/qwen_sft_final")
    
    print("训练完成！")