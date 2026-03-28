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
    chat_list 是一个完整的对话片段，格式为：
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...,
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}  # 最后一条是 assistant
    ]
    """
    # 分离：前 n-1 条作为输入，最后一条作为目标
    input_messages = chat_list[:-1]
    target_message = chat_list[-1]
    
    # 1. 构造输入部分的文本（不添加 generation prompt）
    input_text = tokenizer.apply_chat_template(
        input_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # 2. 构造目标部分的文本：单独的 assistant 消息（包括其 role 标记）
    target_text = tokenizer.apply_chat_template(
        [target_message],
        tokenize=False,
        add_generation_prompt=False
    )
    
    target_text = target_text + tokenizer.eos_token
    # 3. 分别 tokenize
    input_ids = tokenizer.encode(input_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    
    # 4. 拼接
    full_input_ids = input_ids + target_ids
    
    # 5. 截断
    if len(full_input_ids) > max_length:
        # 优先保留目标部分
        if len(target_ids) >= max_length:
            full_input_ids = target_ids[-max_length:]
            input_len = 0
        else:
            max_input_len = max_length - len(target_ids)
            input_ids = input_ids[-max_input_len:]
            full_input_ids = input_ids + target_ids
            input_len = len(input_ids)
    else:
        input_len = len(input_ids)
    
    # 6. 构造 labels：只对目标部分计算损失
    labels = [-100] * len(full_input_ids)
    if input_len < len(full_input_ids):
        labels[input_len:] = full_input_ids[input_len:]
    
    return {
        "input_ids": full_input_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_input_ids)
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
    # trainer.save_model("../output_sft/qwen_sft_final")
    # tokenizer.save_pretrained("../output_sft/qwen_sft_final")
    
    print("训练完成！")