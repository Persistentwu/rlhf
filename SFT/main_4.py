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
    改进版：基于 Token 索引切片构造 Labels，避免字符串匹配的不稳定性
    """
    # 1. 分离输入和目标
    # chat_list 的最后一条是 assistant，前面的是 history (包含 user)
    input_messages = chat_list[:-1]
    target_message_content = chat_list[-1]["content"]
    
    # 2. 处理输入部分
    # add_generation_prompt=True 会加上类似 "<|im_start|>assistant\n" 的前缀，
    # 这正是我们需要的，因为我们马上要接 target_message_content
    input_text = tokenizer.apply_chat_template(
        input_messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 3. 分别 Tokenize
    # 注意：这里不需要 add_special_tokens，因为 template 已经处理了
    tokenized_input = tokenizer(input_text, add_special_tokens=False, return_tensors="pt")
    tokenized_target = tokenizer(target_message_content, add_special_tokens=False, return_tensors="pt")

    input_ids = tokenized_input["input_ids"][0]
    target_ids = tokenized_target["input_ids"][0]

    # 4. 拼接 Input IDs
    # 将输入和目标拼在一起，形成完整的序列
    full_input_ids = torch.cat([input_ids, target_ids], dim=-1)

    # 5. 截断处理
    # 如果超过最大长度，优先截断输入部分，保留目标部分
    if len(full_input_ids) > max_length:
        # 计算保留目标部分需要的长度
        target_len = len(target_ids)
        # 计算输入部分允许的最大长度
        max_input_len = max_length - target_len
        # 如果输入部分太长，只保留尾部
        if max_input_len < 0:
            # 极端情况：目标本身太长，只能截断目标
            full_input_ids = full_input_ids[-max_length:]
            input_len = 0
        else:
            full_input_ids = torch.cat([input_ids[-max_input_len:], target_ids], dim=-1)
            input_len = max_input_len
    else:
        input_len = len(input_ids)

    # 6. 构造 Labels
    # 创建全为 -100 的 labels (忽略 Loss)
    labels = torch.full_like(full_input_ids, -100)
    # 将目标部分的 labels 设为对应的 input_ids (计算 Loss)
    # 注意：这里我们只学习 target_message_content 的内容
    if input_len < len(labels):
        labels[input_len:] = full_input_ids[input_len:]

    # 7. 构造 Attention Mask
    attention_mask = torch.ones_like(full_input_ids)

    return {
        "input_ids": full_input_ids.tolist(),
        "labels": labels.tolist(),
        "attention_mask": attention_mask.tolist()
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

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.enable_input_require_grads()

    # ----------------------
    # 数据：长对话 → 拆成多条训练样本
    # ----------------------
    def prepare_all_samples(file_path):
        raw_convs = load_json_lines(file_path)
        all_chats = []
        for item in raw_convs:
            convs = item["conversations"]
            chat_samples = build_chat_samples(convs)
            all_chats.extend(chat_samples)
        return all_chats


    train_chats = prepare_all_samples(TRAIN_DATA_PATH)
    test_chats = prepare_all_samples(TEST_DATA_PATH)

    train_dataset = Dataset.from_list([{"chat": c} for c in train_chats])
    test_dataset = Dataset.from_list([{"chat": c} for c in test_chats])

    # 预处理
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

    import random
    
    # 将 Dataset 转换为列表进行 shuffle（对于大数据集这步可能比较慢）
    train_data_list = train_dataset.to_list()
    random.shuffle(train_data_list)
    
    # 将打乱后的列表重新转回 Dataset
    train_dataset = Dataset.from_list(train_data_list)
    # 训练参数
    training_args = TrainingArguments(
        output_dir="../output_sft",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        logging_steps=20,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=400,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        # save_strategy='no', # 修改为 steps 以配合 save_steps
        report_to=["swanlab"],
        run_name="qwen3-multi-chat-sft",
        
        
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model("../output_sft/qwen_sft_final")
    tokenizer.save_pretrained("../output_sft/qwen_sft_final")

    swanlab.finish()
