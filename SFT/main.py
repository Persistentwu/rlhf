import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from functools import partial

# ==================== SwanLab 配置 ====================
api_key = os.environ.get("SWANLAB_api")
swanlab.login(api_key=api_key, save=True)
swanlab.config.update({
    "model": "Qwen/Qwen3-1.7B",
})

# ==================== 全局参数 ====================
MAX_LENGTH = 512

# ==================== 数据处理核心函数 ====================
def convert_feature(sample, tokenizer):
    """
    使用 apply_chat_template 保证训练和推理格式严格一致。
    关键设计：只对最后一轮 assistant 的回复计算 loss。
    """
    conversations = sample.get("conversations", [])
    if not conversations:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    # 1. 使用官方模板拼接对话 (不加特殊token，因为模板里自带了)
    text = tokenizer.apply_chat_template(
        conversations, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # 2. 对整段文本进行 tokenize
    tokenized_full = tokenizer(text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)
    input_ids = tokenized_full["input_ids"]
    
    # 3. 构造 labels，默认全部设为 -100 (不计算损失)
    labels = [-100] * len(input_ids)
    
    # 获取 "<|im_start|>assistant\n" 的 token 序列，用于定位最后一轮回答的起始位置
    assistant_token_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    
    # 倒序寻找最后一次出现 assistant 标记的位置
    start_idx = -1
    for i in range(len(input_ids) - len(assistant_token_ids), -1, -1):
        if input_ids[i:i+len(assistant_token_ids)] == assistant_token_ids:
            start_idx = i + len(assistant_token_ids) # 跳过标记本身，只保留回复内容
            break
            
    # 如果找到了 assistant 回复，将该部分的 labels 设为真实的 token_id
    if start_idx != -1:
        for i in range(start_idx, len(input_ids)):
            labels[i] = input_ids[i] 
            
    attention_mask = [1] * len(input_ids)
    
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def load_json_lines(file_path):
    """从 JSON Lines 文件加载数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ==================== 主函数 ====================
if __name__ == "__main__":
    model_path = "../model_1.7"
    train_dataset_path = "./data/train_qwen.jsonl"
    test_dataset_path = "./data/test_qwen.jsonl"

    print("正在加载 Tokenizer 和模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    model.enable_input_require_grads() # 开启输入梯度，某些加速方法需要

    print("正在加载和处理数据集...")
    train_data = load_json_lines(train_dataset_path)
    test_data = load_json_lines(test_dataset_path)

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    # 使用 partial 固定 tokenizer 参数
    preprocess_func = partial(convert_feature, tokenizer=tokenizer)

    # 多进程应用预处理
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

    print("配置训练参数...")
    training_args = TrainingArguments(
        output_dir="../output_sft",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=20,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        # load_best_model_at_end=True,
        report_to=["swanlab"],
    )

    # DataCollator 会自动将 batch 内的序列 pad 到同等长度，并在 labels 中将 padding 部分设为 -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    print("开始训练...")
    trainer.train()
    print("训练完成！")
