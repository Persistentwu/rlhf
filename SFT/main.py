import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab
from functools import partial

api_key=os.environ.get("SWANLAB_api")
swanlab.login(api_key=api_key, save=True)

swanlab.config.update({
    "model": "Qwen/Qwen3-0.6B",
    })


MAX_LENGTH = 512
def process_func(example, tokenizer):
    """将数据集预处理"""
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(
        
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = (
        instruction["attention_mask"] + response["attention_mask"] + [1]
    )
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]


    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}



def convert_feature(sample, tokenizer):
    """
    将多轮对话转换为模型输入
    
    关键设计：
    1. 只对assistant的回复计算损失（通过标签设为回复的token_id）
    2. 将user的问题标记为-100，不计算梯度
    3. 在user和assistant之间可添加分隔符提高模型理解
    
    示例转换过程：
    输入: [
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答1"},
    ]
    
    输出:
    input_ids:    [问题1_tokens..., 回答1_tokens...]
    labels:       [-100, -100..., 回答1_tokens...]  <- 只有回答部分计算损失
    attention_mask: [1, 1, 1, ...]
    """
    conversations = sample.get("conversations", [])
    if not conversations:
        return None, None, None

    input_ids = []
    labels = []
    '''
    # 添加系统提示，无需添加提示
    system_prompt = sample.get("system", "")
    if system_prompt:
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=False)
        input_ids.extend(system_tokens)
        # 系统提示不计算损失
        labels.extend([-100] * len(system_tokens))
    '''
    # 处理每个对话轮次
    for i, message in enumerate(conversations):
        role = message.get("role", "")
        content = message.get("content", "")
        
        if not content:
            continue
        
        # tokenize内容
        content_tokens = tokenizer(content, add_special_tokens=False)["input_ids"]
        
        if role == "user":
            # user的消息不计算损失
            input_ids.extend(content_tokens)
            labels.extend([-100] * len(content_tokens))
            input_ids.append(tokenizer.pad_token_id)
            labels.append(tokenizer.pad_token_id)
            
        elif role == "assistant":
            # assistant的消息计算损失
            input_ids.extend(content_tokens)
            labels.extend(content_tokens)
            input_ids.append(tokenizer.pad_token_id)
            labels.append(tokenizer.pad_token_id)
    
        
    # 检查长度
    if len(input_ids) > MAX_LENGTH:
        # 截断，优先保留后面的内容（更重要的对话轮次）
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        attention_mask = [1] * len(input_ids)

    # 如果为空，则返回None
    if len(input_ids) == 0:
        return {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }

    attention_mask = [1] * len(input_ids) 
    
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# 从JSON Lines文件加载数据
def load_json_lines(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


if __name__ == "__main__":
    
    model_path = "../models"
    train_dataset_path = "./film/sft_train.json"
    test_dataset_path = "./film/sft_test.json"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map = "auto",trust_remote_code = True)

    model.enable_input_require_grads()
    train_data = load_json_lines(train_dataset_path)
    test_data = load_json_lines(test_dataset_path)

   #train_data = train_data[:16]
   # test_data = test_data[:16]
    # 创建Dataset对象
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    preprocess_func = partial(convert_feature, tokenizer=tokenizer)

    # 应用预处理
    train_dataset = train_dataset.map(
        preprocess_func,
        remove_columns=train_dataset.column_names,  # 移除原始列，只保留处理后的列
        num_proc=4  # 使用多进程加速处理
    )

    test_dataset = test_dataset.map(
        preprocess_func,
        remove_columns=test_dataset.column_names,
        num_proc=4
    )

    training_args = TrainingArguments(
        output_dir="../output_sft",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=2e-5,
        warmup_steps=100,
        logging_steps=20,
        save_steps=25,
        eval_strategy="steps",
        eval_steps=25,
        save_total_limit=2,
        fp16=False,
        bf16=True,
        load_best_model_at_end=True,
        report_to=["swanlab"],  # 使用swanlab记录训练过程
    )

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
        #tokenizer=tokenizer,
    )

    ## 开始训练
    trainer.train()
