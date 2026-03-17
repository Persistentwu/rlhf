import torch
import json
import os
import logging
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)

class MultiRoundDialogueDataset(Dataset):
    """
    电商多轮对话数据集
    数据格式:
    {
        "conversations": [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "问题2"},
            {"role": "assistant", "content": "回答2"}
        ]
    }
    特点：
    1. 只对assistant的回复计算损失
    2. user的问题部分标签设为-100，不参与损失计算
    3. 自动处理对话的拼接和标记
    """
    def __init__(self, tokenizer, max_len, data_dir, data_set_name, path_file=None, is_overwrite=False):
        """
        Args:
            tokenizer: 分词器（Qwen tokenizer）
            max_len: 数据最大长度
            data_dir: 缓存文件保存目录
            data_set_name: 数据集名称
            path_file: 原始数据文件路径
            is_overwrite: 是否重新生成缓存
        """
        self.tokenizer = tokenizer
        self.max_len = max_len  # 生成的最大长度 
        cached_feature_file = os.path.join(data_dir, f"cached_{data_set_name}_{max_len}")

        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info(f"加载缓存文件: {cached_feature_file}")
            self.data_set = torch.load(cached_feature_file)["data_set"]
        else:
            logger.info(f"生成缓存文件: {cached_feature_file}")
            self.data_set = self.load_data(path_file)
            torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):  # qwen3 model
        """加载多轮对话数据"""
        data_set = []
        with open(path_file, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(tqdm(fh, desc="加载数据")):
                sample = json.loads(line.strip())
                input_ids, labels, attention_mask = self.convert_feature(sample)
                if input_ids is None:
                    continue
                data_set.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask
                })
        return data_set  # 加载数据部分用于测试


    def convert_feature(self, sample):
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
            content_tokens = self.tokenizer.encode(content, add_special_tokens=False)
            
            if role == "user":
                # user的消息不计算损失
                input_ids.extend(content_tokens)
                labels.extend([-100] * len(content_tokens))
                input_ids.append(self.tokenizer.pad_token_id)
                labels.append(self.tokenizer.pad_token_id)
            elif role == "assistant":
                # assistant的消息计算损失
                input_ids.extend(content_tokens)
                labels.extend(content_tokens)
                input_ids.append(self.tokenizer.pad_token_id)
                labels.append(self.tokenizer.pad_token_id)
        
            

        # 检查长度
        if len(input_ids) > self.max_len:
            # 截断，优先保留后面的内容（更重要的对话轮次）
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]

        # 如果为空，则返回None
        if len(input_ids) == 0:
            return None, None, None

        attention_mask = [1] * len(input_ids) + [1]
        
        return input_ids, labels, attention_mask

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]


def collate_func(batch_data):
    """
    批处理函数
    支持variable length的输入和标签
    """
    batch_size = len(batch_data)
    if batch_size == 0:
        return {}

    input_ids_list = []
    labels_list = []
    attention_mask_list = []

    for instance in batch_data:
        input_ids_list.append(torch.tensor(instance["input_ids"], dtype=torch.long))
        labels_list.append(torch.tensor(instance["labels"], dtype=torch.long))
        attention_mask_list.append(torch.tensor(instance["attention_mask"], dtype=torch.long))

    # padding到批次中的最大长度
    input_ids = pad_sequence(
        input_ids_list, 
        batch_first=True, 
        padding_value=0  # Qwen tokenizer通常用0作为padding
    )
    labels = pad_sequence(
        labels_list,
        batch_first=True,
        padding_value=0  # 不计算损失的位置
    )
    attention_mask = pad_sequence(
        attention_mask_list,
        batch_first=True,
        padding_value=0
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

