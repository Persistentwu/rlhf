import logging
import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

class RWDataSet(Dataset):
    """多头奖励模型所需的数据类"""
    def __init__(self, tokenizer, max_len:int, query_max_len:int, data_dir, data_set_name, path_file = None, is_overwrite = False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_query_len = query_max_len
        
        # 检查和设置特殊 token ID
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        
        # 如果特殊token不存在，使用备用方案
        if self.cls_token_id is None:
            # 使用 bos_token_id 或 padding_token_id
            self.cls_token_id = tokenizer.bos_token_id or tokenizer.pad_token_id or 101
            logger.info(f"cls_token_id 为 None，使用替代值: {self.cls_token_id}")
        
        if self.sep_token_id is None:
            # 使用 eos_token_id 或其他
            self.sep_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 102
            logger.info(f"sep_token_id 为 None，使用替代值: {self.sep_token_id}")

        cached_feature_file = os.path.join(data_dir, f"cached_{data_set_name}_{max_len}")
        # 强制重新处理数据（避免缓存中的坏数据）
        if os.path.exists(cached_feature_file):
            os.remove(cached_feature_file)
            logger.info(f"删除旧缓存文件: {cached_feature_file}")
        
        logger.info(f"预处理原始数据: {path_file}")
        self.data_set = self.load_data(path_file)
        logger.info(f"保存缓存文件: {cached_feature_file}")
        torch.save({"data_set": self.data_set}, cached_feature_file)

    def load_data(self, path_file):
        data_set = []
        skipped_count = 0
        with open(path_file, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(tqdm(fh, desc="Processing Data")):
                if not line.strip(): 
                    continue
                try:
                    sample = json.loads(line.strip())
                    
                    # 核心逻辑：将 1个正样本 + 4个维度的负样本 索引化
                    input_ids, attention_mask = self.convert_feature(sample)
                    if input_ids is not None and attention_mask is not None:
                        data_set.append({"input_ids": input_ids, "attention_mask": attention_mask})
                    else:
                        skipped_count += 1
                except Exception as e:
                    logger.warning(f"处理第 {idx} 行时出错: {e}")
                    skipped_count += 1
        
        logger.info(f"数据加载完成: 成功 {len(data_set)}, 跳过 {skipped_count}")
        return data_set

    def convert_feature(self, sample):
        """
        处理单条 JSON 数据：提取 1 Chosen + 4 Rejected
        """
        prompt_text = sample.get("prompt", "")
        if not prompt_text:
            logger.debug("空 prompt，跳过")
            return None, None
            
        # 我们按照固定顺序提取，方便后面模型 Head 的对应：
        # 索引 0: Chosen
        # 索引 1: Consistency Rejected
        # 索引 2: Relevance Rejected
        # 索引 3: Coherence Rejected
        # 索引 4: Quality Rejected
        responses = [
            sample.get("chosen", ""),
            sample.get("rejected_consistency", ""),
            sample.get("rejected_relevance", ""),
            sample.get("rejected_coherence", ""),
            sample.get("rejected_quality", "")
        ]
        
        # 检查是否所有字段都存在
        if not all(responses):
            logger.debug(f"字段不完整，存在空值: {[bool(r) for r in responses]}")
            return None, None

        input_ids_list, attention_mask_list = [], []
        
        # 预先对 prompt 分词并转换为 ID
        content_tokens = self.tokenizer.tokenize(prompt_text)
        content_ids = self.tokenizer.convert_tokens_to_ids(content_tokens)
        
        # 验证 content_ids
        if None in content_ids:
            logger.warning(f"Content IDs 中存在 None 值，长度: {len(content_ids)}")
            return None, None

        for resp_idx, resp_text in enumerate(responses):
            # 1. 对回答进行分词并转换为 ID
            query_tokens = self.tokenizer.tokenize(resp_text)
            query_ids = self.tokenizer.convert_tokens_to_ids(query_tokens)
            
            # 验证 query_ids
            if None in query_ids:
                logger.warning(f"Response {resp_idx} 的 query_ids 中存在 None 值")
                return None, None
            
            query_ids = query_ids[:self.max_query_len]

            # 2. 计算 Prompt 的可用长度并截断 (留出 [CLS] [SEP] [SEP] 空间)
            content_max_len = self.max_len - len(query_ids) - 3
            current_content_ids = content_ids[:content_max_len]

            # 3. 拼接序列 - 使用实例变量中的 token IDs
            input_ids = [self.cls_token_id] + \
                        current_content_ids + \
                        [self.sep_token_id] + \
                        query_ids + \
                        [self.sep_token_id]

            # 最终验证 - 检查每个元素是否为 None
            none_indices = [i for i, x in enumerate(input_ids) if x is None]
            if none_indices:
                logger.warning(f"最终 input_ids 中存在 None 值，位置: {none_indices}")
                logger.debug(f"cls_token_id={self.cls_token_id}, sep_token_id={self.sep_token_id}")
                logger.debug(f"content_ids 中 None: {None in current_content_ids}")
                logger.debug(f"query_ids 中 None: {None in query_ids}")
                return None, None

            attention_mask = [1] * len(input_ids)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        return input_ids_list, attention_mask_list

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        return self.data_set[idx]

def collate_fn(batch_data):
    """
    DataLoader 的整理函数
    输入：[Batch_Size] 个字典，每个字典包含 5 组 input_ids
    输出：Tensor 形状为 [Batch_Size * 5, Seq_Len]
    """
    if not batch_data:
        return {}

    all_input_ids = []
    all_attention_masks = []

    for instance in batch_data:
        if instance is None:
            continue
        # instance["input_ids"] 长度为 5
        ids_list = instance.get("input_ids")
        mask_list = instance.get("attention_mask")
        
        if ids_list is None or mask_list is None:
            logger.warning("跳过包含 None 值的实例")
            continue
            
        for ids, mask in zip(ids_list, mask_list):
            if ids is None or mask is None:
                logger.warning("跳过包含 None 的 ids 或 mask")
                continue
            all_input_ids.append(torch.tensor(ids, dtype=torch.long))
            all_attention_masks.append(torch.tensor(mask, dtype=torch.long))

    if not all_input_ids:
        return {}

    # 自动对齐长度并填充
    return {
        "input_ids": pad_sequence(all_input_ids, batch_first=True, padding_value=0),
        "attention_mask": pad_sequence(all_attention_masks, batch_first=True, padding_value=0)
    }