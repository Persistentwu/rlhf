from typing import Optional
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import AutoModelForCausalLM

class QwenMultiRoundModel: 
    """
    Qwen3多轮对话微调模型
    支持多轮对话场景，只对assistant回复计算损失
    """
    def __init__(self, model_name_or_path, device):
        """
        Args:
            model_name_or_path: 模型路径或HuggingFace模型名称
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            dtype=torch.float16,  # 节省显存
            device_map=device
        )
        # 冻结大部分层，只微调最后几层（LoRA推荐）
        # self.model.gradient_checkpointing_enable()

    def forward(self, input_ids=None, attention_mask=None, labels = None):
        """
        input_ids: [batch_size, seq_length] 输入token序列
        attention_mask: [batch_size, seq_length] 注意力掩码，1表示有效，0表示padding
        labels: [batch_size, seq_length] 标签，-100表示不计算损失的位置
        
        多轮对话中：
        - user的问题对应位置设为-100
        - assistant的回复对应位置设为token_id
        """
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )   # [batch_size, seq_length-1, vocab_size]
        shift_logits = logits[:, :-1, :].contiguous()  # 预测下一个token
        shift_labels = labels[:, 1:].contiguous()      # 对齐标签
        loss_fct = CrossEntropyLoss(ignore_index=0)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss




