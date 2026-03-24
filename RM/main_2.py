import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments
)
import swanlab
import json
import os
import random
from peft import LoraConfig, get_peft_model

# ==========================================
# 0. 环境配置
# ==========================================
api_key = os.environ.get("SWANLAB_api")
swanlab.login(api_key=api_key, save=True)
swanlab.config.update({
    "model": "Qwen_RM/Qwen3-1.7B",
    "framework": "PyTorch",
})

# ==========================================
# 1. 定义模型
# ==========================================
class MultiDimensionRewardModel(nn.Module):
    def __init__(self, model_path, device='cuda:0'):
        super().__init__()
        self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32, 
            output_hidden_states=True,
            device_map=device,
            trust_remote_code=True
        )
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8, 
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 针对 Qwen 系列
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()    

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.hidden_size = self.model.config.hidden_size
        
        # 定义四个维度的打分头
        self.score_heads = nn.ModuleDict({
            'consistency': nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
        ),
            'relevance': nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
        ),
            'coherence': nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
        ),
            'quality': nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
        )
        })

    def forward(self, input_ids, attention_mask, dimension='quality'):
        # 显式传入 output_hidden_states=True
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,  # 确保这里显式开启
            return_dict=True            # 确保返回的是对象而不是 tuple
        )
        
        # 针对 PEFT 包装后的模型，安全地获取 hidden_states
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        else:
            raise ValueError("Model output does not contain hidden_states. Check if output_hidden_states=True is effective.")

        # 找到真正的最后一个 token (防止 padding 干扰)
        last_token_indices = attention_mask.sum(dim=1) - 1
        
        # 这种 index 方式在某些混合精度下更稳定
        last_token_hidden = last_hidden_state[torch.arange(last_hidden_state.size(0)), last_token_indices]
        
        # 打分
        score = self.score_heads[dimension](last_token_hidden).squeeze(-1)
        return score
    
    def gradient_checkpointing_enable(self, **kwargs):
        """转发给基座模型"""
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """转发给基座模型"""
        self.model.gradient_checkpointing_disable()

# ==========================================
# 2. 定义 Dataset (修改点：使用 apply_chat_template)
# ==========================================
class PreferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = []
                for line in f:
                    try:
                        # 尝试解析每一行
                        data = json.loads(line)
                        raw_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line: {line}")
                        print(f"Error message: {str(e)}")
                        continue  # 跳过有问题的行
            
            for item in raw_data:
                prompt = item['prompt']
                chosen = item['chosen']
                
                # --- 修改开始 ---
                # 使用 apply_chat_template 替代硬编码的字符串拼接
                # 1. 构造 messages 结构
                messages = [
                    {"role": "user", "content": prompt}
                ]
                
                # 2. 定义格式化函数
                def format_text(messages, response):
                    # 将 assistant 的回复加入 messages
                    full_messages = messages + [{"role": "assistant", "content": response}]
                    # 使用 tokenizer 官方方法生成文本
                    # tokenize=False 返回字符串，add_generation_prompt=False 因为我们要完整的对话
                    text = self.tokenizer.apply_chat_template(
                        full_messages, 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                    return text
                # --- 修改结束 ---

                dimensions = ['consistency', 'relevance', 'coherence', 'quality']
                for dim in dimensions:
                    rejected_key = f'rejected_{dim}'
                    if rejected_key in item:
                        self.data.append({
                            # 调用新的格式化函数
                            'text_chosen': format_text(messages, chosen),
                            'text_rejected': format_text(messages, item[rejected_key]),
                            'dimension': dim
                        })
            
            # 全局打乱数据，防止维度扎堆
            random.seed(42)
            random.shuffle(self.data)
            print(f"Loaded {len(self.data)} samples with Chat Template format.")
        else:
            print(f"Warning: {data_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 3. 定义 Data Collator (修改点：add_special_tokens)
# ==========================================
class RewardDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        dimensions = [item['dimension'] for item in batch]
        chosen_texts = [item['text_chosen'] for item in batch]
        rejected_texts = [item['text_rejected'] for item in batch]
        
        # --- 修改开始 ---
        # 添加 add_special_tokens=False
        # 因为 apply_chat_template 生成的文本通常已经包含了 <BOS>, <EOS> 等特殊 token
        # 再次添加会导致重复，影响模型效果
        c_inputs = self.tokenizer(
            chosen_texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt",
            add_special_tokens=False 
        )
        r_inputs = self.tokenizer(
            rejected_texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt",
            add_special_tokens=False
        )
        # --- 修改结束 ---
        
        return {
            'input_ids_chosen': c_inputs['input_ids'],
            'attention_mask_chosen': c_inputs['attention_mask'],
            'input_ids_rejected': r_inputs['input_ids'],
            'attention_mask_rejected': r_inputs['attention_mask'],
            'dimension': dimensions
        }

# ==========================================
# 4. 定义 Trainer (保持不变)
# ==========================================
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids_chosen = inputs.get('input_ids_chosen').to(model.device)
        attention_mask_chosen = inputs.get('attention_mask_chosen').to(model.device)
        input_ids_rejected = inputs.get('input_ids_rejected').to(model.device)
        attention_mask_rejected = inputs.get('attention_mask_rejected').to(model.device)
        dimensions = inputs.get('dimension')
        
        batch_size = input_ids_chosen.size(0)
        loss_fct = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        
        dim_indices = {}
        for i, dim in enumerate(dimensions):
            dim_indices.setdefault(dim, []).append(i)
        
        for dim, indices in dim_indices.items():
            idx_tensor = torch.tensor(indices).to(model.device)
            c_ids = input_ids_chosen.index_select(0, idx_tensor)
            c_mask = attention_mask_chosen.index_select(0, idx_tensor)
            r_ids = input_ids_rejected.index_select(0, idx_tensor)
            r_mask = attention_mask_rejected.index_select(0, idx_tensor)
            
            chosen_scores = model(input_ids=c_ids, attention_mask=c_mask, dimension=dim)
            rejected_scores = model(input_ids=r_ids, attention_mask=r_mask, dimension=dim)
            
            diff = chosen_scores - rejected_scores
            loss = loss_fct(diff, torch.ones_like(diff))
            total_loss += loss * len(indices)
            
        avg_loss = total_loss / batch_size
        return (avg_loss, None) if return_outputs else avg_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        input_ids_chosen = inputs.get('input_ids_chosen').to(model.device)
        attention_mask_chosen = inputs.get('attention_mask_chosen').to(model.device)
        input_ids_rejected = inputs.get('input_ids_rejected').to(model.device)
        attention_mask_rejected = inputs.get('attention_mask_rejected').to(model.device)
        dimensions = inputs.get('dimension')
        
        with torch.no_grad():
            batch_size = input_ids_chosen.size(0)
            diffs = torch.zeros(batch_size, device=model.device)
            dim_indices = {}
            for i, dim in enumerate(dimensions):
                dim_indices.setdefault(dim, []).append(i)
                
            for dim, indices in dim_indices.items():
                idx_tensor = torch.tensor(indices).to(model.device)
                c_scores = model(input_ids_chosen.index_select(0, idx_tensor), attention_mask_chosen.index_select(0, idx_tensor), dimension=dim)
                r_scores = model(input_ids_rejected.index_select(0, idx_tensor), attention_mask_rejected.index_select(0, idx_tensor), dimension=dim)
                diffs[indices] = c_scores - r_scores
        
        labels = torch.ones_like(diffs)
        loss = nn.BCEWithLogitsLoss()(diffs, labels)
        return (loss, diffs.unsqueeze(-1), labels.unsqueeze(-1))

    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = (logits > 0).astype(float)
        accuracy = (predictions == labels).mean().item()
        return {"eval_accuracy": accuracy}

# ==========================================
# 5. 主流程
# ==========================================
def main():
    model_path = "../output_sft/qwen_sft_final" 
    train_path = "data/neg_train.json"
    test_path = "data/neg_test.json"
    output_dir = "../rm_models"

    model = MultiDimensionRewardModel(model_path)
    # 确保 padding 在右侧，因为我们用了 sum(mask) 逻辑定位最后一个 token
    model.tokenizer.padding_side = 'right' 
    
    train_dataset = PreferenceDataset(train_path, model.tokenizer)
    test_dataset = PreferenceDataset(test_path, model.tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        save_strategy='no',
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        greater_is_better=False,
        remove_unused_columns=False,
        report_to=["swanlab"]
    )
    
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=RewardDataCollator(model.tokenizer, max_length=512)
    )
    
    trainer.train()

    trainer.save_model("../rm_models/best_model")

    # 别忘了保存 Tokenizer，推理时需要用到同样的词表
    model.tokenizer.save_pretrained("../rm_models/best_model")


if __name__ == "__main__":
    main()
