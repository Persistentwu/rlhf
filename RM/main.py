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
#from dotenv import load_dotenv
import os

api_key=os.environ.get("SWANLAB_api")
# ... (swanlab login 等保持不变) ...
#API_KEY=os.getenv("SWANLAB_KEY")
#swanlab.login(api_key=API_KEY, save=True)
swanlab.login(api_key=api_key, save=True)
swanlab.config.update({
    "model": "Qwen_RM/Qwen3-0.6B",
    })
torch.cuda.empty_cache()
# ==========================================
# 1. 定义模型 (保持不变)
# ==========================================
class MultiDimensionRewardModel(nn.Module):
    def __init__(self, model_path, device='cuda:0'):
        super().__init__()
        self.device = device
        
        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32, # 建议使用 torch_dtype 替代 dtype
            output_hidden_states=True,
            device_map=device
        )
        # 冻结基座模型参数（可选）
        # for param in self.model.parameters():
        #     param.requires_grad = False
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.hidden_size = self.model.config.hidden_size
        
        # 定义四个维度的打分头
        self.score_heads = nn.ModuleDict({
            'consistency': nn.Linear(self.hidden_size, 1),
            'relevance': nn.Linear(self.hidden_size, 1),
            'coherence': nn.Linear(self.hidden_size, 1),
            'quality': nn.Linear(self.hidden_size, 1)
        })

    def forward(self, input_ids, attention_mask, dimension='quality'):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.hidden_states[-1]
        
        # 取最后一个 token 的 hidden state
        # 注意：需要根据 attention_mask 找到真正的最后一个 token，防止 padding 干扰
        # 但为了简化且通常 padding 在后，这里直接取 -1
        last_token_hidden = last_hidden_state[:, -1, :]
        
        score = self.score_heads[dimension](last_token_hidden).squeeze(-1)
        return score

# ==========================================
# 2. 定义 Dataset (保持不变)
# ==========================================
class PreferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 加载数据
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = [json.loads(line) for line in f]
            
                #raw_data = raw_data[:16] # 仅用于演示快速跑通
            
            for item in raw_data:
                prompt = item['prompt']
                chosen = item['chosen']
                
                # 为每个维度创建一个训练样本
                dimensions = ['consistency', 'relevance', 'coherence', 'quality']
                for dim in dimensions:
                    rejected_key = f'rejected_{dim}'
                    # 确保数据中存在该字段
                    if rejected_key in item:
                        self.data.append({
                            'prompt': prompt,
                            'chosen': chosen,
                            'rejected': item[rejected_key],
                            'dimension': dim
                        })
            
            print(f"Loaded {len(self.data)} samples from {data_path}.")
        else:
            print(f"Warning: {data_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        text_chosen = item['prompt'] + item['chosen']
        text_rejected = item['prompt'] + item['rejected']
        
        return {
            'text_chosen': text_chosen,
            'text_rejected': text_rejected,
            'dimension': item['dimension']
        }

# ==========================================
# 3. 定义 Data Collator (保持不变)
# ==========================================
class RewardDataCollator:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        dimensions = [item['dimension'] for item in batch]
        
        chosen_texts = [item['text_chosen'] for item in batch]
        rejected_texts = [item['text_rejected'] for item in batch]
        
        chosen_inputs = self.tokenizer(
            chosen_texts, padding=True, truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        )
        
        rejected_inputs = self.tokenizer(
            rejected_texts, padding=True, truncation=True, 
            max_length=self.max_length, return_tensors="pt"
        )
        
        return {
            'input_ids_chosen': chosen_inputs['input_ids'],
            'attention_mask_chosen': chosen_inputs['attention_mask'],
            'input_ids_rejected': rejected_inputs['input_ids'],
            'attention_mask_rejected': rejected_inputs['attention_mask'],
            'dimension': dimensions
        }

# ==========================================
# 4. 定义 Trainer (重写 compute_loss 和 prediction_step)
# ==========================================
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # print(">>> [DEBUG] compute_loss called")
        # ... (保持你原有的 compute_loss 逻辑不变) ...
        input_ids_chosen = inputs.get('input_ids_chosen')
        attention_mask_chosen = inputs.get('attention_mask_chosen')
        input_ids_rejected = inputs.get('input_ids_rejected')
        attention_mask_rejected = inputs.get('attention_mask_rejected')
        dimensions = inputs.get('dimension')
        
        batch_size = input_ids_chosen.size(0)
        loss_fct = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        
        input_ids_chosen = input_ids_chosen.to(model.device)
        attention_mask_chosen = attention_mask_chosen.to(model.device)
        input_ids_rejected = input_ids_rejected.to(model.device)
        attention_mask_rejected = attention_mask_rejected.to(model.device)
        
        dim_indices = {}
        for i, dim in enumerate(dimensions):
            if dim not in dim_indices:
                dim_indices[dim] = []
            dim_indices[dim].append(i)
        
        for dim, indices in dim_indices.items():
            idx_tensor = torch.tensor(indices).to(model.device)
            
            c_ids = input_ids_chosen.index_select(0, idx_tensor)
            c_mask = attention_mask_chosen.index_select(0, idx_tensor)
            r_ids = input_ids_rejected.index_select(0, idx_tensor)
            r_mask = attention_mask_rejected.index_select(0, idx_tensor)
            
            chosen_scores = model(input_ids=c_ids, attention_mask=c_mask, dimension=dim)
            rejected_scores = model(input_ids=r_ids, attention_mask=r_mask, dimension=dim)
            
            diff = chosen_scores - rejected_scores
            labels = torch.ones_like(diff)
            loss = loss_fct(diff, labels)
            
            total_loss += loss * len(indices)
            
        avg_loss = total_loss / batch_size
        return (avg_loss, None) if return_outputs else avg_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        重写预测步骤，用于计算评估指标。
        """
        # print(">>> [DEBUG] prediction_step called") # 加这行
        # 1. 准备输入
        input_ids_chosen = inputs.get('input_ids_chosen')
        attention_mask_chosen = inputs.get('attention_mask_chosen')
        input_ids_rejected = inputs.get('input_ids_rejected')
        attention_mask_rejected = inputs.get('attention_mask_rejected')
        dimensions = inputs.get('dimension')
        
        # 2. 移动到设备
        input_ids_chosen = input_ids_chosen.to(model.device)
        attention_mask_chosen = attention_mask_chosen.to(model.device)
        input_ids_rejected = input_ids_rejected.to(model.device)
        attention_mask_rejected = attention_mask_rejected.to(model.device)
        
        # 3. 计算分数 (逻辑与 compute_loss 类似，但不需要计算 Loss)
        with torch.no_grad():
            dim_indices = {}
            for i, dim in enumerate(dimensions):
                if dim not in dim_indices:
                    dim_indices[dim] = []
                dim_indices[dim].append(i)
            
            all_diffs = []
            # 必须按原始顺序收集 diff，以便与 labels 对齐
            # 这里我们创建一个全0的 tensor 来存放结果，然后填入对应位置的值
            batch_size = input_ids_chosen.size(0)
            diffs = torch.zeros(batch_size, device=model.device)
            
            for dim, indices in dim_indices.items():
                idx_tensor = torch.tensor(indices).to(model.device)
                
                c_ids = input_ids_chosen.index_select(0, idx_tensor)
                c_mask = attention_mask_chosen.index_select(0, idx_tensor)
                r_ids = input_ids_rejected.index_select(0, idx_tensor)
                r_mask = attention_mask_rejected.index_select(0, idx_tensor)
                
                chosen_scores = model(input_ids=c_ids, attention_mask=c_mask, dimension=dim)
                rejected_scores = model(input_ids=r_ids, attention_mask=r_mask, dimension=dim)
                
                diff = chosen_scores - rejected_scores
                # 将计算出的 diff 填回对应位置
                # 注意：index_select 会保持顺序，所以直接赋值即可
                diffs[indices] = diff
        
        loss_fct = nn.BCEWithLogitsLoss()
        labels = torch.ones_like(diffs)
        loss = loss_fct(diffs, labels)
        
        diffs = diffs.unsqueeze(-1) 
        labels = labels.unsqueeze(-1)
        # 5. 返回结果
        # 必须返回 (loss, logits, labels)
        # logits 用于 compute_metrics
        return (loss, diffs, labels)


    def compute_metrics(self, eval_preds):
        """
        计算评估指标。
        eval_preds 是一个 tuple (logits, labels)，但在 prediction_step 中我们返回了 (None, diffs, None)
        所以这里 eval_preds[0] 就是 diffs
        """
        # print(">>> [DEBUG] compute_metrics called") # 加这行
        diffs, labels = eval_preds
        # 计算准确率：diff > 0 的比例
        # 如果 diff > 0，说明 chosen_score > rejected_score，预测正确
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        logits = logits.flatten()
        labels = labels.flatten()    
        # 计算准确率：diff > 0 的比例
        predictions = (logits > 0).astype(float)
        accuracy = (predictions == labels).mean().item()
        
        return {"eval_accuracy": accuracy}

# ==========================================
# 5. 主训练流程
# ==========================================
def main():
    model_path = "../output_sft/sft-model" 
    train_path = "data/neg_train.json"
    test_path = "data/neg_test.json"
    output_dir = "../rm_models"
    

    # 初始化模型和 Tokenizer
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = MultiDimensionRewardModel(model_path, device=device)
    tokenizer = model.tokenizer
    
    # 准备数据集
    train_dataset = PreferenceDataset(train_path, tokenizer)
    test_dataset = PreferenceDataset(test_path, tokenizer)
    
    # 定义 TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        fp16=True,
        logging_steps=10,
        

        # --- 评估配置 ---
        eval_strategy="steps",
        eval_steps=50, # 为了演示，设小一点，你可以设为 1000
        save_steps=200,
        save_total_limit=2,
        
        # --- 关键：加载最佳模型 ---
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss", # 监控准确率
        greater_is_better=False,           # 准确率越高越好
        
        remove_unused_columns=False,
        report_to=["swanlab"]
        # save_safetensors=False
    )
    
    # 初始化 Trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset, # 传入测试集
        data_collator=RewardDataCollator(tokenizer, max_length=512)
    )
    
    # 开始训练
    trainer.train()
    
    # 训练结束后，手动在测试集上跑一次最终评估
    '''
    print("\n=== Running Final Evaluation on Test Set ===")
    metrics = trainer.evaluate(test_dataset)
    print(f"Test Set Metrics: {metrics}")
    '''

if __name__ == "__main__":
    main()
