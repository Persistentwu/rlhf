import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from trl import GRPOConfig, GRPOTrainer
from safetensors.torch import load_file
import swanlab
import json
import os
from copy import deepcopy
from peft import LoraConfig, get_peft_model
import random
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# ==========================================
# 1. 配置与初始化
# ==========================================
api_key = os.environ.get("SWANLAB_api")
swanlab.login(api_key=api_key, save=True)
swanlab.config.update({
    "model": "Qwen_GRPO/Qwen3-1.7B-MultiRM",
})
swanlab.init()

class Config:
    model_name = "../output_sft/checkpoint-1086"
    reward_model_path = "../rm_models/checkpoint-1167"
    data_path = "data/test_qwen.jsonl"
    output_dir = "../gspo_models"

    actor_device = "cuda:0"
    rm_device = "cuda:0"
    learning_rate = 5e-7  # 降低学习率
    batch_size = 4  # 减小batch size
    gradient_accumulation_steps = 16  # 增加累积步数
    
    # GRPO 特定参数
    num_generations = 8  # 增加生成样本数
    max_completion_length = 64
    temperature = 1.0  # 增加温度增加探索
    top_p = 0.95
    repetition_penalty = 1.1
    
    beta = 0.1  # 增加KL惩罚系数（从0.05提升）
    
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_train_epochs = 2
    max_grad_norm = 1.0
    max_length = 512
    
    # 调试选项
    debug_mode = True
    reward_scale = 5.0  # Reward缩放因子

# ==========================================
# 2. Reward Model 定义
# ==========================================
class MultiDimensionRewardModel(nn.Module):
    def __init__(self, model_path, device='cuda:0'):
        super().__init__()
        self.device = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16, 
            output_hidden_states=True,
            device_map=device,
            trust_remote_code=True
        )
        
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=True,
            r=8, 
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.eval()
        
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        
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
        
        self.score_heads.to(torch.float32).to(self.device)  # 改用float32
        self.score_heads.eval()
        
        # 加载预训练权重
        self._load_weights()

    def _load_weights(self):
        """加载预训练的RM权重"""
        weight_path = os.path.join(Config.reward_model_path, "model.safetensors")
        if os.path.exists(weight_path):
            try:
                state_dict = load_file(weight_path)
                filtered_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('score_heads.') or k.startswith('model.'):
                        # 转换权重到float32
                        filtered_state_dict[k] = v.to(torch.float32)
                self.load_state_dict(filtered_state_dict, strict=False)
                print(f"Successfully loaded RM weights from {weight_path}")
            except Exception as e:
                print(f"Warning: Failed to load RM weights: {e}")
        else:
            print(f"Warning: RM weights not found at {weight_path}")

    def forward(self, input_ids, attention_mask, dimension='quality'):
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        else:
            raise ValueError("Model output does not contain hidden_states.")

        # 取最后一个非padding token的hidden state
        last_token_indices = attention_mask.sum(dim=1) - 1
        last_token_hidden = last_hidden_state[torch.arange(last_hidden_state.size(0), device=self.device), last_token_indices]
        
        # 转换为float32再进行评分
        last_token_hidden = last_token_hidden.to(torch.float32)
        
        # 获取对应维度的分数
        score = self.score_heads[dimension](last_token_hidden).squeeze(-1)
        return score

# ==========================================
# 3. 数据处理
# ==========================================
class GRPODataset(Dataset):
    def __init__(self, data_path, tokenizer, shuffle=True):
        self.tokenizer = tokenizer
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): 
                    continue
                item = json.loads(line)
                conversations = item.get("conversations", [])
                
                if len(conversations) < 1:
                    continue
                
                # 提取用户消息
                user_msg = None
                for conv in conversations:
                    if conv['role'] == 'user':
                        user_msg = conv
                        break
                
                if user_msg is None:
                    continue
                
                # 构建只包含用户消息的对话
                messages = [user_msg]
                
                # 使用 apply_chat_template 格式化
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                self.data.append({
                    "prompt": prompt_text,
                    "user_message": user_msg
                })
        
        if shuffle:
            random.shuffle(self.data)
            
        print(f"Loaded {len(self.data)} samples with chat template format")
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"prompt": self.data[idx]["prompt"]}

# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    config = Config()
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    torch.cuda.empty_cache()
    
    print(f"Available devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. Actor Model
    actor_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": config.actor_device}
    )
    
    # 2. Reward Model
    print(f"Loading Reward Model on {config.rm_device}...")
    reward_model = MultiDimensionRewardModel(
        config.model_name, 
        device=config.rm_device
    )
    
    # 冻结所有参数
    for param in reward_model.parameters():
        param.requires_grad = False
    reward_model.eval()

    # 3. 测试RM输出范围
    print("\n" + "="*50)
    print("Testing RM Output Range")
    print("="*50)
    test_prompts = ["你好，请介绍一下自己"]
    test_completions = ["我是AI助手，很高兴为您服务！"]
    
    with torch.no_grad():
        full_text = test_prompts[0] + test_completions[0]
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(config.rm_device)
        print(f"\nTest sample:")
        print(f"Prompt: {test_prompts[0]}")
        print(f"Completion: {test_completions[0]}")
        for dim in ['consistency', 'relevance', 'coherence', 'quality']:
            score = reward_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                dimension=dim
            )
            print(f"{dim}: {score.item():.4f}")
    print("="*50 + "\n")

    # 4. 定义 Reward Function（关键修复）
    step_count = 0
    
    def compute_reward(prompts, completions, **kwargs):
        """
        prompts: List[str] - 已格式化的提示词
        completions: List[str] - 模型生成的回答
        返回: List[float] - 对应的奖励分数
        """
        nonlocal step_count
        step_count += 1
        
        batch_size = len(prompts)
        all_rewards = []
        dimension_scores = {}
        
        # 四个维度的奖励
        dimensions = ['consistency', 'relevance', 'coherence', 'quality']
        
        for dim in dimensions:
            # 构建完整的对话文本
            full_texts = [p + c for p, c in zip(prompts, completions)]
            
            # Tokenize
            inputs = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length + config.max_completion_length,
                add_special_tokens=False
            ).to(config.rm_device)
            
            # 计算该维度的奖励
            with torch.no_grad():
                scores = reward_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    dimension=dim
                )
                all_rewards.append(scores)
                # 修复：先转换为float再转numpy
                dimension_scores[dim] = scores.float().cpu().numpy()
        
        # 对四个维度的奖励取平均
        stacked_rewards = torch.stack(all_rewards, dim=0)  # [4, batch_size]
        final_rewards = stacked_rewards.mean(dim=0)  # [batch_size]
        
        # 关键修复1：放大reward信号
        final_rewards = final_rewards * config.reward_scale
        
        # 关键修复2：标准化reward使其有正有负
        # 这样模型才能区分好和坏的回答
        reward_mean = final_rewards.mean()
        reward_std = final_rewards.std() + 1e-8
        final_rewards = (final_rewards - reward_mean) / reward_std
        
        # 关键修复3：clip防止极端值
        final_rewards = torch.clamp(final_rewards, min=-3.0, max=3.0)
        
        
        
        # 返回float列表
        return final_rewards.float().cpu().tolist()

    # 5. 加载数据集
    dataset = GRPODataset(config.data_path, tokenizer)
    
    # 6. GRPO 配置（关键修复）
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        importance_sampling_level="sequence",
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_steps=10,
        save_steps=200,
        save_total_limit=1,
        save_strategy='steps',
        remove_unused_columns=False,
        
        # GRPO特定参数
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        
        # PPO参数
        beta=config.beta,  # KL惩罚系数
        
        report_to=["swanlab"],
        gradient_checkpointing=True,
        bf16=True,
        
        # 额外的稳定训练参数
        dataloader_num_workers=0,
        optim="adamw_torch",
    )
    

    
    # 7. 初始化Trainer
    print("Initializing GRPO Trainer...")
    grpo_trainer = GRPOTrainer(
        model=actor_model,
        reward_funcs=compute_reward,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # 8. 执行训练
    print("Starting GRPO training...")
    grpo_trainer.train()
    
    # 9. 保存模型
    # print("\nTraining Complete. Saving models...")
    # final_output_dir = os.path.join(config.output_dir, "final_actor_model")
    # grpo_trainer.save_model(final_output_dir)
    # tokenizer.save_pretrained(final_output_dir)
    
    # print(f"Model saved to {final_output_dir}")

if __name__ == "__main__":
    main()