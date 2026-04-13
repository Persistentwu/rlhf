import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorWithPadding
)
from transformers.modeling_outputs import SequenceClassifierOutput
from trl.experimental.ppo import PPOTrainer, PPOConfig
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
    "model": "Qwen_PPO/Qwen3-1.7B-MultiRM",
})
swanlab.init()

class Config:
    # 模型路径
    model_name = "../output_sft/checkpoint-1086"
    reward_model_path = "../rm_models/checkpoint-1167"
    
    # 数据路径
    data_path = "data/test_qwen.jsonl"
    output_dir = "../ppo_models"

    # 显卡分配
    actor_device = "cuda:0"
    critic_device = "cuda:1"
    
    # 训练参数
    learning_rate = 1e-6
    batch_size = 8
    mini_batch_size = 4
    gradient_accumulation_steps = 4
    
    # PPO 特定参数
    ppo_epochs = 2
    init_kl_coef = 0.05
    target_kl = 0.1 
    gamma = 1
    lam = 0.95
    cliprange = 0.2
    cliprange_value = 0.2
    vf_coef = 0.5 
    
    # 训练配置
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.1
    max_length = 512
    response_length = 64
    
    # 生成参数
    gen_kwargs = {
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": response_length, 
        "temperature": 0.7,
        "repetition_penalty": 1.1
    }

# ==========================================
# 2. 统一打分模型
# ==========================================
class MultiDimensionScoreModel(nn.Module):
    base_model_prefix = "model"
    
    def __init__(self, model_path=None, device='cuda:0', is_value_model=False, existing_model=None):
        super().__init__()
        self.device = device
        self.is_value_model = is_value_model
        
        if existing_model is not None:
            self.model = existing_model
        elif model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(device)
        else:
            raise ValueError("必须提供 model_path 或 existing_model 其中之一")
        
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8, 
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)

        self.config = self.model.config
        hidden_size = self.config.hidden_size
        
        self.score_heads = nn.ModuleDict({
            'consistency': nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ),
            'relevance': nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ),
            'coherence': nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ),
            'quality': nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        })
        self.score_heads.to(torch.bfloat16).to(self.device)

    def score(self, hidden_states):
        last_token_hidden = hidden_states[:, -1, :]
        
        scores = []
        for head_name, head in self.score_heads.items():
            score = head(last_token_hidden)
            scores.append(score)
        
        multi_scores = torch.cat(scores, dim=-1)
        
        if self.is_value_model:
            return multi_scores.mean(dim=-1, keepdim=True)
        else:
            return multi_scores.mean(dim=-1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True,
            return_dict=True
        )
        logits = self.score(outputs.hidden_states[-1])
        return SequenceClassifierOutput(logits=logits)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

# ==========================================
# 3. 数据处理
# ==========================================
class PPODataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, shuffle=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f): 
                if not line.strip(): 
                    continue
                    
                try:
                    item = json.loads(line)
                    conversations = item.get("conversations", [])
                    
                    if len(conversations) < 1:
                        continue
                    
                    user_messages = []
                    for conv in conversations:
                        if conv.get('role') == 'user':
                            user_messages.append(conv)
                    
                    if not user_messages:
                        continue
                    
                    user_msg = user_messages[-1]
                    messages = [{"role": "user", "content": user_msg['content']}]
                    
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    inputs = self.tokenizer(
                        prompt_text, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=max_length,
                        add_special_tokens=False
                    )
                    
                    self.data.append({
                        "input_ids": inputs["input_ids"].squeeze(0),
                        "prompt_text": prompt_text,
                    })
                    
                except Exception as e:
                    print(f"Error processing line {i}: {e}")
                    continue
        
        if shuffle:
            random.shuffle(self.data)
            
        print(f"Loaded {len(self.data)} samples")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["input_ids"],
            "prompt_text": self.data[idx]["prompt_text"]
        }

# ==========================================
# 4. 数据整理器
# ==========================================
class PPOCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        
    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        prompt_texts = [f["prompt_text"] for f in features]
        
        max_len = max(len(ids) for ids in input_ids)
        
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) 
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        padded_input_ids = []
        attention_masks = []
        
        for ids in input_ids:
            padding_length = max_len - len(ids)
            padded_ids = torch.cat([
                ids,
                torch.full((padding_length,), self.tokenizer.pad_token_id)
            ])
            mask = torch.cat([
                torch.ones(len(ids)),
                torch.zeros(padding_length)
            ])
            
            padded_input_ids.append(padded_ids)
            attention_masks.append(mask)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(attention_masks),
            "prompt_texts": prompt_texts
        }

# ==========================================
# 5. 主训练流程
# ==========================================
def main():
    config = Config()
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    torch.cuda.empty_cache()
    
    print(f"Available devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Actor Model
    print("Loading Actor Model...")
    actor_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": config.actor_device} 
    )
    
    # Shared Base for RM and Critic
    print(f"Loading Shared Base for RM/Critic on {config.critic_device}...")
    shared_base = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": config.critic_device}
    )

    # Reward Model
    print("Initializing Reward Model...")
    reward_model = MultiDimensionScoreModel(
        None, 
        device=config.critic_device, 
        existing_model=shared_base
    )
    
    weight_path = os.path.join(config.reward_model_path, "model.safetensors")
    if os.path.exists(weight_path):
        try:
            state_dict = load_file(weight_path)
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('score_heads.') or k.startswith('model.'):
                    filtered_state_dict[k] = v
            reward_model.load_state_dict(filtered_state_dict, strict=False)
            print("RM Weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load RM weights: {e}")
    
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    # Critic Model
    print("Initializing Critic Model...")
    critic_model = MultiDimensionScoreModel(
        None, 
        device=config.critic_device, 
        is_value_model=True, 
        existing_model=shared_base
    )
    
    if os.path.exists(weight_path):
        try:
            state_dict = load_file(weight_path)
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('score_heads.') or k.startswith('model.'):
                    filtered_state_dict[k] = v
            critic_model.load_state_dict(filtered_state_dict, strict=False)
            print("Critic Weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Failed to load Critic weights: {e}")
    
    for name, param in critic_model.named_parameters():
        if "score_heads" not in name:
            param.requires_grad = False
    critic_model.train()

    # Dataset and Collator
    dataset = PPODataset(config.data_path, tokenizer, max_length=config.max_length)
    collator = PPOCollator(tokenizer, pad_to_multiple_of=8)

    # PPO Config
    ppo_config = PPOConfig(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.ppo_epochs,
        kl_coef=config.init_kl_coef,
        gamma=config.gamma,
        lam=config.lam,
        cliprange=config.cliprange,
        cliprange_value=config.cliprange_value,
        vf_coef=config.vf_coef,
        response_length=config.response_length,
        save_strategy='steps',
        save_steps=100,
        save_total_limit=1,
        gradient_checkpointing=True,
        report_to=["swanlab"],
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=1.0
    )

    # PPO Trainer
    print("Initializing PPO Trainer...")
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer, 
        model=actor_model,
        ref_model=None,
        reward_model=reward_model,
        value_model=critic_model,
        train_dataset=dataset,
        data_collator=collator
    )

    def dummy_disable_gc(self): 
        pass
    
    if hasattr(ppo_trainer.model, "gradient_checkpointing_disable"):
        ppo_trainer.model.gradient_checkpointing_disable = dummy_disable_gc.__get__(
            ppo_trainer.model, type(ppo_trainer.model)
        )

    # Training
    print("\nStarting PPO training...")
    ppo_trainer.train()

    # # Save Models
    # print("\nTraining Complete. Saving models...")
    # actor_output_dir = os.path.join(config.output_dir, "final_actor_model")
    # actor_model.save_pretrained(actor_output_dir)
    
    # critic_heads_path = os.path.join(config.output_dir, "final_value_heads.pth")
    # torch.save(critic_model.score_heads.state_dict(), critic_heads_path)
    
    # tokenizer.save_pretrained(config.output_dir)
    
    # print(f"Models saved to {config.output_dir}")

if __name__ == "__main__":
    main()