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
    model_name = "../output_sft/qwen_sft_final"
    reward_model_path = "../rm_models/best_model"
    data_path = "data/train.json"
    output_dir = "../grpo_models"

    actor_device = "cuda:0"
    rm_device = "cuda:0" # 建议显式区分设备
    
    learning_rate = 1e-5
    batch_size = 8 # GRPO 生成多倍数据，显存压力大，适当减小 batch_size
    gradient_accumulation_steps = 8
    
    # GRPO 特定参数
    num_generations = 4 
    max_completion_length = 64
    temperature = 0.7
    top_p = 0.9
    repetition_penalty = 1.1
    
    beta = 0.05  # KL惩罚系数
    
    lr_scheduler_type = "cosine"
    warmup_ratio = 0.1
    num_train_epochs = 2
    max_grad_norm = 1.0

# ==========================================
# 2. Reward Model 定义
# ==========================================
class MultiDimensionScoreModel(nn.Module):
    base_model_prefix = "model"
    def __init__(self, model_path=None, device='cuda:0', existing_model=None):
        super().__init__()
        self.device = device
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
        
        # 注意：这里 inference_mode=True，且不需要训练
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=True, 
            r=8, 
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.eval() # 设置为评估模式

        self.config = self.model.config
        hidden_size = self.config.hidden_size
        
        self.score_heads = nn.ModuleDict({
            'consistency': nn.Linear(hidden_size, 1),
            'relevance': nn.Linear(hidden_size, 1),
            'coherence': nn.Linear(hidden_size, 1),
            'quality': nn.Linear(hidden_size, 1)
        })
        self.score_heads.to(torch.bfloat16).to(self.device)
        self.score_heads.eval()

    def score(self, hidden_states):
        scores = []
        for head in self.score_heads.values():
            scores.append(head(hidden_states))
        multi_scores = torch.cat(scores, dim=-1)
        # 返回标量分数
        return multi_scores.mean(dim=-1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = self.score(outputs.hidden_states[-1])
        return SequenceClassifierOutput(logits=logits)

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
                prompt = item["conversations"][0]["content"]
                
                messages = [{"role": "user", "content": prompt}]
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                self.data.append({"prompt": prompt_text})
        
        if shuffle:
            random.shuffle(self.data)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    config = Config()
    
    torch.cuda.empty_cache()
    
    print(f"Available devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
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
    
    # 2. Reward Model (加载并冻结)
    print(f"Loading Reward Model on {config.rm_device}...")
    reward_base = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": config.rm_device}
    )
    
    reward_model_instance = MultiDimensionScoreModel(None, device=config.rm_device, existing_model=reward_base)
    weight_path = os.path.join(config.reward_model_path, "model.safetensors")
    if os.path.exists(weight_path):
        # 注意：load_state_dict 可能需要 strict=False，因为 LoRA 权重结构可能略有不同
        reward_model_instance.load_state_dict(load_file(weight_path), strict=False)
        print("RM Weights loaded.")
    
    # 冻结所有参数
    for param in reward_model_instance.parameters():
        param.requires_grad = False
    reward_model_instance.eval()

    # 3. 定义 Reward Function (关键修改点)
    def compute_reward(prompts, completions, **kwargs):
        """
        prompts: List[str] - 输入的提示词
        completions: List[str] - 模型生成的回答
        返回: List[float] - 对应的奖励分数
        """
        # 将文本转换为模型输入
        # 注意：这里需要拼接 prompt 和 completion
        texts = [p + c for p, c in zip(prompts, completions)]
        
        inputs = tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512 + config.max_completion_length
        ).to(config.rm_device)
        
        with torch.no_grad():
            outputs = reward_model_instance(**inputs)
            # outputs.logits shape: [batch, seq_len, 1] (根据你的 forward 实现)
            # 我们通常取序列最后一个非 padding token 的分数作为整句的分数
            # 这里简化处理，直接取 mean 或最后一个 token
            # 假设 score() 返回的是 [batch, seq_len]
            rewards = outputs.logits.mean(dim=1).squeeze(-1).cpu().tolist()
            
        return rewards

    dataset = GRPODataset(config.data_path, tokenizer)
    
    # GRPO 配置
    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        max_grad_norm=config.max_grad_norm,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        save_strategy='no',
        remove_unused_columns=False,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        beta=config.beta,
        report_to=["swanlab"],
        gradient_checkpointing=True,
        bf16=True,
    )
    
    # 在初始化trainer之后，训练之前添加
    def test_generation():
        """测试当前模型的生成质量"""
        test_prompts = [
            "看过《我是山姆》吗？",
            "推荐一部好看的电影",
            "什么是人工智能？"
        ]
        
        print("\n=== 测试模型生成质量 ===")
        for prompt in test_prompts:
            messages = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt_text, return_tensors="pt").to(config.actor_device)
            
            with torch.no_grad():
                outputs = actor_model.generate(
                    **inputs,
                    max_new_tokens=config.max_completion_length,
                    temperature=0.7,
                    do_sample=False,  # 使用贪婪解码测试
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response}")
            print(f"Length: {len(response)} chars")
            print("-" * 50)

    # 在训练前调用
    test_generation()
    
    print("Initializing GRPO Trainer...")
    grpo_trainer = GRPOTrainer(
        model=actor_model,
        reward_funcs=compute_reward, # <--- 这里传入函数，而不是模型实例
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # 执行训练
    grpo_trainer.train()
    
    # 保存模型
    print("Training Complete. Saving models...")
    grpo_trainer.save_model(os.path.join(config.output_dir, "final_actor_model"))
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()
