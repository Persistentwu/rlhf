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


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'  # 添加这行以避免显存碎片

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
    model_name = "../output_sft/qwen_sft"  # SFT 后的 Actor 基座
    reward_model_path = "../rm_models/best_model" # 你训练好的 RM 路径
    
    # 数据路径
    data_path = "data/train.json"
    output_dir = "../ppo_models"

    # 显卡分配
    actor_device = "cuda:0"
    critic_device = "cuda:1"
    
    # 训练参数
    learning_rate = 1.4e-7
    batch_size = 2  # 减小批次大小
    mini_batch_size = 1  # 减小mini batch大小
    gradient_accumulation_steps = 4  # 增加梯度累积步数
    
    # PPO 特定参数
    ppo_epochs = 2 # 建议先从 2 开始，防止崩坏
    init_kl_coef = 0.1
    target_kl = 0.2 
    gamma = 1
    lam = 0.95
    cliprange = 0.2
    cliprange_value = 0.2
    vf_coef = 0.2 
    
    # 生成参数：关键点是让模型输出 <lm_end>
    gen_kwargs = {
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": 64, 
        "stop_strings": ["<|im_end|>"], 
    }

# ==========================================
# 2. 统一打分模型 (适配你的 Multi-Dimension RM)
# ==========================================
class MultiDimensionScoreModel(nn.Module):
    """
    适配 trl 的打分模型：
    1. 继承自 nn.Module
    2. 实现 .score() 方法处理 token 级别的 hidden_states
    """
    base_model_prefix = "model"
    def __init__(self, model_path=None, device='cuda:0', is_value_model=False, existing_model = None):
        super().__init__()
        self.device = device
        self.is_value_model = is_value_model
        if existing_model is not None:
            # 方案 A: 共享已经存在的模型实例（省显存的关键！）
            self.model = existing_model
        elif model_path is not None:
            # 方案 B: 根据路径加载新模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(device)
        else:
            raise ValueError("必须提供 model_path 或 existing_model 其中之一")
        # # 加载基座 (保持 bfloat16 节省显存)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16,  # 使用 torch_dtype 而不是 dtype
        #     output_hidden_states=True,
        #     trust_remote_code=True,
        #     low_cpu_mem_usage=True,  # 添加低内存使用选项
        # )
        
        # 明确将模型移动到指定设备
        # self.model.to(device)
        
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

        self.config = self.model.config
        hidden_size = self.config.hidden_size
        
        # 必须与 RM 代码中的 key 完全一致
        self.score_heads = nn.ModuleDict({
            'consistency': nn.Linear(hidden_size, 1),
            'relevance': nn.Linear(hidden_size, 1),
            'coherence': nn.Linear(hidden_size, 1),
            'quality': nn.Linear(hidden_size, 1)
        })
        self.score_heads.to(torch.bfloat16).to(self.device)

    def score(self, hidden_states):
        """
        trl 会传入整个序列的 hidden_states [batch, seq_len, hidden_size]
        """
        scores = []
        for head in self.score_heads.values():
            scores.append(head(hidden_states))  # [batch, seq_len, 1]

        # 拼接四个维度的分
        multi_scores = torch.cat(scores, dim=-1)  # [batch, seq_len, 4]

        if self.is_value_model:
            # Critic 需要标量 V(s)
            return multi_scores.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        else:
            # Reward 需要标量 R
            return multi_scores.mean(dim=-1)  # [batch, seq_len]

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = self.score(outputs.hidden_states[-1])
        return SequenceClassifierOutput(logits=logits)

    def gradient_checkpointing_enable(self, **kwargs):
        """转发给基座模型"""
        self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        """转发给基座模型"""
        self.model.gradient_checkpointing_disable()


def load_reward_weights(base_model_path, reward_checkpoint_path, model_class, device="cuda:0", is_value_model=False):
    print(f"Loading weights for {'Value' if is_value_model else 'Reward'} model...")
    model = model_class(base_model_path, device=device, is_value_model=is_value_model)
    
    # 路径指向你 RM 训练保存的 model.safetensors
    weight_path = os.path.join(reward_checkpoint_path, "model.safetensors")
    if os.path.exists(weight_path):
        state_dict = load_file(weight_path)
        # RM 训练时带了 base_model 前缀，load 时注意匹配
        model.load_state_dict(state_dict, strict=False)
        print("RM Weights loaded successfully.")
    else:
        print(f"Warning: {weight_path} not found. Check path!")
    return model

# ==========================================
# 3. 数据处理 (注入 lm_start 引导生成)
# ==========================================
class PPODataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f): 
                if i % 3 != 0: continue
                if not line.strip(): continue
                item = json.loads(line)
                
                # 构造符合你 RM 训练格式的 Prompt
                prompt = item["conversations"][0]["content"] # 假设第一条是 user prompt
                
                # 关键：手动构造 Assistant 的起始部分
                # 这样生成的 response 就会自动接在 <lm_start> 后面
                text = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n<im_start>"
                )

                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512 
                )
                
                self.data.append({
                    "input_ids": inputs["input_ids"].squeeze(0),
                })
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ==========================================
# 4. 主训练流程
# ==========================================
def main():
    config = Config()
    
    # 清空显存
    torch.cuda.empty_cache()
    
    # 检查可用设备
    print(f"Available devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    # PPO 必须左填充
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1. Actor Model - 放在 cuda:0
    actor_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": config.actor_device} 
    )
    
    # --- 步骤 2: 加载唯一的基座给 RM 和 Critic (cuda:1) ---
    print(f"Loading Shared Base for RM/Critic on {config.critic_device}...")
    shared_base = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": config.critic_device}
    )

    # 包装 Reward Model (共享 shared_base)
    reward_model = MultiDimensionScoreModel(None, device=config.critic_device, existing_model=shared_base)
    reward_model.model = shared_base # 关键：直接引用
    # 加载 RM 权重
    weight_path = os.path.join(config.reward_model_path, "model.safetensors")
    if os.path.exists(weight_path):
        reward_model.load_state_dict(load_file(weight_path), strict=False)
    reward_model.eval()

    # 包装 Critic Model (共享 shared_base)
    critic_model = MultiDimensionScoreModel(None, device=config.critic_device, is_value_model=True, existing_model=shared_base)
    critic_model.model = shared_base # 关键：直接引用
    # 加载 Critic 权重 (通常和 RM 一致，只是 Head 不同)
    if os.path.exists(weight_path):
        critic_model.load_state_dict(load_file(weight_path), strict=False)
    
    for name, param in critic_model.named_parameters():
        if "score_heads" not in name:
            param.requires_grad = False
    critic_model.train()

    dataset = PPODataset(config.data_path, tokenizer)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) 

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
        response_length = 64,
        save_strategy='no',
        gradient_checkpointing=True,
        # optimize_device_cache = True,
        report_to=["swanlab"]
    )
    
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

    # 解决一些 TRL 的兼容性小问题
    def dummy_disable_gc(self): pass
    if hasattr(ppo_trainer.model, "gradient_checkpointing_disable"):
        ppo_trainer.model.gradient_checkpointing_disable = dummy_disable_gc.__get__(ppo_trainer.model, type(ppo_trainer.model))

    # 执行训练
    ppo_trainer.train()

    # 5. 保存
    print("Training Complete. Saving models...")
    actor_model.save_pretrained(os.path.join(config.output_dir, "final_actor_model"))
    torch.save(critic_model.score_heads.state_dict(), os.path.join(config.output_dir, "final_value_heads.pth"))
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()
