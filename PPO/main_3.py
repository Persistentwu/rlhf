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
    model_name = "../output_sft/qwen_sft_final"  # SFT 后的 Actor 基座
    reward_model_path = "../rm_models/best_model" # 你训练好的 RM 路径
    
    # 数据路径
    data_path = "data/train.json"
    output_dir = "../ppo_models"

    # 显卡分配
    actor_device = "cuda:0"
    critic_device = "cuda:1"
    
    # 训练参数
    learning_rate = 1e-5
    batch_size = 32  # 减小批次大小
    mini_batch_size = 4  # 减小mini batch大小
    gradient_accumulation_steps = 4  # 增加梯度累积步数
    
    # PPO 特定参数
    ppo_epochs = 2 # 建议先从 2 开始，防止崩坏
    init_kl_coef = 0.05
    target_kl = 0.1 
    gamma = 1
    lam = 0.95
    cliprange = 0.2
    cliprange_value = 0.2
    vf_coef = 0.5 
    
    lr_scheduler_type="cosine"
    warmup_ratio=0.1
    # 生成参数
    # 注意：移除了 stop_strings，因为 apply_chat_template 会自动处理格式
    # 如果模型训练得很好，它会自动生成 <|im_end|>
    gen_kwargs = {
        "top_p": 0.9,
        "do_sample": True,
        "max_new_tokens": 64, 
        "temperature": 0.7,
        "repetition_penalty": 1.1
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
# 3. 数据处理 (修改：使用 apply_chat_template)
# ==========================================
class PPODataset(Dataset):
    def __init__(self, data_path, tokenizer, shuffle=True):
        self.tokenizer = tokenizer
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f): 
                if not line.strip(): continue
                item = json.loads(line)
                
                # 获取 prompt 内容
                prompt = item["conversations"][0]["content"]
                
                # 构造 messages 列表
                messages = [{"role": "user", "content": prompt}]
                
                # 使用 apply_chat_template
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    add_special_tokens=False
                )
                
                self.data.append({
                    "input_ids": inputs["input_ids"].squeeze(0),
                })
        
        # 打乱数据
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
        report_to=["swanlab"],
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_ratio,
        max_grad_norm=1.0  # 添加梯度裁剪
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
