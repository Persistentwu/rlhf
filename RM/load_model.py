import torch
from RM.train_first import MultiDimensionRewardModel
import os
from safetensors.torch import load_file

'''
加载模型
'''
def load_reward_model(save_directory, pre_trained_path, model_class):
    # 1. 创建模型实例（使用预训练路径）
    model = model_class(pre_trained_path)
    
    # 2. 加载 safetensors 权重
    model_path = os.path.join(save_directory, "model.safetensors")
    state_dict = load_file(model_path)
    
    # 3. 加载权重到模型
    model.load_state_dict(state_dict)
    
    return model


if __name__ == '__main__':
    save_directory = r'D:\VsCodeProj\RL\output_rm\checkpoint_2'
    pre_trained_path = r"D:\model\qwen\qwen-0.6b"
    
    # 加载模型
    model = load_reward_model(
        save_directory, 
        pre_trained_path, 
        MultiDimensionRewardModel
    )
    
    # 打印模型结构
    print(model)

'''
MultiDimensionRewardModel(
  (model): Qwen3ForCausalLM(
    (model): Qwen3Model(
      (embed_tokens): Embedding(151936, 1024)
      (layers): ModuleList(
        (0-27): 28 x Qwen3DecoderLayer(
          (self_attn): Qwen3Attention(
            (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
            (k_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=False)
            (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
            (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
            (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
          )
          (mlp): Qwen3MLP(
            (gate_proj): Linear(in_features=1024, out_features=3072, bias=False)       
            (up_proj): Linear(in_features=1024, out_features=3072, bias=False)
            (down_proj): Linear(in_features=3072, out_features=1024, bias=False)       
            (act_fn): SiLUActivation()
          )
          (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
          (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        )
      )
      (norm): Qwen3RMSNorm((1024,), eps=1e-06)
      (rotary_emb): Qwen3RotaryEmbedding()
    )
    (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
  )
  (score_heads): ModuleDict(
    (consistency): Linear(in_features=1024, out_features=1, bias=True)
    (relevance): Linear(in_features=1024, out_features=1, bias=True)
    (coherence): Linear(in_features=1024, out_features=1, bias=True)
    (quality): Linear(in_features=1024, out_features=1, bias=True)
  )
)
'''