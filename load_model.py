import torch
from RM.main import MultiDimensionRewardModel
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
    print(model_path)
    state_dict = load_file(model_path)
    
    # 3. 加载权重到模型
    model.load_state_dict(state_dict)
    
    return model


if __name__ == '__main__':
    pre_trained_path = 'output_sft/sft-model'
    save_path = 'rm_models/rm_model'
    model = MultiDimensionRewardModel(pre_trained_path, device='cuda:0')
    state_dict = load_file(os.path.join(save_path, "model.safetensors"))
    model.load_state_dict(state_dict)
    '''
    model = load_reward_model( 
        pre_trained_path,
        save_path,
        MultiDimensionRewardModel
    )'''
    for name, param in model.score_heads.named_parameters():
      print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
