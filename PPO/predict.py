import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from swanlab import login

# [可选] 登录 SwanLab (如果需要记录日志)
# login(api_key="...", save=True)

# 1. 设置路径
# 请将此路径修改为您实际保存的微调模型路径 (即 output_dir 中的 best model 或 checkpoint)
# 例如: "../output_sft/checkpoint-500"
ppo_model_path = "../ppo_models" 
safetensors_path = "../ppo_models/final_actor_model"

# 2. 加载 Tokenizer 和 Model
tokenizer = AutoTokenizer.from_pretrained(ppo_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    safetensors_path, 
    device_map="auto", 
    trust_remote_code=True,
    use_safetensors=True
)
# 设置为评估模式
model.eval()

def predict(text, max_length=512, top_p=0.7, temperature=0.1):
    """
    输入用户文本，使用加载的模型生成回复
    Args:
        text (str): 用户的输入问题/指令
        max_length (int): 生成序列的最大长度
        top_p (float): 核采样参数
        temperature (float): 温度参数，控制随机性
        
    Returns:
        str: 模型生成的回复
    """
    # 1. 构造 Prompt
    # 注意：这里的 Prompt 格式必须与您训练时的格式完全一致！
    # 根据 convert_feature 函数，训练时并没有显式添加 <|im_start|> 等特殊字符，
    # 而是直接拼接了 user 和 assistant 的内容。
    # 为了让模型知道要回答，我们需要手动拼接，并在最后留出空格或特定标记让模型开始生成。
    # 假设训练时格式简单拼接为 "User内容Assistant内容"，我们这里构造如下：
    prompt = text 
    
    # 2. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 3. 生成
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            max_new_tokens=max_length,     # 最多生成多少个新 token
            do_sample=True,                # 是否使用采样 (False 则是贪婪搜索)
            top_p=top_p,                   # 核采样
            temperature=temperature,       # 温度
            repetition_penalty=1.0,        # 重复惩罚
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 4. 解码
    # 切片操作 [len(inputs["input_ids"][0]):] 是为了去掉输入的 prompt 部分，只保留生成的回复
    s = generation_output[0][len(inputs["input_ids"][0]):]
    output = tokenizer.decode(s, skip_special_tokens=True)
    
    return output

if __name__ == "__main__":
    # 测试预测
    user_input = "帮我介绍七宗罪这部电影"
    
    response = predict(user_input)
    print(f"模型回复: {response}")

    # 针对测试集 JSONL，也可以批量处理
    # test_file = "./film/sft_test.json"
    # with open(test_file, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         data = json.loads(line)
    #         # 假设您的数据格式里有 'conversations' 或者 'input' 字段
    #         # 需要根据实际数据提取出最后一条 user 的输入
    #         query = data['conversations'][-2]['content'] # 获取倒数第二条(通常是user)的内容
    #         print(f"问题: {query}")
    #         res = predict(query)
    #         print(f"回答: {res}\n{'-'*20}")
