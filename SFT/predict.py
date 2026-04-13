import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def chat_with_model():
    # 指向你训练保存的最佳路径 (如果开启了 load_best_model_at_end，通常在 output_dir 下)
    # 如果直接想测试某个特定 step，可以写 "../output_sft/checkpoint-xxxx"
    model_path = "../output_sft/checkpoint-1086" 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"正在加载模型至 {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        dtype=torch.bfloat16, # 注意这里改成了标准的 torch_dtype 参数
        device_map="auto", 
        trust_remote_code=True
    ).eval()

    # 初始化对话历史
    messages = []
    print("--- 已进入电影助手对话模式 (输入 'exit' 退出) ---")

    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit", "退出"]:
            break
        if not user_input:
            continue

        # 将用户输入加入历史
        messages.append({"role": "user", "content": user_input})

        # 构造推理输入 (这里和训练时的 apply_chat_template 完美呼应)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.9,
                temperature=0.6,
                repetition_penalty=1.1,  # 抑制复读
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 截取生成的部分 (去掉 prompt)
        input_ids_len = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[0][input_ids_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        print(f"Assistant: {response}")

        # 将模型的回复加入历史，实现“记忆”
        messages.append({"role": "assistant", "content": response})

        # 限制历史长度，防止多轮对话后超过训练时的 MAX_LENGTH (512) 导致报错或效果下降
        # 粗略估算：保留最近 4 轮对话 (8条消息) 比较安全
        if len(messages) > 10:
            messages = messages[-6:]

if __name__ == "__main__":
    chat_with_model()
