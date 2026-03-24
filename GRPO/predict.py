import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def chat_with_model():
    # 1. 加载模型和分词器
    model_path = "../model_1.7" # 指向你训练保存的路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"正在加载模型至 {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True,
        use_safetensors=True
    ).eval()

    # 2. 初始化对话历史
    # 必须包含 System Prompt，这是模型角色的“定海神针”
    messages = []

    print("--- 已进入电影助手对话模式 (输入 'exit' 退出) ---")

    while True:
        user_input = [
        {
            "role": "system",
            "content": "你是一个电影助手，判断回答质量"
        },
        {
            "role": "user",
            "content": "知道重庆森林这部电影吗？"
        },
        {
            "role": "assistant",
            "content": "知道呀，是一部由王家卫导演的片子。"
        },
        {
            "role": "user",
            "content": "而主演里更是有王菲，一上映便受到追捧。"
        },
        {
            "role": "assistant",
            "content": "所以此片获得了第14届香港电影金像奖最佳影片奖。"
        },
        {
            "role": "user",
            "content": "具体是哪年上映的你还记得吗？"
        },
        {
            "role": "assistant",
            "content": "记得，是在1994年07月14日。"
        },
        {
            "role": "user",
            "content": "妈呀，都过去二十年了嘛，为啥我的记忆好像还是几年前的感觉呢。"
        },
        {
            "role": "assistant",
            "content": "看看主演梁朝伟，1962年出生的他，那时才三十多点，而现在他已经快要六十岁的老人了。"
        },
        {
            "role": "user",
            "content": "啊，不许说我偶像是老人，他在我心中永远是男神。"
        },
        {
            "role": "response1",
            "content": "哈哈，对于他我也很喜欢，而且他也值得我们喜欢，获得过五届【香港电影金像奖最佳男主角】奖。"
        },
        {
            "role": "response2",
            "content": "哈哈，对于他我也很喜欢，而且他也值得我们喜欢，获得过五届【香港电影金像奖最佳女主角】奖。"
        }
        ]
        messages.extend(user_input)
        # user_input = input("User: ").strip()
        # if user_input.lower() in ["exit", "quit", "退出"]:
        #     break
        # if not user_input:
        #     continue

        # # 将用户输入加入历史
        # messages.append({"role": "user", "content": user_input})

        # 3. 构造推理输入
        # apply_chat_template 会自动根据 Qwen 格式拼接所有历史轮次
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
       
        # 4. 生成回复
        # 增加 repetition_penalty 是解决“好的谢谢”复读的关键
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=64,
                do_sample=True,
                top_p=0.8,
                temperature=0.3,
                repetition_penalty=1.15, # 抑制复读
                
            )
        
        # 截取生成的部分
        input_ids_len = model_inputs.input_ids.shape[1]
        response_ids = generated_ids[0][input_ids_len:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        print(f"Assistant: {response}")

        # 5. 将模型的回复加入历史，实现“记忆”
        messages.append({"role": "assistant", "content": response})

        # 限制历史长度，防止超过 MAX_LENGTH (可选)
        # if len(messages) > 11: # 保留最近 5 轮左右的对话
        #     messages = [messages[0]] + messages[-10:]


if __name__ == "__main__":
    chat_with_model()