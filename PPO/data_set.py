import json

def convert_dialogue(input_file, output_file):
    """
    转换对话格式
    
    输入：带attrs的原始对话数据（可能是多个JSON对象）
    输出：Qwen格式的多轮对话（JSONL，每行一条）
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        # 逐行读取并解析每个JSON对象
        data = []
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"解析JSON行时出错: {e}")
                    continue
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for dialogue in data:
            messages = dialogue.get("conversations", [])
            
            conversations = []
            
            # 遍历每条消息
            cnt = 0
            for msg in messages:
                message_text = msg.get("content", "").strip()  # 使用"content"而不是"message"
                
                if not message_text:
                    continue
                
                # 判断role：有attrs的是assistant，没有的是user
                if cnt % 2 == 1:
                    role = "assistant"
                else:
                    role = "user"
                
                conversations.append({
                    "role": role,
                    "content": message_text
                })
                cnt += 1
                output_sample = {
                    "conversations": conversations.copy()
                }
                if cnt % 2 == 0:
                    fout.write(json.dumps(output_sample, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    # 输入输出文件路径
    input_file = "data/ppo_train.json"
    output_file = "data/train.json"
    
    convert_dialogue(input_file, output_file)
