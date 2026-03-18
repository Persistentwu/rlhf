"""
转换数据格式
"""
import json


def convert_dialogue(input_file, output_file):
    """
    转换对话格式
    
    输入：带attrs的原始对话数据（JSON数组）
    输出：Qwen格式的多轮对话（JSONL，每行一条）
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for dialogue in data:
            messages = dialogue.get("messages", [])
            
            conversations = []
            
            # 遍历每条消息
            cnt = 0
            for msg in messages:
                message_text = msg.get("message", "").strip()
                
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
    input_file = "film/test.json"
    output_file = "data/sft_test.json"
    
    
    convert_dialogue(input_file, output_file)
    
    print("输出样本预览（前5条）：")
    with open(output_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= 5:
                break
            data = json.loads(line)
            print(f"\n【第{idx+1}条】")
            for conv in data["conversations"]:
                print(conv)
