import json
import random
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
BASE_URL = os.getenv("DS_BASE_URL")
API_KEY = os.getenv("DS_KEY")

def generate_with_ds(prompt):
    
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    response = client.chat.completions.create(
        model="deepseek-chat",
        temperature= 0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content


def generate_negative_samples(conversation):
    """
    针对给定的对话片段生成 Prompt
    """
    if conversation[-1]['role'] != 'assistant':
        return None
    
    context_list = conversation[:-1]
    target_response = conversation[-1]['content']
    
    context_str = ""
    for turn in context_list:
        context_str += f"{turn['role'].capitalize()}: {turn['content']}\n"

    # 修改后的 Prompt 模板
    prompt_template = """
    你是一个专业的对话数据构造专家，负责生成用于训练奖励模型的负样本。
    我将给你一段对话上下文和一个高质量的正样本回答。
    请你基于这个正样本，伪造出 4 个具有特定缺陷的负样本。

    ### Context:
    {context}

    ### Chosen (正样本):
    {target_response}

    ### 任务：请生成以下四个负样本（注意：生成的负样本长度应尽量与正样本保持一致，不要生成过短的回复）：

    1. Rejected_Consistency (一致性错误): 回复必须与前文提到的事实矛盾（如改错名字、时间、地点或前文已确认的信息）。
    2. Rejected_Relevance (相关性错误): 回复看似在回答，但核心内容跑题，或者一直在重复无关的背景知识。
    3. Rejected_Coherence (连贯性错误): 回复包含逻辑断层、前后矛盾、语序混乱或严重的病句，导致难以理解。
    4. Rejected_Quality (低质量/态度差): 
       - 不要生成“哦”、“嗯”、“不知道”等过短回复。
       - 请生成“废话文学”（车轱辘话，说了很多但没实质内容）、“复读机”（重复用户的问题或前文内容）、或“语气生硬/机械”（像没有感情的机器）的回复。
       - 或者生成包含明显常识性错误但语法正确的回复。

    输出格式要求为严格的 JSON: 
    {{ "consistency": "...", "relevance": "...", "coherence": "...", "quality": "..." }}
    """
    
    return prompt_template.format(context=context_str, target_response=target_response)


def parse_json_res(res_str):
    """
    防止 DeepSeek 加上 ```json 等废话导致解析失败
    """
    try:
        # 尝试直接解析
        return json.loads(res_str)
    except:
        # 提取 JSON 块
        try:
            start = res_str.find('{')
            end = res_str.rfind('}') + 1
            return json.loads(res_str[start:end])
        except Exception as e:
            print(f"JSON解析彻底失败: {e}")
            return None

def process_sliding_window_sampling(raw_conv, step=3):
    """
    实现滑动窗口+固定步长采样+强制保留末尾逻辑
    
    Args:
        raw_conv: 原始对话列表
        step: 采样步长，默认为 3（即每隔 3 个 assistant 回复采一次）
    """
    total_len = len(raw_conv)
    if total_len < 4: 
        return [] # 对话太短不采

    # 1. 找到所有 assistant 的索引
    assistant_indices = [i for i, turn in enumerate(raw_conv) if turn['role'] == 'assistant']
    
    # 2. 筛选出有效的采样索引（必须至少有前两轮对话作为上下文，即索引 >= 2）
    valid_indices = [idx for idx in assistant_indices if idx >= 2]
    
    if not valid_indices:
        return []

    # 3. 确定最后一个 assistant 的索引（强制保留）
    last_assistant_idx = valid_indices[-1]
    
    # 4. 进行步长采样
    # 逻辑：在 valid_indices 列表中，每隔 step 个元素取一个
    # 例如 valid_indices = [2, 4, 6, 8, 10, 12], step=3 -> 取 [2, 8]
    selected_indices = valid_indices[::step]
    
    # 5. 补充逻辑：如果最后一个索引没有被选中，强制加入
    if last_assistant_idx not in selected_indices:
        selected_indices.append(last_assistant_idx)
        
    # 6. 排序，确保按时间顺序处理（虽然最后加入的是最大的，但为了保险起见）
    selected_indices = sorted(list(set(selected_indices)))

    final_data = []
    
    for idx in selected_indices:
        # 截取从 0 到 idx+1 的子对话
        sub_conv = raw_conv[0:idx + 1]

        # 生成 Prompt 并调用
        prompt = generate_negative_samples(sub_conv)
        if not prompt: continue
        
        raw_res = generate_with_ds(prompt)
        parsed_res = parse_json_res(raw_res)

        if parsed_res:
            # 构造最终存储格式
            sample = {
                "prompt": prompt.split("### Context:")[1].split("### Chosen")[0].strip(), # 提取纯净 context
                "chosen": sub_conv[-1]['content'],
                "rejected_consistency": parsed_res.get("consistency", ""),
                "rejected_relevance": parsed_res.get("relevance", ""),
                "rejected_coherence": parsed_res.get("coherence", ""),
                "rejected_quality": parsed_res.get("quality", "")
            }
            final_data.append(sample)
    
    return final_data


'''
raw_conv = [
    {"role": "user", "content": "你听说过比尔·奈伊这个人吗？"},
    {"role": "assistant", "content": "听说过啊，他是很有实力的一位演员。"},  
]
'''

def gen_data(input_data_path, output_data_path):
    data = []
    with open(input_data_path, "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()
            data.append(json.loads(line))

    with open(output_data_path, "a", encoding="utf-8") as f_out:
        for i, item in enumerate(data):
            if i < 89: continue
            if i % 50 == 0:
                print(f"正在处理 {i} 条对话)")
            conv = item['conversations']
            results = process_sliding_window_sampling(conv) # 处理滑动窗口
        
            # 保存为 JSON
            for item in results:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                
            
    print(f"成功生成多头负样本数据")

if __name__ == "__main__":
    train_data_path = r"SFT\film\sft_train.json"
    train_output_path = r"RM\data\neg_train.json"
    test_data_path = r"SFT\film\sft_test.json"
    test_output_path = r"RM\data\neg_test.json"

    # gen_data(train_data_path, train_output_path)
    gen_data(test_data_path, test_output_path)