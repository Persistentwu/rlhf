import os
import json
import re
from dotenv import load_dotenv
from openai import OpenAI

# from zai import ZhipuAiClient

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
# client = ZhipuAiClient(api_key=api_key)

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)
# =========================
# 配置 API
# =========================
if not api_key:
    raise ValueError("ZAI_KEY not found in environment variables.")

MODEL_NAME = "deepseek-chat"


def load_model(prompt, system_prompt=None):
    """调用模型生成回复"""
    if system_prompt is None:
        system_prompt = "你是一个电影数据清洗专家，擅长检测负样本、消除指代词和补充缺失信息。"
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\nAPI 请求失败: {e}")
        return None

def parse_json_response(response_text):
    """解析模型返回的 JSON"""
    if not response_text:
        return None
    
    # 1. 尝试直接解析
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # 2. 尝试去除 Markdown 代码块标记
    cleaned_text = re.sub(r'```json\s*|```', '', response_text.strip())
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    
    # 3. 尝试正则提取 JSON 数组
    match = re.search(r'\[.*\]', cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # 4. 尝试提取 JSON 对象
    match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
            
    return None

def process_qa_pairs(qa_pairs):
    """
    处理问答对：
    1. 检测是否为负样本（label=0）
    2. 消除问题中的指代词
    3. 补充缺失的信息
    """
    prompt = f"""
【任务】处理以下问答对，确保问题完整、无指代词，并检测是否为负样本。

【待处理问答对】：
{json.dumps(qa_pairs, ensure_ascii=False, indent=2)}

【处理要求】：

1. **检测负样本**：
   - 检查答案中是否包含明显错误的信息
   - 如果是负样本，保持label为0
   - 如果不是负样本，将label改为1

2. **消除指代词**：
   - 将问题中的"他"、"她"、"它"、"这部电影"等指代词替换为具体的人名、电影名等
   - 例如："他主演了哪部电影？" → "唐·钱德尔主演了哪部电影？"
   - 例如："这部电影是哪年上映的？" → "《卢旺达饭店》是哪年上映的？"

3. **补充缺失信息**：
   - 如果问题缺少必要的上下文信息，需要补充完整
   - 例如："哪年上映的？" → "《卢旺达饭店》是哪年上映的？"
   - 例如："你知道主演都有谁吗？" → "你知道《卢旺达饭店》的主演都有谁吗？"

4. **保持答案不变**：
   - 只修改问题部分，保持答案和label不变（除非检测出不是负样本）

【输出格式】：
请输出处理后的问答对列表，格式为 JSON 数组，每个元素包含 question、answer 和 label。

请开始处理：
"""
    
    response_text = load_model(prompt)
    processed_pairs = parse_json_response(response_text)
    
    if processed_pairs is not None and isinstance(processed_pairs, list):
        return processed_pairs
    else:
        print(f"  [警告] 处理失败，保留原数据")
        return qa_pairs

def process_dataset(input_file, output_file, verbose=True):
    """
    处理数据集：
    1. 检测负样本
    2. 消除指代词
    3. 补充缺失信息
    4. 确保每四条数据来自同一部电影
    5. 边处理边写入文件
    """
    print(f"开始处理: {input_file} -> {output_file}")

    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except Exception as e:
        print(f"解析输入文件失败，请确保文件是标准的 JSON 格式: {e}")
        return

    # 打开输出文件并写入起始括号
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('[\n')
        
        total = len(full_data)
        processed_count = 0  # 记录已处理的数据总数
        movie_buffer = []  # 用于存储同一部电影的数据
        current_movie = None  # 当前处理的电影
        
        for index, item in enumerate(full_data):
            if verbose:
                print(f"进度: {index + 1}/{total} 条数据...", end='\r')
            
            # 检查是否为问答对格式
            if "question" in item and "answer" in item:
                # 提取电影名（从问题中）
                movie_match = re.search(r'《(.+?)》', item["question"])
                movie_name = movie_match.group(1) if movie_match else None
                
                # 如果是新电影且缓冲区有数据，先写入缓冲区
                if movie_name != current_movie and movie_buffer:
                    # 处理缓冲区中的数据
                    processed_buffer = process_qa_pairs(movie_buffer)
                    
                    # 写入处理后的数据
                    for i, pair in enumerate(processed_buffer):
                        processed_count += 1
                        json_str = json.dumps(pair, ensure_ascii=False, indent=2)
                        
                        # 判断是否需要添加逗号
                        if index < total - 1 or i < len(processed_buffer) - 1:
                            f_out.write(json_str + ',\n')
                        else:
                            f_out.write(json_str + '\n')
                        
                        # 强制刷新缓冲区
                        f_out.flush()
                    
                    # 清空缓冲区
                    movie_buffer = []
                
                # 更新当前电影
                current_movie = movie_name
                
                # 将当前数据加入缓冲区
                movie_buffer.append(item)
                
                # 如果缓冲区已经有4条数据，处理并写入
                if len(movie_buffer) >= 4:
                    # 处理缓冲区中的数据
                    processed_buffer = process_qa_pairs(movie_buffer)
                    
                    # 写入处理后的数据
                    for i, pair in enumerate(processed_buffer):
                        processed_count += 1
                        json_str = json.dumps(pair, ensure_ascii=False, indent=2)
                        
                        # 判断是否需要添加逗号
                        if index < total - 1 or i < len(processed_buffer) - 1:
                            f_out.write(json_str + ',\n')
                        else:
                            f_out.write(json_str + '\n')
                        
                        # 强制刷新缓冲区
                        f_out.flush()
                    
                    # 清空缓冲区
                    movie_buffer = []
        
        # 处理剩余的缓冲区数据
        if movie_buffer:
            # 处理缓冲区中的数据
            processed_buffer = process_qa_pairs(movie_buffer)
            
            # 写入处理后的数据
            for i, pair in enumerate(processed_buffer):
                processed_count += 1
                json_str = json.dumps(pair, ensure_ascii=False, indent=2)
                
                # 判断是否需要添加逗号
                if i < len(processed_buffer) - 1:
                    f_out.write(json_str + ',\n')
                else:
                    f_out.write(json_str + '\n')
                
                # 强制刷新缓冲区
                f_out.flush()
        
        # 写入结束括号
        f_out.write(']')

    print(f"\n处理完成！最终结果保存至: {output_file}")
    print(f"共处理 {processed_count} 条数据")

# =========================
# 主程序入口
# =========================
if __name__ == "__main__":
    input_path = "qwen_negative_samples.json" 
    output_path = "qwen_processed_samples.json"
    
    process_dataset(input_path, output_path)
