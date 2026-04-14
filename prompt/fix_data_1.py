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
        system_prompt = "你是一个数据清洗助手，专门处理对话数据的优化工作。"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API 请求失败: {e}")
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


def filter_invalid_rounds(conversations):
    """
    重写无效轮次，将附和式回答改为主动提供信息
    """
    prompt = f"""
【任务】重写对话中的无效轮次，将模型被动的附和式回答改为主动提供信息

【原始对话】：
{json.dumps(conversations, ensure_ascii=False, indent=2)}

【重写规则】：

1. **识别无效轮次**：
   - 模型回答只有"是的"、"对"、"没错"、"嗯"、"好"、"原来如此"等附和性词语
   - 模型只是简单重复用户的话
   - 模型回答没有提供任何新信息

2. **重写策略**：
   - 如果用户提供了电影信息，模型应该：
     * 确认信息正确（如果确实正确）
     * 补充相关信息（如其他成就、相关作品、合作演员等）
     * 提供有信息量的扩展内容
   
   - 如果用户询问问题，模型应该：
     * 给出准确、完整的答案
     * 适当补充相关背景信息
   
   - 如果用户表达观点/感受，模型应该：
     * 合理回应
     * 可以补充相关事实或延伸讨论

3. **重写示例**：

   ❌ 无效对话：
   User: "唐·钱德尔的演技爆表啊！"
   Assistant: "他的演技确实很出色。"

   ✅ 重写后：
   User: "唐·钱德尔的演技爆表啊！"
   Assistant: "确实，他在《卢旺达饭店》中的表演非常出色，凭借这个角色获得了奥斯卡最佳男主角提名。他还出演了《钢铁侠》系列中的罗迪上校。"

4. **事实准确性要求**：
   - 所有补充的信息必须准确
   - 如果不确定某个信息，不要编造
   - 优先补充与当前话题相关的信息

5. **保持对话连贯性**：
   - 重写后的回答要与上下文自然衔接
   - 不要改变用户的消息内容
   - 只修改 assistant 的回答

【输出格式】：
请输出重写后的完整对话列表，格式为 JSON 数组，每个元素包含 role 和 content。
只修改 assistant 的无效回答，保持其他内容不变。

请开始重写：
"""
    response_text = load_model(prompt, "你是一个数据清洗助手，专门判断和过滤对话中的无效轮次。")
    rewritten_conversations = parse_json_response(response_text)

    if rewritten_conversations is not None:
        return rewritten_conversations
    else:
        print(f"  [警告] 重写失败，保留原数据")
        return conversations


def detect_and_remove_duplicate_info(conversations):
    """
    检测并移除重复的信息询问
    基于已出现的事实信息进行去重
    """
    prompt = f"""
【任务】检测并移除对话中重复询问已出现信息的部分。

【原始对话】：
{json.dumps(conversations, ensure_ascii=False, indent=2)}

【检测规则】：
1. 扫描整个对话，提取所有模型已经提供的事实信息（如：上映年份、导演、演员、奖项等）
2. 检查用户后续是否重复询问相同信息
3. 如果用户重复询问且模型重复回答相同信息，删除重复的问答对

【信息类型】：
- 上映年份/时间
- 导演/制片方
- 主要演员
- 奖项/荣誉
- 剧情简介
- 系列/续作信息
- 技术参数（片长、语言等）

【处理示例】：

输入对话：
[
  {{"role": "user", "content": "哪年上映的？"}},
  {{"role": "assistant", "content": "2004年上映。"}},
  {{"role": "user", "content": "好看吗？"}},
  {{"role": "assistant", "content": "非常震撼人心，是一部经典作品。"}},
  {{"role": "user", "content": "它是哪一年上映的来着？"}},
  {{"role": "assistant", "content": "2004年上映的。"}}
]

输出（删除重复的第5、6轮）：
[
  {{"role": "user", "content": "哪年上映的？"}},
  {{"role": "assistant", "content": "2004年上映。"}},
  {{"role": "user", "content": "好看吗？"}},
  {{"role": "assistant", "content": "非常震撼人心，是一部经典作品。"}}
]

【输出格式】：
输出删除重复问答对后的完整对话列表，JSON数组格式。

注意：
- 保留第一次出现的信息
- 只删除完全重复的问答对
- 如果重复询问但模型提供了额外新信息，则保留

请开始处理：
"""
    response_text = load_model(prompt, "你是一个对话数据去重专家，专门检测并删除重复的信息询问。")
    cleaned_conversations = parse_json_response(response_text)

    if cleaned_conversations is not None:
        return cleaned_conversations
    else:
        return conversations


def fact_consistency_and_cot_rewrite(conversations):
    """
    对对话进行事实一致性检查和CoT推理重写
    """
    prompt = f"""
【任务】对对话进行事实一致性检查和优化，确保信息准确、无重复、逻辑连贯。

【原始对话】：
{json.dumps(conversations, ensure_ascii=False, indent=2)}

【检查与重写规则】：

## 1. 事实一致性检查
对于每个事实性陈述，检查：
- 信息是否准确（上映时间、奖项、演员信息等）
- 同一对话中是否有矛盾信息
- 发现错误时修正，同时保持上下文连贯

## 2. 信息整合优化
- 如果模型需要回答多个问题，合并为一段连贯的回答
- 如果用户连续询问相关信息，模型应一次性提供完整信息，避免碎片化
- 删除"嗯"、"对"等单纯附和且无信息增量的轮次

## 3. CoT推理优化
对模型回答进行思维链优化：
- 确保回答有信息增量，不重复已有信息
- 上下文感知：如果用户引用之前的信息，模型应确认并在此基础上扩展
- 自然过渡：回答要与前后对话自然衔接

## 4. 优化示例

❌ 优化前：
## 5. 保持对话结构
- 只修改 assistant 的回答
- 用户消息保持不变
- 确保删除冗余轮次后，剩余对话依然连贯

【输出格式】：
输出重写后的完整对话列表，JSON数组格式。
要求：
1. 按时间顺序输出所有轮次
2. 优化模型的回答内容
3. 保持对话的自然流畅性

请开始重写：
"""
    response_text = load_model(prompt, "你是一个对话数据优化专家，擅长进行事实一致性检查和思维链推理优化。")
    rewritten_conversations = parse_json_response(response_text)

    if rewritten_conversations is not None:
        return rewritten_conversations
    else:
        print(f"  [警告] 事实一致性重写失败，保留原数据")
        return conversations


def process_conversations_with_filter(conversations, verbose=False):
    """
    处理单组对话，过滤无效轮次
    """
    original_rounds = len(conversations) // 2

    # 过滤无效轮次
    filtered = filter_invalid_rounds(conversations)

    # 确保返回的是列表
    if not isinstance(filtered, list):
        filtered = []

    filtered_rounds = len(filtered) // 2

    if verbose and filtered_rounds < original_rounds:
        print(f"  [过滤] 原始 {original_rounds} 轮 → 保留 {filtered_rounds} 轮")

    return filtered


def process_conversations_with_consistency(conversations, verbose=False):
    """
    综合处理：先检测删除重复，再进行事实一致性重写
    """
    original_rounds = len(conversations) // 2

    # 第一步：删除重复的问答对
    deduped = detect_and_remove_duplicate_info(conversations)
    if not isinstance(deduped, list):
        deduped = conversations

    deduped_rounds = len(deduped) // 2

    # 第二步：事实一致性和CoT重写
    rewritten = fact_consistency_and_cot_rewrite(deduped)
    if not isinstance(rewritten, list):
        rewritten = deduped

    final_rounds = len(rewritten) // 2

    if verbose and (deduped_rounds < original_rounds or final_rounds < deduped_rounds):
        print(f"  [优化] 原始 {original_rounds} 轮 → 去重后 {deduped_rounds} 轮 → 重写后 {final_rounds} 轮")

    return rewritten


def fix_dataset_stream_with_filter(input_file, output_file, verbose=True):
    """
    第一阶段：带无效轮次过滤的数据处理
    """
    print(f"=== 第一阶段：无效轮次过滤 ===")
    print(f"正在处理: {input_file} -> {output_file}")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        f_out.write('[\n')

        try:
            data_list = json.load(f_in)
        except Exception as e:
            print(f"读取文件失败: {e}")
            return

        total_count = len(data_list)
        total_original_rounds = 0
        total_filtered_rounds = 0

        for index, item in enumerate(data_list):
            if verbose:
                print(f"正在处理第 {index + 1}/{total_count} 条...", end='\r')

            if "conversations" in item:
                original_convs = item["conversations"]
                original_rounds = len(original_convs) // 2
                total_original_rounds += original_rounds

                # 过滤无效轮次
                filtered_convs = process_conversations_with_filter(original_convs, verbose)
                filtered_rounds = len(filtered_convs) // 2
                total_filtered_rounds += filtered_rounds

                # 更新数据
                if filtered_convs:
                    item["conversations"] = filtered_convs
                else:
                    # 如果全部被过滤，跳过这条数据
                    continue

            # 写入文件
            json_str = json.dumps(item, ensure_ascii=False, indent=2)

            if index < total_count - 1:
                f_out.write(json_str + ',\n')
            else:
                f_out.write(json_str + '\n')

            f_out.flush()

        f_out.write(']')
        f_out.flush()

    print(f"\n第一阶段处理完成！")
    print(f"原始对话数: {total_count}")
    print(f"原始总轮次: {total_original_rounds}")
    print(f"过滤后总轮次: {total_filtered_rounds}")
    print(f"保留率: {total_filtered_rounds / total_original_rounds * 100:.1f}%")
    print(f"结果已保存至: {output_file}")


def fix_dataset_with_consistency(input_file, output_file, verbose=True):
    """
    第二阶段：带事实一致性和去重处理的数据优化
    """
    print(f"\n=== 第二阶段：事实一致性检查和CoT优化 ===")
    print(f"正在处理: {input_file} -> {output_file}")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        f_out.write('[\n')

        try:
            data_list = json.load(f_in)
        except Exception as e:
            print(f"读取文件失败: {e}")
            return

        total_count = len(data_list)
        total_original_rounds = 0
        total_final_rounds = 0

        for index, item in enumerate(data_list):
            if verbose:
                print(f"正在处理第 {index + 1}/{total_count} 条...", end='\r')

            if "conversations" in item:
                original_convs = item["conversations"]
                original_rounds = len(original_convs) // 2
                total_original_rounds += original_rounds

                # 优化对话
                optimized_convs = process_conversations_with_consistency(original_convs, verbose)
                final_rounds = len(optimized_convs) // 2
                total_final_rounds += final_rounds

                # 更新数据
                if optimized_convs:
                    item["conversations"] = optimized_convs
                else:
                    # 如果全部被删除，跳过这条数据
                    continue

            # 写入文件
            json_str = json.dumps(item, ensure_ascii=False, indent=2)

            if index < total_count - 1:
                f_out.write(json_str + ',\n')
            else:
                f_out.write(json_str + '\n')

            f_out.flush()

        f_out.write(']')
        f_out.flush()

    print(f"\n第二阶段处理完成！")
    print(f"原始对话数: {total_count}")
    print(f"原始总轮次: {total_original_rounds}")
    print(f"优化后总轮次: {total_final_rounds}")
    print(f"压缩率: {(total_original_rounds - total_final_rounds) / total_original_rounds * 100:.1f}%")
    print(f"结果已保存至: {output_file}")


def process_dataset_full_pipeline(input_file, output_file, verbose=True):
    """
    完整的数据处理流程
    """
    # 生成临时文件名
    temp_file = output_file.replace('.json', '_temp.json')

    # 第一阶段：过滤无效轮次
    fix_dataset_stream_with_filter(input_file, temp_file, verbose)

    # 第二阶段：事实一致性检查和优化
    fix_dataset_with_consistency(temp_file, output_file, verbose)

    # 删除临时文件
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"\n已删除临时文件: {temp_file}")

    print(f"\n=== 全部处理完成 ===")
    print(f"最终结果已保存至: {output_file}")


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    input_file = "../data_builder/train_rm_20.json"
    output_file = "filtered_rm.json"

    if os.path.exists(input_file):
        print(f"\n开始处理数据文件: {input_file}")
        process_dataset_full_pipeline(input_file, output_file, verbose=True)
    else:
        print(f"\n输入文件 {input_file} 不存在")
