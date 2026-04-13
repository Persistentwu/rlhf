import json
import random
import os

def split_conversations_to_samples(conversations, save_dir=".", train_ratio=0.9, seed=42):
    """
    将对话数据拆分成训练集和测试集，按照递增历史格式保存
    
    例如：对话有3轮 (u1-a1, u2-a2, u3-a3)
    生成：
    - 样本1: [u1-a1]
    - 样本2: [u1-a1, u2-a2]
    - 样本3: [u1-a1, u2-a2, u3-a3]
    
    Args:
        conversations: 原始对话列表（每条包含 conversations 字段）
        save_dir: 保存目录
        train_ratio: 训练集比例
        seed: 随机种子
    """
    random.seed(seed)
    
    train_samples = []
    test_samples = []
    total_samples = 0
    
    for item in conversations:
        convs = item.get("conversations", [])
        if not convs or len(convs) < 2:
            continue
        
        # 生成递增历史样本
        history = []
        for i in range(0, len(convs), 2):
            if i + 1 >= len(convs):
                break
            print(i)
            user_msg = convs[i]
            assistant_msg = convs[i + 1]
            
            # 构建当前样本
            sample = {
                "conversations": history + [user_msg, assistant_msg]
            }
            
            # 随机分配到训练集或测试集
            if random.random() < train_ratio:
                train_samples.append(sample)
            else:
                test_samples.append(sample)
            
            # 更新历史
            history.extend([user_msg, assistant_msg])
            total_samples += 1
    
    # 保存为 JSONL 格式（每行一个 JSON 对象，无外层数组）
    train_file = os.path.join(save_dir, "train_qwen.jsonl")
    test_file = os.path.join(save_dir, "test_qwen.jsonl")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for sample in test_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"处理完成！")
    print(f"总样本数: {total_samples}")
    print(f"训练集: {len(train_samples)} 条 ({len(train_samples)/total_samples*100:.1f}%)")
    print(f"测试集: {len(test_samples)} 条 ({len(test_samples)/total_samples*100:.1f}%)")
    print(f"训练集保存至: {train_file}")
    print(f"测试集保存至: {test_file}")
    
    return train_samples, test_samples


def split_and_save(input_file, save_dir=".", train_ratio=0.9, seed=42):
    """
    从文件读取对话数据，拆分并保存为 JSONL 格式（每行一个 JSON 对象）
    
    Args:
        input_file: 输入文件路径（包含 conversations 的 JSON 数组）
        save_dir: 保存目录
        train_ratio: 训练集比例
        seed: 随机种子
    """
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 拆分
    train_samples, test_samples = split_conversations_to_samples(
        data, save_dir, train_ratio, seed
    )
    
    return train_samples, test_samples


if __name__ == "__main__":
    input_file = "qwen_fix_data_qwen.json"
    save_dir = "."
    
    if os.path.exists(input_file):
        print(f"开始处理数据文件: {input_file}")
        split_and_save(input_file, save_dir, train_ratio=0.9)
    else:
        print(f"输入文件 {input_file} 不存在")