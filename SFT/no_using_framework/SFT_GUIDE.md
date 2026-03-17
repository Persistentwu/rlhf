"""
多轮对话SFT微调完整指南

本文档涵盖：
1. 数据格式和准备
2. Dataset的构建
3. 损失函数的原理
4. 完整的训练流程
5. 最佳实践建议
"""

# ============================================================================
# 第一部分: 数据格式说明
# ============================================================================

"""
标准的多轮对话数据格式（JSONL）：

{
    "conversations": [
        {"role": "user", "content": "用户问题1"},
        {"role": "assistant", "content": "助手回答1"},
        {"role": "user", "content": "用户问题2"},
        {"role": "assistant", "content": "助手回答2"}
    ]
}

{
    "conversations": [
        {"role": "user", "content": "知道重庆森林这部电影吗？"},
        {"role": "assistant", "content": "知道呀，是一部由王家卫导演的片子。"}
    ]
}

关键点：
- 每行是一条完整的对话（JSONL格式）
- role: "user" 表示用户消息，"assistant" 表示模型回答
- content: 实际的文本内容
- 可以包含任意多个user-assistant对话轮次
"""


# ============================================================================
# 第二部分: 使用split_dialogue_data.py分割数据
# ============================================================================

"""
步骤1：基础分割
-------

from split_dialogue_data import split_dialogue_dataset

# 将数据分为50% SFT和50% PPO
split_dialogue_dataset(
    input_file="原始数据.jsonl",
    output_dir="split_output",
    sft_ratio=0.5,
    seed=42
)

输出文件：
- split_output/sft_data.jsonl      (50% 用于SFT微调)
- split_output/ppo_data.jsonl      (50% 用于PPO强化学习)


步骤2：验证数据质量
-------

from split_dialogue_data import validate_split_data

validation = validate_split_data(
    "split_output/sft_data.jsonl",
    "split_output/ppo_data.jsonl"
)

# 输出统计信息：
# - 数据条数
# - 平均对话轮数
# - 最小/最大对话轮数


步骤3：进一步分割SFT数据为训练和验证集
-------

from split_dialogue_data import create_train_test_split

create_train_test_split(
    sft_file="split_output/sft_data.jsonl",
    output_dir="split_output",
    train_ratio=0.8,  # 80% 训练，20% 验证
    seed=42
)

输出文件：
- split_output/sft_train.jsonl     (80% 训练数据)
- split_output/sft_test.jsonl      (20% 验证数据)
"""


# ============================================================================
# 第三部分: 使用data_set.py构建Dataset
# ============================================================================

"""
1. MultiRoundDialogueDataset的工作原理

from transformers import AutoTokenizer
from data_set import MultiRoundDialogueDataset, collate_func

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat")

# 创建Dataset
train_dataset = MultiRoundDialogueDataset(
    tokenizer=tokenizer,
    max_len=2048,  # 最大序列长度
    data_dir="cache_dir",  # 缓存目录
    data_set_name="train",
    path_file="split_output/sft_train.jsonl",
    is_overwrite=False  # 如果缓存存在则使用缓存
)

# 创建DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_func  # 重要！使用自定义的batch处理函数
)


2. 数据转换流程详解

原始对话:
{
    "conversations": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"}
    ]
}

转换为:
input_ids:     [你好的tokens] + [你好呀的tokens]
               = [7688, 2513, 29892, 8592, 33391, 29943, 34798]  # 示例

labels:        [-100, -100, -100] + [8592, 33391, 29943, 34798]
               ^--------user部分--------^  ^----assistant部分-----^
               不计算损失                计算损失

attention_mask: [1, 1, 1, 1, 1, 1, 1]  # 全1表示没有padding


3. collate_func的作用

batch的多个样本长度不同时：
输入: [
    {"input_ids": [1,2,3], "labels": [-100,-100,5], ...},
    {"input_ids": [7,8,9,10], "labels": [-100,8,9,10], ...}
]

输出后padding到最大长度(4):
{
    "input_ids": [[1,2,3,0], [7,8,9,10]],
    "labels": [[-100,-100,5,-100], [-100,8,9,10]],
    "attention_mask": [[1,1,1,0], [1,1,1,1]]
}

关键点：
- padding位置用0填充input_ids
- padding位置用-100填充labels（不计算损失）
- attention_mask为0表示padding位置
"""


# ============================================================================
# 第四部分: 损失函数详解
# ============================================================================

"""
1. 损失函数选择

在SFT中使用CrossEntropyLoss，特点：
- 自动计算分类任务的损失
- 支持ignore_index参数，可忽略特定类别
- 我们使用ignore_index=-100来忽略user部分


2. 损失计算流程（伪代码）

model_output = model(
    input_ids=input_ids,           # [batch, seq_len]
    attention_mask=attention_mask   # [batch, seq_len]
)

logits = model_output.logits       # [batch, seq_len, vocab_size]
                                    # 例如 [2, 10, 151936] for Qwen

# 模型内部或手动计算损失
loss = CrossEntropyLoss(ignore_index=-100)(
    logits.view(-1, vocab_size),   # [batch*seq_len, vocab_size]
    labels.view(-1)                # [batch*seq_len]
)

计算步骤：
1. 对每个有效位置（labels != -100），计算softmax
2. 取真实token的log概率
3. 对所有有效位置求平均


3. 数学公式

对于单个token的交叉熵损失：
L = -log(P(correct_token))
  = -log(exp(score_correct) / sum(exp(score_all)))
  = log(sum(exp(score_all))) - score_correct

对于整个批次（忽略-100标记的位置）：
Total_Loss = sum(L_i for all i where labels[i] != -100) / count

困惑度 (Perplexity) = exp(Total_Loss)


4. 为什么用-100？

CrossEntropyLoss的ignore_index参数的工作原理：
- 标记为-100的位置在损失计算中被跳过
- 这些位置的梯度不会被反向传播
- PyTorch官方约定使用-100表示"忽略"


5. 实际训练中的损失变化

第1个epoch:   loss = 4.5234  (模型刚开始学习，损失很高)
第5个epoch:   loss = 2.1456  (模型学到一些模式)
第10个epoch:  loss = 1.2345  (收敛，损失进一步降低)
第20个epoch:  loss = 0.9876  (接近收敛)
第30个epoch:  loss = 0.9432  (基本收敛)

困惑度(Perplexity) = exp(loss):
第1个epoch:   PPL = 92.3
第30个epoch:  PPL = 2.57  (越低越好)
"""


# ============================================================================
# 第五部分: 完整的训练代码
# ============================================================================

"""
完整的训练脚本示例：

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from data_set import MultiRoundDialogueDataset, collate_func
from sft_complete_training import train_sft_epoch, evaluate_sft

# 1. 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Qwen/Qwen-7B-Chat"
train_file = "split_output/sft_train.jsonl"
eval_file = "split_output/sft_test.jsonl"
output_dir = "fine_tuned_model"

# 2. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 3. 准备数据
train_dataset = MultiRoundDialogueDataset(
    tokenizer=tokenizer,
    max_len=2048,
    data_dir="cache",
    data_set_name="train",
    path_file=train_file,
    is_overwrite=False
)

eval_dataset = MultiRoundDialogueDataset(
    tokenizer=tokenizer,
    max_len=2048,
    data_dir="cache",
    data_set_name="eval",
    path_file=eval_file,
    is_overwrite=False
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=4, 
    shuffle=True, 
    collate_fn=collate_func
)

eval_loader = DataLoader(
    eval_dataset, 
    batch_size=8, 
    collate_fn=collate_func
)

# 4. 优化器和学习率调度
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

# 5. 训练循环
for epoch in range(num_epochs):
    train_loss = train_sft_epoch(
        model, train_loader, optimizer, device, epoch
    )
    
    eval_loss, eval_ppl = evaluate_sft(model, eval_loader, device)
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Eval Loss={eval_loss:.4f}, PPL={eval_ppl:.2f}")

# 6. 保存模型
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
"""


# ============================================================================
# 第六部分: 最佳实践
# ============================================================================

"""
✓ 数据准备
  1. 确保数据格式正确（JSONL，每行一条对话）
  2. 检查数据质量，移除过短或过长的对话
  3. 对话应包含自然的多轮交互，而不仅仅是单轮问答
  4. 验证user和assistant的角色标记正确

✓ 数据分割
  1. 使用固定的随机种子保证可复现性
  2. 训练/验证/测试集不应有重叠
  3. 保持类别分布的均衡
  4. 如果用于PPO，确保有足够的独立数据

✓ 超参数配置
  1. 学习率：通常5e-5到2e-4（从小到大尝试）
  2. batch_size：4-16（根据显存调整）
  3. max_len：通常2048-4096
  4. 梯度裁剪norm：1.0-2.0
  5. warmup比例：总step的5%-10%

✓ 监控训练
  1. 记录训练/验证损失，绘制曲线
  2. 监控困惑度(PPL)，应该持续下降
  3. 定期保存最佳模型
  4. 检查梯度norm，异常值表示问题

✓ 模型保存
  1. 保存训练过程中验证集上最好的模型
  2. 同时保存分词器
  3. 保存训练配置和超参数
  4. 记录训练日志

✓ 推理和评估
  1. 训练完成后在独立测试集上评估
  2. 使用人工评估验证回答质量
  3. 检查模型是否过拟合或欠拟合
  4. 与基础模型进行对比

✗ 常见错误
  1. 遗漏collate_fn参数 -> 数据格式错误
  2. 将padding_value设置为0而labels设为-100不一致
  3. 没有梯度裁剪 -> 训练不稳定
  4. 学习率过高 -> 损失振荡
  5. max_len过小 -> 对话被截断
"""


# ============================================================================
# 第七部分: 故障排除
# ============================================================================

"""
问题1: RuntimeError: CUDA out of memory
解决:
- 减少batch_size
- 减少max_len
- 启用梯度累积
- 使用混合精度训练（torch.float16）

问题2: 损失不下降或NaN
解决:
- 检查数据是否正确加载
- 减少学习率
- 添加梯度裁剪
- 检查标签中是否有错误值

问题3: 模型输出重复或无关
解决:
- 检查数据质量
- 增加训练数据量
- 增加训练轮数
- 调整learning_rate

问题4: 验证集损失上升（过拟合）
解决:
- 增加训练数据
- 使用早停（Early Stopping）
- 减少模型参数（如果可能）
- 添加正则化

问题5: 训练速度太慢
解决:
- 使用更好的GPU
- 启用梯度累积减少显存使用
- 使用混合精度训练
- 减少max_len
"""


# ============================================================================
# 第八部分: 下一步建议
# ============================================================================

"""
1. SFT之后的PPO阶段：
   - 使用分割出的PPO数据
   - 需要奖励模型(Reward Model)
   - 使用PPO算法进行强化学习

2. 模型评估：
   - BLEU、ROUGE等自动指标
   - 人工评估（Likert scale）
   - 与基础模型的对比测试

3. 部署和推理：
   - 量化模型以减少显存
   - 使用推理框架（如vLLM）
   - API服务化

4. 持续改进：
   - 收集模型的失败案例
   - 使用DPO等新方法替代PPO
   - 定期更新训练数据
"""
