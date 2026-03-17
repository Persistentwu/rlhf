"""
SFT训练脚本 - 多轮对话有监督微调
"""
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from torch.utils.data import DataLoader
from model import QwenMultiRoundModel
from tqdm import tqdm
import logging
import argparse
import os
import numpy as np


from data_set import MultiRoundDialogueDataset, collate_func

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def train_sft_epoch(
    model: QwenMultiRoundModel,
    train_dataloader: DataLoader,   # 
    optimizer, device, epochs: int, output_dir, max_norm=1.0):

    device = "cuda:" + device
    model.train()
    total_loss = 0
    # total_tokens = 0
    progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epochs}")
    
    for epoch in range(1, epochs + 1):  ## epoch
        total_loss = 0
        for batch_idx, batch_data in enumerate(progress_bar):
            # 准备数据
            input_ids = batch_data['input_ids'].to(device)
            labels = batch_data['labels'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)

            loss = model.forward(
                input_ids=input_ids,            
                attention_mask=attention_mask,  # 模型自动计算CrossEntropyLoss    
                labels=labels
            )

            loss.backward()
            optimizer.step()        
            # 梯度裁剪（防止梯度爆炸）
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            # 参数更新

            optimizer.zero_grad()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)

        logger.info(f"Epoch {epoch} - Loss: {loss:.4f} - Avg Loss: {avg_loss:.4f}")
        # save checkpoint per epoch 
        ckpt_dir = os.path.join(output_dir, f"epoch-{epoch}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        # tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    return avg_loss


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    
    # === 模型和数据相关参数 ===
    # 修改：使用 parser.add_argument 而不是 model_args.add_argument
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="D:/model/qwen/qwen-0.6b",
        help="预训练模型的名称或路径"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/film/sft_train.json",
        help="训练数据文件路径（JSON格式）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_model",
        help="模型保存目录"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="cache/",
        help="缓存目录，用于保存处理后的数据"
    )
    
    # === 训练参数 ===
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="训练轮数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,  # 修改：补充缺失的 batch_size 参数
        help="训练批次大小"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,  # 修改：补充缺失的 learning_rate 参数
        help="学习率"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,  # 修改：补充缺失的 warmup_ratio 参数
        help="预热步数比例"
    )
    parser.add_argument(
        "--gradient_clip_norm",
        type=float,
        default=1.0,
        help="梯度裁剪的最大范数"
    )
 
    # === 数据处理参数 ===
    parser.add_argument(
        "--max_len",
        type=int,
        default=768,
        help="最大序列长度"
    )

    parser.add_argument(
        "--epoches",
        type=int,
        default = 1,
        help="训练次数"
    )

    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="是否重新生成缓存文件"
    )
   
    # === 计算相关参数 ===
    parser.add_argument(
        "--device",
        type=str,
        default="0",  # 修改：默认值设为字符串 "0" 以匹配后面的 f'cuda:{args.device}'
        help="计算设备。0表示cuda:0，cpu表示cpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )

    return parser.parse_args()
    

def setup_seed(seed):
    """设置随机种子以保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"已设置随机种子: {seed}")


def main():
    """主训练函数"""
    
    # 设置
    print("start training")
    args = parse_arguments()
    
    setup_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # 加载模型和分词器
    logger.info(f"\n加载分词器: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    logger.info(f"加载模型: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 加载数据
    logger.info(f"\n加载训练数据: {args.train_file}")
    train_dataset = MultiRoundDialogueDataset(
        tokenizer=tokenizer,
        max_len=args.max_len,
        data_dir=args.data_dir,
        data_set_name="train",
        path_file=args.train_file,
        is_overwrite=False
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_func
    )

    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    logger.info("开始训练")
    '''   
    model: QwenMultiRoundModel,
    train_dataloader: DataLoader,   # 
    optimizer, device, epochs: int, labels, output_dir, max_norm=1.0):
    '''
    # 训练
    train_loss = train_sft_epoch(
        model, train_dataloader, optimizer, args.device,
        epochs = args.epoches, output_dir=args.output_dir
    )
           
    logger.info("\n" + "="*80)
    logger.info("训练完成！")
    logger.info(f"模型已保存到: {args.output_dir}")


if __name__ == "__main__":
    print("start training")
    
    main()
