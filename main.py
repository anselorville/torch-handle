"""
主入口文件

【构建顺序：Layer 6 - 最上层】

功能：
1. 训练流程编排
2. 支持单卡、DDP、DeepSpeed 三种训练模式
3. 命令行参数解析
4. 配置文件加载

使用示例：
    # 单卡训练
    python main.py --mode single
    
    # DDP 训练（单机 8 卡）
    torchrun --nproc_per_node=8 main.py --mode ddp
    
    # DeepSpeed 训练（ZeRO Stage 2）
    torchrun --nproc_per_node=8 main.py --mode deepspeed --deepspeed config/ds_config_stage2.json
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

# 导入模型和数据模块
from model import Transformer
from data import create_demo_dataset, create_distributed_dataloader
from data.dataset import TranslationDataset

# 导入训练器
from train import DDPTrainer, DeepSpeedTrainer, ddp_setup, cleanup

# 导入工具
from utils import apply_gradient_checkpointing


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Transformer 分布式训练')
    
    # 训练模式
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'ddp', 'deepspeed'],
                        help='训练模式：single（单卡）、ddp（分布式数据并行）、deepspeed（DeepSpeed ZeRO）')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=512, help='模型嵌入维度')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数量')
    parser.add_argument('--num_encoder_layers', type=int, default=6, help='编码器层数')
    parser.add_argument('--num_decoder_layers', type=int, default=6, help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='FFN 隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 概率')
    parser.add_argument('--use_checkpoint', action='store_true', help='启用梯度检查点')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='训练 epoch 数')
    parser.add_argument('--batch_size', type=int, default=32, help='每个 GPU 的批次大小')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='学习率预热步数')
    
    # DeepSpeed 配置
    parser.add_argument('--deepspeed', type=str, default=None, help='DeepSpeed 配置文件路径')
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint 保存目录')
    parser.add_argument('--resume_from', type=str, default=None, help='恢复训练的 checkpoint 路径')
    parser.add_argument('--save_every', type=int, default=1, help='每隔多少个 epoch 保存一次 checkpoint')
    
    # 其他
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 工作进程数')
    parser.add_argument('--enable_profiler', action='store_true', help='启用性能分析')
    
    args = parser.parse_args()
    return args


def get_lr_scheduler(optimizer, warmup_steps, d_model):
    """
    创建学习率调度器（Transformer 原始论文的调度策略）
    
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    def lr_lambda(step):
        if step == 0:
            step = 1
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    
    return LambdaLR(optimizer, lr_lambda)


def train_single_gpu(args):
    """单卡训练"""
    print("=" * 60)
    print("单卡训练模式")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建演示数据集
    dataset = create_demo_dataset()
    src_vocab_size = len(dataset.src_vocab)
    tgt_vocab_size = len(dataset.tgt_vocab)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"源语言词汇表大小: {src_vocab_size}")
    print(f"目标语言词汇表大小: {tgt_vocab_size}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_checkpoint=args.use_checkpoint
    ).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建优化器和损失函数
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_pad_idx)
    
    # 创建数据加载器（非分布式）
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: TranslationDataset.collate_fn(batch, dataset.tgt_pad_idx),
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 简单的训练循环
    print("\n开始训练...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]
            
            output = model(src, tgt_input)
            output = output.reshape(-1, output.size(-1))
            tgt_label = tgt_label.reshape(-1)
            
            loss = criterion(output, tgt_label)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    
    print("\n训练完成！")


def train_ddp(args):
    """DDP 分布式训练"""
    print("=" * 60)
    print("DDP 分布式训练模式")
    print("=" * 60)
    
    # 初始化 DDP
    rank, local_rank, world_size = ddp_setup()
    device = torch.device(f'cuda:{local_rank}')
    
    # 创建演示数据集
    dataset = create_demo_dataset()
    src_vocab_size = len(dataset.src_vocab)
    tgt_vocab_size = len(dataset.tgt_vocab)
    
    if rank == 0:
        print(f"数据集大小: {len(dataset)}")
        print(f"源语言词汇表大小: {src_vocab_size}")
        print(f"目标语言词汇表大小: {tgt_vocab_size}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_checkpoint=args.use_checkpoint
    )
    
    if rank == 0:
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建优化器和损失函数
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_pad_idx)
    
    # 创建分布式数据加载器
    train_loader, train_sampler = create_distributed_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda batch: TranslationDataset.collate_fn(batch, dataset.tgt_pad_idx),
        is_distributed=True
    )
    
    # 创建训练器
    trainer = DDPTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        device=device,
        rank=rank,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # 开始训练
    if rank == 0:
        print("\n开始训练...")
    
    trainer.train(num_epochs=args.epochs)
    
    # 清理
    cleanup()


def train_deepspeed(args):
    """DeepSpeed 训练"""
    print("=" * 60)
    print("DeepSpeed 训练模式")
    print("=" * 60)
    
    if args.deepspeed is None:
        raise ValueError("DeepSpeed 模式需要指定 --deepspeed 配置文件")
    
    # 初始化 DDP（DeepSpeed 基于 DDP）
    rank, local_rank, world_size = ddp_setup()
    
    # 创建演示数据集
    dataset = create_demo_dataset()
    src_vocab_size = len(dataset.src_vocab)
    tgt_vocab_size = len(dataset.tgt_vocab)
    
    if rank == 0:
        print(f"数据集大小: {len(dataset)}")
        print(f"源语言词汇表大小: {src_vocab_size}")
        print(f"目标语言词汇表大小: {tgt_vocab_size}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_checkpoint=args.use_checkpoint
    )
    
    if rank == 0:
        print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.tgt_pad_idx)
    
    # 创建分布式数据加载器
    train_loader, train_sampler = create_distributed_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=lambda batch: TranslationDataset.collate_fn(batch, dataset.tgt_pad_idx),
        is_distributed=True
    )
    
    # 创建 DeepSpeed 训练器
    trainer = DeepSpeedTrainer(
        model=model,
        train_loader=train_loader,
        config=args.deepspeed,
        criterion=criterion,
        rank=rank,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # 开始训练
    if rank == 0:
        print("\n开始训练...")
    
    trainer.train(num_epochs=args.epochs)
    
    # 清理
    cleanup()


def main():
    """主函数"""
    args = get_args()
    
    if args.mode == 'single':
        train_single_gpu(args)
    elif args.mode == 'ddp':
        train_ddp(args)
    elif args.mode == 'deepspeed':
        train_deepspeed(args)
    else:
        raise ValueError(f"不支持的训练模式: {args.mode}")


if __name__ == '__main__':
    main()

