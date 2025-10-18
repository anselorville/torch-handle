"""
DDP (DistributedDataParallel) 训练器模块

【构建顺序：Layer 3.1 - 第 7 步】

核心功能：
1. ddp_setup: 初始化 DDP 进程组
2. cleanup: 清理 DDP 资源
3. DDPTrainer: DDP 训练器类

关键技术：
- 使用 torchrun 注入的环境变量（RANK, LOCAL_RANK, WORLD_SIZE）
- DDP 模型封装：DDP(model, device_ids=[local_rank])
- Rank 0 独占的 Checkpoint 保存
- 梯度 Bucketing 和通信/计算重叠（DDP 自动实现）
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional
from tqdm import tqdm


def ddp_setup():
    """
    初始化 DDP 进程组
    
    从 torchrun 注入的环境变量中读取配置：
    - RANK: 全局进程秩（0 到 WORLD_SIZE-1）
    - LOCAL_RANK: 节点内的本地秩（0 到 nproc_per_node-1）
    - WORLD_SIZE: 总进程数
    - MASTER_ADDR: 主节点地址
    - MASTER_PORT: 主节点端口
    
    返回：
        rank: 全局秩
        local_rank: 本地秩
        world_size: 总进程数
    """
    # 检查是否在 torchrun 环境中
    if 'RANK' not in os.environ:
        raise RuntimeError(
            "未检测到 torchrun 环境变量。\n"
            "请使用 torchrun 启动：\n"
            "  torchrun --nproc_per_node=N your_script.py"
        )
    
    # 获取环境变量
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    # 初始化进程组（使用 NCCL 后端，适用于 GPU 通信）
    dist.init_process_group(backend='nccl')
    
    # 将当前进程绑定到指定的 GPU
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print(f"[DDP Setup] 初始化完成")
        print(f"  - RANK: {rank}")
        print(f"  - LOCAL_RANK: {local_rank}")
        print(f"  - WORLD_SIZE: {world_size}")
        print(f"  - MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'N/A')}")
        print(f"  - MASTER_PORT: {os.environ.get('MASTER_PORT', 'N/A')}")
    
    return rank, local_rank, world_size


def cleanup():
    """
    清理 DDP 资源
    
    在训练结束时调用，确保进程组被正确销毁。
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        print("[DDP Cleanup] 进程组已销毁")


class DDPTrainer:
    """
    DDP 分布式训练器
    
    封装完整的 DDP 训练流程，包括：
    - 模型初始化和 DDP 封装
    - 训练循环
    - Checkpoint 保存和加载
    - 日志记录（仅 Rank 0）
    
    参数：
        model: PyTorch 模型
        optimizer: 优化器
        criterion: 损失函数
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        device: 设备（通常是 'cuda:local_rank'）
        rank: 全局秩
        save_every: 每隔多少个 epoch 保存一次 checkpoint
        checkpoint_dir: checkpoint 保存目录
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader,
        val_loader=None,
        device: torch.device = None,
        rank: int = 0,
        save_every: int = 1,
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else torch.device('cuda')
        self.rank = rank
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        
        # 将模型移到设备并封装为 DDP
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.device.index])
        
        # 创建 checkpoint 目录（仅 Rank 0）
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """
        训练一个 epoch
        
        参数：
            epoch: 当前 epoch 编号
        
        返回：
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 使用 tqdm 显示进度（仅 Rank 0）
        if self.rank == 0:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            progress_bar = self.train_loader
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            # 将数据移到设备（支持异步传输）
            src = src.to(self.device, non_blocking=True)
            tgt = tgt.to(self.device, non_blocking=True)
            
            # 前向传播
            # 注意：对于 Seq2Seq 任务，通常使用 tgt[:, :-1] 作为输入
            # 使用 tgt[:, 1:] 作为标签（Teacher Forcing）
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]
            
            output = self.model(src, tgt_input)  # shape: (batch, seq_len-1, vocab_size)
            
            # 计算损失
            # 需要重塑张量以匹配 CrossEntropyLoss 的输入格式
            output = output.reshape(-1, output.size(-1))  # (batch * seq_len, vocab_size)
            tgt_label = tgt_label.reshape(-1)  # (batch * seq_len)
            
            loss = self.criterion(output, tgt_label)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器更新
            self.optimizer.step()
            
            # 累积损失
            total_loss += loss.item()
            
            # 更新进度条（仅 Rank 0）
            if self.rank == 0:
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """
        验证模型
        
        返回：
            avg_loss: 平均验证损失
        """
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src = src.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)
                
                tgt_input = tgt[:, :-1]
                tgt_label = tgt[:, 1:]
                
                output = self.model(src, tgt_input)
                output = output.reshape(-1, output.size(-1))
                tgt_label = tgt_label.reshape(-1)
                
                loss = self.criterion(output, tgt_label)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None):
        """
        保存 checkpoint（仅 Rank 0）
        
        参数：
            epoch: 当前 epoch
            val_loss: 验证损失（可选）
        """
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # 访问 DDP 内部的原始模型
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"[Rank {self.rank}] Checkpoint 已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载 checkpoint
        
        参数：
            checkpoint_path: checkpoint 文件路径
        """
        # 使用 map_location 确保模型加载到正确的设备
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        
        if self.rank == 0:
            print(f"[Rank {self.rank}] Checkpoint 已加载: {checkpoint_path}")
            print(f"  - 恢复训练从 Epoch {start_epoch} 开始")
        
        return start_epoch
    
    def train(self, num_epochs: int, start_epoch: int = 0):
        """
        完整训练流程
        
        参数：
            num_epochs: 训练的总 epoch 数
            start_epoch: 起始 epoch（用于恢复训练）
        """
        for epoch in range(start_epoch, num_epochs):
            # 【关键】设置数据加载器的 epoch（确保随机性）
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            # 训练一个 epoch
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss = self.validate()
            
            # 日志记录（仅 Rank 0）
            if self.rank == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # 保存 checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch, val_loss)
        
        if self.rank == 0:
            print("训练完成！")

