"""
DeepSpeed 训练器模块

【构建顺序：Layer 3.2 - 第 8 步】

核心功能：
1. DeepSpeedTrainer: 集成 DeepSpeed 的训练器
2. 支持 ZeRO Stage 1/2/3
3. 支持混合精度训练（FP16/BF16）
4. 支持参数/优化器卸载

关键技术：
- 使用 deepspeed.initialize() 封装模型和优化器
- 使用 engine.backward() 和 engine.step() 替代原生 PyTorch
- 支持 Gradient Accumulation（梯度累积）
"""

import os
import torch
import torch.nn as nn
from typing import Optional
from tqdm import tqdm

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("[Warning] DeepSpeed 未安装。请运行: pip install deepspeed")


class DeepSpeedTrainer:
    """
    DeepSpeed 训练器
    
    封装完整的 DeepSpeed 训练流程，支持：
    - ZeRO Stage 1/2/3（参数/梯度/优化器状态分片）
    - 混合精度训练（FP16/BF16）
    - 梯度累积
    - Checkpoint 保存和加载
    
    参数：
        model: PyTorch 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器（可选）
        config: DeepSpeed 配置字典或配置文件路径
        criterion: 损失函数
        rank: 全局秩
        save_every: 每隔多少个 epoch 保存一次 checkpoint
        checkpoint_dir: checkpoint 保存目录
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        config=None,
        criterion: nn.Module = None,
        rank: int = 0,
        save_every: int = 1,
        checkpoint_dir: str = './checkpoints_deepspeed'
    ):
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed 未安装，请运行: pip install deepspeed")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.rank = rank
        self.save_every = save_every
        self.checkpoint_dir = checkpoint_dir
        
        # 初始化 DeepSpeed 引擎
        # deepspeed.initialize() 会自动处理模型、优化器和学习率调度器
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=config,
            model_parameters=model.parameters()
        )
        
        # 获取设备
        self.device = self.model_engine.device
        
        # 创建 checkpoint 目录（仅 Rank 0）
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"[DeepSpeed] 初始化完成")
            print(f"  - ZeRO Stage: {self.model_engine.zero_optimization_stage()}")
            print(f"  - FP16 Enabled: {self.model_engine.fp16_enabled()}")
            print(f"  - BF16 Enabled: {self.model_engine.bfloat16_enabled()}")
    
    def train_epoch(self, epoch: int) -> float:
        """
        训练一个 epoch
        
        参数：
            epoch: 当前 epoch 编号
        
        返回：
            avg_loss: 平均损失
        """
        self.model_engine.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 使用 tqdm 显示进度（仅 Rank 0）
        if self.rank == 0:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            progress_bar = self.train_loader
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            # 将数据移到设备
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            # 前向传播
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]
            
            output = self.model_engine(src, tgt_input)
            
            # 计算损失
            output = output.reshape(-1, output.size(-1))
            tgt_label = tgt_label.reshape(-1)
            
            loss = self.criterion(output, tgt_label)
            
            # 反向传播（DeepSpeed 自动处理梯度累积和混合精度）
            self.model_engine.backward(loss)
            
            # 优化器更新（DeepSpeed 会自动处理梯度同步）
            self.model_engine.step()
            
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
        
        self.model_engine.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for src, tgt in self.val_loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_label = tgt[:, 1:]
                
                output = self.model_engine(src, tgt_input)
                output = output.reshape(-1, output.size(-1))
                tgt_label = tgt_label.reshape(-1)
                
                loss = self.criterion(output, tgt_label)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, epoch: int, val_loss: Optional[float] = None):
        """
        保存 checkpoint（DeepSpeed 格式）
        
        DeepSpeed 会自动处理分片模型的保存，支持 ZeRO Stage 3。
        
        参数：
            epoch: 当前 epoch
            val_loss: 验证损失（可选）
        """
        tag = f"epoch_{epoch}"
        
        # DeepSpeed 的 save_checkpoint 会自动在所有 Rank 上同步
        self.model_engine.save_checkpoint(
            save_dir=self.checkpoint_dir,
            tag=tag,
            client_state={'epoch': epoch, 'val_loss': val_loss}
        )
        
        if self.rank == 0:
            print(f"[Rank {self.rank}] DeepSpeed Checkpoint 已保存: {self.checkpoint_dir}/{tag}")
    
    def load_checkpoint(self, tag: str):
        """
        加载 checkpoint（DeepSpeed 格式）
        
        参数：
            tag: checkpoint 标签（例如 'epoch_10'）
        
        返回:
            start_epoch: 下一个 epoch 的编号
        """
        _, client_state = self.model_engine.load_checkpoint(
            load_dir=self.checkpoint_dir,
            tag=tag
        )
        
        start_epoch = client_state.get('epoch', 0) + 1
        
        if self.rank == 0:
            print(f"[Rank {self.rank}] DeepSpeed Checkpoint 已加载: {self.checkpoint_dir}/{tag}")
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


def get_deepspeed_config(
    stage: int = 2,
    micro_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    fp16_enabled: bool = True,
    bf16_enabled: bool = False,
    offload_optimizer: bool = False,
    offload_param: bool = False
) -> dict:
    """
    生成 DeepSpeed 配置字典
    
    参数：
        stage: ZeRO 优化阶段（0, 1, 2, 3）
        micro_batch_size: 微批次大小（每个 GPU 的批次大小）
        gradient_accumulation_steps: 梯度累积步数
        fp16_enabled: 是否启用 FP16 混合精度
        bf16_enabled: 是否启用 BF16 混合精度
        offload_optimizer: 是否将优化器状态卸载到 CPU
        offload_param: 是否将参数卸载到 CPU（仅 Stage 3）
    
    返回：
        config: DeepSpeed 配置字典
    """
    config = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "steps_per_print": 100,
        "zero_optimization": {
            "stage": stage
        }
    }
    
    # 混合精度配置
    if fp16_enabled:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,  # 动态损失缩放
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    
    if bf16_enabled:
        config["bf16"] = {"enabled": True}
    
    # ZeRO Stage 2/3 特定配置
    if stage >= 2:
        config["zero_optimization"]["allgather_bucket_size"] = 5e8
        config["zero_optimization"]["reduce_bucket_size"] = 5e8
    
    # 优化器卸载（Stage 2/3）
    if offload_optimizer and stage >= 2:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # 参数卸载（Stage 3）
    if offload_param and stage == 3:
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # Stage 3 额外配置
    if stage == 3:
        config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = True
    
    return config

