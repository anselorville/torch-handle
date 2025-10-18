"""
Checkpoint 管理模块

【构建顺序：Layer 4.1 - 第 9 步】

核心功能：
1. 保存和加载模型 Checkpoint
2. 支持 DDP 和 DeepSpeed 格式
3. 支持训练恢复（从中断点继续）
"""

import os
import torch
from typing import Optional, Dict, Any


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    save_path: str,
    scheduler=None,
    additional_state: Optional[Dict[str, Any]] = None,
    is_ddp: bool = False
):
    """
    保存训练 checkpoint
    
    参数：
        model: PyTorch 模型（可以是 DDP 包装的模型）
        optimizer: 优化器
        epoch: 当前 epoch
        save_path: 保存路径
        scheduler: 学习率调度器（可选）
        additional_state: 额外的状态信息（可选）
        is_ddp: 是否为 DDP 模型
    """
    # 获取模型的 state_dict
    if is_ddp:
        # DDP 模型需要访问 module 属性
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    # 构建 checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # 添加学习率调度器状态
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 添加额外状态
    if additional_state is not None:
        checkpoint.update(additional_state)
    
    # 保存到文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint 已保存: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer=None,
    scheduler=None,
    device=None,
    is_ddp: bool = False
) -> Dict[str, Any]:
    """
    加载训练 checkpoint
    
    参数：
        checkpoint_path: checkpoint 文件路径
        model: PyTorch 模型（可以是 DDP 包装的模型）
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 设备（用于 map_location）
        is_ddp: 是否为 DDP 模型
    
    返回：
        checkpoint: 完整的 checkpoint 字典
    """
    # 加载 checkpoint
    if device is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_path)
    
    # 加载模型状态
    if is_ddp:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载学习率调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint 已加载: {checkpoint_path}")
    print(f"  - Epoch: {epoch}")
    
    return checkpoint


def get_latest_checkpoint(checkpoint_dir: str, prefix: str = 'checkpoint_epoch_') -> Optional[str]:
    """
    获取目录中最新的 checkpoint
    
    参数：
        checkpoint_dir: checkpoint 目录
        prefix: checkpoint 文件名前缀
    
    返回：
        最新 checkpoint 的路径，如果没有则返回 None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    # 查找所有 checkpoint 文件
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith('.pt')]
    
    if not checkpoints:
        return None
    
    # 提取 epoch 编号并排序
    def extract_epoch(filename):
        try:
            # 假设格式为 'checkpoint_epoch_10.pt'
            epoch_str = filename.replace(prefix, '').replace('.pt', '')
            return int(epoch_str)
        except:
            return -1
    
    checkpoints.sort(key=extract_epoch, reverse=True)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    
    print(f"找到最新的 checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def clean_old_checkpoints(checkpoint_dir: str, keep_last_n: int = 3, prefix: str = 'checkpoint_epoch_'):
    """
    清理旧的 checkpoint，只保留最新的 N 个
    
    参数：
        checkpoint_dir: checkpoint 目录
        keep_last_n: 保留最新的 N 个 checkpoint
        prefix: checkpoint 文件名前缀
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # 查找所有 checkpoint 文件
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(prefix) and f.endswith('.pt')]
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # 提取 epoch 编号并排序
    def extract_epoch(filename):
        try:
            epoch_str = filename.replace(prefix, '').replace('.pt', '')
            return int(epoch_str)
        except:
            return -1
    
    checkpoints.sort(key=extract_epoch, reverse=True)
    
    # 删除旧的 checkpoint
    for checkpoint in checkpoints[keep_last_n:]:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        os.remove(checkpoint_path)
        print(f"已删除旧的 checkpoint: {checkpoint_path}")

