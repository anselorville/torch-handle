"""
训练逻辑模块

包含以下子模块：
- ddp_trainer: DDP 分布式训练器
- deepspeed_trainer: DeepSpeed 训练器
"""

from .ddp_trainer import DDPTrainer, ddp_setup, cleanup
from .deepspeed_trainer import DeepSpeedTrainer

__all__ = [
    'DDPTrainer',
    'ddp_setup',
    'cleanup',
    'DeepSpeedTrainer',
]

