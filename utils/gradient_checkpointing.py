"""
梯度检查点（Gradient Checkpointing）辅助函数模块

【构建顺序：Layer 4.3 - 第 11 步】

核心功能：
1. apply_gradient_checkpointing: 为模型应用梯度检查点
2. 减少激活值内存消耗（从 O(N) 降至 O(√N)）

关键技术：
- 使用 torch.utils.checkpoint.checkpoint()
- 推荐使用 use_reentrant=False 模式
- 应用于 Transformer Block（MHA + FFN）
"""

import torch
from torch.utils.checkpoint import checkpoint


def apply_gradient_checkpointing(model, checkpoint_blocks: bool = True):
    """
    为 Transformer 模型应用梯度检查点
    
    梯度检查点是一种以计算换内存的技术：
    - 在前向传播时，不保存所有中间激活值
    - 在反向传播时，从最近的检查点重新计算激活值
    - 内存消耗从 O(N) 降至 O(√N)，代价是增加 30-50% 的计算时间
    
    参数：
        model: Transformer 模型
        checkpoint_blocks: 是否为 Transformer Block 启用梯度检查点
    
    使用示例：
        ```python
        model = Transformer(...)
        apply_gradient_checkpointing(model, checkpoint_blocks=True)
        ```
    """
    if not checkpoint_blocks:
        return
    
    # 为编码器层启用梯度检查点
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        for layer in model.encoder.layers:
            _enable_gradient_checkpointing_for_layer(layer)
        print(f"[Gradient Checkpointing] 已为 {len(model.encoder.layers)} 个编码器层启用")
    
    # 为解码器层启用梯度检查点
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        for layer in model.decoder.layers:
            _enable_gradient_checkpointing_for_layer(layer)
        print(f"[Gradient Checkpointing] 已为 {len(model.decoder.layers)} 个解码器层启用")


def _enable_gradient_checkpointing_for_layer(layer):
    """
    为单个 Transformer 层启用梯度检查点
    
    通过替换层的 forward 方法，使其使用 checkpoint()。
    """
    original_forward = layer.forward
    
    def checkpointed_forward(*args, **kwargs):
        # 使用 checkpoint 包装原始的 forward
        # use_reentrant=False 模式更加内存高效
        return checkpoint(
            original_forward,
            *args,
            use_reentrant=False,
            **kwargs
        )
    
    # 替换 forward 方法
    layer.forward = checkpointed_forward


def checkpoint_sequential(functions, segments, *inputs):
    """
    对顺序模块应用梯度检查点（分段）
    
    将模型分割为多个段，每个段作为一个检查点。
    
    参数：
        functions: 模型的各层（例如 nn.ModuleList）
        segments: 分段数量
        inputs: 输入张量
    
    返回：
        output: 输出张量
    
    使用示例：
        ```python
        # 将 12 层 Transformer 分为 4 段，每段 3 层
        output = checkpoint_sequential(model.layers, segments=4, x)
        ```
    """
    if segments <= 0:
        raise ValueError("segments 必须大于 0")
    
    # 计算每段的层数
    num_functions = len(functions)
    segment_size = (num_functions + segments - 1) // segments  # 向上取整
    
    def run_segment(start, end, *args):
        """运行一个段的所有层"""
        for i in range(start, end):
            args = (functions[i](*args),) if not isinstance(args, tuple) else (functions[i](*args),)
        return args[0]
    
    # 对每个段应用 checkpoint
    output = inputs
    for i in range(0, num_functions, segment_size):
        end = min(i + segment_size, num_functions)
        output = checkpoint(run_segment, i, end, output, use_reentrant=False)
    
    return output


def get_checkpoint_memory_stats():
    """
    获取当前的内存统计信息（用于验证梯度检查点的效果）
    
    返回：
        stats: 内存统计字典
    """
    if not torch.cuda.is_available():
        return {}
    
    stats = {
        'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
        'reserved': torch.cuda.memory_reserved() / 1024**3,  # GB
        'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
    }
    
    return stats


def print_memory_stats(prefix: str = ""):
    """
    打印当前的内存统计信息
    
    参数：
        prefix: 前缀字符串
    """
    stats = get_checkpoint_memory_stats()
    
    if not stats:
        print(f"{prefix}[内存] CUDA 不可用")
        return
    
    print(f"{prefix}[内存] 已分配: {stats['allocated']:.2f} GB, "
          f"已保留: {stats['reserved']:.2f} GB, "
          f"峰值: {stats['max_allocated']:.2f} GB")


def reset_memory_stats():
    """重置内存统计信息"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print("[内存] 峰值内存统计已重置")

