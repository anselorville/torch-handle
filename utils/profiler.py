"""
性能分析工具模块

【构建顺序：Layer 4.2 - 第 10 步】

核心功能：
1. ProfilerWrapper: PyTorch Profiler 的封装
2. 分析 CPU/GPU 时间、通信/计算重叠
3. 导出 TensorBoard 或 Chrome Trace 格式

使用示例：
    ```python
    profiler = ProfilerWrapper(log_dir='./logs')
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            # 训练代码
            forward()
            backward()
            step()
            
            # 通知 profiler 步骤结束
            profiler.step()
    
    profiler.export()
    ```
"""

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import os


class ProfilerWrapper:
    """
    PyTorch Profiler 封装类
    
    简化 Profiler 的使用，自动处理调度和导出。
    
    参数：
        log_dir: TensorBoard 日志目录
        wait_steps: 等待步数（跳过初始化阶段）
        warmup_steps: 预热步数
        active_steps: 活跃步数（实际记录的步数）
        repeat: 重复次数
        record_shapes: 是否记录张量形状
        profile_memory: 是否记录内存使用
        with_stack: 是否记录 Python 调用栈
        enabled: 是否启用 profiler
    """
    
    def __init__(
        self,
        log_dir: str = './logs',
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 3,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        enabled: bool = True
    ):
        self.log_dir = log_dir
        self.enabled = enabled
        
        if not self.enabled:
            self.profiler = None
            return
        
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建 profiler 调度器
        self.schedule = schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=repeat
        )
        
        # 创建 profiler
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=self.schedule,
            on_trace_ready=tensorboard_trace_handler(log_dir),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack
        )
        
        # 启动 profiler
        self.profiler.__enter__()
        
        print(f"[Profiler] 已启用，日志保存到: {log_dir}")
        print(f"  - Wait: {wait_steps}, Warmup: {warmup_steps}, Active: {active_steps}, Repeat: {repeat}")
    
    def step(self):
        """
        通知 profiler 当前步骤结束
        
        必须在每个训练步骤结束时调用。
        """
        if self.enabled and self.profiler is not None:
            self.profiler.step()
    
    def export(self, output_path: str = None):
        """
        导出 profiler 结果
        
        参数：
            output_path: 导出路径（Chrome Trace 格式）
        """
        if not self.enabled or self.profiler is None:
            return
        
        # 停止 profiler
        self.profiler.__exit__(None, None, None)
        
        # 导出 Chrome Trace（如果指定了路径）
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.profiler.export_chrome_trace(output_path)
            print(f"[Profiler] Chrome Trace 已导出: {output_path}")
        
        print(f"[Profiler] 分析完成，使用以下命令查看 TensorBoard:")
        print(f"  tensorboard --logdir={self.log_dir}")
    
    def print_summary(self, sort_by: str = 'cuda_time_total', row_limit: int = 10):
        """
        打印性能摘要
        
        参数：
            sort_by: 排序方式（'cpu_time_total', 'cuda_time_total', 'self_cpu_time_total' 等）
            row_limit: 显示的行数
        """
        if not self.enabled or self.profiler is None:
            return
        
        print(f"\n[Profiler] 性能摘要（按 {sort_by} 排序）:")
        print(self.profiler.key_averages().table(sort_by=sort_by, row_limit=row_limit))


def create_simple_profiler(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes: bool = True,
    profile_memory: bool = True
):
    """
    创建一个简单的 profiler（不使用调度器）
    
    使用示例：
        ```python
        with create_simple_profiler() as prof:
            # 训练代码
            forward()
            backward()
            step()
        
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
        prof.export_chrome_trace('trace.json')
        ```
    """
    return profile(
        activities=activities,
        record_shapes=record_shapes,
        profile_memory=profile_memory
    )


def analyze_communication_overlap(profiler):
    """
    分析通信与计算的重叠情况
    
    在 DDP/DeepSpeed 训练中，验证 NCCL 通信（如 AllReduce）是否与计算并行。
    
    参数：
        profiler: PyTorch Profiler 对象
    
    返回：
        analysis: 分析结果字典
    """
    events = profiler.key_averages()
    
    # 查找 NCCL 通信事件
    nccl_events = [e for e in events if 'nccl' in e.key.lower() or 'allreduce' in e.key.lower()]
    
    # 查找计算事件（例如 MatMul）
    compute_events = [e for e in events if 'matmul' in e.key.lower() or 'gemm' in e.key.lower()]
    
    analysis = {
        'nccl_time_total': sum(e.cuda_time_total for e in nccl_events),
        'compute_time_total': sum(e.cuda_time_total for e in compute_events),
        'nccl_count': len(nccl_events),
        'compute_count': len(compute_events),
    }
    
    print("\n[通信/计算重叠分析]")
    print(f"  - NCCL 总时间: {analysis['nccl_time_total'] / 1000:.2f} ms")
    print(f"  - 计算总时间: {analysis['compute_time_total'] / 1000:.2f} ms")
    print(f"  - NCCL 操作数: {analysis['nccl_count']}")
    print(f"  - 计算操作数: {analysis['compute_count']}")
    
    # 理想情况下，NCCL 时间应该被计算时间"隐藏"
    if analysis['nccl_time_total'] > 0:
        overlap_ratio = analysis['compute_time_total'] / analysis['nccl_time_total']
        print(f"  - 计算/通信比率: {overlap_ratio:.2f}x")
        if overlap_ratio > 1.0:
            print("  ✓ 通信与计算有效重叠")
        else:
            print("  ✗ 通信可能成为瓶颈，建议检查通信/计算重叠")
    
    return analysis

