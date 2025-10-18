# Torch-Handle: 大规模 Transformer 分布式训练工程框架

> 基于 PyTorch 1.10+、torchrun 与 DeepSpeed 的工业级大模型训练实践项目

## 📖 项目简介

本项目是一个**从零开始手搓 Transformer**，并结合**分布式训练（DDP）**和**大规模优化（DeepSpeed ZeRO）**的完整工程实践框架。项目旨在帮助开发者深入理解大模型训练的底层原理，掌握从单卡到多卡、从 DDP 到 DeepSpeed 的完整技术栈。

### 核心特性

- ✅ **完整 Transformer 实现**：从位置编码、多头注意力到完整的 Encoder-Decoder 架构
- ✅ **分布式训练支持**：基于 torchrun 的 DDP 实现，支持多卡/多节点训练
- ✅ **DeepSpeed 集成**：支持 ZeRO Stage 1/2/3，实现超大模型训练
- ✅ **内存优化技术**：梯度检查点、混合精度训练、参数卸载
- ✅ **性能分析工具**：集成 PyTorch Profiler，诊断训练瓶颈
- ✅ **工程最佳实践**：模块化设计、容错机制、Checkpoint 管理

---

## 🏗️ 项目架构：底层到上层的模块设计

本项目采用**分层模块化设计**，严格遵循从底层组件到上层训练逻辑的构建顺序：

```
torch-handle/
│
├── 📁 model/                          # 【Layer 1: 底层模型组件】
│   ├── embeddings.py                  # 词嵌入 + 位置编码（Positional Encoding）
│   ├── attention.py                   # 多头注意力机制（Multi-Head Attention）
│   ├── layers.py                      # FFN、残差连接、LayerNorm
│   └── transformer.py                 # 完整 Transformer（Encoder + Decoder）
│
├── 📁 data/                           # 【Layer 2: 数据处理层】
│   ├── dataset.py                     # 自定义数据集（支持 Tokenization）
│   └── dataloader.py                  # 分布式数据加载器（DistributedSampler）
│
├── 📁 train/                          # 【Layer 3: 训练逻辑层】
│   ├── ddp_trainer.py                 # DDP 训练器（单机多卡 / 多节点）
│   └── deepspeed_trainer.py           # DeepSpeed 训练器（ZeRO 优化）
│
├── 📁 utils/                          # 【Layer 4: 工具与优化】
│   ├── checkpoint.py                  # Checkpoint 保存/加载/容错
│   ├── profiler.py                    # 性能分析工具（PyTorch Profiler）
│   └── gradient_checkpointing.py      # 梯度检查点辅助函数
│
├── 📁 config/                         # 【Layer 5: 配置管理】
│   ├── ds_config_stage1.json          # DeepSpeed Stage 1 配置
│   ├── ds_config_stage2.json          # DeepSpeed Stage 2 配置
│   ├── ds_config_stage3.json          # DeepSpeed Stage 3 配置（LLM 训练）
│   └── model_config.yaml              # 模型超参数配置
│
├── 📁 scripts/                        # 【Layer 6: 启动脚本】
│   ├── train_ddp_single_node.sh       # 单机多卡 DDP 训练
│   ├── train_ddp_multi_node.sh        # 多节点 DDP 训练
│   └── train_deepspeed.sh             # DeepSpeed 训练（支持 ZeRO）
│
├── main.py                            # 【主入口】训练流程编排
├── requirements.txt                   # 依赖管理
└── README.md                          # 项目文档
```

---

## 📚 模块详解：从底层到上层的构建逻辑

### Layer 1: 底层模型组件 (`model/`)

这是项目的**核心**，实现 Transformer 的所有基础组件。

#### 1.1 `embeddings.py` - 词嵌入与位置编码
**构建顺序：第 1 步**

- **功能**：
  - 实现 `nn.Embedding` 的词嵌入层
  - 实现基于正弦/余弦函数的 **Positional Encoding**
  - 使用 `register_buffer()` 确保位置编码在分布式环境下的正确性

- **关键技术**：
  - 位置编码公式：$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$
  - 非训练参数的 Buffer 管理（适配 DDP/DeepSpeed）

#### 1.2 `attention.py` - 多头注意力机制
**构建顺序：第 2 步**

- **功能**：
  - 实现 **Scaled Dot-Product Attention**
  - 实现 **Multi-Head Attention** 的分头并行计算
  - 支持 Padding Mask 和 Causal Mask

- **关键技术**：
  - 注意力公式：$\text{Attention}(Q, K, V) = \text{Softmax}(\frac{QK^T}{\sqrt{d_k}})V$
  - 张量重塑：`(batch, seq_len, d_model)` → `(batch, num_heads, seq_len, d_k)`

#### 1.3 `layers.py` - 前馈网络与残差连接
**构建顺序：第 3 步**

- **功能**：
  - 实现 **Position-wise Feed-Forward Networks (FFN)**
  - 实现 **残差连接 + Layer Normalization**
  - 支持梯度检查点（Gradient Checkpointing）

- **关键技术**：
  - FFN 结构：`Linear(d_model → d_ff) → ReLU/GELU → Linear(d_ff → d_model)`
  - 残差连接：`LayerNorm(x + Sublayer(x))`

#### 1.4 `transformer.py` - 完整 Transformer 架构
**构建顺序：第 4 步**

- **功能**：
  - 组装 **Encoder Block** 和 **Decoder Block**
  - 堆叠多层 Transformer 层
  - 实现 Seq2Seq 任务的输出层

- **依赖关系**：
  - 依赖 `embeddings.py`、`attention.py`、`layers.py`

---

### Layer 2: 数据处理层 (`data/`)

#### 2.1 `dataset.py` - 自定义数据集
**构建顺序：第 5 步**

- **功能**：
  - 实现 `torch.utils.data.Dataset` 接口
  - 支持文本 Tokenization（使用 `torchtext` 或 `tokenizers`）
  - 支持 Seq2Seq 数据对（源语言 + 目标语言）

#### 2.2 `dataloader.py` - 分布式数据加载
**构建顺序：第 6 步**

- **功能**：
  - 封装 `torch.utils.data.DataLoader`
  - 使用 `DistributedSampler` 实现数据分片
  - 支持 `pin_memory` 和异步传输

- **关键技术**：
  - **必须在每个 epoch 调用 `sampler.set_epoch(epoch)`**（避免数据分片重复）
  - 优化：`num_workers > 0`、`pin_memory=True`、`non_blocking=True`

---

### Layer 3: 训练逻辑层 (`train/`)

#### 3.1 `ddp_trainer.py` - DDP 训练器
**构建顺序：第 7 步**

- **功能**：
  - 实现 `ddp_setup()` 和 `destroy_process_group()`
  - 封装模型为 `DDP(model, device_ids=[local_rank])`
  - 实现训练循环：前向 → 损失 → 反向 → 优化
  - 实现 Rank 0 独占的 Checkpoint 保存

- **关键技术**：
  - 初始化进程组：`dist.init_process_group(backend="nccl")`
  - 获取环境变量：`RANK`、`LOCAL_RANK`、`WORLD_SIZE`（由 torchrun 注入）

#### 3.2 `deepspeed_trainer.py` - DeepSpeed 训练器
**构建顺序：第 8 步**

- **功能**：
  - 使用 `deepspeed.initialize()` 封装模型和优化器
  - 支持 ZeRO Stage 1/2/3
  - 支持混合精度训练（FP16/BF16）
  - 支持参数/优化器状态卸载到 CPU/NVMe

- **关键技术**：
  - 读取 `ds_config.json` 配置
  - 使用 `engine.backward()` 和 `engine.step()` 替代原生 PyTorch

---

### Layer 4: 工具与优化 (`utils/`)

#### 4.1 `checkpoint.py` - Checkpoint 管理
**构建顺序：第 9 步**

- **功能**：
  - 保存/加载模型、优化器、训练状态
  - 支持 DDP 和 DeepSpeed 的 Checkpoint 格式
  - 实现训练恢复（从中断点继续）

#### 4.2 `profiler.py` - 性能分析
**构建顺序：第 10 步**

- **功能**：
  - 集成 PyTorch Profiler
  - 导出 TensorBoard 或 Chrome Trace 格式
  - 分析 CPU/GPU 时间、通信/计算重叠

#### 4.3 `gradient_checkpointing.py` - 梯度检查点
**构建顺序：第 11 步**

- **功能**：
  - 封装 `torch.utils.checkpoint.checkpoint()`
  - 应用于 Transformer Block（MHA + FFN）
  - 推荐使用 `use_reentrant=False` 模式

---

### Layer 5: 配置管理 (`config/`)

**构建顺序：第 12 步**

- **`ds_config_stage1.json`**：仅分片优化器状态（适合中等模型）
- **`ds_config_stage2.json`**：分片优化器 + 梯度（内存节省 3-4x）
- **`ds_config_stage3.json`**：全分片（P + G + O），适合 LLM 训练

---

### Layer 6: 启动脚本 (`scripts/`)

**构建顺序：第 13 步**

提供一键启动的 Bash 脚本，封装复杂的 `torchrun` 命令。

---

## 🚀 快速开始

### 环境准备

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd torch-handle

# 2. 安装依赖
pip install -r requirements.txt

# 3. （可选）安装 DeepSpeed
pip install deepspeed
```

### 单机多卡训练（DDP）

```bash
# 使用 8 张 GPU
bash scripts/train_ddp_single_node.sh
```

等价于：
```bash
torchrun --nproc_per_node=8 main.py --mode ddp --epochs 10 --batch_size 32
```

### 多节点训练（DDP）

在**每个节点**上分别执行：

```bash
# 节点 0（主节点）
bash scripts/train_ddp_multi_node.sh --node_rank 0

# 节点 1
bash scripts/train_ddp_multi_node.sh --node_rank 1
```

### DeepSpeed 训练（ZeRO Stage 3）

```bash
bash scripts/train_deepspeed.sh
```

等价于：
```bash
torchrun --nproc_per_node=8 main.py \
    --mode deepspeed \
    --deepspeed config/ds_config_stage3.json \
    --epochs 10
```

---

## 📊 性能分析

启用 PyTorch Profiler 进行性能诊断：

```bash
python main.py --mode ddp --enable_profiler --profile_steps 10
```

查看 TensorBoard：
```bash
tensorboard --logdir=./logs
```

关注指标：
- **通信/计算重叠**：NCCL AllReduce 是否与 CUDA Kernel 并行
- **数据加载效率**：CPU wait_time 是否过高
- **内存峰值**：验证梯度检查点的效果

---

## 🎯 学习路径建议

### 初级：理解核心组件
1. 阅读 `model/embeddings.py` 和 `model/attention.py`
2. 理解 Transformer 的基本计算流程
3. 在单 GPU 上运行 `python main.py --mode single`

### 中级：掌握分布式训练
1. 学习 `train/ddp_trainer.py` 的 DDP 样板代码
2. 理解 `DistributedSampler` 的数据分片机制
3. 运行单机多卡训练并观察加速比

### 高级：大规模模型优化
1. 阅读 `train/deepspeed_trainer.py` 和 DeepSpeed 配置文件
2. 理解 ZeRO Stage 1/2/3 的分片策略
3. 启用梯度检查点并分析内存节省
4. 使用 Profiler 诊断训练瓶颈

---

## 📖 参考资料

本项目基于以下核心技术和论文：

- **Transformer 原理**：*Attention Is All You Need* (Vaswani et al., 2017)
- **DDP 机制**：[PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- **torchrun 启动器**：[Fault-tolerant Distributed Training](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html)
- **DeepSpeed ZeRO**：[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- **梯度检查点**：[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

在提交代码前，请确保：
1. 代码遵循 PEP 8 规范
2. 添加必要的注释和文档字符串
3. 通过基础功能测试

---

## 📜 License

MIT License

---

## 🙏 致谢

感谢 PyTorch、DeepSpeed 和 Hugging Face 社区的开源贡献。

本项目的设计灵感来源于《基于 PyTorch 1.10+、torchrun 与 DeepSpeed 的大规模 Transformer 训练工程指南》。

