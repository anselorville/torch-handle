# 快速开始指南

本文档提供了 `torch-handle` 项目的快速上手指南，帮助您在 5 分钟内运行第一个训练任务。

---

## 🚀 第一步：环境准备

### 1.1 克隆项目

```bash
cd /path/to/your/workspace
git clone <your-repo-url>
cd torch-handle
```

### 1.2 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

**最小依赖**（不使用 DeepSpeed）：
```bash
pip install torch>=1.10.0 torchvision numpy tqdm pyyaml tensorboard
```

---

## 🎯 第二步：运行您的第一个训练

### 2.1 单卡训练（最简单）

无需任何分布式配置，直接运行：

```bash
python main.py --mode single --epochs 5 --batch_size 32
```

**预期输出**：
```
==========================================
单卡训练模式
==========================================
使用设备: cuda
数据集大小: 5
源语言词汇表大小: 18
目标语言词汇表大小: 19
模型参数量: 8.76M

开始训练...
Epoch 0: Loss = 2.8543
Epoch 1: Loss = 2.4321
...
```

---

### 2.2 单机多卡训练（DDP）

如果您有多张 GPU（例如 2 张或 8 张）：

```bash
# 方式 1：使用启动脚本（推荐）
bash scripts/train_ddp_single_node.sh

# 方式 2：手动指定 GPU 数量
torchrun --nproc_per_node=2 main.py --mode ddp --epochs 5 --batch_size 32
```

**重要提示**：
- `--nproc_per_node` 应等于您的 GPU 数量
- 批次大小是**每个 GPU** 的批次大小，总批次大小 = `batch_size × GPU数量`

---

### 2.3 DeepSpeed 训练（内存优化）

如果您想训练更大的模型或使用更大的批次：

```bash
# 方式 1：使用启动脚本（推荐）
bash scripts/train_deepspeed.sh

# 方式 2：手动指定配置
torchrun --nproc_per_node=2 main.py \
    --mode deepspeed \
    --deepspeed config/ds_config_stage2.json \
    --epochs 5 \
    --batch_size 16
```

**ZeRO Stage 选择**：
- Stage 1：适合中等模型，内存节省 ~1.5x
- Stage 2：适合大型模型，内存节省 ~3-4x（**推荐**）
- Stage 3：适合超大模型（LLM），内存节省 >10x

---

## 📊 第三步：查看训练结果

### 3.1 Checkpoint 位置

训练过程中，模型会自动保存到：
- 单卡/DDP：`./checkpoints/checkpoint_epoch_N.pt`
- DeepSpeed：`./checkpoints_deepspeed/epoch_N/`

### 3.2 查看日志

如果启用了 TensorBoard：

```bash
tensorboard --logdir=./logs
```

然后在浏览器中打开 `http://localhost:6006`

---

## 🔧 第四步：自定义您的训练

### 4.1 修改模型参数

```bash
python main.py \
    --mode single \
    --d_model 256 \           # 嵌入维度（默认 512）
    --num_heads 4 \           # 注意力头数量（默认 8）
    --num_encoder_layers 3 \  # 编码器层数（默认 6）
    --num_decoder_layers 3 \  # 解码器层数（默认 6）
    --d_ff 1024 \             # FFN 隐藏层维度（默认 2048）
    --dropout 0.2             # Dropout 概率（默认 0.1）
```

### 4.2 启用梯度检查点（减少内存）

```bash
python main.py --mode ddp --use_checkpoint
```

这会将激活内存从 O(N) 降至 O(√N)，代价是增加 30-50% 的计算时间。

### 4.3 使用真实数据集

**准备数据**：
```bash
# 创建数据目录
mkdir -p data

# 将您的平行语料库放入 data/ 目录
# 例如：data/train.src（源语言）和 data/train.tgt（目标语言）
```

**修改 `main.py`**：

将 `create_demo_dataset()` 替换为：

```python
from data.dataset import load_parallel_corpus, create_vocabulary, TranslationDataset

# 加载数据
src_sentences, tgt_sentences = load_parallel_corpus(
    src_file='data/train.src',
    tgt_file='data/train.tgt'
)

# 创建词汇表
src_vocab = create_vocabulary(src_sentences, min_freq=2)
tgt_vocab = create_vocabulary(tgt_sentences, min_freq=2)

# 创建数据集
dataset = TranslationDataset(
    src_sentences=src_sentences,
    tgt_sentences=tgt_sentences,
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
    max_len=100
)
```

---

## 🛠️ 常见问题

### Q1: `RuntimeError: CUDA out of memory`

**解决方案**：
1. 减小批次大小：`--batch_size 16` → `--batch_size 8`
2. 启用梯度检查点：`--use_checkpoint`
3. 使用 DeepSpeed ZeRO Stage 2 或 3

### Q2: `RuntimeError: 未检测到 torchrun 环境变量`

**原因**：您在使用 DDP 模式时直接运行 `python main.py`。

**解决方案**：使用 `torchrun` 启动：
```bash
torchrun --nproc_per_node=1 main.py --mode ddp
```

### Q3: 训练速度很慢

**排查步骤**：
1. 检查是否使用了 GPU：`nvidia-smi`
2. 增加 DataLoader 工作进程：`--num_workers 8`
3. 启用性能分析：`--enable_profiler`
4. 确认数据加载不是瓶颈（使用 Profiler）

### Q4: 多节点训练时进程卡住

**常见原因**：
1. 网络配置错误：检查 `MASTER_ADDR` 和 `MASTER_PORT`
2. 防火墙阻止通信：开放端口 `29500`
3. 不同节点的代码不一致：确保所有节点使用相同的代码版本

---

## 📚 下一步学习

完成快速开始后，您可以：

1. **阅读完整文档**：`README.md`
2. **理解项目结构**：`PROJECT_STRUCTURE.md`
3. **学习训练指南**：`docs/基于 PyTorch 1.10+、torchrun 与 DeepSpeed 的大规模 Transformer 训练工程指南.md`
4. **深入源码**：从 `model/embeddings.py` 开始逐步阅读

---

## 🎓 推荐学习路径

```
1. 单卡训练（理解基本流程）
   ↓
2. 单机多卡 DDP（理解分布式）
   ↓
3. DeepSpeed Stage 2（理解 ZeRO 优化）
   ↓
4. 梯度检查点（理解内存优化）
   ↓
5. 性能分析（理解训练瓶颈）
   ↓
6. 多节点训练（理解大规模训练）
```

---

**祝您训练愉快！** 🚀

如有问题，请提交 Issue 或参考训练指南文档。

