# 项目结构详解

本文档详细说明了 `torch-handle` 项目的目录结构和模块设计原则。

## 📂 目录结构

```
torch-handle/
│
├── 📁 model/                          # 【Layer 1: 底层模型组件】
│   ├── __init__.py                    # 模块导出
│   ├── embeddings.py                  # 词嵌入 + 位置编码
│   ├── attention.py                   # 多头注意力机制
│   ├── layers.py                      # FFN、残差连接、LayerNorm
│   └── transformer.py                 # 完整 Transformer（Encoder + Decoder）
│
├── 📁 data/                           # 【Layer 2: 数据处理层】
│   ├── __init__.py
│   ├── dataset.py                     # 自定义数据集（Tokenization、词汇表）
│   └── dataloader.py                  # 分布式数据加载器（DistributedSampler）
│
├── 📁 train/                          # 【Layer 3: 训练逻辑层】
│   ├── __init__.py
│   ├── ddp_trainer.py                 # DDP 训练器（单机/多节点）
│   └── deepspeed_trainer.py           # DeepSpeed 训练器（ZeRO 优化）
│
├── 📁 utils/                          # 【Layer 4: 工具与优化】
│   ├── __init__.py
│   ├── checkpoint.py                  # Checkpoint 保存/加载/管理
│   ├── profiler.py                    # 性能分析工具（PyTorch Profiler）
│   └── gradient_checkpointing.py      # 梯度检查点辅助函数
│
├── 📁 config/                         # 【Layer 5: 配置管理】
│   ├── ds_config_stage1.json          # DeepSpeed Stage 1 配置
│   ├── ds_config_stage2.json          # DeepSpeed Stage 2 配置
│   ├── ds_config_stage3.json          # DeepSpeed Stage 3 配置（LLM）
│   └── model_config.yaml              # 模型超参数配置
│
├── 📁 scripts/                        # 【Layer 6: 启动脚本】
│   ├── train_ddp_single_node.sh       # 单机多卡 DDP 训练
│   ├── train_ddp_multi_node.sh        # 多节点 DDP 训练
│   └── train_deepspeed.sh             # DeepSpeed 训练（ZeRO）
│
├── main.py                            # 【主入口】训练流程编排
├── requirements.txt                   # 依赖管理
├── README.md                          # 项目文档
├── PROJECT_STRUCTURE.md               # 本文件
└── .gitignore                         # Git 忽略文件
```

---

## 🔄 模块依赖关系

本项目采用**自底向上**的模块设计，遵循以下依赖原则：

```
┌─────────────────────────────────────────────────────────────┐
│                     Layer 6: 主入口                          │
│                     main.py                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│              Layer 5: 配置管理                                │
│         config/*.json, config/*.yaml                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│              Layer 4: 工具与优化                              │
│     utils/ (checkpoint, profiler, gradient_checkpointing)    │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────────┐
│              Layer 3: 训练逻辑层                              │
│          train/ (ddp_trainer, deepspeed_trainer)             │
└─────────┬────────────────────────────────────────────────────┘
          │
┌─────────┴──────────────────┬─────────────────────────────────┐
│      Layer 2: 数据处理层    │                                 │
│       data/ (dataset,      │                                 │
│         dataloader)        │                                 │
└────────────────────────────┘                                 │
                                                                │
┌───────────────────────────────────────────────────────────────┘
│                   Layer 1: 底层模型组件                        │
│       model/ (embeddings, attention, layers, transformer)     │
└───────────────────────────────────────────────────────────────┘
```

### 依赖规则

1. **上层依赖下层**：每一层只能依赖比它更底层的模块
2. **同层独立**：同一层的模块之间尽量保持独立
3. **接口清晰**：每个模块通过 `__init__.py` 明确导出的接口

---

## 📝 各层详细说明

### Layer 1: 底层模型组件 (`model/`)

**构建顺序**：1 → 2 → 3 → 4

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `embeddings.py` | 词嵌入和位置编码 | `PositionalEncoding`, `TransformerEmbedding` |
| `attention.py` | 多头注意力机制 | `ScaledDotProductAttention`, `MultiHeadAttention` |
| `layers.py` | Transformer 层组件 | `PositionwiseFeedForward`, `TransformerBlock` |
| `transformer.py` | 完整 Transformer 架构 | `Transformer`, `TransformerEncoder`, `TransformerDecoder` |

**依赖关系**：
- `transformer.py` 依赖 `embeddings.py`, `attention.py`, `layers.py`
- `layers.py` 依赖 `attention.py`
- `embeddings.py` 和 `attention.py` 互相独立

---

### Layer 2: 数据处理层 (`data/`)

**构建顺序**：5 → 6

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `dataset.py` | 自定义数据集 | `TranslationDataset`, `create_vocabulary`, `load_parallel_corpus` |
| `dataloader.py` | 分布式数据加载 | `create_distributed_dataloader`, `DataLoaderWrapper` |

**关键技术**：
- 使用 `DistributedSampler` 实现数据分片
- 必须在每个 epoch 调用 `sampler.set_epoch(epoch)`

---

### Layer 3: 训练逻辑层 (`train/`)

**构建顺序**：7 → 8

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `ddp_trainer.py` | DDP 训练器 | `DDPTrainer`, `ddp_setup`, `cleanup` |
| `deepspeed_trainer.py` | DeepSpeed 训练器 | `DeepSpeedTrainer`, `get_deepspeed_config` |

**关键技术**：
- DDP: `DistributedDataParallel` 封装，梯度 Bucketing，通信/计算重叠
- DeepSpeed: ZeRO Stage 1/2/3，混合精度，参数卸载

---

### Layer 4: 工具与优化 (`utils/`)

**构建顺序**：9 → 10 → 11

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `checkpoint.py` | Checkpoint 管理 | `save_checkpoint`, `load_checkpoint`, `get_latest_checkpoint` |
| `profiler.py` | 性能分析 | `ProfilerWrapper`, `analyze_communication_overlap` |
| `gradient_checkpointing.py` | 梯度检查点 | `apply_gradient_checkpointing`, `checkpoint_sequential` |

---

### Layer 5: 配置管理 (`config/`)

**构建顺序**：12

| 文件 | 功能 | 适用场景 |
|------|------|---------|
| `ds_config_stage1.json` | ZeRO Stage 1 配置 | 仅分片优化器状态 |
| `ds_config_stage2.json` | ZeRO Stage 2 配置 | 分片优化器 + 梯度（推荐） |
| `ds_config_stage3.json` | ZeRO Stage 3 配置 | 全分片（LLM 训练） |
| `model_config.yaml` | 模型超参数配置 | 统一管理模型、训练、数据配置 |

---

### Layer 6: 启动脚本 (`scripts/`)

**构建顺序**：13

| 文件 | 功能 | 使用场景 |
|------|------|---------|
| `train_ddp_single_node.sh` | 单机多卡训练 | 8 张 GPU，标准 DDP |
| `train_ddp_multi_node.sh` | 多节点训练 | 2+ 节点，需在每个节点运行 |
| `train_deepspeed.sh` | DeepSpeed 训练 | ZeRO Stage 2，启用梯度检查点 |

---

## 🔧 使用建议

### 1. 学习路径

**初学者**：
1. 阅读 `model/embeddings.py` 和 `model/attention.py`
2. 运行单卡训练：`python main.py --mode single`

**进阶者**：
1. 学习 `train/ddp_trainer.py`
2. 运行单机多卡训练：`bash scripts/train_ddp_single_node.sh`

**专家**：
1. 研究 `train/deepspeed_trainer.py` 和 ZeRO 配置
2. 启用梯度检查点并分析内存节省
3. 使用 Profiler 优化训练性能

### 2. 扩展建议

**添加新模型**：
- 在 `model/` 中创建新文件，继承 `nn.Module`
- 在 `model/__init__.py` 中导出

**添加新数据集**：
- 在 `data/dataset.py` 中添加新的 `Dataset` 类
- 更新 `main.py` 中的数据集加载逻辑

**添加新优化器**：
- 在 `train/` 中创建新的 Trainer 类
- 在 `main.py` 中添加对应的训练模式

---

## 📊 内存与性能优化路线图

```
┌─────────────────────────────────────────────────────────────┐
│  基准：DDP (Stage 0)                                         │
│  内存: 1x, 速度: 1x                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  优化 1: ZeRO Stage 1（分片优化器状态）                       │
│  内存: ~1.5x, 速度: 0.95x                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  优化 2: ZeRO Stage 2（分片优化器 + 梯度）                    │
│  内存: ~3-4x, 速度: 0.85x                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  优化 3: ZeRO Stage 3（全分片）                               │
│  内存: >10x, 速度: 0.7x                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  优化 4: Stage 3 + 梯度检查点                                 │
│  内存: >20x, 速度: 0.5x                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  优化 5: Stage 3 + GC + CPU/NVMe 卸载                        │
│  内存: >50x, 速度: 0.3x                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 参考文献

- PyTorch DDP: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
- DeepSpeed ZeRO: https://www.deepspeed.ai/tutorials/zero/
- Transformer 论文: *Attention Is All You Need* (Vaswani et al., 2017)

---

**最后更新**: 2025-10-18

