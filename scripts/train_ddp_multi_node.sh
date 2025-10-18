#!/bin/bash

# 多节点 DDP 训练脚本
# 需要在每个节点上分别运行此脚本

echo "=========================================="
echo "多节点 DDP 训练"
echo "=========================================="

# 节点配置
NPROC_PER_NODE=8      # 每个节点的 GPU 数量
NNODES=2              # 总节点数
NODE_RANK=${1:-0}     # 当前节点的秩（从命令行参数获取，默认为 0）

# 主节点配置
MASTER_ADDR="hostname1"  # 主节点的 IP 地址或主机名
MASTER_PORT=29500

# 训练参数
EPOCHS=10
BATCH_SIZE=32
LR=0.0001

echo "节点配置："
echo "  - NNODES: $NNODES"
echo "  - NODE_RANK: $NODE_RANK"
echo "  - NPROC_PER_NODE: $NPROC_PER_NODE"
echo "  - MASTER_ADDR: $MASTER_ADDR"
echo "  - MASTER_PORT: $MASTER_PORT"

# 启动训练
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    main.py \
    --mode ddp \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --checkpoint_dir ./checkpoints_ddp_multi \
    --save_every 1

echo "训练完成！"

