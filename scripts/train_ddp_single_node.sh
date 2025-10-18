#!/bin/bash

# 单机多卡 DDP 训练脚本
# 使用 8 张 GPU 进行训练

echo "=========================================="
echo "单机多卡 DDP 训练"
echo "=========================================="

# GPU 数量
NPROC_PER_NODE=8

# 训练参数
EPOCHS=10
BATCH_SIZE=32
LR=0.0001

# 启动训练
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    main.py \
    --mode ddp \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --checkpoint_dir ./checkpoints_ddp \
    --save_every 1

echo "训练完成！"

