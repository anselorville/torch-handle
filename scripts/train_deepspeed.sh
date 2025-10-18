#!/bin/bash

# DeepSpeed 训练脚本
# 使用 ZeRO Stage 2 进行训练

echo "=========================================="
echo "DeepSpeed 训练（ZeRO Stage 2）"
echo "=========================================="

# GPU 数量
NPROC_PER_NODE=8

# 训练参数
EPOCHS=10
BATCH_SIZE=16
LR=0.0001

# DeepSpeed 配置文件
DS_CONFIG="config/ds_config_stage2.json"

echo "配置："
echo "  - GPU 数量: $NPROC_PER_NODE"
echo "  - DeepSpeed 配置: $DS_CONFIG"

# 启动训练
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    main.py \
    --mode deepspeed \
    --deepspeed $DS_CONFIG \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --checkpoint_dir ./checkpoints_deepspeed \
    --save_every 1 \
    --use_checkpoint

echo "训练完成！"

