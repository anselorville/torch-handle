"""
词嵌入与位置编码模块

【构建顺序：Layer 1.1 - 第 1 步】

核心功能：
1. PositionalEncoding: 基于正弦/余弦函数的位置编码
2. TransformerEmbedding: 词嵌入 + 位置编码的组合

关键技术：
- 使用 register_buffer() 确保位置编码在分布式环境下的正确性
- 位置编码公式：PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
                  PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    位置编码层
    
    使用正弦和余弦函数生成固定的位置编码，用于注入序列的位置信息。
    
    参数：
        d_model (int): 模型的嵌入维度
        max_len (int): 支持的最大序列长度，默认 5000
        dropout (float): Dropout 概率，默认 0.1
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵：shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        
        # 计算 div_term: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用 cos
        
        # 增加 batch 维度: (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 【关键】使用 register_buffer 注册为非训练参数
        # 这确保了在 DDP/DeepSpeed 环境下，位置编码能够：
        # 1. 正确地随模型移动到 GPU
        # 2. 在 state_dict 中被保存和加载
        # 3. 不会被优化器更新
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量，shape (batch_size, seq_len, d_model)
        
        返回：
            添加了位置编码的张量，shape (batch_size, seq_len, d_model)
        """
        # 将位置编码加到输入嵌入上
        # self.pe[:, :x.size(1), :] 确保位置编码的长度与输入序列匹配
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Transformer 嵌入层
    
    组合词嵌入和位置编码，是 Transformer 模型的输入层。
    
    参数：
        vocab_size (int): 词汇表大小
        d_model (int): 模型的嵌入维度
        max_len (int): 支持的最大序列长度
        dropout (float): Dropout 概率
        padding_idx (int): 填充 token 的索引，默认 0
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        padding_idx: int = 0,
        position_embedding_type: str = 'sinusoidal'
    ):
        super(TransformerEmbedding, self).__init__()
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(
            vocab_size, 
            d_model, 
            padding_idx=padding_idx
        )
        
        # 位置编码类型：'sinusoidal' 时启用固定位置相加；'alibi'/'t5_relative' 等相对方案在注意力中处理
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == 'sinusoidal':
            self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        else:
            self.positional_encoding = None
        
        # 缩放因子（按照原始论文，嵌入需要乘以 sqrt(d_model)）
        self.scale = math.sqrt(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入的 token 索引，shape (batch_size, seq_len)
        
        返回：
            嵌入向量 + 位置编码，shape (batch_size, seq_len, d_model)
        """
        # 词嵌入并缩放
        token_emb = self.token_embedding(x) * self.scale
        
        # 添加或跳过位置编码
        if self.positional_encoding is not None:
            return self.positional_encoding(token_emb)
        return token_emb

