"""
多头注意力机制模块

【构建顺序：Layer 1.2 - 第 2 步】

核心功能：
1. ScaledDotProductAttention: 缩放点积注意力
2. MultiHeadAttention: 多头注意力机制

关键技术：
- 注意力公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
- 支持 Padding Mask 和 Causal Mask
- 张量重塑：(batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    
    实现公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    参数：
        dropout (float): Dropout 概率，默认 0.1
    """
    
    def __init__(self, dropout: float = 0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        前向传播
        
        参数：
            query: 查询张量，shape (batch, num_heads, seq_len_q, d_k)
            key: 键张量，shape (batch, num_heads, seq_len_k, d_k)
            value: 值张量，shape (batch, num_heads, seq_len_v, d_v)
            mask: 掩码张量，shape (batch, 1, seq_len_q, seq_len_k) 或 (batch, 1, 1, seq_len_k)
        
        返回：
            output: 注意力输出，shape (batch, num_heads, seq_len_q, d_v)
            attention_weights: 注意力权重，shape (batch, num_heads, seq_len_q, seq_len_k)
        """
        d_k = query.size(-1)
        
        # 计算注意力分数：QK^T / sqrt(d_k)
        # scores shape: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码（如果提供）
        if mask is not None:
            # 将掩码位置的分数设置为一个很大的负数，使 softmax 后接近 0
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算加权值：attention_weights × V
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    将输入投影到多个表示子空间，在每个子空间中并行计算注意力。
    
    参数：
        d_model (int): 模型的嵌入维度
        num_heads (int): 注意力头的数量
        dropout (float): Dropout 概率，默认 0.1
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q, K, V 的线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性层
        self.W_o = nn.Linear(d_model, d_model)
        
        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将张量分割为多个注意力头
        
        参数：
            x: 输入张量，shape (batch, seq_len, d_model)
        
        返回：
            重塑后的张量，shape (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.size()
        
        # 重塑：(batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 转置：(batch, seq_len, num_heads, d_k) → (batch, num_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多个注意力头合并回原始维度
        
        参数：
            x: 输入张量，shape (batch, num_heads, seq_len, d_k)
        
        返回：
            合并后的张量，shape (batch, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.size()
        
        # 转置：(batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
        x = x.transpose(1, 2)
        
        # 重塑：(batch, seq_len, num_heads, d_k) → (batch, seq_len, d_model)
        return x.contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple:
        """
        前向传播
        
        参数：
            query: 查询张量，shape (batch, seq_len_q, d_model)
            key: 键张量，shape (batch, seq_len_k, d_model)
            value: 值张量，shape (batch, seq_len_v, d_model)
            mask: 掩码张量，用于 Padding Mask 或 Causal Mask
        
        返回：
            output: 注意力输出，shape (batch, seq_len_q, d_model)
            attention_weights: 注意力权重（用于可视化）
        """
        batch_size = query.size(0)
        
        # 1. 线性投影：(batch, seq_len, d_model) → (batch, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # 2. 分头：(batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. 调整 mask 的维度以匹配多头
        if mask is not None:
            # mask shape: (batch, 1, seq_len) or (batch, seq_len_q, seq_len_k)
            # 扩展为: (batch, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)
        
        # 4. 计算缩放点积注意力
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # 5. 合并多头：(batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        attn_output = self.combine_heads(attn_output)
        
        # 6. 最终线性投影
        output = self.W_o(attn_output)
        
        return output, attention_weights

