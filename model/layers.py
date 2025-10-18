"""
Transformer 层组件模块

【构建顺序：Layer 1.3 - 第 3 步】

核心功能：
1. PositionwiseFeedForward: 位置前馈网络（FFN）
2. TransformerBlock: 完整的 Transformer 块（包含 MHA + FFN + 残差连接）

关键技术：
- FFN 结构：Linear(d_model → d_ff) → ReLU/GELU → Linear(d_ff → d_model)
- 残差连接：LayerNorm(x + Sublayer(x))
- 支持梯度检查点（Gradient Checkpointing）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络（FFN）
    
    实现两层全连接网络，对序列的每个位置独立应用。
    结构：Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)
    
    参数：
        d_model (int): 模型的嵌入维度
        d_ff (int): 前馈网络的隐藏层维度，通常是 d_model 的 4 倍
        dropout (float): Dropout 概率，默认 0.1
        activation (str): 激活函数类型，'relu' 或 'gelu'，默认 'relu'
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super(PositionwiseFeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量，shape (batch, seq_len, d_model)
        
        返回：
            输出张量，shape (batch, seq_len, d_model)
        """
        # FFN(x) = Linear2(Dropout(Activation(Linear1(x))))
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer 编码器/解码器块
    
    完整的 Transformer 层，包含：
    1. 多头自注意力子层（Self-Attention）
    2. 位置前馈网络子层（FFN）
    3. 每个子层后的残差连接和层归一化
    
    参数：
        d_model (int): 模型的嵌入维度
        num_heads (int): 注意力头的数量
        d_ff (int): FFN 的隐藏层维度
        dropout (float): Dropout 概率
        activation (str): FFN 的激活函数类型
        is_decoder (bool): 是否为解码器块（解码器块需要额外的交叉注意力层）
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        is_decoder: bool = False,
        self_attention_type: str = 'full',
        self_attention_window: int = None,
        cross_attention_type: str = 'full',
        cross_attention_window: int = None,
        position_embedding_type: str = 'sinusoidal',
        attn_impl: str = 'auto',
        rope_theta: float = 10000.0,
        rope_scaling_type: str = 'none',
        rope_scaling_factor: float = 1.0
    ):
        super(TransformerBlock, self).__init__()
        
        self.is_decoder = is_decoder
        
        # 1. 自注意力子层
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout,
            attention_type=self_attention_type,
            window_size=self_attention_window,
            position_embedding_type=position_embedding_type,
            attn_impl=attn_impl,
            rope_theta=rope_theta,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. 交叉注意力子层（仅用于解码器）
        if is_decoder:
            self.cross_attn = MultiHeadAttention(
                d_model, num_heads, dropout,
                attention_type=cross_attention_type,
                window_size=cross_attention_window,
                position_embedding_type=position_embedding_type,
                attn_impl=attn_impl,
                rope_theta=rope_theta,
                rope_scaling_type=rope_scaling_type,
                rope_scaling_factor=rope_scaling_factor
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)
        
        # 3. 前馈网络子层
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm3 = nn.LayerNorm(d_model) if is_decoder else nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor = None,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入张量，shape (batch, seq_len, d_model)
            encoder_output: 编码器输出（仅解码器需要），shape (batch, src_seq_len, d_model)
            src_mask: 源序列掩码（用于 Padding Mask）
            tgt_mask: 目标序列掩码（用于 Causal Mask，仅解码器需要）
        
        返回：
            输出张量，shape (batch, seq_len, d_model)
        """
        # 1. 自注意力子层 + 残差连接 + 层归一化
        # Sublayer: Self-Attention
        attn_output, _ = self.self_attn(x, x, x, mask=tgt_mask if self.is_decoder else src_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 2. 交叉注意力子层（仅解码器）
        if self.is_decoder and encoder_output is not None:
            # Sublayer: Cross-Attention (Query from decoder, Key/Value from encoder)
            cross_attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
            x = self.norm2(x + self.dropout2(cross_attn_output))
        
        # 3. 前馈网络子层 + 残差连接 + 层归一化
        # Sublayer: Feed-Forward Network
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        
        return x


class TransformerEncoderLayer(TransformerBlock):
    """
    Transformer 编码器层（TransformerBlock 的别名，明确表示为编码器）
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        self_attention_type: str = 'full',
        self_attention_window: int = None,
        position_embedding_type: str = 'sinusoidal'
    ):
        super().__init__(
            d_model, num_heads, d_ff, dropout, activation,
            is_decoder=False,
            self_attention_type=self_attention_type,
            self_attention_window=self_attention_window,
            cross_attention_type='full',
            cross_attention_window=None,
            position_embedding_type=position_embedding_type
        )


class TransformerDecoderLayer(TransformerBlock):
    """
    Transformer 解码器层（TransformerBlock 的别名，明确表示为解码器）
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        self_attention_type: str = 'full',
        self_attention_window: int = None,
        cross_attention_type: str = 'full',
        cross_attention_window: int = None,
        position_embedding_type: str = 'sinusoidal'
    ):
        super().__init__(
            d_model, num_heads, d_ff, dropout, activation,
            is_decoder=True,
            self_attention_type=self_attention_type,
            self_attention_window=self_attention_window,
            cross_attention_type=cross_attention_type,
            cross_attention_window=cross_attention_window,
            position_embedding_type=position_embedding_type
        )

