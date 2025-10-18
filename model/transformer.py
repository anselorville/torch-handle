"""
完整 Transformer 架构模块

【构建顺序：Layer 1.4 - 第 4 步】

核心功能：
1. TransformerEncoder: Transformer 编码器（堆叠多个编码器层）
2. TransformerDecoder: Transformer 解码器（堆叠多个解码器层）
3. Transformer: 完整的 Seq2Seq Transformer 模型

依赖关系：
- 依赖 embeddings.py、attention.py、layers.py

关键技术：
- 支持梯度检查点（Gradient Checkpointing）减少激活值内存
- 生成 Causal Mask（下三角矩阵）用于自回归解码
- 支持 Padding Mask 忽略填充位置
"""

import torch
import torch.nn as nn
from .embeddings import TransformerEmbedding
from .layers import TransformerEncoderLayer, TransformerDecoderLayer
from typing import Optional


class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    
    堆叠多个编码器层，用于处理源序列。
    
    参数：
        num_layers (int): 编码器层的数量
        d_model (int): 模型的嵌入维度
        num_heads (int): 注意力头的数量
        d_ff (int): FFN 的隐藏层维度
        dropout (float): Dropout 概率
        activation (str): FFN 的激活函数类型
        use_checkpoint (bool): 是否使用梯度检查点
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_checkpoint: bool = False
    ):
        super(TransformerEncoder, self).__init__()
        
        self.use_checkpoint = use_checkpoint
        
        # 堆叠多个编码器层
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 输入嵌入，shape (batch, src_seq_len, d_model)
            src_mask: 源序列掩码，shape (batch, 1, src_seq_len)
        
        返回：
            编码器输出，shape (batch, src_seq_len, d_model)
        """
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # 使用梯度检查点（仅在训练时启用）
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, None, src_mask, None, use_reentrant=False
                )
            else:
                x = layer(x, src_mask=src_mask)
        
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """
    Transformer 解码器
    
    堆叠多个解码器层，用于自回归生成目标序列。
    
    参数：
        num_layers (int): 解码器层的数量
        d_model (int): 模型的嵌入维度
        num_heads (int): 注意力头的数量
        d_ff (int): FFN 的隐藏层维度
        dropout (float): Dropout 概率
        activation (str): FFN 的激活函数类型
        use_checkpoint (bool): 是否使用梯度检查点
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_checkpoint: bool = False
    ):
        super(TransformerDecoder, self).__init__()
        
        self.use_checkpoint = use_checkpoint
        
        # 堆叠多个解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout, activation)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        参数：
            x: 目标序列嵌入，shape (batch, tgt_seq_len, d_model)
            encoder_output: 编码器输出，shape (batch, src_seq_len, d_model)
            src_mask: 源序列掩码（用于交叉注意力）
            tgt_mask: 目标序列掩码（因果掩码，用于自注意力）
        
        返回：
            解码器输出，shape (batch, tgt_seq_len, d_model)
        """
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # 使用梯度检查点（仅在训练时启用）
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, encoder_output, src_mask, tgt_mask, use_reentrant=False
                )
            else:
                x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)


class Transformer(nn.Module):
    """
    完整的 Transformer 模型（Seq2Seq）
    
    包含编码器、解码器和输出层，用于序列到序列任务（如机器翻译）。
    
    参数：
        src_vocab_size (int): 源语言词汇表大小
        tgt_vocab_size (int): 目标语言词汇表大小
        d_model (int): 模型的嵌入维度，默认 512
        num_heads (int): 注意力头的数量，默认 8
        num_encoder_layers (int): 编码器层数，默认 6
        num_decoder_layers (int): 解码器层数，默认 6
        d_ff (int): FFN 的隐藏层维度，默认 2048
        max_len (int): 支持的最大序列长度，默认 5000
        dropout (float): Dropout 概率，默认 0.1
        activation (str): FFN 的激活函数类型，默认 'relu'
        src_pad_idx (int): 源序列的填充索引，默认 0
        tgt_pad_idx (int): 目标序列的填充索引，默认 0
        use_checkpoint (bool): 是否使用梯度检查点，默认 False
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
        activation: str = 'relu',
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
        use_checkpoint: bool = False
    ):
        super(Transformer, self).__init__()
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        
        # 源序列和目标序列的嵌入层
        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_len, dropout, src_pad_idx)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_len, dropout, tgt_pad_idx)
        
        # 编码器和解码器
        self.encoder = TransformerEncoder(
            num_encoder_layers, d_model, num_heads, d_ff, dropout, activation, use_checkpoint
        )
        self.decoder = TransformerDecoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout, activation, use_checkpoint
        )
        
        # 输出投影层（将 d_model 映射到目标词汇表大小）
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # 参数初始化
        self._init_parameters()
    
    def _init_parameters(self):
        """Xavier 均匀初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        生成源序列的 Padding Mask
        
        参数：
            src: 源序列，shape (batch, src_seq_len)
        
        返回：
            src_mask: shape (batch, 1, src_seq_len)
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1)  # (batch, 1, src_seq_len)
        return src_mask
    
    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        生成目标序列的 Causal Mask（下三角矩阵）
        
        用于防止解码器在预测当前位置时看到未来的信息。
        
        参数：
            tgt: 目标序列，shape (batch, tgt_seq_len)
        
        返回：
            tgt_mask: shape (batch, tgt_seq_len, tgt_seq_len)
        """
        batch_size, tgt_len = tgt.size()
        
        # 1. Padding Mask: (batch, 1, tgt_len)
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1)
        
        # 2. Causal Mask: (tgt_len, tgt_len) 下三角矩阵
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        
        # 3. 合并两种掩码: (batch, tgt_len, tgt_len)
        tgt_mask = tgt_pad_mask & causal_mask
        
        return tgt_mask
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        参数：
            src: 源序列，shape (batch, src_seq_len)
            tgt: 目标序列，shape (batch, tgt_seq_len)
        
        返回：
            output: 输出 logits，shape (batch, tgt_seq_len, tgt_vocab_size)
        """
        # 生成掩码
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # 嵌入
        src_emb = self.src_embedding(src)
        tgt_emb = self.tgt_embedding(tgt)
        
        # 编码
        encoder_output = self.encoder(src_emb, src_mask)
        
        # 解码
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
        
        # 输出投影
        output = self.output_projection(decoder_output)
        
        return output
    
    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """
        仅编码（用于推理时缓存编码器输出）
        
        参数：
            src: 源序列，shape (batch, src_seq_len)
        
        返回：
            encoder_output: shape (batch, src_seq_len, d_model)
        """
        src_mask = self.make_src_mask(src)
        src_emb = self.src_embedding(src)
        return self.encoder(src_emb, src_mask)
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        仅解码（用于推理时的自回归生成）
        
        参数：
            tgt: 目标序列（已生成的部分），shape (batch, tgt_seq_len)
            encoder_output: 编码器输出，shape (batch, src_seq_len, d_model)
            src_mask: 源序列掩码
        
        返回：
            output: 输出 logits，shape (batch, tgt_seq_len, tgt_vocab_size)
        """
        tgt_mask = self.make_tgt_mask(tgt)
        tgt_emb = self.tgt_embedding(tgt)
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)

