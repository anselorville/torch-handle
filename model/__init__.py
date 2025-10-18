"""
Transformer 模型核心组件模块

包含以下子模块：
- embeddings: 词嵌入和位置编码
- attention: 多头注意力机制
- layers: 前馈网络和残差连接
- transformer: 完整的 Transformer 架构
"""

from .embeddings import PositionalEncoding, TransformerEmbedding
from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward, TransformerBlock
from .transformer import Transformer, TransformerEncoder, TransformerDecoder

__all__ = [
    'PositionalEncoding',
    'TransformerEmbedding',
    'MultiHeadAttention',
    'PositionwiseFeedForward',
    'TransformerBlock',
    'Transformer',
    'TransformerEncoder',
    'TransformerDecoder',
]

