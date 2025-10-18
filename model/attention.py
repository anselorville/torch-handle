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
try:
    from xformers.ops import memory_efficient_attention
    _XFORMERS_AVAILABLE = True
except Exception:
    _XFORMERS_AVAILABLE = False


class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力
    
    实现公式：Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    参数：
        dropout (float): Dropout 概率，默认 0.1
    """
    
    def __init__(self, dropout: float = 0.1, attn_impl: str = 'auto'):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attn_impl = attn_impl
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
        attn_bias: torch.Tensor = None
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
        # 优先使用 PyTorch SDPA 或 xFormers（当可用、无 attn_bias 且非局部分块场景）
        if attn_bias is None and mask is not None:
            # 转换为 PyTorch SDPA 需要的“True 为屏蔽”掩码
            sdpa_mask = ~mask
        else:
            sdpa_mask = None
        if attn_bias is None:
            # 尝试 SDPA
            try:
                attn = F.scaled_dot_product_attention(
                    query, key, value,
                    attn_mask=sdpa_mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=False
                )
                return attn, None
            except Exception:
                pass
            # 尝试 xFormers
            if _XFORMERS_AVAILABLE:
                b, h, ql, d = query.shape
                kl = key.size(-2)
                q_ = query.reshape(b*h, ql, d)
                k_ = key.reshape(b*h, kl, d)
                v_ = value.reshape(b*h, kl, d)
                attn = memory_efficient_attention(q_, k_, v_, attn_bias=None, p=self.dropout.p if self.training else 0.0)
                attn = attn.reshape(b, h, ql, d)
                return attn, None
        # 退化到朴素实现
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        if attn_bias is not None:
            scores = scores + attn_bias
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
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
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, attention_type: str = 'full', window_size: int = None, position_embedding_type: str = 'sinusoidal', attn_impl: str = 'auto', rope_theta: float = 10000.0, rope_scaling_type: str = 'none', rope_scaling_factor: float = 1.0):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # 注意力实现类型：'full'（全局）或 'local'（滑窗）
        if attention_type not in ('full', 'local'):
            raise ValueError(f"不支持的注意力类型: {attention_type}")
        self.attention_type = attention_type
        self.window_size = window_size
        self.position_embedding_type = position_embedding_type
        self.attn_impl = attn_impl
        self.rope_theta = rope_theta
        self.rope_scaling_type = rope_scaling_type
        self.rope_scaling_factor = rope_scaling_factor
        
        # Q, K, V 的线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出线性层
        self.W_o = nn.Linear(d_model, d_model)
        
        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(dropout, attn_impl=self.attn_impl)
        
        self.dropout = nn.Dropout(dropout)

        # 预计算 ALiBi 斜率并缓存为 buffer（仅在启用 alibi 时）
        if self.position_embedding_type == 'alibi':
            import math as _m
            def _get_slopes_power_of_2(n: int):
                start = 2 ** (-2 ** -1)
                return [_m.pow(start, i + 1) for i in range(n)]
            if _m.log2(self.num_heads).is_integer():
                slopes_list = _get_slopes_power_of_2(self.num_heads)
            else:
                closest = 2 ** _m.floor(_m.log2(self.num_heads))
                slopes_list = _get_slopes_power_of_2(closest)
                extra = _get_slopes_power_of_2(2 * closest)[0::2]
                slopes_list += extra[: self.num_heads - closest]
            slopes = torch.tensor(slopes_list, dtype=torch.float32).view(1, self.num_heads, 1, 1)
            self.register_buffer('alibi_slopes', slopes, persistent=False)
        else:
            self.alibi_slopes = None
    
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
        # 期望 mask 为布尔型，True 表示可见，False 表示不可见
        if mask is not None:
            # mask shape: (batch, 1, seq_len_k) or (batch, seq_len_q, seq_len_k)
            # 扩展为: (batch, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)
        
        # 4. 构造可选的注意力偏置（ALiBi），以及 RoPE 旋转
        attn_bias = None
        if self.position_embedding_type == 'alibi':
            # 复用预计算的 slopes，并适配当前设备与数据类型
            slopes = self.alibi_slopes.to(device=Q.device, dtype=Q.dtype)
            Lq = Q.size(-2)
            Lk = K.size(-2)
            if self.attention_type == 'full':
                q_pos = torch.arange(Lq, device=Q.device).view(1, 1, Lq, 1)
                k_pos = torch.arange(Lk, device=Q.device).view(1, 1, 1, Lk)
                relative = k_pos - q_pos
                attn_bias = slopes * relative
        
        # RoPE（仅作用于自注意力的 Q/K 旋转）
        if self.position_embedding_type == 'rope':
            Q, K = self._apply_rope(Q, K)
        
        if self.attention_type == 'local':
            # 分块滑窗计算，避免构造全量 O(n^2) 分数矩阵
            B, H, Lq, _ = Q.size()
            Lk = K.size(-2)
            if self.window_size is None or self.window_size <= 0:
                raise ValueError("local 注意力需要有效的 window_size (> 0)")
            block_size = min(1024, Lq)
            pos_q_all = torch.arange(Lq, device=Q.device)
            pos_k_all = torch.arange(Lk, device=Q.device)
            outputs = []
            weights = []
            for q_start in range(0, Lq, block_size):
                q_end = min(q_start + block_size, Lq)
                k_start = max(0, q_start - self.window_size)
                k_end = min(Lk, q_end + self.window_size)
                Qb = Q[:, :, q_start:q_end, :]
                Kb = K[:, :, k_start:k_end, :]
                Vb = V[:, :, k_start:k_end, :]
                b = q_end - q_start
                k_sub = k_end - k_start
                # 局部可见性掩码 (1,1,b,k_sub)
                q_pos = pos_q_all[q_start:q_end].unsqueeze(1)
                k_pos = pos_k_all[k_start:k_end].unsqueeze(0)
                local_visible = (q_pos - k_pos).abs() <= self.window_size
                local_visible = local_visible.view(1, 1, b, k_sub)
                # 外部 mask 子块
                block_mask = local_visible
                if mask is not None:
                    if mask.size(2) == 1:
                        # (B,1,1,Lk) → 取子块并扩展到 b
                        ext = mask[:, :, :, k_start:k_end]
                        block_mask = block_mask & ext.expand(-1, -1, b, -1)
                    else:
                        # (B,1,Lq,Lk)
                        ext = mask[:, :, q_start:q_end, k_start:k_end]
                        block_mask = block_mask & ext
                # ALiBi 子块偏置
                attn_bias_sub = None
                if self.position_embedding_type == 'alibi' and self.alibi_slopes is not None:
                    slopes_b = self.alibi_slopes.to(device=Qb.device, dtype=Qb.dtype)
                    rel = k_pos - q_pos  # (b, k_sub)
                    attn_bias_sub = slopes_b * rel.view(1, 1, b, k_sub)
                out_b, w_b = self.attention(Qb, Kb, Vb, block_mask, attn_bias_sub)
                outputs.append(out_b)
                weights.append(w_b)
            attn_output = torch.cat(outputs, dim=2)
            attention_weights = None if any(w is None for w in weights) else torch.cat(weights, dim=2)
        else:
            attn_output, attention_weights = self.attention(Q, K, V, mask, attn_bias)
        
        # 合并多头并映射回 d_model
        attn_output = self.combine_heads(attn_output)
        output = self.W_o(attn_output)
        return output, attention_weights

