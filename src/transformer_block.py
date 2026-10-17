"""Transformer encoder block: MHA + FFN with residual and LayerNorm."""
from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from src.feed_forward import FeedForward
from src.layer_norm import LayerNorm
from src.multi_head_attention import MultiHeadAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm_first: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.norm_first = norm_first
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        x:    [B, T, D]
        mask: bool, broadcastable to [B, H, T, T]; True = may attend.
        Returns (output [B, T, D], attention weights [B, H, T, T]).
        """
        if self.norm_first:
            # Pre-LN: x + Sublayer(LN(x)); the residual path stays untouched.
            h = self.norm1(x)
            attn_out, weights = self.attention(h, h, h, mask)
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            # Post-LN (original paper): LN(x + Sublayer(x)).
            attn_out, weights = self.attention(x, x, x, mask)
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, weights