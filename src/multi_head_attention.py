"""Multi-head attention: split heads, attend, concat, project."""
from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from src.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # One big [D, D] projection == H separate [D, Dh] projections, but faster.
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        # Output projection: mixes information across the concatenated heads.
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def _split_heads(self, x: Tensor) -> Tensor:
        """[B, T, D] -> [B, H, T, Dh]"""
        B, T, _ = x.shape
        # First view as [B, T, H, Dh] (D is laid out as H consecutive chunks of Dh),
        # then move the head axis in front of the time axis.
        return x.view(B, T, self.num_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """[B, H, T, Dh] -> [B, T, D]"""
        B, H, T, Dh = x.shape
        # transpose() makes the tensor non-contiguous, so view() needs contiguous().
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            query: [B, T_q, D]
            key:   [B, T_k, D]
            value: [B, T_k, D]
            mask:  bool, broadcastable to [B, H, T_q, T_k]; True = may attend.
        Returns:
            output:  [B, T_q, D]
            weights: [B, H, T_q, T_k]
        For self-attention pass the same tensor as query, key and value.
        """
        q = self._split_heads(self.w_q(query))   # [B, H, T_q, Dh]
        k = self._split_heads(self.w_k(key))     # [B, H, T_k, Dh]
        v = self._split_heads(self.w_v(value))   # [B, H, T_k, Dh]

        # Attention runs independently inside every head (scaled by sqrt(Dh)).
        out, weights = scaled_dot_product_attention(q, k, v, mask)  # [B, H, T_q, Dh]

        out = self._merge_heads(out)             # [B, T_q, D]
        out = self.w_o(out)                      # [B, T_q, D]
        return out, weights