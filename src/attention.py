"""Scaled dot-product attention with optional mask."""
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    """
    Args:
        q: [..., T_q, d_k]
        k: [..., T_k, d_k]
        v: [..., T_k, d_v]
        mask: bool tensor broadcastable to [..., T_q, T_k]; True = may attend.
    Returns:
        output:  [..., T_q, d_v]
        weights: [..., T_q, T_k]
    Works for both [B, T, d] and [B, H, T, d] inputs.
    """
    d_k = q.size(-1)

    # [..., T_q, d_k] @ [..., d_k, T_k] -> [..., T_q, T_k]
    # Transpose only the last two dims so batch/head dims stay untouched.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # Forbidden positions (mask == False) get -inf, so softmax gives them 0.
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)   # each row sums to 1
    output = weights @ v                  # [..., T_q, d_v]
    return output, weights


def make_padding_mask(token_ids: Tensor, pad_idx: int) -> Tensor:
    """[B, T] -> [B, 1, 1, T] bool, True where the token is NOT padding."""
    return (token_ids != pad_idx)[:, None, None, :]


def make_causal_mask(T: int, device=None) -> Tensor:
    """[T, T] bool lower-triangular, True where attending is allowed."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))