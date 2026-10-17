import pytest
import torch

from src.embeddings import TokenEmbedding
from src.multi_head_attention import MultiHeadAttention
from src.positional_encoding import PositionalEncoding

torch.manual_seed(0)


@pytest.mark.parametrize(
    "B,T,D,H", [(1, 1, 8, 1), (2, 5, 16, 4), (3, 7, 32, 8), (4, 10, 64, 2)]
)
def test_mha_self_attention_shapes(B, T, D, H):
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, T, D)
    out, weights = mha(x, x, x)
    assert out.shape == (B, T, D)
    assert weights.shape == (B, H, T, T)


def test_mha_cross_attention_shapes():
    mha = MultiHeadAttention(16, 4)
    q = torch.randn(2, 3, 16)
    kv = torch.randn(2, 7, 16)
    out, weights = mha(q, kv, kv)
    assert out.shape == (2, 3, 16)            # follows the query length
    assert weights.shape == (2, 4, 3, 7)      # [B, H, T_q, T_k]


def test_full_front_pipeline_shapes():
    B, T, D, H, V = 2, 9, 32, 4, 100
    tokens = torch.randint(1, V, (B, T))
    x = TokenEmbedding(V, D, padding_idx=0)(tokens)   # [B, T, D]
    x = PositionalEncoding(D, max_len=50)(x)          # [B, T, D]
    out, weights = MultiHeadAttention(D, H)(x, x, x)
    assert out.shape == (B, T, D)
    assert weights.shape == (B, H, T, T)