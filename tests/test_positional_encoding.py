import math

import pytest
import torch

from src.embeddings import TokenEmbedding
from src.positional_encoding import PositionalEncoding

torch.manual_seed(0)


def test_pe_shape_and_buffer():
    pe = PositionalEncoding(d_model=16, max_len=50)
    x = torch.zeros(3, 10, 16)
    assert pe(x).shape == (3, 10, 16)
    assert list(pe.parameters()) == []        # nothing to learn
    assert len(list(pe.buffers())) == 1       # but it is a buffer


def test_pe_known_values():
    table = PositionalEncoding(d_model=16, max_len=50).pe[0]   # [50, 16]
    assert torch.all(table[0, 0::2] == 0)                      # sin(0)
    assert torch.all(table[0, 1::2] == 1)                      # cos(0)
    assert math.isclose(table[1, 0].item(), math.sin(1.0), rel_tol=1e-5)
    assert table.abs().max() <= 1.0 + 1e-6


def test_pe_positions_are_distinct():
    table = PositionalEncoding(d_model=16, max_len=50).pe[0]
    assert len(torch.unique(table, dim=0)) == 50


def test_pe_rejects_odd_d_model():
    with pytest.raises(ValueError):
        PositionalEncoding(d_model=15)


def test_embedding_shape_scale_and_padding():
    D = 16
    emb = TokenEmbedding(vocab_size=1000, d_model=D, padding_idx=0)
    tokens = torch.randint(1, 1000, (32, 50))
    out = emb(tokens)
    assert out.shape == (32, 50, D)
    assert torch.allclose(out, emb.embedding(tokens) * math.sqrt(D))
    assert 0.8 < out.std().item() < 1.2       # init std D**-0.5 times sqrt(D)

    padded = torch.zeros(2, 4, dtype=torch.long)
    assert torch.all(emb(padded) == 0)        # padding vector stays zero