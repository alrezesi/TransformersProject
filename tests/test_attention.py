import torch
import torch.nn.functional as F

from src.attention import (
    make_causal_mask,
    make_padding_mask,
    scaled_dot_product_attention,
)

torch.manual_seed(0)


def test_output_shapes_3d_and_4d():
    q = k = v = torch.randn(2, 5, 8)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 5, 8) and w.shape == (2, 5, 5)

    q = k = v = torch.randn(2, 4, 5, 8)
    out, w = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 5, 8) and w.shape == (2, 4, 5, 5)


def test_weights_sum_to_one():
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    _, w = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 4, 6), atol=1e-6)


def test_matches_pytorch_reference():
    # The PyTorch function is used ONLY as a reference, never in our model.
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    out, _ = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(out, ref, atol=1e-5)


def test_causal_mask_matches_reference_and_blocks_future():
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    mask = make_causal_mask(6)
    out, w = scaled_dot_product_attention(q, k, v, mask=mask)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.allclose(out, ref, atol=1e-5)
    # Everything above the diagonal (the future) must get zero weight.
    assert torch.all(w.triu(diagonal=1) == 0)


def test_padding_mask_gives_zero_weight_to_pad_tokens():
    tokens = torch.tensor([[5, 6, 7, 0, 0], [1, 2, 3, 4, 5]])
    mask = make_padding_mask(tokens, pad_idx=0)           # [2, 1, 1, 5]
    q, k, v = (torch.randn(2, 3, 5, 8) for _ in range(3))
    _, w = scaled_dot_product_attention(q, k, v, mask=mask)
    assert torch.all(w[0, :, :, 3:] == 0)                 # pads in sample 0
    assert torch.allclose(w.sum(dim=-1), torch.ones(2, 3, 5), atol=1e-6)


def test_self_attention_is_permutation_equivariant():
    x = torch.randn(1, 6, 8)
    perm = torch.randperm(6)
    out, _ = scaled_dot_product_attention(x, x, x)
    out_perm, _ = scaled_dot_product_attention(x[:, perm], x[:, perm], x[:, perm])
    assert torch.allclose(out_perm, out[:, perm], atol=1e-5)