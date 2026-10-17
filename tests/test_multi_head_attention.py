import pytest
import torch
import torch.nn as nn

from src.attention import make_causal_mask, make_padding_mask
from src.multi_head_attention import MultiHeadAttention

torch.manual_seed(0)


def _copy_weights_to_reference(mha: MultiHeadAttention, ref: nn.MultiheadAttention):
    """Copy our weights into torch's MultiheadAttention so outputs are comparable."""
    with torch.no_grad():
        ref.in_proj_weight.copy_(
            torch.cat([mha.w_q.weight, mha.w_k.weight, mha.w_v.weight], dim=0)
        )
        ref.in_proj_bias.copy_(
            torch.cat([mha.w_q.bias, mha.w_k.bias, mha.w_v.bias], dim=0)
        )
        ref.out_proj.weight.copy_(mha.w_o.weight)
        ref.out_proj.bias.copy_(mha.w_o.bias)


def test_rejects_indivisible_d_model():
    with pytest.raises(ValueError):
        MultiHeadAttention(d_model=10, num_heads=4)


def test_split_then_merge_is_identity():
    mha = MultiHeadAttention(16, 4)
    x = torch.randn(2, 5, 16)
    assert torch.equal(mha._merge_heads(mha._split_heads(x)), x)


def test_split_heads_layout():
    # Head h must see exactly the feature slice [h*Dh : (h+1)*Dh].
    mha = MultiHeadAttention(16, 4)
    x = torch.randn(2, 5, 16)
    heads = mha._split_heads(x)                       # [2, 4, 5, 4]
    for h in range(4):
        assert torch.equal(heads[:, h], x[..., h * 4:(h + 1) * 4])


def test_parameter_count_does_not_depend_on_heads():
    D = 32
    counts = {sum(p.numel() for p in MultiHeadAttention(D, H).parameters())
              for H in (1, 2, 4, 8)}
    assert counts == {4 * D * D + 4 * D}


def test_matches_torch_reference():
    # torch's MultiheadAttention is used ONLY as a reference.
    D, H = 16, 4
    mha = MultiHeadAttention(D, H).eval()
    ref = nn.MultiheadAttention(D, H, batch_first=True).eval()
    _copy_weights_to_reference(mha, ref)

    x = torch.randn(2, 6, D)
    out, weights = mha(x, x, x)
    ref_out, ref_weights = ref(x, x, x, need_weights=True, average_attn_weights=False)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(weights, ref_weights, atol=1e-5)


def test_matches_torch_reference_with_causal_mask():
    D, H, T = 16, 4, 6
    mha = MultiHeadAttention(D, H).eval()
    ref = nn.MultiheadAttention(D, H, batch_first=True).eval()
    _copy_weights_to_reference(mha, ref)

    x = torch.randn(2, T, D)
    mask = make_causal_mask(T)                        # True = may attend (ours)
    out, weights = mha(x, x, x, mask=mask)
    # torch uses the opposite convention for boolean masks: True = NOT allowed.
    ref_out, _ = ref(x, x, x, attn_mask=~mask)
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.all(weights.triu(diagonal=1) == 0)


def test_padding_mask_zero_weight_on_pad_tokens():
    tokens = torch.tensor([[5, 6, 7, 0, 0], [1, 2, 3, 4, 5]])
    mask = make_padding_mask(tokens, pad_idx=0)       # [2, 1, 1, 5]
    mha = MultiHeadAttention(16, 4)
    x = torch.randn(2, 5, 16)
    _, weights = mha(x, x, x, mask=mask)
    assert torch.all(weights[0, :, :, 3:] == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 5), atol=1e-6)


def test_gradients_reach_all_parameters():
    mha = MultiHeadAttention(16, 4)
    x = torch.randn(2, 5, 16)
    out, _ = mha(x, x, x)
    out.sum().backward()
    for name, p in mha.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        if name == "w_k.bias":
            # Softmax is shift-invariant, so the key bias has zero gradient.
            continue
        assert p.grad.abs().sum() > 0, f"zero gradient for {name}"