"""Tests for the encoder block and the full model."""
import pytest
import torch
import torch.nn as nn

from src.model import TransformerClassifier
from src.transformer_block import TransformerEncoderBlock

torch.manual_seed(0)


@pytest.mark.parametrize("norm_first", [False, True])
def test_block_shapes(norm_first):
    block = TransformerEncoderBlock(32, 4, 128, dropout=0.0, norm_first=norm_first)
    out, weights = block(torch.randn(2, 9, 32))
    assert out.shape == (2, 9, 32)
    assert weights.shape == (2, 4, 9, 9)


def test_pre_ln_block_is_identity_when_sublayers_output_zero():
    # Residual connection: if the sublayers output 0, the block returns its input.
    block = TransformerEncoderBlock(16, 4, 64, dropout=0.0, norm_first=True).eval()
    with torch.no_grad():
        for layer in (block.attention.w_o, block.ffn.linear2):
            layer.weight.zero_()
            layer.bias.zero_()
    x = torch.randn(2, 5, 16)
    out, _ = block(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_post_ln_block_output_is_normalized():
    block = TransformerEncoderBlock(32, 4, 128, dropout=0.0, norm_first=False).eval()
    out, _ = block(torch.randn(2, 7, 32) * 5 + 3)
    assert torch.allclose(out.mean(dim=-1), torch.zeros(2, 7), atol=1e-5)


@pytest.mark.parametrize("norm_first", [False, True])
def test_model_output_shapes(norm_first):
    model = TransformerClassifier(
        vocab_size=100, num_classes=3, d_model=32, num_heads=4,
        d_ff=64, num_layers=3, norm_first=norm_first,
    )
    tokens = torch.randint(1, 100, (4, 12))
    logits, weights = model(tokens, return_attention=True)
    assert logits.shape == (4, 3)
    assert len(weights) == 3
    assert all(w.shape == (4, 4, 12, 12) for w in weights)


def test_model_ignores_padding():
    # Extra padding tokens must not change the prediction.
    model = TransformerClassifier(
        vocab_size=50, num_classes=2, d_model=32, num_heads=4,
        d_ff=64, num_layers=2, dropout=0.0, pad_idx=0,
    ).eval()
    short = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0]])
    assert torch.allclose(model(short), model(padded), atol=1e-5)


def test_model_without_positional_encoding_is_order_blind():
    # Experiment 2 preview: no PE + mean pooling => token order cannot matter.
    kwargs = dict(vocab_size=50, num_classes=2, d_model=32, num_heads=4,
                  d_ff=64, num_layers=2, dropout=0.0)
    tokens = torch.tensor([[5, 6, 7, 8]])
    shuffled = torch.tensor([[8, 6, 5, 7]])

    no_pe = TransformerClassifier(use_positional_encoding=False, **kwargs).eval()
    assert torch.allclose(no_pe(tokens), no_pe(shuffled), atol=1e-5)

    with_pe = TransformerClassifier(use_positional_encoding=True, **kwargs).eval()
    assert not torch.allclose(with_pe(tokens), with_pe(shuffled), atol=1e-5)


def test_gradients_reach_all_parameters():
    model = TransformerClassifier(
        vocab_size=50, num_classes=2, d_model=32, num_heads=4,
        d_ff=64, num_layers=2, dropout=0.0,
    )
    tokens = torch.randint(1, 50, (4, 8))
    model(tokens).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        if name.endswith("w_k.bias"):
            continue    # softmax is shift-invariant => zero gradient by design
        assert p.grad.abs().sum() > 0, f"zero gradient for {name}"


def test_model_can_overfit_tiny_batch():
    # Classic sanity check: a correct model memorizes 8 random examples.
    torch.manual_seed(0)
    model = TransformerClassifier(
        vocab_size=50, num_classes=2, d_model=32, num_heads=4,
        d_ff=64, num_layers=2, dropout=0.0,
    )
    tokens = torch.randint(1, 50, (8, 6))
    labels = torch.randint(0, 2, (8,))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    first_loss = None
    for step in range(300):
        optimizer.zero_grad()
        loss = loss_fn(model(tokens), labels)
        if step == 0:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()

    assert loss.item() < first_loss
    assert loss.item() < 0.1