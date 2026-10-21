import pytest
import torch
from torch.utils.data import DataLoader

from experiments.common import (A_ID, B_ID, fit, make_order_dataset,
                                summarize, swap_markers)
from experiments.sequence_length import loglog_slope
from src.model import TransformerClassifier

MODEL_KW = dict(vocab_size=13, num_classes=2, d_model=32, num_heads=4,
                d_ff=64, num_layers=2, max_len=16, dropout=0.0)


def marker_positions(tokens):
    return (tokens == A_ID).long().argmax(1), (tokens == B_ID).long().argmax(1)


def test_order_task_is_well_formed():
    tokens, labels = make_order_dataset(500, seq_len=10, seed=0).tensors
    assert tokens.shape == (500, 10)
    assert ((tokens == A_ID).sum(1) == 1).all() and ((tokens == B_ID).sum(1) == 1).all()
    pos_a, pos_b = marker_positions(tokens)
    assert torch.equal(labels, (pos_a < pos_b).long())
    assert 0.4 < labels.float().mean() < 0.6                 # roughly balanced


def test_swapping_markers_flips_label_and_keeps_the_bag_of_tokens():
    tokens, labels = make_order_dataset(200, seed=1).tensors
    swapped = swap_markers(tokens)
    assert torch.equal(swapped.sort(dim=1).values, tokens.sort(dim=1).values)
    pos_a, pos_b = marker_positions(swapped)
    assert torch.equal((pos_a < pos_b).long(), 1 - labels)


def test_model_without_pe_cannot_see_marker_order():
    tokens, _ = make_order_dataset(64, seed=2).tensors
    torch.manual_seed(0)
    no_pe = TransformerClassifier(use_positional_encoding=False, **MODEL_KW).eval()
    with torch.no_grad():
        assert torch.allclose(no_pe(tokens), no_pe(swap_markers(tokens)), atol=1e-5)
    with_pe = TransformerClassifier(use_positional_encoding=True, **MODEL_KW).eval()
    with torch.no_grad():
        assert not torch.allclose(with_pe(tokens), with_pe(swap_markers(tokens)), atol=1e-6)


def test_summarize():
    assert summarize([1.0, 2.0, 3.0]) == (2.0, 1.0)
    assert summarize([5.0]) == (5.0, 0.0)


def test_loglog_slope():
    xs = [64, 128, 256, 512]
    assert loglog_slope(xs, [x ** 2 for x in xs]) == pytest.approx(2.0)
    assert loglog_slope(xs, [3 * x for x in xs]) == pytest.approx(1.0)


def test_fit_smoke():
    torch.manual_seed(0)
    train = make_order_dataset(128, seed=3)
    dev = make_order_dataset(64, seed=4)
    test = make_order_dataset(64, seed=5)
    loaders = {"train": DataLoader(train, batch_size=32, shuffle=True),
               "dev": DataLoader(dev, batch_size=64), "test": DataLoader(test, batch_size=64)}
    res = fit(TransformerClassifier(**MODEL_KW), loaders, epochs=2, lr=1e-3)
    assert 0.0 <= res["test_acc"] <= 1.0
    assert res["best_epoch"] in (1, 2) and len(res["history"]) == 2