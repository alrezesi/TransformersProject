import random

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import Collator, SentimentDataset
from src.model import TransformerClassifier
from src.tokenizer import PAD_ID, SimpleTokenizer
from train import make_scheduler, run_epoch, set_seed

FILLER = "the a movie film story plot actor scene really very quite".split()


def make_toy(n, seed):
    """Label 1 iff the sentence contains 'good' (otherwise it contains 'bad')."""
    rng = random.Random(seed)
    texts, labels = [], []
    for _ in range(n):
        words = [rng.choice(FILLER) for _ in range(rng.randint(4, 10))]
        label = rng.randint(0, 1)
        words.insert(rng.randint(0, len(words)), "good" if label else "bad")
        texts.append(" ".join(words))
        labels.append(label)
    return texts, labels


def test_scheduler_warmup_then_decay():
    param = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([param], lr=1.0)
    sched = make_scheduler(opt, total_steps=100, warmup_steps=10)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert all(a < b for a, b in zip(lrs[:9], lrs[1:10]))      # rising during warmup
    assert max(lrs) == pytest.approx(1.0)
    assert lrs[-1] < 0.05                                      # decayed close to zero


def test_training_loop_learns_a_separable_task():
    set_seed(0)
    train_t, train_l = make_toy(600, seed=1)
    dev_t, dev_l = make_toy(200, seed=2)
    tok = SimpleTokenizer.build(train_t, min_freq=1)
    collate = Collator(PAD_ID)
    train_loader = DataLoader(SentimentDataset(train_t, train_l, tok, 16),
                              batch_size=32, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(SentimentDataset(dev_t, dev_l, tok, 16),
                            batch_size=100, collate_fn=collate)

    model = TransformerClassifier(vocab_size=len(tok), num_classes=2, d_model=32,
                                  num_heads=4, d_ff=64, num_layers=1,
                                  max_len=16, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss()

    first = run_epoch(model, train_loader, loss_fn, "cpu", optimizer)
    for _ in range(9):
        last = run_epoch(model, train_loader, loss_fn, "cpu", optimizer)
    dev = run_epoch(model, dev_loader, loss_fn, "cpu")

    assert last["loss"] < first["loss"]
    assert dev["acc"] > 0.9