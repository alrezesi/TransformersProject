"""Shared helpers for the experiments: data setup, a compact training run, statistics."""
import copy
import random
import statistics
import time
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.data import Collator, SentimentDataset, load_sst2_texts
from src.tokenizer import PAD_ID, SimpleTokenizer
from train import make_scheduler, run_epoch, set_seed

A_ID, B_ID = 1, 2              # marker tokens of the synthetic order task
FIRST_FILLER_ID = 3            # filler tokens start here (0 is the pad id)


# ------------------------------------------------------------------ SST-2 data
def build_data(subset: int, max_len: int, min_freq: int = 2):
    """Tokenizer (built from the training subset only) and the three datasets."""
    splits = load_sst2_texts()
    train_texts, train_labels = splits["train"]
    if subset:
        keep = random.Random(0).sample(range(len(train_texts)), min(subset, len(train_texts)))
        train_texts = [train_texts[i] for i in keep]
        train_labels = [train_labels[i] for i in keep]
    tokenizer = SimpleTokenizer.build(train_texts, min_freq=min_freq)
    data = {"train": (train_texts, train_labels), "dev": splits["dev"], "test": splits["test"]}
    datasets = {
        name: SentimentDataset(texts, labels, tokenizer, max_len)
        for name, (texts, labels) in data.items()
    }
    return tokenizer, datasets


def make_loaders(datasets, batch_size: int, seed: int):
    """Fresh DataLoaders per run so the shuffling order depends only on the seed."""
    collate = Collator(PAD_ID)
    generator = torch.Generator().manual_seed(seed)
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            collate_fn=collate, generator=generator),
        "dev": DataLoader(datasets["dev"], batch_size=256, collate_fn=collate),
        "test": DataLoader(datasets["test"], batch_size=256, collate_fn=collate),
    }


def shuffled_copy(dataset: SentimentDataset, seed: int = 0) -> SentimentDataset:
    """Same sentences with the word order inside every sentence randomly permuted."""
    rng = random.Random(seed)
    clone = copy.copy(dataset)
    clone.ids = [rng.sample(ids, len(ids)) for ids in dataset.ids]
    return clone


# ------------------------------------------------------------- synthetic task
def make_order_dataset(n: int, seq_len: int = 10, n_fillers: int = 10, seed: int = 0):
    """
    Every row has random filler tokens plus exactly one A and one B marker.
    Label = 1 iff A comes BEFORE B. Swapping the two markers flips the label but
    keeps the bag of tokens identical, so a model that cannot see order is at chance.
    """
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(FIRST_FILLER_ID, FIRST_FILLER_ID + n_fillers,
                           (n, seq_len), generator=g)
    order = torch.rand(n, seq_len, generator=g).argsort(dim=1)
    pos_a, pos_b = order[:, 0], order[:, 1]               # two distinct random positions
    rows = torch.arange(n)
    tokens[rows, pos_a] = A_ID
    tokens[rows, pos_b] = B_ID
    return TensorDataset(tokens, (pos_a < pos_b).long())


def swap_markers(tokens: Tensor) -> Tensor:
    """Exchange the positions of A and B in every row."""
    out = tokens.clone()
    rows = torch.arange(tokens.size(0))
    pos_a = (tokens == A_ID).long().argmax(dim=1)
    pos_b = (tokens == B_ID).long().argmax(dim=1)
    out[rows, pos_a] = B_ID
    out[rows, pos_b] = A_ID
    return out


# ------------------------------------------------------------------- training
def fit(model, loaders, epochs: int, lr: float, weight_decay: float = 0.01,
        warmup_ratio: float = 0.1, grad_clip: float = 1.0) -> Dict:
    """
    Compact training run: the best epoch is chosen on dev, then evaluated once on
    test. Nothing is written to disk (unlike train.py), so it is cheap to repeat.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(loaders["train"])
    scheduler = make_scheduler(optimizer, total_steps, int(warmup_ratio * total_steps))
    loss_fn = nn.CrossEntropyLoss()

    history, best_dev, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_m = run_epoch(model, loaders["train"], loss_fn, "cpu",
                            optimizer, scheduler, grad_clip)
        train_seconds = time.time() - start
        dev_m = run_epoch(model, loaders["dev"], loss_fn, "cpu")
        history.append({
            "epoch": epoch, "train_loss": train_m["loss"], "train_acc": train_m["acc"],
            "dev_loss": dev_m["loss"], "dev_acc": dev_m["acc"],
            "train_seconds": train_seconds,
        })
        if dev_m["acc"] > best_dev:
            best_dev, best_epoch = dev_m["acc"], epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_m = run_epoch(model, loaders["test"], loss_fn, "cpu")
    model.eval()
    return {
        "best_epoch": best_epoch, "best_dev_acc": best_dev,
        "test_acc": test_m["acc"], "test_f1": test_m["f1"],
        "final_train_loss": history[-1]["train_loss"],   # measured with dropout active
        "final_dev_loss": history[-1]["dev_loss"],
        "seconds_per_epoch": statistics.mean(h["train_seconds"] for h in history),
        "history": history,
    }


# ----------------------------------------------------------------- statistics
def summarize(values: Sequence[float]) -> Tuple[float, float]:
    """(mean, sample standard deviation); the std is 0 when there is a single value."""
    values = list(values)
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)