"""Dataset, collate function and SST-2 loading."""
import random
from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.tokenizer import PAD_ID, SimpleTokenizer


class SentimentDataset(Dataset):
    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: SimpleTokenizer,
        max_len: int,
    ):
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        self.ids: List[List[int]] = [tokenizer.encode(t, max_len) for t in texts]
        self.labels = [int(label) for label in labels]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[List[int], int]:
        return self.ids[index], self.labels[index]


class Collator:
    """Pads every sequence of a batch to the longest one in that batch."""

    def __init__(self, pad_id: int = PAD_ID):
        self.pad_id = pad_id

    def __call__(self, batch) -> Tuple[Tensor, Tensor]:
        sequences, labels = zip(*batch)
        longest = max(len(seq) for seq in sequences)
        tokens = torch.full((len(sequences), longest), self.pad_id, dtype=torch.long)
        for row, seq in enumerate(sequences):
            tokens[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return tokens, torch.tensor(labels, dtype=torch.long)    # [B, T], [B]


def load_sst2_texts(split_seed: int = 0) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Returns {"train", "dev", "test"} -> (texts, labels).
    GLUE hides the SST-2 test labels, so the validation set is split in half:
    dev (pick the best epoch) and test (final report only). The fixed seed keeps
    the split identical across every experiment and the Hugging Face comparison.
    """
    from datasets import load_dataset     # imported lazily: unit tests need no network

    raw = load_dataset("nyu-mll/glue", "sst2")
    train, validation = raw["train"], raw["validation"]
    val_texts, val_labels = list(validation["sentence"]), list(validation["label"])

    order = list(range(len(val_texts)))
    random.Random(split_seed).shuffle(order)
    half = len(order) // 2
    dev_idx, test_idx = order[:half], order[half:]

    def pick(indices):
        return [val_texts[i] for i in indices], [val_labels[i] for i in indices]

    return {
        "train": (list(train["sentence"]), list(train["label"])),
        "dev": pick(dev_idx),
        "test": pick(test_idx),
    }