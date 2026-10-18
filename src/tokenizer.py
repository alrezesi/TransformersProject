"""Simple word-level tokenizer built from scratch."""
import re
from collections import Counter
from typing import Iterable, List, Optional

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

# A word (optionally with an apostrophe part) or a single punctuation mark.
_TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class SimpleTokenizer:
    def __init__(self, itos: List[str]):
        if list(itos[:2]) != [PAD, UNK]:
            raise ValueError("vocab must start with the <pad> and <unk> tokens")
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def build(
        cls, texts: Iterable[str], min_freq: int = 2, max_size: Optional[int] = None
    ) -> "SimpleTokenizer":
        counter: Counter = Counter()
        for text in texts:
            counter.update(tokenize(text))
        # Most frequent first; ties broken alphabetically so the vocab is deterministic.
        items = sorted(
            ((tok, c) for tok, c in counter.items() if c >= min_freq),
            key=lambda item: (-item[1], item[0]),
        )
        if max_size is not None:
            items = items[: max_size - 2]
        return cls([PAD, UNK] + [tok for tok, _ in items])

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        ids = [self.stoi.get(tok, UNK_ID) for tok in tokenize(text)]
        if max_len is not None:
            ids = ids[:max_len]
        # Never return an empty sequence: a fully padded row would make the
        # attention softmax see only -inf and produce NaN.
        return ids or [UNK_ID]