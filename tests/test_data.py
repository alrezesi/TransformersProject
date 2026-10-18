import pytest
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data import Collator, SentimentDataset
from src.metrics import binary_metrics
from src.tokenizer import PAD_ID, UNK_ID, SimpleTokenizer


def make_tokenizer():
    return SimpleTokenizer.build(["good good movie", "bad movie", "good film"], min_freq=2)


def test_special_tokens_and_frequency_order():
    tok = make_tokenizer()
    assert tok.itos == ["<pad>", "<unk>", "good", "movie"]    # good=3, movie=2
    assert (PAD_ID, UNK_ID) == (0, 1)


def test_encode_unknown_truncate_and_never_empty():
    tok = make_tokenizer()
    assert tok.encode("Good BAD movie") == [2, UNK_ID, 3]      # lowercased, "bad" unknown
    assert tok.encode("good good good movie", max_len=2) == [2, 2]
    assert tok.encode("") == [UNK_ID]
    assert tok.encode("...") == [UNK_ID] * 3


def test_vocab_must_start_with_special_tokens():
    with pytest.raises(ValueError):
        SimpleTokenizer(["good", "bad"])


def test_collator_pads_to_longest_in_batch():
    tokens, labels = Collator()([([2, 3], 1), ([2], 0)])
    assert tokens.tolist() == [[2, 3], [2, 0]]
    assert labels.tolist() == [1, 0] and labels.dtype == torch.long


def test_no_row_is_fully_padded():
    tok = make_tokenizer()
    ds = SentimentDataset(["", "bad"], [0, 1], tok, max_len=8)
    tokens, _ = Collator()([ds[0], ds[1]])
    assert (tokens != PAD_ID).any(dim=1).all()                 # avoids NaN in attention


def test_metrics_match_sklearn():
    torch.manual_seed(0)
    preds, labels = torch.randint(0, 2, (200,)), torch.randint(0, 2, (200,))
    m = binary_metrics(preds, labels)
    assert m["acc"] == pytest.approx(accuracy_score(labels, preds))
    assert m["precision"] == pytest.approx(precision_score(labels, preds))
    assert m["recall"] == pytest.approx(recall_score(labels, preds))
    assert m["f1"] == pytest.approx(f1_score(labels, preds))


def test_metrics_edge_case_no_positive_predictions():
    m = binary_metrics(torch.zeros(5, dtype=torch.long), torch.tensor([1, 0, 1, 0, 0]))
    assert m["precision"] == 0.0 and m["f1"] == 0.0            # no ZeroDivisionError