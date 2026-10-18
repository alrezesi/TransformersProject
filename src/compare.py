"""Helpers for comparing two classifiers evaluated on the same examples."""
from math import comb
from typing import Dict, Sequence


def sign_test_p(only_a_correct: int, only_b_correct: int) -> float:
    """
    Two-sided exact sign test (McNemar's exact test) on the discordant pairs, i.e.
    the examples where exactly one of the two models is right. Under the null
    hypothesis (both models equally good) each discordant example is a fair coin flip.
    """
    n = only_a_correct + only_b_correct
    if n == 0:
        return 1.0
    k = min(only_a_correct, only_b_correct)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def agreement_table(
    correct_a: Sequence[bool], correct_b: Sequence[bool]
) -> Dict[str, int]:
    """Counts of: both right / only A right / only B right / both wrong."""
    if len(correct_a) != len(correct_b):
        raise ValueError("both models must be evaluated on the same examples")
    pairs = list(zip(correct_a, correct_b))
    return {
        "both_correct": sum(a and b for a, b in pairs),
        "only_a": sum(a and not b for a, b in pairs),
        "only_b": sum(b and not a for a, b in pairs),
        "both_wrong": sum(not a and not b for a, b in pairs),
    }


def check_label_mapping(id2label: Dict[int, str]) -> None:
    """
    Our labels are 0 = negative, 1 = positive (GLUE SST-2). If the model uses a
    different mapping every metric would be silently wrong, so fail loudly instead.
    """
    names = {int(i): str(name).lower() for i, name in id2label.items()}
    if names.get(0, "").startswith("neg") and names.get(1, "").startswith("pos"):
        return
    raise ValueError(f"unexpected label mapping: {id2label}")