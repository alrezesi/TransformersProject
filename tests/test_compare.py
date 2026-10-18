import pytest

from src.compare import agreement_table, check_label_mapping, sign_test_p


def test_sign_test_known_values():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 5) == 1.0                        # perfectly balanced
    assert sign_test_p(0, 10) == pytest.approx(2 / 1024)   # 2 * (1 / 2^10)
    assert sign_test_p(2, 8) == pytest.approx(2 * (1 + 10 + 45) / 1024)
    assert sign_test_p(8, 2) == sign_test_p(2, 8)          # symmetric


def test_agreement_table():
    a = [True, True, False, False, True]
    b = [True, False, True, False, True]
    assert agreement_table(a, b) == {
        "both_correct": 2, "only_a": 1, "only_b": 1, "both_wrong": 1,
    }


def test_agreement_table_rejects_different_lengths():
    with pytest.raises(ValueError):
        agreement_table([True], [True, False])


def test_label_mapping_accepts_expected_and_rejects_others():
    check_label_mapping({0: "NEGATIVE", 1: "POSITIVE"})
    check_label_mapping({"0": "negative", "1": "positive"})   # keys may be strings
    with pytest.raises(ValueError):
        check_label_mapping({0: "POSITIVE", 1: "NEGATIVE"})   # flipped: would corrupt metrics
    with pytest.raises(ValueError):
        check_label_mapping({0: "LABEL_0", 1: "LABEL_1"})