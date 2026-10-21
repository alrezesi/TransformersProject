from experiments.make_report import build_report

CORE = "attention core (QK^T, softmax, AV)"


def fake_exp1():
    return {
        "config": {"tag": "exp1_heads", "d_model": 64, "seeds": [0, 1]},
        "runs": [{"heads": 1, "seed": 0, "test_acc": 0.80, "best_epoch": 2,
                  "final_train_loss": 0.3}],
        "summary": [
            {"heads": h, "d_head": 64 // h, "params": 16640, "test_acc_mean": acc,
             "test_acc_std": 0.01, "dev_acc_mean": acc, "train_loss_mean": 0.3,
             "seconds_per_epoch": 10.0}
            for h, acc in ((1, 0.80), (4, 0.805))
        ],
    }


def fake_exp3():
    return {
        "config": {"tag": "exp3_sequence_length", "lengths": [64, 128], "batch": 2,
                   "heads": 4, "d_model": 64, "repeats": 5},
        "forward_seconds": {CORE: [0.001, 0.003]},
        "train_seconds": {CORE: [0.002, 0.008]},
        "slopes": {CORE: {"all": 2.0, "last3": 2.0}},
    }


def test_missing_files_are_reported_not_crashing():
    assert build_report(None, None, None).count("Not run yet") == 3


def test_experiment_1_numbers_and_noise_check():
    text = build_report(fake_exp1(), None, None)
    assert "80.0 ± 1.0" in text
    assert "not established" in text          # gap 0.5 < std 1.0


def test_experiment_3_ratio():
    text = build_report(None, None, fake_exp3())
    assert "4.00" in text                     # 0.008 / 0.002


def test_experiment_2_with_only_the_synthetic_part():
    exp2 = {"config": {}, "synthetic": [
        {"positional_encoding": True, "test_acc": 0.9, "swap_prob_diff": 0.5},
        {"positional_encoding": False, "test_acc": 0.5, "swap_prob_diff": 1e-7},
    ]}
    text = build_report(None, exp2, None)
    assert "without PE" in text and "Not run in this experiment file" in text