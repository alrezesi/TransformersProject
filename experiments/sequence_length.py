"""Experiment 3: how the cost of self-attention grows with the sequence length T."""
import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.attention import scaled_dot_product_attention
from src.multi_head_attention import MultiHeadAttention
from src.transformer_block import TransformerEncoderBlock

ROOT = Path(__file__).resolve().parents[1]


def loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Least-squares exponent p in y ~ x^p (the slope on a log-log plot)."""
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def measure(fn: Callable[[], None], repeats: int, warmup: int = 2) -> float:
    """Median wall-clock seconds of fn()."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def make_workloads(T: int, B: int, D: int, H: int):
    Dh = D // H
    q, k, v = (torch.randn(B, H, T, Dh, requires_grad=True) for _ in range(3))
    x = torch.randn(B, T, D, requires_grad=True)
    linear = nn.Linear(D, D)
    mha = MultiHeadAttention(D, H)
    block = TransformerEncoderBlock(D, H, 4 * D, dropout=0.0)
    return {
        "attention core (QK^T, softmax, AV)": lambda: scaled_dot_product_attention(q, k, v)[0],
        "linear projection (D -> D)": lambda: linear(x),
        "multi-head attention module": lambda: mha(x, x, x)[0],
        "encoder block": lambda: block(x)[0],
    }


def plot(lengths, train_times, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, times in train_times.items():
        ax.loglog(lengths, times, marker="o", label=name)
    core = train_times["attention core (QK^T, softmax, AV)"]
    lin = train_times["linear projection (D -> D)"]
    ax.loglog(lengths, [core[0] * (t / lengths[0]) ** 2 for t in lengths],
              "k--", alpha=0.4, label="reference: T^2")
    ax.loglog(lengths, [lin[0] * (t / lengths[0]) for t in lengths],
              "k:", alpha=0.4, label="reference: T")
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("forward + backward time (s)")
    ax.set_title("Cost of attention vs. sequence length (log-log)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--tag", default="exp3_sequence_length")
    args = p.parse_args()
    torch.manual_seed(0)

    names = list(make_workloads(8, args.batch, args.d_model, args.heads))
    fwd = {n: [] for n in names}
    train = {n: [] for n in names}
    for T in args.lengths:
        workloads = make_workloads(T, args.batch, args.d_model, args.heads)
        for name, fn in workloads.items():
            def forward_only():
                with torch.no_grad():
                    fn()
            fwd[name].append(measure(forward_only, args.repeats))
            train[name].append(measure(lambda: fn().sum().backward(), args.repeats))
        print(f"T={T} done")

    core = "attention core (QK^T, softmax, AV)"
    print(f"\nB={args.batch}, D={args.d_model}, H={args.heads} (Dh={args.d_model // args.heads})")
    print("\n| T | attention matrix (MB) | core fwd (ms) | core fwd+bwd (ms) | x vs previous T |")
    print("|---:|---:|---:|---:|---:|")
    for i, T in enumerate(args.lengths):
        mb = args.batch * args.heads * T * T * 4 / 1024 ** 2      # float32 weights tensor
        ratio = "-" if i == 0 else f"{train[core][i] / train[core][i - 1]:.2f}"
        print(f"| {T} | {mb:.2f} | {fwd[core][i] * 1e3:.2f} | "
              f"{train[core][i] * 1e3:.2f} | {ratio} |")

    print("\nlog-log slope of forward+backward time (all points / last 3 points):")
    slopes = {}
    for name in names:
        all_s = loglog_slope(args.lengths, train[name])
        tail_s = loglog_slope(args.lengths[-3:], train[name][-3:])
        slopes[name] = {"all": all_s, "last3": tail_s}
        print(f"  {name:<38} {all_s:5.2f} / {tail_s:5.2f}")

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"{args.tag}.json").write_text(json.dumps(
        {"config": vars(args), "forward_seconds": fwd, "train_seconds": train, "slopes": slopes},
        indent=2), encoding="utf-8")
    plot(args.lengths, train, reports / f"{args.tag}.png")
    print(f"\nsaved reports/{args.tag}.json and .png")


if __name__ == "__main__":
    main()