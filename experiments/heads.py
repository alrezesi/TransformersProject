"""Experiment 1: number of attention heads (D fixed, so Dh = D / H)."""
import argparse
import json
import time
from pathlib import Path

from experiments.common import build_data, fit, make_loaders, summarize
from src.model import TransformerClassifier
from src.tokenizer import PAD_ID
from train import set_seed

ROOT = Path(__file__).resolve().parents[1]


def plot(summary, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs = range(len(summary))
    ax.errorbar(xs, [r["test_acc_mean"] * 100 for r in summary],
                yerr=[r["test_acc_std"] * 100 for r in summary], marker="o", capsize=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"H={r['heads']}\nDh={r['d_head']}" for r in summary])
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Effect of the number of heads (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--subset", type=int, default=20000)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--tag", default="exp1_heads")
    args = p.parse_args()

    for h in args.heads:
        if args.d_model % h:
            raise SystemExit(f"d_model {args.d_model} is not divisible by {h} heads")

    tokenizer, datasets = build_data(args.subset, args.max_len)
    print(f"vocab={len(tokenizer)}  train={len(datasets['train'])}  "
          f"dev={len(datasets['dev'])}  test={len(datasets['test'])}")

    runs, total, started = [], len(args.heads) * len(args.seeds), time.time()
    for h in args.heads:
        for seed in args.seeds:
            set_seed(seed)       # same seed => identical initial weights for every H
            loaders = make_loaders(datasets, args.batch_size, seed)
            model = TransformerClassifier(
                vocab_size=len(tokenizer), num_classes=2, d_model=args.d_model,
                num_heads=h, d_ff=args.d_ff, num_layers=args.layers,
                max_len=args.max_len, dropout=0.1, pad_idx=PAD_ID,
            )
            res = fit(model, loaders, args.epochs, args.lr)
            res.update(heads=h, seed=seed, d_head=args.d_model // h,
                       params=sum(q.numel() for q in model.parameters()))
            res.pop("history")
            runs.append(res)
            elapsed = time.time() - started
            eta = elapsed / len(runs) * (total - len(runs))
            print(f"H={h} seed={seed} | test acc {res['test_acc']:.4f} | "
                  f"train loss {res['final_train_loss']:.4f} | "
                  f"{res['seconds_per_epoch']:.0f}s/epoch | "
                  f"run {len(runs)}/{total}, ETA {eta / 60:.1f} min")

    summary = []
    for h in args.heads:
        rows = [r for r in runs if r["heads"] == h]
        acc_m, acc_s = summarize([r["test_acc"] for r in rows])
        dev_m, _ = summarize([r["best_dev_acc"] for r in rows])
        loss_m, _ = summarize([r["final_train_loss"] for r in rows])
        sec_m, _ = summarize([r["seconds_per_epoch"] for r in rows])
        summary.append({"heads": h, "d_head": args.d_model // h, "params": rows[0]["params"],
                        "test_acc_mean": acc_m, "test_acc_std": acc_s, "dev_acc_mean": dev_m,
                        "train_loss_mean": loss_m, "seconds_per_epoch": sec_m})

    print("\n| H | Dh | Params | Test acc (mean ± std) | Best dev acc | Final train loss | s/epoch |")
    print("|:-:|:-:|---:|:-:|:-:|:-:|:-:|")
    for r in summary:
        print(f"| {r['heads']} | {r['d_head']} | {r['params']:,} | "
              f"{r['test_acc_mean'] * 100:.1f} ± {r['test_acc_std'] * 100:.1f} | "
              f"{r['dev_acc_mean'] * 100:.1f} | {r['train_loss_mean']:.3f} | "
              f"{r['seconds_per_epoch']:.0f} |")

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"{args.tag}.json").write_text(
        json.dumps({"config": vars(args), "runs": runs, "summary": summary}, indent=2),
        encoding="utf-8")
    plot(summary, reports / f"{args.tag}.png")
    print(f"\nsaved reports/{args.tag}.json and .png")


if __name__ == "__main__":
    main()