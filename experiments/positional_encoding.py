"""Experiment 2: remove positional encoding (real task + synthetic order task)."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from experiments.common import (FIRST_FILLER_ID, build_data, fit, make_loaders,
                                make_order_dataset, shuffled_copy, summarize, swap_markers)
from src.data import Collator
from src.model import TransformerClassifier
from src.tokenizer import PAD_ID
from train import run_epoch, set_seed

ROOT = Path(__file__).resolve().parents[1]
SYN_VOCAB = FIRST_FILLER_ID + 10


def build_model(vocab_size, args, use_pe, max_len, dropout):
    return TransformerClassifier(
        vocab_size=vocab_size, num_classes=2, d_model=args.d_model, num_heads=args.heads,
        d_ff=args.d_ff, num_layers=args.layers, max_len=max_len, dropout=dropout,
        pad_idx=PAD_ID, use_positional_encoding=use_pe,
    )


def run_sst2(args):
    tokenizer, datasets = build_data(args.subset, args.max_len)
    shuffled_loader = DataLoader(shuffled_copy(datasets["test"]), batch_size=256,
                                 collate_fn=Collator(PAD_ID))
    loss_fn = nn.CrossEntropyLoss()
    runs = []
    for use_pe in (True, False):
        for seed in args.seeds:
            set_seed(seed)
            loaders = make_loaders(datasets, args.batch_size, seed)
            model = build_model(len(tokenizer), args, use_pe, args.max_len, 0.1)
            res = fit(model, loaders, args.epochs, args.lr)
            # Same test sentences, but the words of each sentence are shuffled.
            res["shuffled_test_acc"] = run_epoch(model, shuffled_loader, loss_fn, "cpu")["acc"]
            res.update(positional_encoding=use_pe, seed=seed)
            res.pop("history")
            runs.append(res)
            print(f"[SST-2] PE={use_pe!s:5} seed={seed} | test acc {res['test_acc']:.4f} | "
                  f"shuffled-words acc {res['shuffled_test_acc']:.4f}")
    return runs


def run_synthetic(args):
    train = make_order_dataset(args.syn_train, seed=100)
    dev = make_order_dataset(1000, seed=101)
    test = make_order_dataset(1000, seed=102)
    probe_tokens = test.tensors[0][:200]
    runs = []
    for use_pe in (True, False):
        for seed in args.seeds:
            set_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            loaders = {
                "train": DataLoader(train, batch_size=64, shuffle=True, generator=generator),
                "dev": DataLoader(dev, batch_size=500),
                "test": DataLoader(test, batch_size=500),
            }
            model = build_model(SYN_VOCAB, args, use_pe, max_len=16, dropout=0.0)
            res = fit(model, loaders, args.syn_epochs, args.syn_lr)
            with torch.inference_mode():
                p_orig = torch.softmax(model(probe_tokens), dim=-1)[:, 1]
                p_swap = torch.softmax(model(swap_markers(probe_tokens)), dim=-1)[:, 1]
            res["swap_prob_diff"] = (p_orig - p_swap).abs().mean().item()
            res.update(positional_encoding=use_pe, seed=seed)
            res.pop("history")
            runs.append(res)
            print(f"[synthetic] PE={use_pe!s:5} seed={seed} | test acc {res['test_acc']:.4f} | "
                  f"mean |p(orig) - p(swapped)| = {res['swap_prob_diff']:.2e}")
    return runs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--skip-sst2", action="store_true")
    p.add_argument("--skip-synthetic", action="store_true")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--layers", type=int, default=2)
    # real task
    p.add_argument("--subset", type=int, default=20000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-len", type=int, default=64)
    # synthetic task
    p.add_argument("--syn-train", type=int, default=6000)
    p.add_argument("--syn-epochs", type=int, default=30)
    p.add_argument("--syn-lr", type=float, default=1e-3)
    p.add_argument("--tag", default="exp2_positional_encoding")
    args = p.parse_args()

    results = {"config": vars(args)}
    if not args.skip_synthetic:
        results["synthetic"] = run_synthetic(args)
    if not args.skip_sst2:
        results["sst2"] = run_sst2(args)

    print("\n| Task | Variant | Test acc | Shuffled-words acc / swap prob. diff |")
    print("|---|---|:-:|:-:|")
    for task, key in (("SST-2", "sst2"), ("Synthetic A-before-B", "synthetic")):
        for use_pe in (True, False):
            rows = [r for r in results.get(key, []) if r["positional_encoding"] == use_pe]
            if not rows:
                continue
            acc_m, acc_s = summarize([r["test_acc"] for r in rows])
            extra_key = "shuffled_test_acc" if key == "sst2" else "swap_prob_diff"
            extra_m, _ = summarize([r[extra_key] for r in rows])
            extra = f"{extra_m * 100:.1f}" if key == "sst2" else f"{extra_m:.2e}"
            print(f"| {task} | {'with PE' if use_pe else 'without PE'} | "
                  f"{acc_m * 100:.1f} ± {acc_s * 100:.1f} | {extra} |")

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"{args.tag}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved reports/{args.tag}.json")


if __name__ == "__main__":
    main()