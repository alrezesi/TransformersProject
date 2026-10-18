"""Train the from-scratch Transformer classifier on SST-2."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.data import Collator, SentimentDataset, load_sst2_texts
from src.metrics import binary_metrics
from src.model import TransformerClassifier
from src.tokenizer import PAD_ID, SimpleTokenizer

ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR:
    """Linear warmup to the base lr, then linear decay to zero."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None, grad_clip=1.0):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, total_n = 0.0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(training):
        for tokens, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens)                       # [B, num_classes]
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss += loss.item() * labels.size(0)
            total_n += labels.size(0)
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())

    metrics = binary_metrics(torch.cat(all_preds), torch.cat(all_labels))
    metrics["loss"] = total_loss / total_n
    return metrics


def prepare_data(args):
    """Build tokenizer (from the training split only) and the three DataLoaders."""
    splits = load_sst2_texts()
    train_texts, train_labels = splits["train"]
    if args.subset:
        keep = random.Random(0).sample(
            range(len(train_texts)), min(args.subset, len(train_texts))
        )
        train_texts = [train_texts[i] for i in keep]
        train_labels = [train_labels[i] for i in keep]

    tokenizer = SimpleTokenizer.build(train_texts, min_freq=args.min_freq)
    collate = Collator(PAD_ID)
    data = {
        "train": (train_texts, train_labels),
        "dev": splits["dev"],
        "test": splits["test"],
    }
    datasets = {
        name: SentimentDataset(texts, labels, tokenizer, args.max_len)
        for name, (texts, labels) in data.items()
    }
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate, generator=generator),
        "dev": DataLoader(datasets["dev"], batch_size=256, collate_fn=collate),
        "test": DataLoader(datasets["test"], batch_size=256, collate_fn=collate),
    }
    return tokenizer, loaders


def plot_history(history, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")                 # write a file, no GUI window
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(epochs, [r["train_loss"] for r in history], label="train")
    ax_loss.plot(epochs, [r["dev_loss"] for r in history], label="dev")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("loss"); ax_loss.legend()
    ax_acc.plot(epochs, [r["train_acc"] for r in history], label="train")
    ax_acc.plot(epochs, [r["dev_acc"] for r in history], label="dev")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy"); ax_acc.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_model(args, data=None):
    """Full training run. `data` can be passed in so experiments reuse the loaders."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, loaders = data if data is not None else prepare_data(args)

    model_cfg = dict(
        vocab_size=len(tokenizer), num_classes=2, d_model=args.d_model,
        num_heads=args.heads, d_ff=args.d_ff, num_layers=args.layers,
        max_len=args.max_len, dropout=args.dropout, pad_idx=PAD_ID,
        norm_first=args.norm_first, use_positional_encoding=not args.no_pos_enc,
        activation=args.activation,
    )
    model = TransformerClassifier(**model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  vocab={len(tokenizer)}  params={n_params:,}  "
          f"train batches/epoch={len(loaders['train'])}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(loaders["train"])
    scheduler = make_scheduler(optimizer, total_steps, int(args.warmup_ratio * total_steps))
    loss_fn = nn.CrossEntropyLoss()

    ckpt_path = ROOT / "checkpoints" / f"{args.tag}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history, best_acc, bad_epochs = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_m = run_epoch(model, loaders["train"], loss_fn, device,
                            optimizer, scheduler, args.grad_clip)
        dev_m = run_epoch(model, loaders["dev"], loss_fn, device)
        seconds = time.time() - start

        history.append({
            "epoch": epoch,
            "train_loss": train_m["loss"], "train_acc": train_m["acc"],
            "dev_loss": dev_m["loss"], "dev_acc": dev_m["acc"], "dev_f1": dev_m["f1"],
            "seconds": seconds,
        })
        print(f"epoch {epoch:2d} | train loss {train_m['loss']:.4f} acc {train_m['acc']:.4f}"
              f" | dev loss {dev_m['loss']:.4f} acc {dev_m['acc']:.4f} f1 {dev_m['f1']:.4f}"
              f" | {seconds:.0f}s")

        if dev_m["acc"] > best_acc:                    # keep only the best model
            best_acc, bad_epochs = dev_m["acc"], 0
            torch.save({
                "model_state": model.state_dict(),
                "model_cfg": model_cfg,
                "itos": tokenizer.itos,
                "epoch": epoch,
                "dev_acc": best_acc,
                "args": vars(args),
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stopping: no dev improvement for {args.patience} epochs")
                break

    # Final report on the untouched test split, using the best checkpoint.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    test_m = run_epoch(model, loaders["test"], loss_fn, device)
    print(f"\nbest dev acc {best_acc:.4f} | TEST acc {test_m['acc']:.4f} "
          f"f1 {test_m['f1']:.4f}")

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / f"{args.tag}_history.json").write_text(
        json.dumps({"config": vars(args), "params": n_params, "history": history,
                    "best_dev_acc": best_acc, "test": test_m}, indent=2),
        encoding="utf-8",
    )
    plot_history(history, reports / f"{args.tag}_curves.png")
    return {"history": history, "best_dev_acc": best_acc, "test": test_m,
            "params": n_params, "checkpoint": ckpt_path}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="phase5", help="name used for checkpoint and reports")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--subset", type=int, default=None, help="use only N train samples (debug)")
    p.add_argument("--norm-first", action="store_true", help="Pre-LN instead of Post-LN")
    p.add_argument("--no-pos-enc", action="store_true", help="remove positional encoding")
    return p


if __name__ == "__main__":
    train_model(build_parser().parse_args())