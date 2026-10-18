"""Evaluate a trained checkpoint and predict on custom sentences."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import Collator, SentimentDataset, load_sst2_texts
from src.model import TransformerClassifier
from src.tokenizer import PAD_ID, SimpleTokenizer
from train import run_epoch

ROOT = Path(__file__).resolve().parent


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model = TransformerClassifier(**ckpt["model_cfg"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, SimpleTokenizer(ckpt["itos"]), ckpt


@torch.no_grad()
def predict_texts(model, tokenizer, texts, max_len, device="cpu"):
    """Returns class probabilities [N, 2] (column 1 = positive)."""
    model.eval()
    batch = [(tokenizer.encode(text, max_len), 0) for text in texts]
    tokens, _ = Collator(PAD_ID)(batch)
    return torch.softmax(model(tokens.to(device)), dim=-1).cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "phase5.pt"))
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--text", action="append", default=[], help="sentence to classify")
    args = parser.parse_args()

    model, tokenizer, ckpt = load_checkpoint(args.checkpoint)
    max_len = ckpt["model_cfg"]["max_len"]
    print(f"checkpoint: epoch {ckpt['epoch']}, dev acc {ckpt['dev_acc']:.4f}")

    texts, labels = load_sst2_texts()[args.split]
    loader = DataLoader(
        SentimentDataset(texts, labels, tokenizer, max_len),
        batch_size=256, collate_fn=Collator(PAD_ID),
    )
    m = run_epoch(model, loader, nn.CrossEntropyLoss(), "cpu")
    print(f"\n{args.split}: acc {m['acc']:.4f}  precision {m['precision']:.4f}  "
          f"recall {m['recall']:.4f}  f1 {m['f1']:.4f}")
    print("\nconfusion matrix (rows = true, cols = predicted)")
    print(f"          neg   pos\n  neg  {m['tn']:5d} {m['fp']:5d}\n  pos  {m['fn']:5d} {m['tp']:5d}")

    if args.text:
        probs = predict_texts(model, tokenizer, args.text, max_len)
        print()
        for text, p in zip(args.text, probs):
            label = "positive" if p[1] > 0.5 else "negative"
            print(f"{label:8s} (p_pos={p[1]:.2f})  {text}")


if __name__ == "__main__":
    main()