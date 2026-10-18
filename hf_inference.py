"""
Phase 6: Hugging Face model WITHOUT pipeline().
Steps: 1) tokenize  2) encode  3) model inference  4) prediction  5) evaluation,
then a head-to-head comparison with our own from-scratch model on the same split.
"""
import argparse
import json
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.compare import agreement_table, check_label_mapping, sign_test_p
from src.data import Collator, load_sst2_texts
from src.metrics import binary_metrics
from src.model import TransformerClassifier
from src.tokenizer import PAD_ID, SimpleTokenizer

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

TOKENIZATION_DEMO = [
    "a wonderful and moving film",
    "an unwatchable, unbelievably tedious mess",
    "not bad at all",
]
PROBE_SENTENCES = [
    "a wonderful and moving film",
    "a dull, lifeless mess",
    "not bad at all",
    "this movie is not good",
]


# ------------------------------------------------------------ Hugging Face side
def load_hf(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()                                   # dropout off: deterministic output
    check_label_mapping(model.config.id2label)     # make sure 1 really means positive
    return tokenizer, model


def show_tokenization(tokenizer, sentences: List[str]) -> None:
    """Steps 1 and 2: text -> tokens -> ids, with padding and the attention mask."""
    enc = tokenizer(
        sentences, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    print("encoded keys:", list(enc.keys()))
    for i, sentence in enumerate(sentences):
        print(f"\n{sentence!r}")
        print("  tokens        :", tokenizer.convert_ids_to_tokens(enc["input_ids"][i]))
        print("  input_ids     :", enc["input_ids"][i].tolist())
        print("  attention_mask:", enc["attention_mask"][i].tolist())
    print("\nbatch shapes:", {k: tuple(v.shape) for k, v in enc.items()})


def inspect_model(model) -> None:
    cfg = model.config
    print("\n--- model structure ---")
    for name in ("n_layers", "n_heads", "dim", "hidden_dim", "vocab_size",
                 "max_position_embeddings"):
        print(f"{name:>24}: {getattr(cfg, name, 'n/a')}")
    print(f"{'id2label':>24}: {cfg.id2label}")
    print("\nfirst transformer layer:")
    print(model.distilbert.transformer.layer[0])


@torch.inference_mode()
def hf_predict(tokenizer, model, texts, batch_size=64, max_length=128) -> torch.Tensor:
    """Returns class probabilities, shape [N, 2]."""
    probs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        # Steps 1-2: tokenize + encode (pad to the longest sentence in this batch).
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        logits = model(**enc).logits               # step 3: [B, 2]
        probs.append(torch.softmax(logits, dim=-1))  # step 4: probabilities
    return torch.cat(probs)


# ---------------------------------------------------------------- our own model
def load_own(path: Path):
    ckpt = torch.load(path, map_location="cpu")
    model = TransformerClassifier(**ckpt["model_cfg"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, SimpleTokenizer(ckpt["itos"]), ckpt["model_cfg"]["max_len"]


@torch.inference_mode()
def own_predict(model, tokenizer, texts, max_len, batch_size=256) -> torch.Tensor:
    collate = Collator(PAD_ID)
    probs = []
    for start in range(0, len(texts), batch_size):
        batch = [(tokenizer.encode(t, max_len), 0) for t in texts[start:start + batch_size]]
        tokens, _ = collate(batch)
        probs.append(torch.softmax(model(tokens), dim=-1))
    return torch.cat(probs)


# ------------------------------------------------------------------- reporting
def report(name: str, probs: torch.Tensor, labels: torch.Tensor, seconds: float):
    preds = probs.argmax(dim=-1)                   # step 4: predicted class id
    m = binary_metrics(preds, labels)              # step 5: evaluation
    print(f"\n[{name}] acc {m['acc']:.4f}  precision {m['precision']:.4f}  "
          f"recall {m['recall']:.4f}  f1 {m['f1']:.4f}  ({seconds:.1f}s)")
    print("confusion matrix (rows = true, cols = predicted)")
    print(f"          neg   pos\n  neg  {m['tn']:5d} {m['fp']:5d}\n  pos  {m['fn']:5d} {m['tp']:5d}")
    return preds, m


def show_examples(title, texts, labels, indices, hf_probs, own_probs, limit=8):
    print(f"\n{title} ({len(indices)} total, showing {min(limit, len(indices))})")
    for i in indices[:limit]:
        gold = "pos" if labels[i] == 1 else "neg"
        print(f"  gold={gold}  hf p_pos={hf_probs[i, 1]:.2f}  own p_pos={own_probs[i, 1]:.2f}"
              f"  | {texts[i][:90]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "phase5.pt"))
    parser.add_argument("--inspect", action="store_true", help="print the model structure")
    args = parser.parse_args()

    tokenizer, model = load_hf(args.model)
    hf_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.model}\nparameters: {hf_params:,}")

    show_tokenization(tokenizer, TOKENIZATION_DEMO)
    if args.inspect:
        inspect_model(model)

    texts, label_list = load_sst2_texts()[args.split]   # same fixed-seed split as phase 5
    labels = torch.tensor(label_list)
    print(f"\nevaluating on the {args.split} split: {len(texts)} sentences")

    start = time.time()
    hf_probs = hf_predict(tokenizer, model, texts, args.batch_size, args.max_length)
    hf_seconds = time.time() - start
    hf_preds, hf_m = report("Hugging Face", hf_probs, labels, hf_seconds)

    results = {
        "split": args.split, "n": len(texts), "model": args.model,
        "hf": {**hf_m, "params": hf_params, "seconds": hf_seconds},
    }

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"\ncheckpoint {ckpt_path} not found: skipping the comparison")
    else:
        own_model, own_tok, max_len = load_own(ckpt_path)
        own_params = sum(p.numel() for p in own_model.parameters())
        start = time.time()
        own_probs = own_predict(own_model, own_tok, texts, max_len)
        own_seconds = time.time() - start
        own_preds, own_m = report("Own model", own_probs, labels, own_seconds)
        results["own"] = {**own_m, "params": own_params, "seconds": own_seconds}

        # ---- head-to-head on identical sentences
        hf_ok = (hf_preds == labels).tolist()
        own_ok = (own_preds == labels).tolist()
        table = agreement_table(hf_ok, own_ok)
        p_value = sign_test_p(table["only_a"], table["only_b"])
        results["agreement"] = {**table, "sign_test_p": p_value}

        print("\n--- head-to-head (same sentences) ---")
        print(f"both correct        : {table['both_correct']}")
        print(f"only HF correct     : {table['only_a']}")
        print(f"only own correct    : {table['only_b']}")
        print(f"both wrong          : {table['both_wrong']}")
        print(f"exact sign test p   : {p_value:.2e}")
        print(f"parameters          : HF {hf_params:,} vs own {own_params:,} "
              f"({hf_params / own_params:.0f}x)")

        only_hf = [i for i, (a, b) in enumerate(zip(hf_ok, own_ok)) if a and not b]
        only_own = [i for i, (a, b) in enumerate(zip(hf_ok, own_ok)) if b and not a]
        both_wrong = [i for i, (a, b) in enumerate(zip(hf_ok, own_ok)) if not a and not b]
        show_examples("HF right, own wrong", texts, label_list, only_hf, hf_probs, own_probs)
        show_examples("own right, HF wrong", texts, label_list, only_own, hf_probs, own_probs)
        show_examples("both wrong", texts, label_list, both_wrong, hf_probs, own_probs)

        # ---- the sentences we probed by hand in phase 5
        hf_p = hf_predict(tokenizer, model, PROBE_SENTENCES)
        own_p = own_predict(own_model, own_tok, PROBE_SENTENCES, max_len)
        print("\n--- hand-written probes (probability of 'positive') ---")
        print(f"{'sentence':<30}{'hf':>6}{'own':>7}")
        for s, a, b in zip(PROBE_SENTENCES, hf_p[:, 1], own_p[:, 1]):
            print(f"{s:<30}{a:>6.2f}{b:>7.2f}")

    out = ROOT / "reports" / "hf_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()