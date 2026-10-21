"""Share of the encoder-block time spent in the attention core, per sequence length."""
import json
from pathlib import Path

path = Path(__file__).parent / "exp3_sequence_length.json"
data = json.loads(path.read_text(encoding="utf-8"))
core = data["train_seconds"]["attention core (QK^T, softmax, AV)"]
block = data["train_seconds"]["encoder block"]

print(f"{'T':>6} {'core (ms)':>10} {'block (ms)':>11} {'share':>7}")
for T, c, b in zip(data["config"]["lengths"], core, block):
    print(f"{T:>6} {c * 1e3:>10.2f} {b * 1e3:>11.2f} {c / b:>7.0%}")