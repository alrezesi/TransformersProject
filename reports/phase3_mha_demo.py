"""Phase 3 demo: shape trace through multi-head attention, and cost vs. H."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.multi_head_attention import MultiHeadAttention

torch.manual_seed(0)


def show(name: str, t: torch.Tensor) -> None:
    print(f"{name:<26}{str(tuple(t.shape)):<20}contiguous={t.is_contiguous()}")


# ---------------------------------------------------------------- Part A
print("=== A) Shape trace (B=2, T=5, D=16, H=4 -> Dh=4) ===")
B, T, D, H = 2, 5, 16, 4
mha = MultiHeadAttention(D, H).eval()
x = torch.randn(B, T, D)

show("input x", x)
q_full = mha.w_q(x)
show("after W_q", q_full)
q = mha._split_heads(q_full)
k = mha._split_heads(mha.w_k(x))
v = mha._split_heads(mha.w_v(x))
show("split heads (Q)", q)                    # note: NOT contiguous after transpose
scores = q @ k.transpose(-2, -1) / math.sqrt(mha.d_head)
show("scores", scores)
weights = F.softmax(scores, dim=-1)
show("attention weights", weights)
head_out = weights @ v
show("per-head output", head_out)
merged = mha._merge_heads(head_out)
show("concat heads", merged)
out = mha.w_o(merged)
show("after W_o", out)

module_out, _ = mha(x, x, x)
print("\nmanual trace equals module:", torch.allclose(out, module_out, atol=1e-6))
print("heads attend differently  :", not torch.allclose(weights[:, 0], weights[:, 1]))

# ---------------------------------------------------------------- Part B
print("\n=== B) What changes when H changes? (D=64, T=128, B=1) ===")
print(f"{'H':>3} {'Dh':>4} {'params':>8} {'weights numel (B*H*T*T)':>26}")
for heads in (1, 2, 4, 8):
    m = MultiHeadAttention(64, heads)
    params = sum(p.numel() for p in m.parameters())
    y = torch.randn(1, 128, 64)
    _, w = m(y, y, y)
    print(f"{heads:>3} {m.d_head:>4} {params:>8} {w.numel():>26}")