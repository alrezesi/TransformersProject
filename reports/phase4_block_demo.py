"""Phase 4 demo: LayerNorm statistics, residual gradient flow, block shapes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.layer_norm import LayerNorm
from src.transformer_block import TransformerEncoderBlock

torch.manual_seed(0)

# ---------------------------------------------------------------- Part A
print("=== A) What LayerNorm does to a badly scaled input ===")
x = torch.randn(2, 4, 64) * 3 + 5
y = LayerNorm(64)(x)
print(f"before: mean={x.mean():6.3f}  std={x.std():6.3f}")
print(f"after : mean={y.mean():6.3f}  std={y.std():6.3f}")


# ---------------------------------------------------------------- Part B
def input_grad_norm(depth: int, residual: bool) -> float:
    """Gradient norm at the input of a deep stack of Linear+tanh layers."""
    torch.manual_seed(0)
    layers = [nn.Linear(64, 64) for _ in range(depth)]
    inp = torch.randn(1, 64, requires_grad=True)
    h = inp
    for layer in layers:
        f = torch.tanh(layer(h))
        h = h + f if residual else f
    h.sum().backward()
    return inp.grad.norm().item()


print("\n=== B) Gradient reaching the input of a deep stack ===")
print(f"{'depth':>6} {'no residual':>14} {'with residual':>14}")
for depth in (5, 20, 50):
    print(f"{depth:>6} {input_grad_norm(depth, False):>14.3e} "
          f"{input_grad_norm(depth, True):>14.3e}")

# ---------------------------------------------------------------- Part C
print("\n=== C) One encoder block: shapes and parameter count ===")
D, H, d_ff = 64, 4, 256
block = TransformerEncoderBlock(D, H, d_ff, dropout=0.0).eval()
inp = torch.randn(2, 10, D)
out, weights = block(inp)
print("input  :", tuple(inp.shape))
print("output :", tuple(out.shape))
print("weights:", tuple(weights.shape))

total = sum(p.numel() for p in block.parameters())
attention = 4 * D * D + 4 * D
ffn = 2 * D * d_ff + d_ff + D
norms = 2 * (2 * D)
print(f"\nparams: attention={attention}  ffn={ffn}  layernorms={norms}")
print(f"total {total} == formula {attention + ffn + norms}: "
      f"{total == attention + ffn + norms}")
print(f"ffn share of the block: {ffn / total:.1%}")