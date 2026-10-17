"""Phase 2 demo: PE heatmap, and PE breaks permutation equivariance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import torch

from src.attention import scaled_dot_product_attention
from src.positional_encoding import PositionalEncoding

torch.manual_seed(0)

# 1) Heatmap: rows = positions, columns = embedding dimensions.
pe = PositionalEncoding(d_model=64, max_len=100).pe[0]
plt.figure(figsize=(8, 4))
plt.imshow(pe.numpy(), aspect="auto", cmap="RdBu")
plt.xlabel("embedding dimension")
plt.ylabel("position")
plt.colorbar()
plt.title("Sinusoidal positional encoding")
plt.savefig(Path(__file__).parent / "pe_heatmap.png", dpi=150, bbox_inches="tight")

# 2) Without PE attention only permutes the output; with PE it no longer does.
x = torch.randn(1, 6, 64)
perm = torch.randperm(6)
add_pe = PositionalEncoding(d_model=64, max_len=100)

def attend(t):
    return scaled_dot_product_attention(t, t, t)[0]

same_no_pe = torch.allclose(attend(x[:, perm]), attend(x)[:, perm], atol=1e-5)
same_pe = torch.allclose(attend(add_pe(x[:, perm])), attend(add_pe(x))[:, perm], atol=1e-5)
print("without PE, output just permuted:", same_no_pe)   # expected True
print("with PE,    output just permuted:", same_pe)      # expected False