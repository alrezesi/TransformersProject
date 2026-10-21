"""Phase 8 demo: causal mask (decoder-style) and cross-attention with our own modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.attention import make_causal_mask
from src.multi_head_attention import MultiHeadAttention

torch.manual_seed(0)
torch.set_printoptions(precision=2, sci_mode=False)

D, H, T = 16, 4, 5
mha = MultiHeadAttention(D, H).eval()
x = torch.randn(1, T, D)
causal = make_causal_mask(T)

# ---------------------------------------------------------------- Part A
print("=== A) Encoder-style vs decoder-style self-attention (head 0) ===")
_, w_enc = mha(x, x, x)
_, w_dec = mha(x, x, x, mask=causal)
print("encoder (every token sees every token):\n", w_enc[0, 0])
print("decoder (causal mask, zeros above the diagonal):\n", w_dec[0, 0])

# ---------------------------------------------------------------- Part B
print("\n=== B) Can a later token change earlier outputs? ===")
x2 = x.clone()
x2[:, -1] = torch.randn(D)                       # change only the LAST token
out_dec, _ = mha(x, x, x, mask=causal)
out_dec2, _ = mha(x2, x2, x2, mask=causal)
out_enc, _ = mha(x, x, x)
out_enc2, _ = mha(x2, x2, x2)
print("decoder: earlier outputs unchanged:",
      torch.allclose(out_dec[:, :-1], out_dec2[:, :-1], atol=1e-6))   # expected True
print("encoder: earlier outputs unchanged:",
      torch.allclose(out_enc[:, :-1], out_enc2[:, :-1], atol=1e-6))   # expected False

# ---------------------------------------------------------------- Part C
print("\n=== C) Cross-attention: queries from one sequence, keys/values from another ===")
memory = torch.randn(1, 7, D)                    # e.g. encoder output: 7 tokens
target = torch.randn(1, 3, D)                    # e.g. decoder states: 3 tokens
out, w = mha(target, memory, memory)
print("output :", tuple(out.shape), "(follows the query length)")
print("weights:", tuple(w.shape), "= [B, H, T_query, T_key]")