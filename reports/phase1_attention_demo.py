"""Phase 1: self-attention by hand, scaling demo, and permutation check."""
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=3, sci_mode=False)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def self_attention(X, W_q, W_k, W_v):
    """Return (Q, K, V, scores, scaled, weights, out) for X of shape [T, D]."""
    Q, K, V = X @ W_q, X @ W_k, X @ W_v          # each [T, d_k]
    d_k = K.shape[-1]
    scores = Q @ K.T                              # [T, T]
    scaled = scores / math.sqrt(d_k)              # [T, T]
    weights = F.softmax(scaled, dim=-1)           # [T, T], each row sums to 1
    out = weights @ V                             # [T, d_k]
    return Q, K, V, scores, scaled, weights, out


# ---------------------------------------------------------------- Part A
section("A) Step-by-step self-attention (T=4 tokens, D=4)")
T, D = 4, 4
X = torch.randn(T, D)
W_q, W_k, W_v = (torch.randn(D, D) for _ in range(3))

Q, K, V, scores, scaled, weights, out = self_attention(X, W_q, W_k, W_v)

for name, t in [("X", X), ("Q", Q), ("K", K), ("V", V),
                ("scores = Q K^T", scores), ("scaled", scaled),
                ("weights = softmax", weights), ("output", out)]:
    print(f"\n{name}  shape={tuple(t.shape)}\n{t}")

print("\nrow sums of weights:", weights.sum(dim=-1))  # must all be 1

# Sanity check against PyTorch's reference implementation.
# Used here ONLY for verification, never inside our own model.
ref = F.scaled_dot_product_attention(Q[None], K[None], V[None])[0]
print("matches PyTorch reference:", torch.allclose(out, ref, atol=1e-6))

# ---------------------------------------------------------------- Part B
section("B) Why divide by sqrt(d_k)?")
n_trials, n_keys = 2000, 10
for d_k in (4, 64, 512):
    q = torch.randn(n_trials, 1, d_k)
    k = torch.randn(n_trials, n_keys, d_k)
    s = (q * k).sum(dim=-1)                       # [n_trials, n_keys]
    max_raw = F.softmax(s, dim=-1).max(dim=-1).values.mean()
    max_scaled = F.softmax(s / math.sqrt(d_k), dim=-1).max(dim=-1).values.mean()
    print(f"d_k={d_k:4d} | score std={s.std():6.2f} (sqrt(d_k)={math.sqrt(d_k):5.2f})"
          f" | avg max weight: unscaled={max_raw:.3f}, scaled={max_scaled:.3f}")

# ---------------------------------------------------------------- Part C
section("C) Self-attention ignores token order (permutation equivariance)")
perm = torch.randperm(T)
out_perm = self_attention(X[perm], W_q, W_k, W_v)[-1]
print("permuting inputs just permutes outputs:",
      torch.allclose(out_perm, out[perm], atol=1e-6))