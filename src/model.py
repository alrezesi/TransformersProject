"""Small Transformer encoder model for text classification."""
from typing import List, Tuple, Union

import torch.nn as nn
from torch import Tensor

from src.attention import make_padding_mask
from src.embeddings import TokenEmbedding
from src.layer_norm import LayerNorm
from src.positional_encoding import PositionalEncoding
from src.transformer_block import TransformerEncoderBlock


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 256,
        num_layers: int = 2,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        norm_first: bool = False,
        use_positional_encoding: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = TokenEmbedding(vocab_size, d_model, padding_idx=pad_idx)
        # The flag lets experiment 2 remove positional information easily.
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(d_model, max_len, dropout=0.0)
        else:
            self.pos_encoding = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model, num_heads, d_ff, dropout, norm_first, activation
                )
                for _ in range(num_layers)
            ]
        )
        # Pre-LN leaves the last residual stream un-normalized, so add a final LN.
        self.final_norm = LayerNorm(d_model) if norm_first else nn.Identity()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, tokens: Tensor, return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        tokens: [B, T] (int64, padded with pad_idx)
        Returns logits [B, num_classes] (and a list of [B, H, T, T] weights).
        """
        mask = make_padding_mask(tokens, self.pad_idx)       # [B, 1, 1, T]

        x = self.embedding(tokens)                           # [B, T, D]
        x = self.pos_encoding(x)                             # [B, T, D]
        x = self.dropout(x)

        all_weights = []
        for layer in self.layers:
            x, weights = layer(x, mask)                      # [B, T, D]
            all_weights.append(weights)
        x = self.final_norm(x)

        # Masked mean pooling: padding tokens must not take part in the average.
        keep = (tokens != self.pad_idx).unsqueeze(-1).to(x.dtype)   # [B, T, 1]
        pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)  # [B, D]

        logits = self.classifier(self.dropout(pooled))       # [B, num_classes]
        if return_attention:
            return logits, all_weights
        return logits