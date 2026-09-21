"""Scoring head.

The score is a relative ordering *inside a set-level quality band*. If every
alternative on a shortlist is poor, the whole shortlist should land low rather
than the best of a bad lot receiving a high score. So the head sees three
things: the alternative in context, the summary of the set it sits in, and the
conditioning. The set summary is what lets the band move; the contextualised
alternative is what orders within it.

A score of 0.6 therefore means 0.6 *in this context, against these neighbours*.
It is not a globally fixed quantity, and the two value channels on every token
exist to make that representable.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ScoringHead(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(
        self,
        alternatives: torch.Tensor,  # (B, A, D)
        summary: torch.Tensor,       # (B, D)
        conditioning: torch.Tensor,  # (B, D)
        mask: torch.Tensor,          # (B, A)
    ) -> torch.Tensor:
        batch, alts, dim = alternatives.shape
        broadcast_summary = summary.unsqueeze(1).expand(batch, alts, dim)
        broadcast_condition = conditioning.unsqueeze(1).expand(batch, alts, dim)

        logits = self.net(
            torch.cat([alternatives, broadcast_summary, broadcast_condition], dim=-1)
        ).squeeze(-1)

        return torch.sigmoid(logits) * mask.to(logits.dtype)
