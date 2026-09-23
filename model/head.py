"""Scoring head. The set summary places the band; the alternative orders within it."""

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
