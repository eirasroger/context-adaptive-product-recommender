"""Attention across alternatives. It sees alternative embeddings and conditioning only."""

from __future__ import annotations

import torch
import torch.nn as nn

from model.encoder import AttentionBlock


class Comparator(nn.Module):
    """Contextualise each alternative against its competitors."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        blocks: int = 2,
        ffn_factor: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.condition = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList(
            AttentionBlock(dim, heads, ffn_factor, dropout) for _ in range(blocks)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        alternatives: torch.Tensor,  # (B, A, D)
        mask: torch.Tensor,          # (B, A) bool, True where real
        conditioning: torch.Tensor,  # (B, D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, alts, dim = alternatives.shape

        broadcast = conditioning.unsqueeze(1).expand(batch, alts, dim)
        x = self.condition(torch.cat([alternatives, broadcast], dim=-1))

        padding = ~mask
        for block in self.blocks:
            x = block(x, padding)
        x = self.norm(x)

        # Padding would otherwise make a score depend on the other sets in the batch.
        weights = mask.unsqueeze(-1).to(x.dtype)
        summary = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return x * weights, summary
