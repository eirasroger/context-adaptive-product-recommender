"""Attention across alternatives.

This is the transferable part of the network. It receives alternative
embeddings and a conditioning vector and nothing else -- **it never sees an
indicator**. That boundary is what makes comparing three alternatives in one
category and three in another literally the same operation, and it is what makes
the claim of transfer between categories mechanically checkable rather than
merely asserted.

Set size is not capped. Attention handles variable sets natively, so a shortlist
of two and a catalogue of fifty go through the same code.
"""

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
        """Return contextualised alternatives and the set-level summary."""
        batch, alts, dim = alternatives.shape

        broadcast = conditioning.unsqueeze(1).expand(batch, alts, dim)
        x = self.condition(torch.cat([alternatives, broadcast], dim=-1))

        padding = ~mask
        for block in self.blocks:
            x = block(x, padding)
        x = self.norm(x)

        # Padded positions must not leak into the set summary, or a batch with
        # ragged set sizes would score differently depending on its neighbours.
        weights = mask.unsqueeze(-1).to(x.dtype)
        summary = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return x * weights, summary
