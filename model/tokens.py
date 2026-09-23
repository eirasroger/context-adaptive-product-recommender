"""Indicator tokens: family plus indicator identity, level, and numeric channels."""

from __future__ import annotations

import torch
import torch.nn as nn

from core.encoding import N_CHANNELS, NO_LEVEL

#: Row 0 of the level table stands for "no level".
LEVEL_OFFSET = 1


class TokenEmbedder(nn.Module):

    def __init__(
        self,
        n_families: int,
        n_indicators: int,
        n_levels: int,
        dim: int,
        dropout: float = 0.1,
        use_family_prior: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_family_prior = use_family_prior

        self.family = nn.Embedding(n_families, dim)
        self.indicator = nn.Embedding(n_indicators, dim)
        self.level = nn.Embedding(n_levels + LEVEL_OFFSET, dim)

        # Zero, so a newly added indicator starts exactly at its family.
        nn.init.zeros_(self.indicator.weight)
        nn.init.zeros_(self.level.weight)
        nn.init.normal_(self.family.weight, std=0.02)

        self.channels = nn.Sequential(
            nn.Linear(N_CHANNELS, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        indicator_slots: torch.Tensor,   # (B, T)
        family_slots: torch.Tensor,      # (B, T)
        level_slots: torch.Tensor,       # (B, A, T)
        channels: torch.Tensor,          # (B, A, T, C)
    ) -> torch.Tensor:
        identity = self.indicator(indicator_slots)
        if self.use_family_prior:
            identity = identity + self.family(family_slots)
        identity = identity.unsqueeze(1)

        levels = self.level(
            torch.where(
                level_slots == NO_LEVEL,
                torch.zeros_like(level_slots),
                level_slots + LEVEL_OFFSET,
            )
        )

        tokens = identity + levels + self.channels(channels)
        return self.dropout(self.norm(tokens))

    @torch.no_grad()
    def seed_identity_from_text(self, vectors: torch.Tensor, scale: float = 1.0) -> None:
        """Initialise indicator identities from embeddings of their definition text."""
        if vectors.shape != self.indicator.weight.shape:
            raise ValueError(
                f"expected {tuple(self.indicator.weight.shape)}, got {tuple(vectors.shape)}"
            )
        self.indicator.weight.copy_(vectors * scale)
