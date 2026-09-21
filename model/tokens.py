"""Indicator tokens.

Each token is one indicator's statement about one alternative. Its identity is
composed as ``family + indicator_specific``, which is the mechanism that makes
open-ended growth affordable: an indicator nobody has seen before starts from
its family's learned prior -- "a performance indicator behaves like this when
the context pulls on it" -- rather than from noise. Its specific part is
initialised at zero, so on the first step a new indicator *is* its family.

Nothing here knows what a category is. A token is addressed by slot.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from core.encoding import N_CHANNELS, NO_LEVEL

#: Row 0 of the level table stands for "this indicator has no levels", so the
#: sentinel can be shifted into a valid index instead of branching.
LEVEL_OFFSET = 1


class TokenEmbedder(nn.Module):
    """Build token vectors from identity slots and numeric channels."""

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

        # Zero-initialised specifics mean a newly added indicator, level or
        # family member begins exactly at its family prior rather than at a
        # random point, which is what makes the tail of a grown embedding table
        # usable rather than merely present.
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
        """Return token vectors of shape (B, A, T, dim)."""
        identity = self.indicator(indicator_slots)
        if self.use_family_prior:
            identity = identity + self.family(family_slots)
        identity = identity.unsqueeze(1)  # broadcast over alternatives

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
        """Initialise indicator identities from their definition text.

        The registry requires every indicator to carry a written definition
        precisely so this is possible. It is what gives a held-out indicator
        somewhere to start from beyond its family average.
        """
        if vectors.shape != self.indicator.weight.shape:
            raise ValueError(
                f"expected {tuple(self.indicator.weight.shape)}, got {tuple(vectors.shape)}"
            )
        self.indicator.weight.copy_(vectors * scale)
