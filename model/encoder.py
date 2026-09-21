"""Attention within an alternative, over its indicator tokens.

This is the only part of the network that sees indicators at all. It reduces a
variable-length bag of indicator tokens to one fixed-width alternative
embedding, conditioned on who is asking and what for.

Conditioning enters here rather than being concatenated to every token: the
pooling query is built from the category, the active stakeholders and the active
contexts, so the same tokens can be summarised differently depending on what the
question is. The declared direction channel already tells each token which way
its context pulls; the learned context residual is what carries the effects
nobody wrote down.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    """Pre-norm self-attention with a feed-forward residual."""

    def __init__(self, dim: int, heads: int, ffn_factor: float, dropout: float):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(dim * ffn_factor)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * ffn_factor), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm_attention(x)
        attended, _ = self.attention(
            normed, normed, normed, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attended
        return x + self.ffn(self.norm_ffn(x))


class Conditioning(nn.Module):
    """Pool the category, stakeholders and contexts into one vector.

    Stakeholders and contexts are sets, not single choices -- a decision can be
    made under several archetypes at once, and can sit in more than one context
    -- so they are mean-pooled over whatever is active. Each is an embedding
    looked up by slot, so adding a ninth archetype or a fifth context appends a
    row and leaves every existing one alone.
    """

    def __init__(
        self,
        n_categories: int,
        n_stakeholders: int,
        n_contexts: int,
        dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.category = nn.Embedding(n_categories, dim)
        self.stakeholder = nn.Embedding(n_stakeholders, dim)
        # The context residual starts at zero: with no evidence, a context is
        # exactly what its declarations say it is, and the residual only earns
        # its weight from effects the declarations miss.
        self.context = nn.Embedding(n_contexts, dim)
        nn.init.zeros_(self.context.weight)
        nn.init.normal_(self.category.weight, std=0.02)
        nn.init.normal_(self.stakeholder.weight, std=0.02)

        self.project = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def _masked_mean(
        embedded: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(embedded.dtype)
        total = (embedded * weights).sum(dim=1)
        count = weights.sum(dim=1).clamp(min=1.0)
        return total / count

    def forward(
        self,
        category_slots: torch.Tensor,     # (B,)
        stakeholder_slots: torch.Tensor,  # (B, S)
        stakeholder_mask: torch.Tensor,
        context_slots: torch.Tensor,      # (B, X)
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        category = self.category(category_slots)
        stakeholders = self._masked_mean(
            self.stakeholder(stakeholder_slots), stakeholder_mask
        )
        contexts = self._masked_mean(self.context(context_slots), context_mask)
        return self.project(torch.cat([category, stakeholders, contexts], dim=-1))


class AlternativeEncoder(nn.Module):
    """Indicator tokens in, one alternative embedding out."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        blocks: int = 2,
        ffn_factor: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            AttentionBlock(dim, heads, ffn_factor, dropout) for _ in range(blocks)
        )
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(
        self,
        tokens: torch.Tensor,        # (B, A, T, D)
        token_mask: torch.Tensor,    # (B, T)
        conditioning: torch.Tensor,  # (B, D)
    ) -> torch.Tensor:
        batch, alts, n_tokens, dim = tokens.shape

        flat = tokens.reshape(batch * alts, n_tokens, dim)
        padding = ~token_mask.unsqueeze(1).expand(batch, alts, n_tokens)
        padding = padding.reshape(batch * alts, n_tokens)

        for block in self.blocks:
            flat = block(flat, padding)
        flat = self.norm(flat)

        # Attention pooling against a conditioned query: which indicators matter
        # for this alternative depends on who is asking and what for.
        query = self.query(conditioning)                       # (B, D)
        query = query.unsqueeze(1).expand(batch, alts, dim)
        query = query.reshape(batch * alts, 1, dim)

        logits = torch.bmm(flat, query.transpose(1, 2)).squeeze(-1) / (dim**0.5)
        logits = logits.masked_fill(padding, float("-inf"))
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)

        pooled = (flat * weights).sum(dim=1)
        return self.out(pooled).reshape(batch, alts, dim)
