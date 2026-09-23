"""The assembled model: indicator tokens, encoder, comparator, head."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from core.dataset import Batch
from core.registry import Registry
from model.comparator import Comparator
from model.encoder import AlternativeEncoder, Conditioning
from model.head import ScoringHead
from model.tokens import TokenEmbedder


@dataclass
class ModelConfig:
    """Architecture shape, with table sizes taken from the registry."""

    dim: int = 96
    encoder_heads: int = 4
    encoder_blocks: int = 2
    comparator_heads: int = 4
    comparator_blocks: int = 2
    ffn_factor: float = 4.0
    dropout: float = 0.1

    use_family_prior: bool = True
    use_context_residual: bool = True

    n_families: int = 1
    n_indicators: int = 1
    n_levels: int = 1
    n_categories: int = 1
    n_contexts: int = 1
    n_stakeholders: int = 1

    @classmethod
    def for_registry(cls, registry: Registry, **overrides) -> "ModelConfig":
        sizes = registry.table_sizes
        return cls(
            n_families=sizes.get("indicator_family", 1),
            n_indicators=sizes.get("indicator", 1),
            n_levels=sizes.get("indicator_level", 1),
            n_categories=sizes.get("category", 1),
            n_contexts=sizes.get("context", 1),
            n_stakeholders=sizes.get("stakeholder", 1),
            **overrides,
        )

    def to_dict(self) -> dict:
        return asdict(self)


#: Module path of each embedding table that grows with the registry, and its size field.
GROWABLE = {
    "tokens.family": "n_families",
    "tokens.indicator": "n_indicators",
    "tokens.level": "n_levels",
    "conditioning.category": "n_categories",
    "conditioning.context": "n_contexts",
    "conditioning.stakeholder": "n_stakeholders",
}


class Recommender(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tokens = TokenEmbedder(
            n_families=config.n_families,
            n_indicators=config.n_indicators,
            n_levels=config.n_levels,
            dim=config.dim,
            dropout=config.dropout,
            use_family_prior=config.use_family_prior,
        )
        self.conditioning = Conditioning(
            n_categories=config.n_categories,
            n_stakeholders=config.n_stakeholders,
            n_contexts=config.n_contexts,
            dim=config.dim,
            dropout=config.dropout,
        )
        if not config.use_context_residual:
            self.conditioning.context.weight.requires_grad_(False)

        self.encoder = AlternativeEncoder(
            dim=config.dim,
            heads=config.encoder_heads,
            blocks=config.encoder_blocks,
            ffn_factor=config.ffn_factor,
            dropout=config.dropout,
        )
        self.comparator = Comparator(
            dim=config.dim,
            heads=config.comparator_heads,
            blocks=config.comparator_blocks,
            ffn_factor=config.ffn_factor,
            dropout=config.dropout,
        )
        self.head = ScoringHead(dim=config.dim, dropout=config.dropout)

    def forward(self, batch: Batch) -> torch.Tensor:
        tokens = self.tokens(
            batch.indicator_slots,
            batch.family_slots,
            batch.level_slots,
            batch.channels,
        )
        conditioning = self.conditioning(
            batch.category_slots,
            batch.stakeholder_slots,
            batch.stakeholder_mask,
            batch.context_slots,
            batch.context_mask,
        )
        alternatives = self.encoder(tokens, batch.token_mask, conditioning)
        compared, summary = self.comparator(
            alternatives, batch.alternative_mask, conditioning
        )
        return self.head(compared, summary, conditioning, batch.alternative_mask)

    def parameter_count(self) -> dict[str, int]:
        return {
            name: sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name, module in (
                ("tokens", self.tokens),
                ("conditioning", self.conditioning),
                ("encoder", self.encoder),
                ("comparator", self.comparator),
                ("head", self.head),
            )
        }
