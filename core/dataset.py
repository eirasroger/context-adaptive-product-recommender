"""Dataset and collation.

A batch is a ragged thing: comparison sets hold different numbers of
alternatives, categories hold different numbers of indicator tokens, and a
decision can be made under several stakeholders and several contexts at once.
Everything is padded to the batch maximum and carries its own mask. Nothing is
capped at a fixed set size -- attention handles variable sets natively, and a
catalogue category could present far more alternatives than the corpus does
today.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from core.encoding import NO_LEVEL
from core.prepare import Prepared


@dataclass
class Batch:
    """One padded batch of comparison sets."""

    indicator_slots: torch.Tensor   # (B, T) long
    family_slots: torch.Tensor      # (B, T) long
    level_slots: torch.Tensor       # (B, A, T) long, NO_LEVEL where absent
    channels: torch.Tensor          # (B, A, T, C) float
    token_mask: torch.Tensor        # (B, T) bool
    alternative_mask: torch.Tensor  # (B, A) bool

    category_slots: torch.Tensor    # (B,) long
    stakeholder_slots: torch.Tensor # (B, S) long
    stakeholder_mask: torch.Tensor  # (B, S) bool
    context_slots: torch.Tensor     # (B, X) long
    context_mask: torch.Tensor      # (B, X) bool

    pref: torch.Tensor              # (B, A) float, NaN where unlabelled
    conf: torch.Tensor              # (B, A) float, NaN where unstated
    provenance: tuple[str, ...]
    category_keys: tuple[str, ...]
    set_indices: torch.Tensor       # (B,) long, index back into Prepared

    def to(self, device: torch.device | str) -> "Batch":
        moved = {}
        for name, value in self.__dict__.items():
            moved[name] = value.to(device) if torch.is_tensor(value) else value
        return Batch(**moved)

    def __len__(self) -> int:
        return self.channels.shape[0]


class ComparisonSetDataset(Dataset):
    """Serves prepared comparison sets, optionally restricted to one fold."""

    def __init__(
        self,
        prepared: Prepared,
        fold: str | None = None,
        require_labels: bool = True,
        indices: Sequence[int] | None = None,
    ):
        self.prepared = prepared
        if indices is not None:
            selected = list(indices)
        else:
            selected = [
                index
                for index in range(prepared.n_sets)
                if fold is None or prepared.set_fold[index] == fold
            ]
        if require_labels:
            selected = [index for index in selected if self._has_label(index)]
        self.indices = np.asarray(selected, dtype=np.int64)

    def _has_label(self, index: int) -> bool:
        start = int(self.prepared.set_start[index])
        count = int(self.prepared.set_count[index])
        return bool(np.isfinite(self.prepared.member_pref[start : start + count]).any())

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = int(self.indices[position])
        prepared = self.prepared
        start = int(prepared.set_start[index])
        count = int(prepared.set_count[index])
        category = int(prepared.set_category[index])
        n_tokens = int(prepared.token_valid[category].sum())

        return {
            "index": index,
            "channels": prepared.channels_for(index),
            "level_slots": prepared.product_level_slot[start : start + count, :n_tokens],
            "indicator_slots": prepared.token_indicator_slot[category, :n_tokens],
            "family_slots": prepared.token_family_slot[category, :n_tokens],
            "category_slot": int(prepared.category_slot[category]),
            "category_key": prepared.category_keys[category],
            "stakeholder_slots": prepared.set_stakeholder_slots[index],
            "context_slots": prepared.combo_context_slots[int(prepared.set_combo[index])],
            "pref": prepared.member_pref[start : start + count],
            "conf": prepared.member_conf[start : start + count],
            "provenance": prepared.set_provenance[index],
        }


def collate(items: Sequence[dict]) -> Batch:
    """Pad a list of sets into one batch."""
    batch_size = len(items)
    max_alts = max(item["channels"].shape[0] for item in items)
    max_tokens = max(item["channels"].shape[1] for item in items)
    n_channels = items[0]["channels"].shape[2]
    max_stakeholders = max(1, max(len(item["stakeholder_slots"]) for item in items))
    max_contexts = max(1, max(len(item["context_slots"]) for item in items))

    channels = np.zeros(
        (batch_size, max_alts, max_tokens, n_channels), dtype=np.float32
    )
    level_slots = np.full((batch_size, max_alts, max_tokens), NO_LEVEL, dtype=np.int64)
    indicator_slots = np.zeros((batch_size, max_tokens), dtype=np.int64)
    family_slots = np.zeros((batch_size, max_tokens), dtype=np.int64)
    token_mask = np.zeros((batch_size, max_tokens), dtype=bool)
    alternative_mask = np.zeros((batch_size, max_alts), dtype=bool)
    category_slots = np.zeros(batch_size, dtype=np.int64)
    stakeholder_slots = np.zeros((batch_size, max_stakeholders), dtype=np.int64)
    stakeholder_mask = np.zeros((batch_size, max_stakeholders), dtype=bool)
    context_slots = np.zeros((batch_size, max_contexts), dtype=np.int64)
    context_mask = np.zeros((batch_size, max_contexts), dtype=bool)
    pref = np.full((batch_size, max_alts), np.nan, dtype=np.float32)
    conf = np.full((batch_size, max_alts), np.nan, dtype=np.float32)

    for row, item in enumerate(items):
        n_alts, n_tokens, _ = item["channels"].shape
        channels[row, :n_alts, :n_tokens] = item["channels"]
        level_slots[row, :n_alts, :n_tokens] = item["level_slots"]
        indicator_slots[row, :n_tokens] = item["indicator_slots"]
        family_slots[row, :n_tokens] = item["family_slots"]
        token_mask[row, :n_tokens] = True
        alternative_mask[row, :n_alts] = True
        category_slots[row] = item["category_slot"]

        for position, slot in enumerate(item["stakeholder_slots"]):
            stakeholder_slots[row, position] = slot
            stakeholder_mask[row, position] = True
        for position, slot in enumerate(item["context_slots"]):
            context_slots[row, position] = slot
            context_mask[row, position] = True

        pref[row, :n_alts] = item["pref"]
        conf[row, :n_alts] = item["conf"]

    return Batch(
        indicator_slots=torch.from_numpy(indicator_slots),
        family_slots=torch.from_numpy(family_slots),
        level_slots=torch.from_numpy(level_slots),
        channels=torch.from_numpy(channels),
        token_mask=torch.from_numpy(token_mask),
        alternative_mask=torch.from_numpy(alternative_mask),
        category_slots=torch.from_numpy(category_slots),
        stakeholder_slots=torch.from_numpy(stakeholder_slots),
        stakeholder_mask=torch.from_numpy(stakeholder_mask),
        context_slots=torch.from_numpy(context_slots),
        context_mask=torch.from_numpy(context_mask),
        pref=torch.from_numpy(pref),
        conf=torch.from_numpy(conf),
        provenance=tuple(item["provenance"] for item in items),
        category_keys=tuple(item["category_key"] for item in items),
        set_indices=torch.tensor([item["index"] for item in items], dtype=torch.long),
    )
