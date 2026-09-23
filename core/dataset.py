"""Dataset and collation of ragged comparison sets into padded, masked batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from core.scoring import pad

if TYPE_CHECKING:
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
        prepared: "Prepared",
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
    arrays = pad(items)
    return Batch(
        **{name: torch.from_numpy(value) for name, value in arrays.items()},
        provenance=tuple(item["provenance"] for item in items),
        category_keys=tuple(item["category_key"] for item in items),
        set_indices=torch.tensor([item["index"] for item in items], dtype=torch.long),
    )
