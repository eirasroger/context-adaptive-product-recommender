"""The arrays a model scores, and the interface every scoring engine offers.

Torch trains and evaluates; ONNX Runtime serves. Both read the same padded
arrays and return the same (shortlists, alternatives) scores, so nothing that
builds a shortlist needs to know which engine answers it.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

import numpy as np

from core.encoding import NO_LEVEL, EncodedSet

MODEL_INPUTS = (
    "indicator_slots",
    "family_slots",
    "level_slots",
    "channels",
    "token_mask",
    "alternative_mask",
    "category_slots",
    "stakeholder_slots",
    "stakeholder_mask",
    "context_slots",
    "context_mask",
)
SCORES = "scores"

#: What an exported model carries beside its weights, so serving needs one file.
REGISTRY_BLOB = "registry_blob"
REGISTRY_VERSION = "registry_version"
REGISTRY_CONTENT_HASH = "registry_content_hash"
SNAPSHOT_HASH = "snapshot_hash"
SOURCE_SHA256 = "source_sha256"


class Scorer(Protocol):
    def __call__(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        """Scores of shape (shortlists, alternatives), zero where padded."""
        ...


def item(encoded: EncodedSet, category_key: str) -> dict:
    """One encoded shortlist, unlabelled, in the shape `pad` reads."""
    n = encoded.channels.shape[0]
    return {
        "index": 0,
        "channels": encoded.channels,
        "level_slots": encoded.level_slots,
        "indicator_slots": encoded.indicator_slots[0],
        "family_slots": encoded.family_slots[0],
        "category_slot": encoded.category_slot,
        "category_key": category_key,
        "stakeholder_slots": encoded.stakeholder_slots,
        "context_slots": encoded.context_slots,
        "pref": np.full(n, np.nan, dtype=np.float32),
        "conf": np.full(n, np.nan, dtype=np.float32),
        "provenance": "control",
    }


def pad(items: Sequence[dict]) -> dict[str, np.ndarray]:
    """Pad shortlists of different sizes into one set of arrays."""
    batch_size = len(items)
    max_alts = max(entry["channels"].shape[0] for entry in items)
    max_tokens = max(entry["channels"].shape[1] for entry in items)
    n_channels = items[0]["channels"].shape[2]
    max_stakeholders = max(1, max(len(entry["stakeholder_slots"]) for entry in items))
    max_contexts = max(1, max(len(entry["context_slots"]) for entry in items))

    arrays = {
        "channels": np.zeros((batch_size, max_alts, max_tokens, n_channels), dtype=np.float32),
        "level_slots": np.full((batch_size, max_alts, max_tokens), NO_LEVEL, dtype=np.int64),
        "indicator_slots": np.zeros((batch_size, max_tokens), dtype=np.int64),
        "family_slots": np.zeros((batch_size, max_tokens), dtype=np.int64),
        "token_mask": np.zeros((batch_size, max_tokens), dtype=bool),
        "alternative_mask": np.zeros((batch_size, max_alts), dtype=bool),
        "category_slots": np.zeros(batch_size, dtype=np.int64),
        "stakeholder_slots": np.zeros((batch_size, max_stakeholders), dtype=np.int64),
        "stakeholder_mask": np.zeros((batch_size, max_stakeholders), dtype=bool),
        "context_slots": np.zeros((batch_size, max_contexts), dtype=np.int64),
        "context_mask": np.zeros((batch_size, max_contexts), dtype=bool),
        "pref": np.full((batch_size, max_alts), np.nan, dtype=np.float32),
        "conf": np.full((batch_size, max_alts), np.nan, dtype=np.float32),
    }

    for row, entry in enumerate(items):
        n_alts, n_tokens, _ = entry["channels"].shape
        arrays["channels"][row, :n_alts, :n_tokens] = entry["channels"]
        arrays["level_slots"][row, :n_alts, :n_tokens] = entry["level_slots"]
        arrays["indicator_slots"][row, :n_tokens] = entry["indicator_slots"]
        arrays["family_slots"][row, :n_tokens] = entry["family_slots"]
        arrays["token_mask"][row, :n_tokens] = True
        arrays["alternative_mask"][row, :n_alts] = True
        arrays["category_slots"][row] = entry["category_slot"]

        for position, slot in enumerate(entry["stakeholder_slots"]):
            arrays["stakeholder_slots"][row, position] = slot
            arrays["stakeholder_mask"][row, position] = True
        for position, slot in enumerate(entry["context_slots"]):
            arrays["context_slots"][row, position] = slot
            arrays["context_mask"][row, position] = True

        arrays["pref"][row, :n_alts] = entry["pref"]
        arrays["conf"][row, :n_alts] = entry["conf"]

    return arrays
