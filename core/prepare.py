"""Turn a snapshot into arrays the training loop can index into cheaply.

"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.encoding import CHANNEL_INDEX, CHANNELS, NO_LEVEL, N_CHANNELS
from core.registry import Registry

PRODUCT_CHANNELS = (
    "v_ref",
    "v_ref_saturated",
    "v_withinset",
    "v_withinset_defined",
    "present",
    "ideal_distance",
    "has_ideal",
)
CONTEXT_CHANNELS = ("relevant", "direction", "priority")

PRODUCT_CHANNEL_IDS = tuple(CHANNEL_INDEX[name] for name in PRODUCT_CHANNELS)
CONTEXT_CHANNEL_IDS = tuple(CHANNEL_INDEX[name] for name in CONTEXT_CHANNELS)

CACHE_FILE = "prepared.pkl"

#: Bumped whenever the encoding changes shape or meaning. The cache is keyed on
#: it as well as on the data, so a fix here can never be masked by a stale file
#: computed under the old behaviour.
CACHE_VERSION = 3


@dataclass(frozen=True)
class Prepared:
    """A snapshot in array form, addressed by set index."""

    #: category-level token identity, (n_categories, max_tokens)
    token_indicator_slot: np.ndarray
    token_family_slot: np.ndarray
    token_valid: np.ndarray
    category_slot: np.ndarray            # (n_categories,)
    category_keys: tuple[str, ...]

    #: per product, (n_products, max_tokens, len(PRODUCT_CHANNELS))
    product_channels: np.ndarray
    product_level_slot: np.ndarray       # (n_products, max_tokens)

    #: per distinct set of active contexts, (n_combos, max_tokens, 3)
    combo_channels: np.ndarray
    combo_context_slots: tuple[tuple[int, ...], ...]

    #: per comparison set
    set_ids: np.ndarray
    set_external_ids: tuple[str, ...]
    set_category: np.ndarray             # index into category_keys
    set_combo: np.ndarray                # index into combo_channels
    set_start: np.ndarray                # first member row
    set_count: np.ndarray                # number of members
    set_fold: tuple[str, ...]
    set_provenance: tuple[str, ...]
    set_generator: tuple[str | None, ...]
    set_semantics: tuple[str, ...]
    set_stakeholder_slots: tuple[tuple[int, ...], ...]

    #: per member, ordered by (set, position); rows align with product arrays
    member_pref: np.ndarray
    member_conf: np.ndarray
    member_local_key: tuple[str, ...]

    registry_hash: str | None

    @property
    def n_sets(self) -> int:
        return len(self.set_ids)

    @property
    def max_tokens(self) -> int:
        return self.token_valid.shape[1]

    def channels_for(self, set_index: int) -> np.ndarray:
        """Assemble the full channel block for one set, (n_alts, n_tokens, C)."""
        start = int(self.set_start[set_index])
        count = int(self.set_count[set_index])
        category = int(self.set_category[set_index])
        n_tokens = int(self.token_valid[category].sum())

        block = np.zeros((count, n_tokens, N_CHANNELS), dtype=np.float32)
        block[:, :, PRODUCT_CHANNEL_IDS] = self.product_channels[
            start : start + count, :n_tokens, :
        ]
        block[:, :, CONTEXT_CHANNEL_IDS] = self.combo_channels[
            int(self.set_combo[set_index]), :n_tokens, :
        ]
        return block


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def prepare(registry: Registry, snapshot_dir: Path) -> Prepared:
    """Compute the prepared arrays for a snapshot."""
    sets = pd.read_parquet(snapshot_dir / "sets.parquet")
    members = pd.read_parquet(snapshot_dir / "members.parquet")
    values = pd.read_parquet(snapshot_dir / "values.parquet")

    members = members.sort_values(["set_id", "position"], kind="stable").reset_index(
        drop=True
    )
    sets = sets.sort_values("set_id", kind="stable").reset_index(drop=True)

    category_keys = tuple(sorted(sets["category_key"].unique()))
    category_index = {key: index for index, key in enumerate(category_keys)}
    token_index: dict[str, dict[str, int]] = {
        key: {
            indicator_key: position
            for position, indicator_key in enumerate(
                registry.category(key).token_order
            )
        }
        for key in category_keys
    }
    max_tokens = max(len(mapping) for mapping in token_index.values())

    token_indicator_slot = np.zeros((len(category_keys), max_tokens), dtype=np.int64)
    token_family_slot = np.zeros((len(category_keys), max_tokens), dtype=np.int64)
    token_valid = np.zeros((len(category_keys), max_tokens), dtype=bool)
    category_slot = np.zeros(len(category_keys), dtype=np.int64)

    for key, index in category_index.items():
        category = registry.category(key)
        category_slot[index] = category.slot
        for position, indicator_key in enumerate(category.token_order):
            indicator = registry.indicator(indicator_key)
            token_indicator_slot[index, position] = indicator.slot
            token_family_slot[index, position] = indicator.family_slot
            token_valid[index, position] = True

    # -- set bookkeeping ---------------------------------------------------
    set_order = {set_id: index for index, set_id in enumerate(sets["set_id"].tolist())}
    member_set_index = members["set_id"].map(set_order).to_numpy()
    boundaries = np.flatnonzero(np.diff(member_set_index, prepend=-1))
    set_start = np.zeros(len(sets), dtype=np.int64)
    set_count = np.zeros(len(sets), dtype=np.int64)
    ends = np.append(boundaries[1:], len(members))
    for start, end in zip(boundaries, ends):
        index = int(member_set_index[start])
        set_start[index] = start
        set_count[index] = end - start

    set_category = sets["category_key"].map(category_index).to_numpy()

    # -- product channels --------------------------------------------------
    product_row = {
        product_id: row
        for row, product_id in enumerate(members["product_id"].tolist())
    }
    n_products = len(members)

    product_channels = np.zeros(
        (n_products, max_tokens, len(PRODUCT_CHANNELS)), dtype=np.float32
    )
    product_level_slot = np.full((n_products, max_tokens), NO_LEVEL, dtype=np.int64)

    member_category = set_category[member_set_index]

    for key, index in category_index.items():
        rows = np.flatnonzero(member_category == index)
        if rows.size == 0:
            continue
        _fill_category_block(
            registry,
            key,
            token_index[key],
            values,
            rows,
            product_row,
            members,
            member_set_index,
            product_channels,
            product_level_slot,
        )

    # -- context combinations ---------------------------------------------
    combo_labels = (
        sets["category_key"].astype(str) + "@" + sets["context_keys"].astype(str)
    )
    unique_labels = sorted(combo_labels.unique())
    combo_lookup = {label: index for index, label in enumerate(unique_labels)}
    combo_channels = np.zeros(
        (len(unique_labels), max_tokens, len(CONTEXT_CHANNELS)), dtype=np.float32
    )
    combo_context_slots: list[tuple[int, ...]] = []

    for label, index in combo_lookup.items():
        category_key, _, joined = label.partition("@")
        context_keys = [key for key in joined.split("|") if key]
        combo_context_slots.append(
            tuple(registry.contexts[key].slot for key in context_keys)
        )
        for indicator_key, position in token_index[category_key].items():
            direction, priority = registry.resolve_direction(
                category_key, indicator_key, context_keys
            )
            relevant = registry.is_relevant(category_key, indicator_key, context_keys)
            combo_channels[index, position] = (float(relevant), direction, priority)

    set_combo = combo_labels.map(combo_lookup).to_numpy()

    semantics = members.groupby("set_id")["scale_semantics"].first()
    stakeholders = tuple(
        tuple(
            registry.stakeholders[key].slot
            for key in (joined.split("|") if joined else [])
        )
        for joined in sets["stakeholder_keys"].fillna("").tolist()
    )

    return Prepared(
        token_indicator_slot=token_indicator_slot,
        token_family_slot=token_family_slot,
        token_valid=token_valid,
        category_slot=category_slot,
        category_keys=category_keys,
        product_channels=product_channels,
        product_level_slot=product_level_slot,
        combo_channels=combo_channels,
        combo_context_slots=tuple(combo_context_slots),
        set_ids=sets["set_id"].to_numpy(),
        set_external_ids=tuple(sets["external_id"].astype(str).tolist()),
        set_category=set_category,
        set_combo=set_combo,
        set_start=set_start,
        set_count=set_count,
        set_fold=tuple(sets["fold"].tolist()),
        set_provenance=tuple(sets["provenance_key"].tolist()),
        set_generator=tuple(sets["generator_key"].tolist()),
        set_semantics=tuple(
            sets["set_id"].map(semantics).fillna("within_set_relative").tolist()
        ),
        set_stakeholder_slots=stakeholders,
        member_pref=members["pref"].to_numpy(dtype=np.float32, na_value=np.nan),
        member_conf=members["conf"].to_numpy(dtype=np.float32, na_value=np.nan),
        member_local_key=tuple(members["local_key"].astype(str).tolist()),
        registry_hash=registry.content_hash,
    )


def _fill_category_block(
    registry: Registry,
    category_key: str,
    token_index: dict[str, int],
    values: pd.DataFrame,
    rows: np.ndarray,
    product_row: dict[int, int],
    members: pd.DataFrame,
    member_set_index: np.ndarray,
    product_channels: np.ndarray,
    product_level_slot: np.ndarray,
) -> None:
    """Scatter one category's values into the product arrays, then normalise."""
    product_ids = members["product_id"].to_numpy()[rows]
    wanted = set(product_ids.tolist())

    block = values[values["product_id"].isin(wanted)]
    if block.empty:
        return

    target_row = block["product_id"].map(product_row).to_numpy()
    target_col = block["indicator_key"].map(token_index).to_numpy()
    keep = ~pd.isna(target_col)
    target_row = target_row[keep].astype(np.int64)
    target_col = target_col[keep].astype(np.int64)

    raw = np.full(product_channels.shape[:2], np.nan, dtype=np.float64)
    present = np.zeros(product_channels.shape[:2], dtype=bool)

    numeric = block["value_num"].to_numpy(dtype=np.float64, na_value=np.nan)[keep]
    level_keys = block["level_key"].to_numpy()[keep]
    is_present = block["present"].to_numpy()[keep].astype(bool)

    raw[target_row, target_col] = numeric
    present[target_row, target_col] = is_present

    # Ordinal and nominal values arrive as level keys; an ordinal level's
    # declared position is the comparable scalar, a nominal one has none.
    level_slot_lookup: dict[tuple[str, str], int] = {}
    level_position: dict[tuple[str, str], float] = {}
    for indicator_key in token_index:
        indicator = registry.indicator(indicator_key)
        for level in indicator.levels:
            level_slot_lookup[(indicator_key, level.key)] = level.slot
            if level.normalised_position is not None:
                level_position[(indicator_key, level.key)] = level.normalised_position

    if level_slot_lookup:
        has_level = pd.notna(level_keys)
        if has_level.any():
            pairs = list(
                zip(
                    block["indicator_key"].to_numpy()[keep][has_level],
                    level_keys[has_level],
                )
            )
            slots = np.array(
                [level_slot_lookup.get(pair, NO_LEVEL) for pair in pairs], dtype=np.int64
            )
            positions = np.array(
                [level_position.get(pair, np.nan) for pair in pairs], dtype=np.float64
            )
            product_level_slot[target_row[has_level], target_col[has_level]] = slots
            raw[target_row[has_level], target_col[has_level]] = positions

    # Derived indicators are recomputed from their sources rather than stored,
    # so correcting a derivation never means rewriting the corpus. A derived
    # value exists only where every source it needs is present.
    for indicator_key, column in token_index.items():
        indicator = registry.indicator(indicator_key)
        if not indicator.is_derived or not indicator.sources:
            continue
        total = np.zeros(raw.shape[0], dtype=np.float64)
        complete = np.ones(raw.shape[0], dtype=bool)
        for source_key, coefficient in indicator.sources:
            source_column = token_index.get(source_key)
            if source_column is None:
                complete[:] = False
                break
            complete &= present[:, source_column]
            total += coefficient * np.nan_to_num(raw[:, source_column])
        raw[:, column] = np.where(complete, total, np.nan)
        present[:, column] = complete

    category = registry.category(category_key)
    normalised = np.zeros_like(raw)
    saturated = np.zeros_like(raw)
    ideal_distance = np.zeros_like(raw)
    has_ideal = np.zeros_like(raw)

    for indicator_key, column in token_index.items():
        indicator = registry.indicator(indicator_key)
        spec = category.members[indicator_key].reference_range
        values_column = raw[:, column]

        if indicator.is_scale or spec is None:
            normalised[:, column] = np.nan_to_num(values_column)
            continue

        if spec.scale == "log":
            safe = np.where(values_column > 0.0, values_column, np.nan)
            scaled = (np.log(safe) - np.log(spec.ref_low)) / (
                np.log(spec.ref_high) - np.log(spec.ref_low)
            )
        else:
            scaled = (values_column - spec.ref_low) / (spec.ref_high - spec.ref_low)

        outside = (scaled < 0.0) | (scaled > 1.0) | (
            np.isnan(scaled) & ~np.isnan(values_column)
        )
        clipped = np.clip(np.nan_to_num(scaled), 0.0, 1.0)
        normalised[:, column] = clipped
        saturated[:, column] = outside.astype(np.float64)

        if spec.shape != "monotone":
            distance = _ideal_distance_column(clipped, spec)
            if distance is not None:
                ideal_distance[:, column] = distance
                has_ideal[:, column] = 1.0

    # Missing values contribute nothing on the value channels.
    normalised = np.where(present, normalised, 0.0)
    saturated = np.where(present, saturated, 0.0)
    ideal_distance = np.where(present, ideal_distance, 0.0)
    has_ideal = np.where(present, has_ideal, 0.0)

    within, within_defined = _within_set(
        np.where(present, raw, np.nan), member_set_index
    )

    stack = {
        "v_ref": normalised,
        "v_ref_saturated": saturated,
        "v_withinset": within,
        "v_withinset_defined": within_defined,
        "present": present.astype(np.float64),
        "ideal_distance": ideal_distance,
        "has_ideal": has_ideal,
    }
    for position, name in enumerate(PRODUCT_CHANNELS):
        product_channels[rows, :, position] = stack[name][rows].astype(np.float32)


def _ideal_distance_column(normalised: np.ndarray, spec) -> np.ndarray | None:
    from core.encoding import normalise_reference

    if spec.shape == "ideal_point":
        if spec.ideal_value is None:
            return None
        ideal, _ = normalise_reference(spec.ideal_value, spec)
        return np.abs(normalised - ideal)
    if spec.ideal_low is None or spec.ideal_high is None:
        return None
    low, _ = normalise_reference(spec.ideal_low, spec)
    high, _ = normalise_reference(spec.ideal_high, spec)
    return np.maximum(0.0, np.maximum(low - normalised, normalised - high))


def _within_set(
    raw: np.ndarray, member_set_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise each indicator against the other alternatives in its own set.

    Needed alongside the reference channel: within-set normalisation alone
    cannot say that a whole shortlist is poor, and reference normalisation alone
    loses resolution when the alternatives sit close together.
    """
    starts = np.flatnonzero(np.diff(member_set_index, prepend=-1))
    counts = np.diff(np.append(starts, len(member_set_index)))

    lows = np.minimum.reduceat(np.nan_to_num(raw, nan=np.inf), starts, axis=0)
    highs = np.maximum.reduceat(np.nan_to_num(raw, nan=-np.inf), starts, axis=0)
    present_counts = np.add.reduceat((~np.isnan(raw)).astype(np.int32), starts, axis=0)

    span = highs - lows
    usable = (present_counts >= 2) & (span > 0.0)

    lows_expanded = np.repeat(lows, counts, axis=0)
    span_expanded = np.repeat(np.where(usable, span, 1.0), counts, axis=0)
    usable_expanded = np.repeat(usable, counts, axis=0)

    values = (raw - lows_expanded) / span_expanded
    defined = usable_expanded & ~np.isnan(raw)
    return np.where(defined, values, 0.0), defined.astype(np.float64)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def snapshot_hash(snapshot_dir: Path) -> str:
    """The snapshot's own content hash, from its manifest.

    A real snapshot is stored in a directory named after this, but a fixture
    kept under a fixed name is not, and keying a cache on the directory name
    would serve the previous contents back after a rebuild.
    """
    import json

    manifest = snapshot_dir / "manifest.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text(encoding="utf-8")).get("content_hash")
        if recorded:
            return str(recorded)
    return snapshot_dir.name


def cache_key(registry: Registry, snapshot_dir: Path) -> str:
    digest = hashlib.sha256()
    digest.update(str(CACHE_VERSION).encode("utf-8"))
    digest.update((registry.content_hash or "").encode("utf-8"))
    digest.update(snapshot_hash(snapshot_dir).encode("utf-8"))
    digest.update("|".join(CHANNELS).encode("utf-8"))
    return digest.hexdigest()[:16]


def load_or_prepare(registry: Registry, snapshot_dir: Path) -> Prepared:
    """Prepare once, reuse thereafter.

    Keyed by both the snapshot and the registry hash, because the same data
    under changed semantics is different input.
    """
    import pickle

    path = snapshot_dir / f"{cache_key(registry, snapshot_dir)}-{CACHE_FILE}"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)

    prepared = prepare(registry, snapshot_dir)
    with path.open("wb") as handle:
        pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return prepared
