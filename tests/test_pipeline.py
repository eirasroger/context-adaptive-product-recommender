"""End to end: generate, write, snapshot, prepare, encode, train a few steps."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from core.dataset import ComparisonSetDataset, collate
from core.encoding import encode_set
from core.prepare import prepare
from db.session import session_factory
from ingest.generators.writer import generate_for_category, write_cases
from ingest.split import assign
from snapshot.build import build


@pytest.fixture(scope="module")
def corpus(engine, seeded, registry, category_key, tmp_path_factory):
    """A small generated corpus, snapshotted and prepared."""
    from db import release as release_module

    session = session_factory(engine)()
    cases = generate_for_category(
        registry,
        category_key,
        stakeholders=sorted(registry.stakeholders)[:2],
        steps=4,
    )
    write_cases(session, registry, cases)
    session.commit()

    release_module.create_release(
        session,
        "test-pipeline",
        "for tests",
        mirror_dir=tmp_path_factory.mktemp("mirror"),
    )
    assign(session, "pipeline", val_fraction=0.2, test_fraction=0.2)
    session.commit()

    digest, directory = build(
        session, "pipeline", "test-pipeline", root=tmp_path_factory.mktemp("snapshots")
    )
    session.commit()

    from core import registry as registry_module

    released = registry_module.from_release(session, "test-pipeline")
    prepared = prepare(released, directory)
    return released, prepared, cases


def test_snapshot_round_trips_every_case(corpus):
    _, prepared, cases = corpus
    assert prepared.n_sets == len(cases)
    assert set(prepared.set_provenance) == {"control"}


def test_the_two_encoding_paths_agree(corpus):
    """Training reads core.prepare and serving reads core.encoding."""
    registry, prepared, _ = corpus

    external_to_case = {}
    for index in range(prepared.n_sets):
        external_to_case[prepared.set_external_ids[index]] = index

    checked = 0
    for index in range(0, prepared.n_sets, max(1, prepared.n_sets // 25)):
        category_key = prepared.category_keys[int(prepared.set_category[index])]
        start = int(prepared.set_start[index])
        count = int(prepared.set_count[index])

        alternatives = _alternatives_from_prepared(registry, prepared, index)
        context_keys = _context_keys(registry, prepared, index)
        stakeholder_keys = _stakeholder_keys(registry, prepared, index)

        expected = encode_set(
            registry, category_key, alternatives, context_keys, stakeholder_keys
        )
        actual = prepared.channels_for(index)

        assert actual.shape == expected.channels.shape
        np.testing.assert_allclose(actual, expected.channels, atol=1e-5)
        np.testing.assert_array_equal(
            prepared.product_level_slot[start : start + count, : actual.shape[1]],
            expected.level_slots,
        )
        checked += 1

    assert checked >= 5


def _alternatives_from_prepared(registry, prepared, index):
    """Rebuild the raw alternatives a prepared set came from, by inverting the normalisation."""
    from core.encoding import AlternativeInput, CHANNEL_INDEX

    category_key = prepared.category_keys[int(prepared.set_category[index])]
    category = registry.category(category_key)
    start = int(prepared.set_start[index])
    count = int(prepared.set_count[index])
    channels = prepared.channels_for(index)

    alternatives = []
    for row in range(count):
        values, levels = {}, {}
        for token, indicator_key in enumerate(category.token_order):
            if channels[row, token, CHANNEL_INDEX["present"]] < 0.5:
                continue
            indicator = registry.indicator(indicator_key)
            if indicator.is_scale:
                slot = int(prepared.product_level_slot[start + row, token])
                level = next(
                    level for level in indicator.levels if level.slot == slot
                )
                levels[indicator_key] = level.key
                continue
            if indicator.is_derived:
                continue
            spec = category.members[indicator_key].reference_range
            normalised = float(channels[row, token, CHANNEL_INDEX["v_ref"]])
            values[indicator_key] = _denormalise(normalised, spec)
        alternatives.append(
            AlternativeInput(key=f"a{row}", values=values, levels=levels)
        )
    return alternatives


def _denormalise(normalised: float, spec) -> float:
    import math

    if spec is None:
        return normalised
    if spec.scale == "log":
        return math.exp(
            math.log(spec.ref_low)
            + normalised * (math.log(spec.ref_high) - math.log(spec.ref_low))
        )
    return spec.ref_low + normalised * (spec.ref_high - spec.ref_low)


def _context_keys(registry, prepared, index):
    slots = prepared.combo_context_slots[int(prepared.set_combo[index])]
    by_slot = {context.slot: key for key, context in registry.contexts.items()}
    return [by_slot[slot] for slot in slots]


def _stakeholder_keys(registry, prepared, index):
    slots = prepared.set_stakeholder_slots[index]
    by_slot = {s.slot: key for key, s in registry.stakeholders.items()}
    return [by_slot[slot] for slot in slots]


def test_generated_labels_follow_the_declared_direction(corpus):
    registry, _, cases = corpus
    for case in cases:
        order = np.argsort(case.quality)
        prefs = np.asarray(case.prefs)[order]
        assert np.all(np.diff(prefs) >= -1e-9), (
            f"{case.indicator_key} under {case.context_keys}: better quality "
            "did not get a better label"
        )


def test_band_width_varies_by_stakeholder(registry, category_key):
    from ingest.generators.parametric import band_width

    widths = {}
    for indicator_key in registry.category(category_key).token_order:
        family = registry.indicator(indicator_key).family_key
        spread = {
            key: band_width(registry, key, indicator_key)
            for key in registry.stakeholders
        }
        widths[family] = max(spread.values()) - min(spread.values())

    assert any(width > 0.05 for width in widths.values()), widths


def test_a_few_training_steps_reduce_the_loss(corpus):
    registry, prepared, _ = corpus
    from model.loss import LossWeights, composite_loss
    from model.recommender import ModelConfig, Recommender

    torch.manual_seed(0)
    dataset = ComparisonSetDataset(prepared, fold=None)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=collate
    )
    model = Recommender(
        ModelConfig.for_registry(
            registry, dim=32, encoder_blocks=1, comparator_blocks=1, dropout=0.0
        )
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    weights = LossWeights(
        provenance={
            key: dict(c.provenance_weights) for key, c in registry.categories.items()
        }
    )

    losses = []
    for _ in range(6):
        epoch = []
        for batch in loader:
            optimiser.zero_grad()
            scores = model(batch)
            breakdown = composite_loss(
                scores,
                batch.pref,
                batch.conf,
                batch.alternative_mask,
                batch.provenance,
                batch.category_keys,
                weights,
            )
            breakdown.total.backward()
            optimiser.step()
            epoch.append(float(breakdown.total.detach()))
        losses.append(float(np.mean(epoch)))

    assert losses[-1] < losses[0], losses


def test_several_labellers_collapse_to_one_row_per_alternative():
    import pandas as pd

    from snapshot.build import _aggregate_labellers

    members = pd.DataFrame(
        {
            "set_id": [1, 1, 1, 1],
            "position": [0, 0, 1, 1],
            "product_id": [10, 10, 11, 11],
            "local_key": ["a", "a", "b", "b"],
            "pref": [0.9, 0.7, 0.4, 0.4],
            "conf": [1.0, 1.0, 1.0, 1.0],
            "scale_semantics": ["within_set_relative"] * 4,
        }
    )
    collapsed, multi = _aggregate_labellers(members)

    assert multi == 2
    assert len(collapsed) == 2
    assert collapsed.loc[0, "pref"] == pytest.approx(0.8)
    assert collapsed.loc[1, "pref"] == pytest.approx(0.4)
    # The contested alternative carries less weight than the unanimous one.
    assert collapsed.loc[0, "conf"] < collapsed.loc[1, "conf"]


def test_a_single_labeller_passes_through_untouched():
    import pandas as pd

    from snapshot.build import _aggregate_labellers

    members = pd.DataFrame(
        {
            "set_id": [1, 1],
            "position": [0, 1],
            "product_id": [10, 11],
            "local_key": ["a", "b"],
            "pref": [0.9, 0.4],
            "conf": [0.8, 0.8],
            "scale_semantics": ["within_set_relative"] * 2,
        }
    )
    collapsed, multi = _aggregate_labellers(members)
    assert multi == 0
    pd.testing.assert_frame_equal(collapsed, members)
