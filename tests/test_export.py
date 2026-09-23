"""The served ONNX model scores exactly as the checkpoint it came from."""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from core.encoding import AlternativeInput, encode_set  # noqa: E402
from core.scoring import item, pad  # noqa: E402

TOLERANCE = 1e-5


def random_shortlists(registry, category_key, sizes, rng, n_contexts=None, n_stakeholders=None):
    category = registry.category(category_key)
    items = []
    for size in sizes:
        alternatives = []
        for position in range(size):
            values, levels = {}, {}
            for key in category.token_order:
                indicator = registry.indicator(key)
                if indicator.is_derived or rng.random() < 0.2:
                    continue
                if indicator.levels:
                    levels[key] = rng.choice(indicator.levels).key
                    continue
                spec = category.members[key].reference_range
                low, high = (spec.ref_low, spec.ref_high) if spec else (0.0, 1.0)
                values[key] = rng.uniform(low - 0.1 * (high - low), high + 0.1 * (high - low))
            alternatives.append(AlternativeInput(f"a{position}", values, levels))

        contexts = rng.sample(
            sorted(category.available_contexts), n_contexts or rng.choice([1, 2])
        )
        stakeholders = rng.sample(
            sorted(registry.stakeholders), n_stakeholders or rng.choice([1, 2, 3])
        )
        encoded = encode_set(registry, category_key, alternatives, contexts, stakeholders)
        items.append(item(encoded, category_key))
    return pad(items)


@pytest.fixture(scope="module")
def engines(checkpoint, served):
    from model import checkpoint as checkpoint_module
    from model.export import TorchScorer
    from serve.engine import ServedModel

    model, _, _ = checkpoint_module.load(checkpoint, device="cpu")
    return TorchScorer(model), ServedModel(served).scorer


def assert_same_scores(arrays, engines):
    torch_scorer, onnx_scorer = engines
    expected, got = torch_scorer(arrays), onnx_scorer(arrays)
    real = arrays["alternative_mask"]

    np.testing.assert_allclose(got[real], expected[real], atol=TOLERANCE)
    for row, count in enumerate(real.sum(axis=1)):
        assert list(np.argsort(-expected[row, :count])) == list(np.argsort(-got[row, :count]))


@pytest.mark.parametrize(
    ("sizes", "n_contexts", "n_stakeholders"),
    [
        ([2], 1, 1),
        ([5], 1, 3),
        ([2, 5, 3, 4], None, None),
        ([3] * 16, None, None),
    ],
    ids=["smallest", "widest", "mixed-sizes", "many-shortlists"],
)
def test_the_served_model_scores_as_the_checkpoint_does(
    registry, category_key, engines, sizes, n_contexts, n_stakeholders
):
    """The export's example input has two of every dimension, so these cases
    include shapes it never saw: one shortlist, one context, one stakeholder,
    five alternatives. A dimension fixed during export fails here."""
    rng = random.Random(len(sizes) * 100 + sum(sizes))
    arrays = random_shortlists(registry, category_key, sizes, rng, n_contexts, n_stakeholders)
    assert_same_scores(arrays, engines)


def test_the_served_model_carries_its_registry_and_its_source(checkpoint, served):
    import onnxruntime
    import torch

    from core import scoring
    from serve.release import digest

    carried = (
        onnxruntime.InferenceSession(str(served), providers=["CPUExecutionProvider"])
        .get_modelmeta()
        .custom_metadata_map
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    assert carried[scoring.REGISTRY_BLOB] == payload["registry_blob"]
    assert carried[scoring.REGISTRY_VERSION] == payload["meta"]["registry_version"]
    assert carried[scoring.SNAPSHOT_HASH] == payload["meta"]["snapshot_hash"]
    assert carried[scoring.SOURCE_SHA256] == digest(checkpoint)
