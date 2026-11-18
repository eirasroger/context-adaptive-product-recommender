import numpy as np
import pytest

from eval.metrics import Predictions, top1_agreement, top1_regret


def _shortlists(*sets: tuple[list[float], list[float]]) -> Predictions:
    set_index = np.concatenate([np.full(len(predicted), index) for index, (predicted, _) in enumerate(sets)])
    predicted = np.concatenate([np.asarray(p, dtype=float) for p, _ in sets])
    actual = np.concatenate([np.asarray(a, dtype=float) for _, a in sets])
    return Predictions(set_index, np.zeros(len(predicted)), predicted, actual, np.ones(len(predicted)))


def test_regret_is_the_label_score_lost_at_the_first_choice():
    near_tie_miss = ([0.81, 0.80, 0.20], [0.79, 0.80, 0.20])
    clear_miss = ([0.90, 0.40], [0.30, 0.70])
    hit = ([0.70, 0.10], [0.75, 0.05])
    predictions = _shortlists(near_tie_miss, clear_miss, hit)

    assert top1_agreement(predictions) == pytest.approx(1 / 3)
    assert top1_regret(predictions) == pytest.approx((0.01 + 0.40 + 0.0) / 3)
