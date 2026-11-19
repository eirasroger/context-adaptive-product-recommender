import numpy as np
import pytest

from eval.metrics import Predictions, mean_reciprocal_rank, ndcg, top1_agreement, top1_regret


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


def test_reciprocal_rank_is_one_over_the_model_rank_of_the_top_labelled_product():
    first = ([0.9, 0.5, 0.1], [0.8, 0.6, 0.2])
    third = ([0.9, 0.5, 0.1], [0.2, 0.6, 0.8])
    predictions = _shortlists(first, third)

    assert mean_reciprocal_rank(predictions) == pytest.approx((1 + 1 / 3) / 2)


def test_ndcg_discounts_the_top_labelled_product_by_its_model_rank():
    second_of_two = ([0.9, 0.5], [0.3, 0.7])
    tied_top_labels = ([0.9, 0.5, 0.1], [0.7, 0.2, 0.7])
    predictions = _shortlists(second_of_two, tied_top_labels)

    tied = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert ndcg(predictions) == pytest.approx((1 / np.log2(3) + tied) / 2)
