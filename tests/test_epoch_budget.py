from experiments.epoch_budget import DECIDING_SCOPES, METRICS, shortest_level_budget


def _row(mean: float, sd: float) -> dict:
    row = {}
    for metric in METRICS:
        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = sd
    return row


def _summary(rows: dict[int, dict]) -> dict:
    return {budget: {scope: row for scope in DECIDING_SCOPES} for budget, row in rows.items()}


def test_the_shortest_budget_within_seed_noise_of_the_best_is_chosen():
    summary = _summary({30: _row(0.0798, 0.001), 45: _row(0.0795, 0.001), 60: _row(0.0790, 0.001)})
    assert shortest_level_budget(summary) == 30


def test_a_short_budget_worse_beyond_seed_noise_is_passed_over():
    summary = _summary({30: _row(0.085, 0.001), 45: _row(0.0795, 0.001), 60: _row(0.0790, 0.001)})
    assert shortest_level_budget(summary) == 45


def test_a_single_budget_chooses_nothing():
    assert shortest_level_budget(_summary({30: _row(0.08, 0.001)})) is None
