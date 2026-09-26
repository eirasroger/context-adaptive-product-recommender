from __future__ import annotations

from pathlib import Path

import pytest

from experiments import monte_carlo_cv
from ingest.split import DEFAULT_SPLIT

CONFIG = Path(__file__).resolve().parents[1] / "train" / "configs" / "default.yaml"


def _results(gap: float, tau: float) -> dict:
    cell = {
        "n_sets": 400, "gap_fidelity": gap, "pointwise_mae": gap / 2,
        "top1_agreement": 0.9, "tie_tolerant_tau": tau,
    }
    return {"overall": cell, "stratified": {"category": {"facade_system": cell}}}


def test_a_cross_validation_run_never_promotes(tmp_path):
    config = monte_carlo_cv.configure(CONFIG, "mccv-1", tmp_path)

    assert config.promote is False
    assert config.split_key == "mccv-1"
    assert config.output_dir == str(tmp_path)


def test_no_split_key_is_the_release_split():
    assert DEFAULT_SPLIT not in monte_carlo_cv.split_keys(50)


def test_later_repeats_extend_an_earlier_run_without_reusing_a_split():
    assert set(monte_carlo_cv.split_keys(5)).isdisjoint(monte_carlo_cv.split_keys(5, first=6))


def test_the_spread_is_measured_in_every_scope():
    summary = monte_carlo_cv.summarise([_results(0.04, 0.80), _results(0.06, 0.84)])

    for scope in ("overall", "category/facade_system"):
        row = summary[scope]
        assert row["runs"] == 2
        assert row["gap_fidelity_mean"] == pytest.approx(0.05)
        assert row["gap_fidelity_sd"] == pytest.approx(0.01414, abs=1e-4)
        assert row["tie_tolerant_tau_min"] == pytest.approx(0.80)
        assert row["gap_cv"] == pytest.approx(0.2828, abs=1e-3)


def test_the_tolerance_grows_with_the_spread():
    assert monte_carlo_cv.gap_tolerance(0.0) == 0.0
    assert monte_carlo_cv.gap_tolerance(0.05) == pytest.approx(0.116, abs=1e-3)


def test_a_split_config_survives_the_file_the_training_process_reads(tmp_path):
    from train.config import TrainConfig

    config = monte_carlo_cv.configure(CONFIG, "mccv-3", tmp_path)
    assert TrainConfig.load(config.dump(tmp_path / "mccv-3.yaml")) == config


def test_a_rerun_split_counts_once_with_its_latest_result(tmp_path):
    import json

    for name, key, gap in (("mccv-1-a", "mccv-1", 0.9), ("mccv-1-b", "mccv-1", 0.04),
                           ("mccv-2-a", "mccv-2", 0.06), ("mccv-3-a", "mccv-3", None)):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text(f"split_key: {key}\n", encoding="utf-8")
        if gap is not None:
            (run_dir / "metrics.json").write_text(json.dumps(_results(gap, 0.8)), encoding="utf-8")

    runs = monte_carlo_cv.finished_runs(tmp_path)

    assert [r["split_key"] for r in runs] == ["mccv-1", "mccv-2"]
    assert runs[0]["results"]["overall"]["gap_fidelity"] == 0.04


GB = 1 << 30


def _capacity(cores, free_ram_gb, free_gpu_gb, run_cores=1.5, run_ram_gb=2.0, run_gpu_gb=0.75):
    return monte_carlo_cv.capacity(
        monte_carlo_cv.Machine(
            physical_cores=cores,
            free_ram_bytes=int(free_ram_gb * GB),
            free_gpu_bytes=None if free_gpu_gb is None else int(free_gpu_gb * GB),
        ),
        monte_carlo_cv.Footprint(
            cores=run_cores, ram_bytes=int(run_ram_gb * GB),
            gpu_bytes=None if free_gpu_gb is None else int(run_gpu_gb * GB),
        ),
    )


def test_a_small_machine_trains_one_split_at_a_time():
    parallel, _ = _capacity(cores=2, free_ram_gb=1, free_gpu_gb=None)
    assert parallel == 1


def test_the_scarcest_resource_sets_the_parallelism():
    parallel, limits = _capacity(cores=32, free_ram_gb=100, free_gpu_gb=2)
    assert limits["gpu"] == parallel == 3
    assert limits["cpu"] > parallel and limits["ram"] > parallel


def test_a_large_machine_trains_many_splits_at_once():
    parallel, _ = _capacity(cores=32, free_ram_gb=100, free_gpu_gb=22)
    assert parallel >= 16


def test_without_a_gpu_only_cores_and_memory_count():
    _, limits = _capacity(cores=8, free_ram_gb=16, free_gpu_gb=None)
    assert set(limits) == {"cpu", "ram"}
