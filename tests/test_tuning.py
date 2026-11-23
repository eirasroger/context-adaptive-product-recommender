from pathlib import Path

import yaml

from experiments.tuning import STUDIES, apply_overrides, load_study, run_name, setting_of, verdict, with_trials
from train.config import TrainConfig

BASELINE = Path("train/configs/default.yaml")


def _row(mean: float, sd: float) -> dict:
    return {"gap_fidelity_mean": mean, "gap_fidelity_sd": sd}


def test_a_difference_within_the_seed_spread_is_level():
    assert verdict(_row(0.0798, 0.001), _row(0.0790, 0.001), "gap_fidelity") == "level"


def test_a_difference_beyond_the_seed_spread_is_better_or_worse():
    assert verdict(_row(0.0760, 0.001), _row(0.0790, 0.001), "gap_fidelity") == "better"
    assert verdict(_row(0.0820, 0.001), _row(0.0790, 0.001), "gap_fidelity") == "worse"


def test_a_setting_survives_its_run_name():
    assert setting_of(run_name("heads-8", 2)) == "heads-8"


def test_every_study_and_every_sampled_trial_names_real_config_keys_and_fits_its_heads():
    for name in yaml.safe_load(STUDIES.read_text(encoding="utf-8")):
        for overrides in load_study(name)["settings"].values():
            config = TrainConfig.load(BASELINE)
            apply_overrides(config, overrides or {})
            assert config.arch.dim % config.arch.encoder_heads == 0, name
            assert config.arch.dim % config.arch.comparator_heads == 0, name


SEARCH = {
    "reference": "baseline",
    "seeds": [0],
    "settings": {"baseline": {}},
    "search": {"seed": 3, "trials": 20, "space": {
        "optim.learning_rate": {"log": [1e-4, 1e-3]},
        "arch.dim": [64, 96],
    }},
}


def test_a_search_draws_the_same_trials_from_the_same_seed():
    assert with_trials(SEARCH)["settings"] == with_trials(SEARCH)["settings"]


def test_search_trials_stay_inside_their_space_and_keep_the_listed_settings():
    settings = with_trials(SEARCH)["settings"]
    assert settings["baseline"] == {}
    trials = [overrides for name, overrides in settings.items() if name.startswith("trial-")]
    assert len(trials) == 20
    assert all(1e-4 <= trial["optim.learning_rate"] <= 1e-3 for trial in trials)
    assert {trial["arch.dim"] for trial in trials} <= {64, 96}
