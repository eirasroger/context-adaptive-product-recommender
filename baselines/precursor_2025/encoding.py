"""The precursor's input encoding, training and prediction, driven by this repo's folds."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval.metrics import Predictions

from .source.data import config as source_config
from .source.data.loader import (
    PERF_MAP,
    _extract_features_from_alt,
    encode_scenario_pref,
    encode_stakeholder_pref,
)
from .source.model.architecture import SetRanker
from .source.model.trainer import train as source_train
from .source.utils.dataset import ScenarioDataset, collate_fn
from .source.utils.set_seed import set_seed

PUBLISHED_CHECKPOINT = Path(__file__).parent / "source" / "stored_models" / "ranking_model.pt"
PAPER_SPLIT_SEED = source_config.RANDOM_STATE


def read_published(frozen_path: Path | str, labelled_path: Path | str) -> tuple[list[dict], dict[str, dict]]:
    frozen = json.loads(Path(frozen_path).read_text(encoding="utf-8"))
    labelled = json.loads(Path(labelled_path).read_text(encoding="utf-8"))
    return frozen, {entry["id"]: entry for entry in labelled}


def encode(frozen: list[dict], labels: dict[str, dict], fit_ids: Iterable[str]) -> list[dict]:
    """`source.data.loader.load_data`, with the min-max scaling fitted on `fit_ids` alone.

    Given every scenario id, the result equals `load_data` exactly; given the training fold, no
    test value reaches the scaling.
    """
    fit_ids = set(fit_ids)
    feature_keys = sorted({
        key
        for scenario in frozen
        for alternative in scenario.get("alternatives", [])
        for key in alternative
        if key not in ("id_prod", "c", "wb")
    })

    rows, owners = [], []
    for scenario in frozen:
        stakeholders = encode_stakeholder_pref(scenario.get("stakeholder_preference", []))
        situations = encode_scenario_pref(scenario.get("scenario_preference", []))
        relevant = {key for situation in scenario.get("situations", []) or [] for key in PERF_MAP.get(situation, [])}
        for alternative in scenario.get("alternatives", []):
            rows.append((_extract_features_from_alt(alternative, feature_keys, relevant), stakeholders, situations))
            owners.append(scenario.get("id"))

    values = np.array([features[0::3] for features, _, _ in rows], dtype=np.float32)
    present = np.array([features[1::3] for features, _, _ in rows], dtype=np.int8)
    relevant = np.array([features[2::3] for features, _, _ in rows], dtype=np.int8)
    fitted = np.array([owner in fit_ids for owner in owners])

    scaled = np.zeros_like(values)
    for column in range(values.shape[1]):
        usable = (present[:, column] == 1) & (relevant[:, column] == 1)
        reference = values[usable & fitted, column]
        if reference.size == 0:
            continue
        low, high = float(reference.min()), float(reference.max())
        if high > low:
            scaled[usable, column] = (values[usable, column] - low) / (high - low)

    scenarios, row = [], 0
    for scenario in frozen:
        sid = scenario.get("id")
        alternatives = scenario.get("alternatives", [])
        if sid not in labels:
            row += len(alternatives)
            continue
        by_product = {entry["id_prod"]: entry for entry in labels[sid].get("labelled_alternatives", [])}
        features, prefs, confs = [], [], []
        for alternative in alternatives:
            _, stakeholders, situations = rows[row]
            combined = []
            for value, flag_present, flag_relevant in zip(scaled[row], present[row], relevant[row]):
                combined += [float(value), int(flag_present), int(flag_relevant)]
            features.append(combined + stakeholders.tolist() + situations.tolist())
            entry = by_product.get(alternative.get("id_prod"), {})
            prefs.append(entry.get("pref"))
            confs.append(entry.get("conf"))
            row += 1
        scenarios.append({
            "id": sid,
            "features": np.array(features, dtype=np.float32),
            "prefs": prefs,
            "confs": confs,
            "feature_keys": feature_keys + ["total_cost"],
        })
    return scenarios


def paper_split(scenarios: list[dict], seed: int = PAPER_SPLIT_SEED) -> tuple[list[dict], list[dict]]:
    """The published 70/30 split, rebuilt by the same sequence of shuffles as the source's `main.py`."""
    set_seed(seed)
    labelled = [s for s in scenarios if any(p is not None for p in s.get("prefs", []))]
    control = [s for s in labelled if str(s["id"]).startswith("control")]
    expert = [s for s in labelled if str(s["id"]).startswith("expert")]
    llm = [s for s in labelled if not str(s["id"]).startswith(("control", "expert"))]
    for group in (control, expert, llm):
        random.shuffle(group)
    train_share = 1 - source_config.TEST_SIZE
    cuts = [int(train_share * len(group)) for group in (control, expert, llm)]
    train = control[: cuts[0]] + expert[: cuts[1]] + llm[: cuts[2]]
    test = control[cuts[0] :] + expert[cuts[1] :] + llm[cuts[2] :]
    random.shuffle(train)
    random.shuffle(test)
    return train, test


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _on_device(scenarios: list[dict]) -> list[dict]:
    device = _device()
    return [
        {
            "id": s["id"],
            "features": torch.tensor(s["features"], dtype=torch.float32, device=device),
            "prefs": torch.tensor([float("nan") if p is None else p for p in s["prefs"]], device=device),
            "confs": torch.tensor([float("nan") if c is None else c for c in s["confs"]], device=device),
        }
        for s in scenarios
    ]


def new_model(feature_count: int) -> SetRanker:
    return SetRanker(
        feat_dim=feature_count,
        scenario_dim=len(source_config.STAKEHOLDER_PREFS) + len(source_config.SCENARIO_PREFS),
        hidden_dims=source_config.HIDDEN_DIM,
        dropout=source_config.DROPOUT,
    ).to(_device())


def published_model(feature_count: int) -> SetRanker:
    model = new_model(feature_count)
    model.load_state_dict(torch.load(PUBLISHED_CHECKPOINT, map_location=_device()))
    return model.eval()


def fit(train_scenarios: list[dict], val_scenarios: list[dict], seed: int) -> SetRanker:
    """Train as the source's `main.py` does, with early stopping on `val_scenarios`."""
    set_seed(seed)
    train_loader = DataLoader(
        ScenarioDataset(_on_device(train_scenarios)), source_config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(ScenarioDataset(_on_device(val_scenarios)), source_config.BATCH_SIZE, collate_fn=collate_fn)
    model = new_model(train_scenarios[0]["features"].shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=source_config.LR, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    source_train(
        model, train_loader, val_loader, optimizer, source_config.EPOCHS, scheduler,
        device="cuda", early_stopping_patience=5, early_stopping_min_delta=1e-6,
    )
    return model.eval()


@torch.no_grad()
def predict(model: SetRanker, scenarios: list[dict]) -> tuple[Predictions, list[str]]:
    """Predictions over labelled alternatives; `set_index` points into the returned scenario ids."""
    loader = DataLoader(ScenarioDataset(_on_device(scenarios)), source_config.BATCH_SIZE, collate_fn=collate_fn)
    set_index, position, predicted, actual, confidence = [], [], [], [], []
    offset = 0
    for features, mask, prefs, confs, _ in loader:
        scores = model(features, mask).cpu().numpy()
        prefs, confs, mask = prefs.cpu().numpy(), confs.cpu().numpy(), mask.cpu().numpy()
        rows, cols = np.nonzero(mask & ~np.isnan(prefs))
        set_index.append(rows + offset)
        position.append(cols)
        predicted.append(scores[rows, cols])
        actual.append(prefs[rows, cols])
        confidence.append(confs[rows, cols])
        offset += len(scores)
    predictions = Predictions(*(np.concatenate(part) for part in (set_index, position, predicted, actual, confidence)))
    return predictions, [s["id"] for s in scenarios]
