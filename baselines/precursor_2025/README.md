# Precursor (2025)

The concrete-only model from the 2025 study that preceded this repository,
kept as the baseline for the generic model. It is a past model: nothing in `serve/`, `train/` or `model/`
uses it, and it is never trained during a normal run.

## Source

- Repository: <https://github.com/eirasroger/concrete-selection-dl-model>,
  commit `d7a214b7681bfccd87a18fe685754f6de0822d6a` (21 September 2026).
- Paper: Vergés, Forcada, Gaspar and Hosseini (2026), *Sustainable Production
  and Consumption* 67, 119-144.
- Data: CORA Research Data Repository, DOI
  [10.34810/DATA3164](https://doi.org/10.34810/DATA3164) (`frozen_dataset.json`
  and `labelled_dataset.json`, 42,874 scenarios).

## Why it lives here

The research question of this repository is whether one generic model, built
to take any product category, matches a concrete-only model on concrete within
noise. Answering it needs the precursor trained on the same scenarios and folds
as the generic model and scored by the same evaluation code. A copy pinned to
one commit keeps that comparison reproducible from this repository alone.

## Layout

| Path | Holds |
|---|---|
| `source/` | The precursor's model, training and encoding code and its published checkpoint, copied from the commit above |
| `encoding.py` | The adapter: encodes the published scenarios, trains and predicts on this repository's folds |
| `../../experiments/generic_vs_precursor.py` | The experiment: reproduction of the paper's table, and the comparison |

`source/` holds only what the model needs (`model/architecture.py`,
`model/trainer.py`, `data/config.py`, `data/loader.py`, `utils/dataset.py`,
`utils/set_seed.py`, `stored_models/ranking_model.pt`). One line differs from
the commit: `model/trainer.py` imports `..utils.dataset` where the original
imports `utils.dataset`, so the package cannot collide with this repository's
own modules. `source/__init__.py` and `source/data/__init__.py` are added.

## How the comparison runs it

The adapter keeps the model as published: the 66 input features, the
architecture, the loss and its group weights, the optimiser, the schedule, the
patience and the batch size. Given every scenario id, `encode` reproduces the
original `load_data` exactly (checked on 2026-11-19: 42,874 scenarios, every
feature equal).

The protocol is this repository's, shared with the generic model:

1. **Folds.** It trains on this repository's training fold and is scored on
   its test fold, the same sets as the generic model.
2. **Early stopping** watches the validation fold, so no test set informs
   training.
3. **Scaling.** The min-max scaling is fitted on the training fold alone.
4. **Metrics.** `eval.metrics` scores MRR and nDCG with one query per set.
   `reproduce` also reports them grouped as the original evaluation script
   grouped them, which is how its published values are recovered.

## Running

```
python -m experiments.generic_vs_precursor reproduce <frozen_dataset.json> <labelled_dataset.json>
python -m experiments.generic_vs_precursor compare <frozen_dataset.json> <labelled_dataset.json> [--seeds 0 1 2 3 4] [--arms ...]
```

`reproduce` scores the published checkpoint on the paper's split and recovers
its Table 2. `compare` trains the precursor, the generic model on the
published concrete scenarios alone, and the generic model on the full joint
corpus, once per seed, and scores all three on the published concrete test
fold. Results go to `runs/generic-vs-precursor-<time>/`.
