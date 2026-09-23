"""Which indicators drive the shipped model's scores, measured by withholding them.

    python -m experiments.attribution

Reads the committed test fold, so it runs from a clean checkout. Writes figures
and the tables behind them to runs/analysis/.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from core.dataset import collate
from core.encoding import AlternativeInput, encode_set
from core.registry import Registry

SNAPSHOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "snapshot"
CHECKPOINT = Path(__file__).resolve().parents[1] / "serve" / "release" / "model.pt"
OUTPUT = Path("runs") / "analysis"
FLOOR = 0.05


@dataclass(frozen=True)
class Shortlist:
    category_key: str
    alternatives: tuple[AlternativeInput, ...]
    contexts: tuple[str, ...]
    stakeholders: tuple[str, ...]


@dataclass(frozen=True)
class Family:
    key: str
    display_name: str
    sort_order: int


@dataclass(frozen=True)
class Effect:
    """One indicator under one condition, summarised over many alternatives."""

    condition: str
    indicator: str
    importance: float
    direction: float
    alternatives: int

    @property
    def signed(self) -> float:
        return self.importance * self.direction


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _item(registry: Registry, shortlist: Shortlist) -> dict:
    encoded = encode_set(
        registry,
        shortlist.category_key,
        shortlist.alternatives,
        shortlist.contexts,
        shortlist.stakeholders,
    )
    n = len(shortlist.alternatives)
    return {
        "index": 0,
        "channels": encoded.channels,
        "level_slots": encoded.level_slots,
        "indicator_slots": encoded.indicator_slots[0],
        "family_slots": encoded.family_slots[0],
        "category_slot": encoded.category_slot,
        "category_key": shortlist.category_key,
        "stakeholder_slots": encoded.stakeholder_slots,
        "context_slots": encoded.context_slots,
        "pref": np.full(n, np.nan, dtype=np.float32),
        "conf": np.full(n, np.nan, dtype=np.float32),
        "provenance": "control",
    }


@torch.no_grad()
def score_shortlists(
    model,
    registry: Registry,
    shortlists: Sequence[Shortlist],
    device: str = "cpu",
    chunk: int = 512,
) -> list[np.ndarray]:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(shortlists), chunk):
        window = shortlists[start : start + chunk]
        scores = model(collate([_item(registry, s) for s in window]).to(device)).cpu().numpy()
        out.extend(row[: len(s.alternatives)] for row, s in zip(scores, window))
    return out


def withhold(alternative: AlternativeInput, indicator_key: str) -> AlternativeInput:
    values = {k: v for k, v in alternative.values.items() if k != indicator_key}
    levels = {k: v for k, v in alternative.levels.items() if k != indicator_key}
    return AlternativeInput(key=alternative.key, values=values, levels=levels)


def supplied_indicators(registry: Registry, shortlist: Shortlist) -> list[str]:
    """Indicators some alternative has a value for. Derived ones are recomputed
    from their sources, so withholding one alone measures nothing."""
    return [
        key
        for key in registry.category(shortlist.category_key).token_order
        if not registry.indicator(key).is_derived
        and any(key in a.values or key in a.levels for a in shortlist.alternatives)
    ]


def attribute_many(
    model, registry: Registry, shortlists: Sequence[Shortlist], device: str = "cpu"
) -> list[dict[str, np.ndarray]]:
    """Per shortlist and indicator, how far knowing it moves each alternative's score.

    The indicator is withheld from every alternative at once and the shortlist
    scored again. A positive shift means knowing the value raised the score.
    """
    variants: list[Shortlist] = []
    plan: list[list[str]] = []
    for shortlist in shortlists:
        keys = supplied_indicators(registry, shortlist)
        plan.append(keys)
        variants.append(shortlist)
        variants.extend(
            replace(
                shortlist,
                alternatives=tuple(withhold(a, key) for a in shortlist.alternatives),
            )
            for key in keys
        )

    scores = iter(score_shortlists(model, registry, variants, device))
    results = []
    for keys in plan:
        baseline = next(scores)
        results.append({key: baseline - next(scores) for key in keys})
    return results


def attribution(
    model, registry: Registry, shortlist: Shortlist, device: str = "cpu"
) -> dict[str, np.ndarray]:
    return attribute_many(model, registry, [shortlist], device)[0]


def _comparable(registry: Registry, alternative: AlternativeInput, key: str) -> float | None:
    if key in alternative.levels:
        return registry.indicator(key).level(alternative.levels[key]).normalised_position
    return alternative.values.get(key)


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values))
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def summarise(
    registry: Registry,
    condition: str,
    shortlists: Sequence[Shortlist],
    attributions: Sequence[dict[str, np.ndarray]],
) -> list[Effect]:
    """Importance is the mean absolute shift. Direction is the rank correlation
    between an indicator's value and its shift: +1 means higher values helped."""
    pooled: dict[str, tuple[list[float], list[float]]] = {}
    for shortlist, shifts in zip(shortlists, attributions):
        for key, delta in shifts.items():
            values, deltas = pooled.setdefault(key, ([], []))
            for alternative, shift in zip(shortlist.alternatives, delta):
                value = _comparable(registry, alternative, key)
                if value is not None:
                    values.append(float(value))
                    deltas.append(float(shift))

    return [
        Effect(
            condition=condition,
            indicator=key,
            importance=float(np.mean(np.abs(deltas))),
            direction=rank_correlation(np.array(values), np.array(deltas)),
            alternatives=len(deltas),
        )
        for key, (values, deltas) in pooled.items()
    ]


def effects_by(
    model,
    registry: Registry,
    shortlists: Sequence[Shortlist],
    field: str,
    conditions: Sequence[str],
    device: str = "cpu",
) -> list[Effect]:
    """Re-run every shortlist with `field` set to each condition in turn,
    keeping everything else as recorded."""
    effects: list[Effect] = []
    for condition in conditions:
        varied = [replace(s, **{field: (condition,)}) for s in shortlists]
        effects.extend(
            summarise(registry, condition, varied, attribute_many(model, registry, varied, device))
        )
    return effects


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_shortlists(
    snapshot_dir: Path, provenances: Sequence[str] = ("llm", "expert")
) -> list[Shortlist]:
    import pandas as pd

    sets = pd.read_parquet(snapshot_dir / "sets.parquet")
    sets = sets[sets["provenance_key"].isin(provenances)]
    members = pd.read_parquet(snapshot_dir / "members.parquet")
    values = pd.read_parquet(snapshot_dir / "values.parquet")

    products: dict[int, tuple[dict, dict]] = {}
    known = values[values["present"] == 1]
    for product_id, key, number, level in zip(
        known["product_id"], known["indicator_key"], known["value_num"], known["level_key"]
    ):
        numbers, levels = products.setdefault(int(product_id), ({}, {}))
        if isinstance(level, str):
            levels[key] = level
        else:
            numbers[key] = float(number)

    by_set = members.sort_values("position").groupby("set_id")["product_id"].apply(list)
    return [
        Shortlist(
            category_key=row.category_key,
            alternatives=tuple(
                AlternativeInput(f"p{pid}", *products.get(int(pid), ({}, {})))
                for pid in by_set[row.set_id]
            ),
            contexts=tuple(k for k in row.context_keys.split("|") if k),
            stakeholders=tuple(k for k in row.stakeholder_keys.split("|") if k),
        )
        for row in sets.itertuples()
        if row.set_id in by_set.index
    ]


def preference_floors(
    registry: Registry, snapshot_dir: Path, threshold: float = FLOOR
) -> dict[str, str]:
    """Indicators where a single value sinks the preference to zero, with that value.

    Read from the control cases, where every other indicator is held at its
    ideal, so a near-zero label can only come from the swept indicator. Such an
    indicator acts as a veto, and its withholding effect dwarfs every other.
    """
    import pandas as pd

    sets = pd.read_parquet(snapshot_dir / "sets.parquet")
    members = pd.read_parquet(snapshot_dir / "members.parquet")
    values = pd.read_parquet(snapshot_dir / "values.parquet")

    control = sets[sets["provenance_key"] == "control"][["set_id", "generator_key"]]
    sunk = members.merge(control, on="set_id")
    sunk = sunk[(sunk["pref"] < threshold) & sunk["generator_key"].isin(registry.indicators)]
    held = sunk.merge(
        values, left_on=["product_id", "generator_key"], right_on=["product_id", "indicator_key"]
    )

    floors: dict[str, str] = {}
    for key, group in held.groupby("generator_key"):
        levels = group["level_key"].dropna()
        if len(levels):
            floors[key] = registry.indicator(key).level(levels.mode()[0]).display_name
        else:
            floors[key] = f"{group['value_num'].min():g}"
    return floors


def masking_note(registry: Registry, masked: dict[str, str]) -> str:
    if not masked:
        return ""
    return "".join(
        f" {registry.indicator(key).display_name} is masked: one value, "
        f"{value.lower()}, sinks the preference to 0, and its effect would swamp "
        "every other indicator."
        for key, value in sorted(masked.items())
    )


def shap_samples(shortlists: Sequence[Shortlist], size: int = 60) -> list[dict]:
    rows = [
        {**a.values, **a.levels} for s in shortlists for a in s.alternatives
    ]
    return random.Random(0).sample(rows, min(size, len(rows)))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
POSITIVE = "#2a78d6"
NEGATIVE = "#e34948"
NEUTRAL = "#f0efec"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _title(fig, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.985, title, fontsize=12, fontweight="semibold", va="top")
    fig.text(0.01, 0.95, subtitle, fontsize=9, color=INK_SECONDARY, va="top", wrap=True)


def _indicator_order(
    registry: Registry, families: dict[str, Family], keys: set[str]
) -> list[str]:
    return sorted(
        keys,
        key=lambda k: (
            families[registry.indicator(k).family_key].sort_order,
            registry.indicator(k).display_name,
        ),
    )


def heatmap(
    registry: Registry,
    families: dict[str, Family],
    effects: Sequence[Effect],
    conditions: Sequence[str],
    labels: dict[str, str],
    title: str,
    subtitle: str,
    path: Path,
    declared: dict[tuple[str, str], int] | None = None,
) -> None:
    plt = _style()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    rows = _indicator_order(registry, families, {e.indicator for e in effects})
    grid = np.full((len(rows), len(conditions)), np.nan)
    for effect in effects:
        grid[rows.index(effect.indicator), list(conditions).index(effect.condition)] = effect.signed
    limit = float(np.nanmax(np.abs(grid))) or 1.0

    import textwrap

    fig, ax = plt.subplots(figsize=(3.0 + 1.05 * len(conditions), 2.4 + 0.26 * len(rows)))
    fig.subplots_adjust(left=2.7 / (3.0 + 1.05 * len(conditions)), right=0.97, top=0.86, bottom=0.2)
    cmap = LinearSegmentedColormap.from_list("signed", [NEGATIVE, NEUTRAL, POSITIVE])
    mesh = ax.pcolormesh(
        grid, cmap=cmap, norm=TwoSlopeNorm(0.0, -limit, limit),
        edgecolors=SURFACE, linewidth=2,
    )
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(rows)) + 0.5, [registry.indicator(k).display_name for k in rows])
    ax.set_xticks(
        np.arange(len(conditions)) + 0.5,
        [textwrap.fill(labels[c], 13, break_long_words=False) for c in conditions],
        fontsize=8,
    )
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)

    family_of = [registry.indicator(k).family_key for k in rows]
    for index in range(1, len(rows)):
        if family_of[index] != family_of[index - 1]:
            ax.axhline(index, color=INK_MUTED, linewidth=0.8)

    if declared:
        for (key, condition), sign in declared.items():
            if key in rows and condition in conditions and sign:
                ax.text(
                    list(conditions).index(condition) + 0.5, rows.index(key) + 0.5,
                    "+" if sign > 0 else "−", ha="center", va="center",
                    fontsize=9, color=INK,
                )

    bar = fig.colorbar(
        mesh, cax=fig.add_axes([0.45, 0.06, 0.4, 0.018]), orientation="horizontal",
    )
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelsize=8)
    bar.set_label("higher values lower the score  ←   →  higher values raise it", fontsize=8)
    _title(fig, title, subtitle)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def family_shares(
    registry: Registry,
    families_by_key: dict[str, Family],
    groups: Sequence[tuple[str, Sequence[Effect], Sequence[str], dict[str, str]]],
    title: str,
    subtitle: str,
    path: Path,
) -> list[dict]:
    plt = _style()

    families = sorted(families_by_key, key=lambda k: families_by_key[k].sort_order)
    colours = dict(zip(families, SERIES))
    present = {
        registry.indicator(e.indicator).family_key
        for _, effects, _, _ in groups
        for e in effects
    }
    families = [f for f in families if f in present]
    rows_out: list[dict] = []

    heights = [len(conditions) for _, _, conditions, _ in groups]
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(7.6, 1.6 + 0.3 * sum(heights)),
        gridspec_kw={"height_ratios": heights},
    )
    fig.subplots_adjust(left=0.27, right=0.97, top=0.76, bottom=0.06, hspace=0.35)
    for ax, (heading, effects, conditions, labels) in zip(np.atleast_1d(axes), groups):
        for y, condition in enumerate(conditions):
            totals = {f: 0.0 for f in families}
            for e in effects:
                if e.condition == condition:
                    totals[registry.indicator(e.indicator).family_key] += e.importance
            whole = sum(totals.values()) or 1.0
            left = 0.0
            for family in families:
                share = totals[family] / whole
                rows_out.append({"group": heading, "condition": condition, "family": family, "share": round(share, 4)})
                ax.barh(y, share, left=left, height=0.62, color=colours[family], edgecolor=SURFACE, linewidth=2)
                if share >= 0.09:
                    ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=8, color=INK)
                left += share
        ax.set_yticks(range(len(conditions)), [labels[c] for c in conditions])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.tick_params(length=0)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(heading, loc="left", fontsize=9, color=INK_SECONDARY, pad=4)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[f]) for f in families]
    fig.legend(
        handles, [families_by_key[f].display_name for f in families],
        loc="upper left", bbox_to_anchor=(0.01, 0.855), ncol=len(families),
        frameon=False, fontsize=8, handlelength=1.0, columnspacing=1.2,
    )
    _title(fig, title, subtitle)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return rows_out


def agreement(
    registry: Registry, pairs: Sequence[dict], masked: dict[str, str], title: str, path: Path
) -> dict[str, float]:
    """Scatter of the two measures over the unmasked indicators. Both
    correlations are returned, with and without the masked ones."""
    plt = _style()

    x = np.array([p["withholding"] for p in pairs])
    y = np.array([p["shap"] for p in pairs])
    shown = np.array([p["indicator"] not in masked for p in pairs])
    rho = {
        "all": rank_correlation(x, y),
        "unmasked": rank_correlation(x[shown], y[shown]),
    }

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.76, bottom=0.12)
    ax.axhline(0, color=GRID, linewidth=1, zorder=0)
    ax.axvline(0, color=GRID, linewidth=1, zorder=0)
    ax.scatter(
        x[shown], y[shown], s=22, color=POSITIVE, edgecolors=SURFACE, linewidths=1, zorder=2,
    )
    ax.set_xlabel("Shift in score when the indicator is withheld")
    ax.set_ylabel("SHAP contribution")
    ax.tick_params(length=0)
    _title(
        fig, title,
        f"One point per alternative and indicator ({int(shown.sum())} points). "
        f"Rank correlation {rho['unmasked']:.2f}." + masking_note(registry, masked),
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return rho


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _effect_rows(registry: Registry, effects: Sequence[Effect]) -> list[dict]:
    return [
        {
            "condition": e.condition,
            "indicator": e.indicator,
            "family": registry.indicator(e.indicator).family_key,
            "importance": round(e.importance, 5),
            "direction": round(e.direction, 3),
            "signed": round(e.signed, 5),
            "alternatives": e.alternatives,
        }
        for e in effects
    ]


def family_table(checkpoint: Path) -> dict[str, Family]:
    """Family names and order, which the checkpoint's registry blob carries and
    the loaded Registry reduces to slots."""
    import yaml

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rows = yaml.safe_load(payload["registry_blob"])["indicator_family"]
    return {
        row["key"]: Family(row["key"], row["display_name"], int(row["sort_order"]))
        for row in rows
    }


def run(
    checkpoint: Path = CHECKPOINT,
    snapshot_dir: Path = SNAPSHOT,
    out: Path = OUTPUT,
    sets: int = 300,
    shap_alternatives: int = 30,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    from model import checkpoint as checkpoint_module

    model, meta, _ = checkpoint_module.load(checkpoint, device=device)
    registry = checkpoint_module.load_registry(checkpoint)
    families = family_table(checkpoint)
    everything = load_shortlists(Path(snapshot_dir))
    masked = preference_floors(registry, Path(snapshot_dir))
    note = masking_note(registry, masked)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "checkpoint": str(checkpoint),
        "registry_version": meta.registry_version,
        "snapshot": str(snapshot_dir),
        "seed": seed,
        "masked": masked,
        "categories": {},
    }

    for category_key in sorted({s.category_key for s in everything}):
        pool = [s for s in everything if s.category_key == category_key]
        chosen = random.Random(seed).sample(pool, min(sets, len(pool)))
        category = registry.category(category_key)
        contexts = sorted(category.available_contexts)
        stakeholders = sorted(registry.stakeholders)
        context_names = {k: registry.contexts[k].display_name for k in contexts}
        stakeholder_names = {k: registry.stakeholders[k].display_name for k in stakeholders}
        basis = f"{len(chosen)} test shortlists labelled by a language model or an expert"

        by_context = effects_by(model, registry, chosen, "contexts", contexts, device)
        by_stakeholder = effects_by(model, registry, chosen, "stakeholders", stakeholders, device)
        _write_csv(out / f"{category_key}-by-context.csv", _effect_rows(registry, by_context))
        _write_csv(out / f"{category_key}-by-stakeholder.csv", _effect_rows(registry, by_stakeholder))

        shown_by_context = [e for e in by_context if e.indicator not in masked]
        shown_by_stakeholder = [e for e in by_stakeholder if e.indicator not in masked]
        declared = {
            (key, context): int(np.sign(registry.resolve_direction(category_key, key, [context])[0]))
            for key in category.token_order
            for context in contexts
            if registry.is_relevant(category_key, key, [context])
        }
        heatmap(
            registry, families, shown_by_context, contexts, context_names,
            "What each indicator does to the score, by context",
            f"Colour: importance times learned direction, over {basis}. "
            "+ and − mark the direction the registry declares." + note,
            out / f"{category_key}-by-context.png", declared,
        )
        heatmap(
            registry, families, shown_by_stakeholder, stakeholders, stakeholder_names,
            "What each indicator does to the score, by stakeholder",
            f"Colour: importance times learned direction, over {basis}." + note,
            out / f"{category_key}-by-stakeholder.png",
        )
        shares = family_shares(
            registry,
            families,
            [
                ("By context", shown_by_context, contexts, context_names),
                ("By stakeholder", shown_by_stakeholder, stakeholders, stakeholder_names),
            ],
            "Share of the score each indicator family accounts for",
            f"Mean absolute shift when withheld, summed per family, over {basis}." + note,
            out / f"{category_key}-family-share.png",
        )
        _write_csv(out / f"{category_key}-family-share.csv", shares)

        summary = {"shortlists": len(chosen)}
        if shap_alternatives:
            summary["shap_rank_correlation"] = _shap_agreement(
                model, registry, chosen, shap_alternatives, seed, device, out,
                category_key, masked,
            )
        report["categories"][category_key] = summary

    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _shap_agreement(
    model, registry, chosen, count, seed, device, out, category_key, masked
) -> dict[str, float]:
    from experiments import shap_slices

    rng = random.Random(seed)
    background = shap_samples(chosen)
    pairs: list[dict] = []
    for shortlist in rng.sample(chosen, min(count, len(chosen))):
        target = rng.randrange(len(shortlist.alternatives))
        withheld = attribution(model, registry, shortlist, device)
        explained = shap_slices.explain(
            model, registry, category_key, shortlist.alternatives, target,
            shortlist.contexts, shortlist.stakeholders,
            background_samples=background, background_origin="test fold", device=device,
        )
        for entry in explained["indicators"]:
            if entry["indicator"] in withheld:
                pairs.append({
                    "indicator": entry["indicator"],
                    "withholding": round(float(withheld[entry["indicator"]][target]), 5),
                    "shap": entry["contribution"],
                })
    _write_csv(out / f"{category_key}-shap-agreement.csv", pairs)
    return agreement(
        registry, pairs, masked, "Withholding compared with SHAP",
        out / f"{category_key}-shap-agreement.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--sets", type=int, default=300, help="shortlists sampled per category")
    parser.add_argument("--shap", type=int, default=30, help="alternatives to explain with SHAP; 0 skips")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    report = run(args.checkpoint, args.snapshot, args.out, args.sets, args.shap, args.seed, args.device)
    print(json.dumps(report, indent=2))
    print(f"figures and tables in {args.out}")


if __name__ == "__main__":
    main()
