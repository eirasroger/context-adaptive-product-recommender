"""Evaluation report rendering and the release gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from core.dataset import ComparisonSetDataset, collate
from core.prepare import Prepared
from core.registry import Registry
from eval import metrics as metrics_module


def evaluate_all(
    model,
    registry: Registry,
    prepared: Prepared,
    device: str | torch.device = "cpu",
    fold: str = "test",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Overall, stratified and per-family metrics for one fold."""
    dataset = ComparisonSetDataset(prepared, fold=fold)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    predictions = metrics_module.predict(model, loader, device)

    return {
        "fold": fold,
        "overall": metrics_module.compute(predictions).as_dict(),
        "stratified": metrics_module.stratified(
            predictions, prepared, registry=registry
        ),
        "by_family": metrics_module.family_breakdown(predictions, prepared, registry),
    }


@dataclass
class Thresholds:
    """What a build must clear to be releasable. None leaves a metric ungated."""

    max_gap_fidelity: float | None = None
    max_band_placement: float | None = None
    #: How much worse any stratum may get against the previous release.
    max_stratum_regression: float | None = None
    require_behavioural: bool = True


@dataclass
class GateResult:
    passed: bool
    reasons: list[str]

    def __str__(self) -> str:
        if self.passed:
            return "gate: PASS"
        return "gate: FAIL\n" + "\n".join(f"  - {reason}" for reason in self.reasons)


def gate(
    results: dict[str, Any],
    thresholds: Thresholds,
    baseline: dict[str, Any] | None = None,
) -> GateResult:
    reasons: list[str] = []
    overall = results.get("overall", {})

    if thresholds.max_gap_fidelity is not None:
        value = overall.get("gap_fidelity")
        if value is None or value > thresholds.max_gap_fidelity:
            reasons.append(
                f"gap fidelity {value} exceeds {thresholds.max_gap_fidelity}"
            )

    if thresholds.max_band_placement is not None:
        value = overall.get("band_placement")
        if value is None or value > thresholds.max_band_placement:
            reasons.append(
                f"band placement {value} exceeds {thresholds.max_band_placement}"
            )

    if thresholds.require_behavioural:
        behavioural = results.get("behavioural")
        if behavioural is None:
            reasons.append("behavioural suite did not run")
        elif not behavioural.get("passed"):
            failures = behavioural.get("failures", [])
            reasons.append(f"{len(failures)} behavioural assertion(s) failed")

    if baseline and thresholds.max_stratum_regression is not None:
        for kind, cells in results.get("stratified", {}).items():
            previous = baseline.get("stratified", {}).get(kind, {})
            for label, cell in cells.items():
                before = previous.get(label, {}).get("gap_fidelity")
                after = cell.get("gap_fidelity")
                if before is None or after is None:
                    continue
                if after - before > thresholds.max_stratum_regression:
                    reasons.append(
                        f"{kind}/{label} gap fidelity worsened "
                        f"{before:.4f} -> {after:.4f}"
                    )

    return GateResult(passed=not reasons, reasons=reasons)


HEADERS = (
    ("n_sets", "sets"),
    ("gap_fidelity", "gap"),
    ("band_placement", "band"),
    ("pointwise_mae", "mae"),
    ("top1_agreement", "top1"),
    ("tie_tolerant_tau", "tau"),
)


def _table(rows: dict[str, dict[str, float]], label: str) -> str:
    if not rows:
        return ""
    widths = max(len(label), *(len(name) for name in rows))
    header = f"| {label.ljust(widths)} | " + " | ".join(
        title.rjust(6) for _, title in HEADERS
    ) + " |"
    divider = f"|{'-' * (widths + 2)}|" + "|".join("-" * 8 for _ in HEADERS) + "|"

    lines = [header, divider]
    for name, cell in rows.items():
        values = []
        for key, _ in HEADERS:
            value = cell.get(key)
            if value is None:
                values.append("     -")
            elif key == "n_sets":
                values.append(f"{int(value):>6d}")
            else:
                values.append(f"{value:>6.3f}")
        lines.append(f"| {name.ljust(widths)} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def render(results: dict[str, Any], name: str = "") -> str:
    parts = [f"# Evaluation{': ' + name if name else ''}", ""]
    parts.append(f"Fold: `{results.get('fold', '?')}`")
    parts.append("")
    parts.append(_table({"overall": results.get("overall", {})}, "scope"))

    behavioural = results.get("behavioural")
    if behavioural is not None:
        parts.append("")
        verdict = "PASS" if behavioural.get("passed") else "FAIL"
        parts.append(f"## Behavioural assertions: {verdict}")
        parts.append("")
        for kind, counts in sorted(behavioural.get("summary", {}).items()):
            parts.append(
                f"- `{kind}`: {counts.get('pass', 0)} passed, "
                f"{counts.get('fail', 0)} failed, "
                f"{counts.get('no_response', 0)} no response"
            )
        failures = behavioural.get("failures", [])
        if failures:
            parts.append("")
            parts.append("Failures -- the model answered, and answered wrongly:")
            parts.append("")
            for line in failures[:40]:
                parts.append(f"    {line}")
            if len(failures) > 40:
                parts.append(f"    ... and {len(failures) - 40} more")

        flat = behavioural.get("no_response", [])
        if flat:
            parts.append("")
            parts.append(
                "No measurable response -- reported, not gated. The model barely "
                "moves when these indicators move, which usually means nothing "
                "in the training data varies them on their own:"
            )
            parts.append("")
            for line in flat[:40]:
                parts.append(f"    {line}")
            if len(flat) > 40:
                parts.append(f"    ... and {len(flat) - 40} more")

    by_family = results.get("by_family") or {}
    if by_family:
        parts.append("")
        parts.append("## By indicator family")
        parts.append("")
        parts.append(_table(by_family, "family"))

    for kind, cells in (results.get("stratified") or {}).items():
        parts.append("")
        parts.append(f"## By {kind}")
        parts.append("")
        parts.append(_table(cells, kind))

    return "\n".join(parts) + "\n"


def load(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
