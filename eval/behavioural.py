"""The behavioural gate: assertions generated from the registry."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from core.dataset import collate
from core.encoding import encode_set
from core.registry import Registry
from ingest.generators import parametric

DEFAULT_STEPS = 9

#: Share of decisive pairs allowed in the wrong order: one of about 36 at nine steps.
MAX_INVERSION_RATE = 0.05

#: Score range a sweep must span before its direction is judged.
MIN_RESPONSE = 0.01

#: Label gap below which two sweep points count as tied, such as near-equivalent levels.
TIE_EPSILON = 0.03

PASS = "pass"
FAIL = "fail"
NO_RESPONSE = "no_response"


@dataclass
class Assertion:
    """One statement about the model. ``no_response`` marks a coverage gap and passes."""

    kind: str
    category: str
    indicator: str
    context: str
    stakeholder: str
    statement: str
    observed: float
    threshold: float
    verdict: str
    response: float = 0.0
    threshold_response: float = MIN_RESPONSE
    inversions: int = 0
    decisive: int = 0

    @property
    def passed(self) -> bool:
        return self.verdict == PASS

    @property
    def failed(self) -> bool:
        return self.verdict == FAIL

    def __str__(self) -> str:
        label = {PASS: "PASS", FAIL: "FAIL", NO_RESPONSE: "FLAT"}[self.verdict]
        head = (
            f"{label}  {self.kind:14s} {self.category}/{self.indicator} "
            f"[{self.context}|{self.stakeholder}]"
        )
        if self.verdict == NO_RESPONSE:
            return (
                f"{head} no measurable response: the score moved "
                f"{self.response:.4f} across the sweep, below "
                f"{self.threshold_response:.4f}"
            )
        if self.decisive:
            return (
                f"{head} {self.statement}: {self.inversions} of "
                f"{self.decisive} decisive pairs inverted, allowing "
                f"{int(self.threshold * self.decisive)}"
            )
        return (
            f"{head} {self.statement}: observed {self.observed:+.3f}, "
            f"threshold {self.threshold:+.3f}"
        )


@dataclass
class Suite:
    assertions: list[Assertion] = field(default_factory=list)

    @property
    def failures(self) -> list[Assertion]:
        return [a for a in self.assertions if a.failed]

    @property
    def flat(self) -> list[Assertion]:
        """Assertions the model did not move enough to judge."""
        return [a for a in self.assertions if a.verdict == NO_RESPONSE]

    @property
    def passed(self) -> bool:
        """Only a wrong answer fails the gate."""
        return not self.failures

    def summary(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {PASS: 0, FAIL: 0, NO_RESPONSE: 0}
        )
        for assertion in self.assertions:
            counts[assertion.kind][assertion.verdict] += 1
        return dict(counts)


@torch.no_grad()
def score_case(
    model,
    registry: Registry,
    case: parametric.ProbeCase,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Score one generated case through the serving encoder."""
    encoded = encode_set(
        registry,
        case.category_key,
        case.alternatives,
        case.context_keys,
        [case.stakeholder_key],
    )
    item = {
        "index": 0,
        "channels": encoded.channels,
        "level_slots": encoded.level_slots,
        "indicator_slots": encoded.indicator_slots[0],
        "family_slots": encoded.family_slots[0],
        "category_slot": encoded.category_slot,
        "category_key": case.category_key,
        "stakeholder_slots": encoded.stakeholder_slots,
        "context_slots": encoded.context_slots,
        "pref": np.full(len(case.alternatives), np.nan, dtype=np.float32),
        "conf": np.full(len(case.alternatives), np.nan, dtype=np.float32),
        "provenance": "control",
    }
    model.eval()
    batch = collate([item]).to(device)
    return model(batch)[0, : len(case.alternatives)].cpu().numpy()


def count_inversions(
    expected: Sequence[float],
    observed: Sequence[float],
    epsilon: float = TIE_EPSILON,
) -> tuple[int, int]:
    """Inverted and decisive pair counts; pairs expected within ``epsilon`` are skipped."""
    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if len(expected) < 2:
        return 0, 0

    rows, cols = np.triu_indices(len(expected), k=1)
    expected_gaps = expected[rows] - expected[cols]
    observed_gaps = observed[rows] - observed[cols]

    decisive = np.abs(expected_gaps) >= epsilon
    if not decisive.any():
        return 0, 0

    disagree = np.sign(expected_gaps[decisive]) != np.sign(observed_gaps[decisive])
    return int(disagree.sum()), int(decisive.sum())


def monotonicity_assertions(
    model,
    registry: Registry,
    category_key: str,
    stakeholder_key: str,
    steps: int = DEFAULT_STEPS,
    threshold: float = MAX_INVERSION_RATE,
    device: str | torch.device = "cpu",
) -> list[Assertion]:
    """Sweep each sweepable indicator and check the score follows its declared direction."""
    out: list[Assertion] = []
    category = registry.category(category_key)

    for context_key in sorted(category.available_contexts):
        for indicator_key in parametric.varied_indicators(
            registry, category_key, [context_key]
        ):
            values = parametric.sweep_values(
                registry, category_key, indicator_key, steps
            )
            case = parametric.make_case(
                registry,
                category_key,
                indicator_key,
                values,
                [context_key],
                stakeholder_key,
            )
            scores = score_case(model, registry, case, device)

            # A flat response has no direction to judge.
            response = float(np.max(scores) - np.min(scores))
            inversions, decisive = count_inversions(case.prefs, scores)

            if response < MIN_RESPONSE or decisive == 0:
                verdict = NO_RESPONSE
            elif inversions <= threshold * decisive:
                verdict = PASS
            else:
                verdict = FAIL

            out.append(
                Assertion(
                    kind="monotonicity",
                    category=category_key,
                    indicator=indicator_key,
                    context=context_key,
                    stakeholder=stakeholder_key,
                    statement="score follows the declared direction",
                    observed=1.0 - (inversions / decisive if decisive else 0.0),
                    threshold=threshold,
                    verdict=verdict,
                    response=response,
                    inversions=inversions,
                    decisive=decisive,
                )
            )
    return out


def sign_flip_assertions(
    model,
    registry: Registry,
    category_key: str,
    stakeholder_key: str,
    steps: int = DEFAULT_STEPS,
    device: str | torch.device = "cpu",
) -> list[Assertion]:
    """Find indicators two contexts pull opposite ways, and check the response inverts."""
    out: list[Assertion] = []
    category = registry.category(category_key)
    contexts = sorted(category.available_contexts)

    for indicator_key in category.token_order:
        if registry.indicator(indicator_key).is_derived:
            continue
        if not category.members[indicator_key].is_sweepable:
            continue

        # Only declared directions count; an indicator's default makes no claim about a context.
        directions: dict[str, float] = {}
        for context_key in contexts:
            declaration = (
                registry.contexts[context_key]
                .for_category(category_key)
                .get(indicator_key)
            )
            if declaration is not None and declaration.direction != 0:
                directions[context_key] = float(declaration.direction)

        opposed = [
            (a, b)
            for index, a in enumerate(directions)
            for b in list(directions)[index + 1 :]
            if directions[a] * directions[b] < 0
        ]
        if not opposed:
            continue

        values = parametric.sweep_values(registry, category_key, indicator_key, steps)
        for context_a, context_b in opposed:
            responses = []
            for context_key in (context_a, context_b):
                case = parametric.make_case(
                    registry,
                    category_key,
                    indicator_key,
                    values,
                    [context_key],
                    stakeholder_key,
                )
                scores = score_case(model, registry, case, device)
                # Against the raw value: quality already folds in the direction.
                responses.append(float(scores[-1] - scores[0]))

            magnitude = min(abs(responses[0]), abs(responses[1]))
            inverted = responses[0] * responses[1] < 0

            if magnitude < MIN_RESPONSE:
                verdict = NO_RESPONSE
            elif inverted:
                verdict = PASS
            else:
                verdict = FAIL

            out.append(
                Assertion(
                    kind="sign_flip",
                    category=category_key,
                    indicator=indicator_key,
                    context=f"{context_a} vs {context_b}",
                    stakeholder=stakeholder_key,
                    statement="response inverts between opposed contexts",
                    observed=magnitude if inverted else -magnitude,
                    threshold=MIN_RESPONSE,
                    verdict=verdict,
                    response=magnitude,
                )
            )
    return out


def disqualification_assertions(
    model,
    registry: Registry,
    category_key: str,
    stakeholder_key: str,
    device: str | torch.device = "cpu",
) -> list[Assertion]:
    """A level the registry says is never selectable must score lowest."""
    out: list[Assertion] = []
    category = registry.category(category_key)
    context_key = category.default_context_key

    for indicator_key in category.token_order:
        indicator = registry.indicator(indicator_key)
        disqualifying = [level for level in indicator.levels if level.is_disqualifying]
        if not disqualifying:
            continue

        values = [level.key for level in indicator.levels]
        case = parametric.make_case(
            registry,
            category_key,
            indicator_key,
            values,
            [context_key],
            stakeholder_key,
        )
        scores = score_case(model, registry, case, device)

        for level in disqualifying:
            position = values.index(level.key)
            others = np.delete(scores, position)
            margin = float(others.min() - scores[position])
            out.append(
                Assertion(
                    kind="disqualification",
                    category=category_key,
                    indicator=indicator_key,
                    context=context_key,
                    stakeholder=stakeholder_key,
                    statement=f"level {level.key} scores below every other level",
                    observed=margin,
                    threshold=0.0,
                    verdict=PASS if margin > 0.0 else FAIL,
                    response=float(np.max(scores) - np.min(scores)),
                )
            )
    return out


def run(
    model,
    registry: Registry,
    categories: Sequence[str] | None = None,
    stakeholders: Sequence[str] | None = None,
    steps: int = DEFAULT_STEPS,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> Suite:
    """Run every assertion the registry implies."""
    random.seed(seed)
    suite = Suite()
    category_keys = list(categories or registry.categories)
    stakeholder_keys = list(stakeholders or registry.stakeholders)

    for category_key in category_keys:
        for stakeholder_key in stakeholder_keys:
            suite.assertions.extend(
                monotonicity_assertions(
                    model, registry, category_key, stakeholder_key, steps, device=device
                )
            )
            suite.assertions.extend(
                sign_flip_assertions(
                    model, registry, category_key, stakeholder_key, steps, device=device
                )
            )
            suite.assertions.extend(
                disqualification_assertions(
                    model, registry, category_key, stakeholder_key, device=device
                )
            )
    return suite
