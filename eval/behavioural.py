"""Behavioural assertions: the hard gate.

Metrics tell you how close the model is on average. These tell you whether it
learned the right thing at all, and they are pass or fail rather than a number
to watch drift.

Every assertion is generated from the registry, so they cover whatever the
registry currently declares. Add an indicator and it gets swept; add a context
and the sign-flip checks extend to it. Nothing here is written per category.

Three kinds:

* **Monotonicity** -- holding everything else ideal, does the score move the way
  the registry says this indicator should?
* **Sign flip** -- when two contexts declare opposite directions over the same
  indicator, does the response actually invert? This is the sharpest available
  evidence that the model learned context rather than memorised a direction.
* **Disqualification** -- does a level the registry marks as never-selectable
  actually score below the alternatives around it?

Each assertion returns one of three verdicts, not two. **Only a wrong answer
fails the gate.** An indicator the model barely responds to is reported
separately as ``no_response``, because that is a coverage problem rather than a
correctness one -- almost always it means nothing in the training data ever
varied that indicator on its own -- and folding it into the failures would hide
one problem inside another that needs a completely different fix.

Indicators the registry marks as not sweepable are not asserted on at all. If a
declared direction does not hold across the whole declared range, there is
nothing to compare the model against, and inventing an expectation would be
worse than admitting the gap.

This tier is what makes the system credible in production rather than only in a
paper.
"""

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

#: Rank correlation a monotone response must reach.
MONOTONE_THRESHOLD = 0.98

#: How far the score must actually move across a sweep before the *direction* of
#: the response means anything at all. Below this the model is not saying
#: anything about the indicator, and rank correlation would turn numerical noise
#: into a confident verdict either way.
MIN_RESPONSE = 0.01

#: The three things an assertion can conclude.
PASS = "pass"
FAIL = "fail"
NO_RESPONSE = "no_response"


@dataclass
class Assertion:
    """One statement about the model's behaviour, with three possible verdicts.

    ``no_response`` is deliberately not a failure. It says the model barely
    moves when this indicator moves, which is a different problem from moving
    the wrong way -- usually it means nothing in the training data ever isolated
    the indicator. Reporting it as a failure would hide a coverage gap inside a
    correctness signal, and the two need different fixes.
    """

    kind: str
    category: str
    indicator: str
    context: str
    stakeholder: str
    statement: str
    observed: float
    threshold: float
    verdict: str
    #: How far the score actually moved across the sweep.
    response: float = 0.0
    threshold_response: float = MIN_RESPONSE

    @property
    def passed(self) -> bool:
        return self.verdict == PASS

    @property
    def failed(self) -> bool:
        return self.verdict == FAIL

    def __str__(self) -> str:
        label = {PASS: "PASS", FAIL: "FAIL", NO_RESPONSE: "FLAT"}[self.verdict]
        if self.verdict == NO_RESPONSE:
            return (
                f"{label}  {self.kind:14s} {self.category}/{self.indicator} "
                f"[{self.context}|{self.stakeholder}] no measurable response "
                f"(score moved {self.response:.4f} across the sweep, "
                f"needs {self.threshold_response:.4f})"
            )
        return (
            f"{label}  {self.kind:14s} {self.category}/{self.indicator} "
            f"[{self.context}|{self.stakeholder}] {self.statement} "
            f"(observed {self.observed:+.3f}, threshold {self.threshold:+.3f})"
        )


@dataclass
class Suite:
    assertions: list[Assertion] = field(default_factory=list)

    @property
    def failures(self) -> list[Assertion]:
        return [a for a in self.assertions if a.failed]

    @property
    def flat(self) -> list[Assertion]:
        """Assertions that could not be judged because the model did not move."""
        return [a for a in self.assertions if a.verdict == NO_RESPONSE]

    @property
    def passed(self) -> bool:
        """Only a wrong answer fails the gate. An unanswered one is reported."""
        return not self.failures

    def summary(self) -> dict[str, dict[str, int]]:
        """Count assertions per kind, keyed by the verdict constants themselves.

        Keyed on ``PASS``/``FAIL``/``NO_RESPONSE`` rather than on hand-written
        words, so a verdict can never be counted under a bucket that does not
        exist.
        """
        counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {PASS: 0, FAIL: 0, NO_RESPONSE: 0}
        )
        for assertion in self.assertions:
            counts[assertion.kind][assertion.verdict] += 1
        return dict(counts)


# ---------------------------------------------------------------------------
# Running probes through the model
# ---------------------------------------------------------------------------


@torch.no_grad()
def score_case(
    model,
    registry: Registry,
    case: parametric.ProbeCase,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Score one generated case, using the same encoder serving uses."""
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


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation, without pulling in a stats dependency for one formula."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = np.sqrt((rank_a**2).sum() * (rank_b**2).sum())
    return float((rank_a * rank_b).sum() / denominator) if denominator else 0.0


# ---------------------------------------------------------------------------
# The assertions
# ---------------------------------------------------------------------------


def monotonicity_assertions(
    model,
    registry: Registry,
    category_key: str,
    stakeholder_key: str,
    steps: int = DEFAULT_STEPS,
    threshold: float = MONOTONE_THRESHOLD,
    device: str | torch.device = "cpu",
) -> list[Assertion]:
    """Sweep each indicator the registry declares safe to sweep, and check the
    response follows the declared direction.

    ``parametric.varied_indicators`` already drops indicators the registry marks
    as not sweepable, so an indicator whose direction does not hold across its
    range is never asserted on. That is the point: there would be nothing to
    compare the model against.
    """
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

            # Establish that the model moved before judging which way it moved.
            # Rank correlation over a flat response is noise wearing a verdict.
            response = float(np.max(scores) - np.min(scores))
            observed = _spearman(case.quality, scores)

            if response < MIN_RESPONSE:
                verdict = NO_RESPONSE
            elif observed >= threshold:
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
                    observed=observed,
                    threshold=threshold,
                    verdict=verdict,
                    response=response,
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
    """Find indicators two contexts pull opposite ways, and check they invert.

    The pairs are discovered from the declarations rather than listed, so this
    grows with the registry.
    """
    out: list[Assertion] = []
    category = registry.category(category_key)
    contexts = sorted(category.available_contexts)

    for indicator_key in category.token_order:
        if registry.indicator(indicator_key).is_derived:
            continue
        if not category.members[indicator_key].is_sweepable:
            continue

        # Both contexts must actually *declare* over the indicator. Setting a
        # declared direction against an indicator's undeclared default is not a
        # context inversion: the second context says nothing about it, so there
        # is no claim to test and no reason to expect a response.
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
                # Slope of the response against the raw value, not against
                # quality: quality already has the direction folded in, which
                # would make the flip untestable.
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
