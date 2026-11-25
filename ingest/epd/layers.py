"""EPD layers summed into a product per functional unit, with the shared physical limits and helpers."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

HEALTH_WEIGHTS = {"h0": 0.05, "h1": 0.35, "h2": 0.35, "h3": 0.10, "h4": 0.08, "h5": 0.05, "h6": 0.02}
#: Share of products whose health status is unknown, so labels and the model see missing values often.
HEALTH_MISSING_SHARE = 0.2

FLOWS = ("gwp", "wdp", "fw", "sm", "recovered", "mer", "hwd")


def number(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


@dataclass(frozen=True)
class Layer:
    """One EPD's values per base unit (kg, m2 or m3, by role), and the mass of that unit."""

    role: str
    kind: str
    uuid: str
    name: str
    values: dict[str, float | None]
    mass: float | None
    thickness_m: float | None = None
    density: float | None = None

    def times(self, factor: float) -> dict[str, float | None]:
        return {k: None if v is None else v * factor for k, v in self.values.items()}


@dataclass
class Build:
    """A system being assembled layer by layer, per m2 of wall."""

    values: dict[str, float | None] = field(default_factory=lambda: dict.fromkeys(FLOWS, 0.0))
    mass: float = 0.0
    sources: list[str] = field(default_factory=list)

    def add(self, layer: Layer | None, factor: float, mass: float) -> None:
        self.mass += mass
        if layer is None:
            return
        self.sources.append(layer.uuid)
        for key, value in layer.times(factor).items():
            current = self.values[key]
            self.values[key] = None if current is None or value is None else current + value

    def add_estimate(self, mass: float, **values: float) -> None:
        self.mass += mass
        for key, value in values.items():
            if self.values[key] is not None:
                self.values[key] += value

    def indicators(self) -> dict[str, float | None]:
        v, mass = self.values, self.mass

        def share(amount: float | None) -> float | None:
            return None if amount is None or mass <= 0 else min(100.0, max(0.0, 100.0 * amount / mass))

        recovered, incinerated, hazardous = share(v["recovered"]), share(v["mer"]), share(v["hwd"])
        landfilled = (
            None if None in (recovered, incinerated, hazardous)
            else max(0.0, 100.0 - recovered - incinerated - hazardous)
        )
        return {
            "gwp": v["gwp"],
            "wdp": v["wdp"],
            "fwu": v["fw"],
            "circ_orig": share(v["sm"]),
            "fu_recyc": recovered,
            "fu_incin": incinerated,
            "fu_haz": hazardous,
            "fu_inert": landfilled,
            "surface_mass": mass,
        }


def per_unit(row: dict, divisor: float) -> dict[str, float | None]:
    raw = {
        "gwp": number(row["gwp_total_a1a3"]),
        "wdp": number(row["wdp_a1a3"]),
        "fw": number(row["fw_a1a3"]),
        "sm": number(row["sm_a1a3"]),
        "recovered": None,
        "mer": number(row["mer_c"]),
        "hwd": number(row["hwd_c"]),
    }
    mfr, cru = number(row["mfr_c"]), number(row["cru_c"])
    raw["recovered"] = None if mfr is None and cru is None else (mfr or 0.0) + (cru or 0.0)
    return {k: None if v is None else v / divisor for k, v in raw.items()}


#: Physical limits per kg of product; a value outside them is a unit error, such as litres entered as m3.
#: Mass flows cannot exceed the product's own mass; biogenic storage stays above about -2 kg CO2e per kg.
LIMITS_PER_KG = {
    "gwp": (-2.0, 40.0),
    "wdp": (-5.0, 5.0),
    "fw": (-0.05, 0.05),
    "sm": (0.0, 1.05),
    "recovered": (0.0, 1.05),
    "mer": (0.0, 1.05),
    "hwd": (0.0, 1.05),
}

def weighted_choice(rng: random.Random, weights: dict):
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def mass_law_reduction(surface_mass: float) -> float:
    """Airborne sound reduction of a single leaf by the CTE DB-HR mass law."""
    if surface_mass <= 150:
        return 16.6 * math.log10(surface_mass) + 5
    return 36.5 * math.log10(surface_mass) - 38.5


def health_status(rng: random.Random) -> str | None:
    return None if rng.random() < HEALTH_MISSING_SHARE else weighted_choice(rng, HEALTH_WEIGHTS)
