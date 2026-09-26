"""Find EN 15804+A2 facade EPDs on open soda4LCA registry nodes and tabulate what the facade indicators need."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path

NODES = {
    "EPD International": "https://data.environdec.com",
    "IBU": "https://ibudata.lca-data.com",
    "EPD Norge": "https://epdnorway.lca-data.com",
}

#: Name fragments per typology, in the registries' main languages.
TERMS = {
    "sandwich_panel": ["sandwich panel", "sandwich element", "sandwichelement", "sandwichpaneel", "insulated panel"],
    "precast": ["precast concrete sandwich", "precast concrete wall", "precast concrete facade", "precast wall",
                "precast facade", "precast façade", "wall element", "sandwich wall",
                "precast concrete cladding", "fassadenelement", "betonfertigteil"],
    "etics": ["etics", "external thermal insulation composite", "wdvs", "wärmedämmverbundsystem", "eifs"],
    "ventilated": ["ventilated facade", "ventilated façade", "rainscreen", "facade panel", "façade panel",
                   "fassadenplatte", "cladding panel", "cladding board", "cladding", "fibre cement", "fiber cement",
                   "faserzement", "hinterlüftet", "ceramic facade", "ceramic cladding", "terracotta",
                   "hpl", "high pressure laminate", "natural stone cladding"],
    "masonry": ["facing brick", "clay brick", "vormauerziegel", "mauerziegel", "clay block", "masonry"],
    "insulation_layer": ["stone wool", "rock wool", "mineral wool", "glass wool", "expanded polystyrene",
                         "polyisocyanurate", "wood fibre insulation", "steinwolle", "mineralwolle"],
    "substructure": ["substructure", "subframe", "sub-frame", "unterkonstruktion", "facade bracket", "fassadenanker"],
    "render": ["exterior render", "external render", "außenputz", "silicone render", "mineral render"],
}

#: Names that match a facade term but describe another product.
EXCLUDED = (
    "window", "door", "floor", "interior", "cold room", "roof tile", "pipe", "duct", "table", "desk",
    "furniture", "worktop", "topcoat", "partition", "retaining wall", "kryssfiner",
)

FIELDS = [
    "typology", "node", "name", "uuid", "geo", "reg_no", "ref_year", "valid_until", "standard", "pcr",
    "declared_amount", "declared_unit", "mass_kg", "grammage_kg_m2", "thickness_m", "density_kg_m3",
    "gwp_total_a1a3", "gwp_total_c3c4", "gwp_total_d", "wdp_a1a3", "fw_a1a3", "sm_a1a3",
    "mfr_c", "cru_c", "mer_c", "hwd_c", "url",
]

#: Material properties an ILCD+EPD flow may carry, by their declared name.
MATERIAL_PROPERTIES = {"grammage": "grammage_kg_m2", "layer thickness": "thickness_m", "gross density": "density_kg_m3"}

PAGE_SIZE = 200

#: Europe, as ISO country codes and the regional codes registries use; GLO is kept since European
#: manufacturers often declare their market as global.
EUROPE = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT",
    "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE", "GB", "UK", "NO", "CH", "IS", "LI", "TR", "RS", "BA", "MK", "AL",
    "ME", "UA", "MD", "RER", "EU", "EUR", "EU-27", "EU-28", "EU+EFTA", "EU+EFTA+UK", "EUROPE", "GLO",
}

UNIT_NAMES = {"qm": "m2", "Stück": "piece", "pcs": "piece", "pcs.": "piece"}


def fetch_json(url: str, attempts: int = 3) -> dict:
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def search(base: str, term: str) -> list[dict]:
    found, start = [], 0
    while True:
        query = urllib.parse.urlencode(
            {"search": "true", "name": term, "pageSize": PAGE_SIZE, "startIndex": start, "format": "JSON"}
        )
        page = fetch_json(f"{base}/resource/processes?{query}")
        found += page.get("data", [])
        start += PAGE_SIZE
        if start >= page.get("totalCount", 0):
            return found


def european(geo: str) -> bool:
    codes = re.split(r"[\s,;/]+", geo.upper().strip()) if geo else []
    return not codes or any(code in EUROPE or code.split("-")[0] in EUROPE for code in codes)


def matches(term: str, name: str) -> bool:
    lowered = name.lower()
    if any(word in lowered for word in EXCLUDED):
        return False
    return re.search(rf"(?<!\w){re.escape(term)}(?:s|es)?(?!\w)", lowered) is not None


def _text(value) -> str:
    if isinstance(value, list):
        english = [v.get("value", "") for v in value if v.get("lang") in ("en", None)]
        return (english or [v.get("value", "") for v in value] or [""])[0]
    return str(value or "")


def _modules(entry: dict) -> dict[str, float]:
    out = {}
    for item in (entry.get("other") or {}).get("anies", []):
        module, value = item.get("module"), item.get("value")
        if module is None or not isinstance(value, str):
            continue
        try:
            out[module] = float(value)
        except ValueError:
            continue
    return out


def _sum(modules: dict[str, float], names: tuple[str, ...]) -> float | None:
    present = [modules[n] for n in names if n in modules]
    return sum(present) if present else None


def material_properties(flow: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for block in ((flow.get("flowInformation") or {}).get("dataSetInformation") or {}).get("other", {}).get("anies", []):
        if not isinstance(block, dict) or "Material" not in block:
            continue
        names = {
            detail.get("id"): str((detail.get("Name") or {}).get("value", "")).lower()
            for detail in (block.get("Metadata") or {}).get("PropertyDetails", [])
        }
        for material in block["Material"]:
            for datum in (material.get("BulkDetails") or {}).get("PropertyData", []):
                field = MATERIAL_PROPERTIES.get(names.get(datum.get("property"), ""))
                try:
                    value = float((datum.get("Data") or {}).get("value"))
                except (TypeError, ValueError):
                    continue
                if field:
                    out[field] = value
    return out


def details(base: str, uuid: str) -> dict:
    data = fetch_json(f"{base}/resource/processes/{uuid}?format=JSON&view=extended")
    exchanges = (data.get("exchanges") or {}).get("exchange", [])
    reference = next((e for e in exchanges if e.get("referenceFlow")), {})
    properties = reference.get("flowProperties", [])
    declared = next((p for p in properties if p.get("referenceFlowProperty")), {})
    declared_mass = next(
        (p.get("meanValue") for p in properties if _text(p.get("name")).lower() in ("mass", "masse")), None
    )

    results = {
        _text(r["referenceToLCIAMethodDataSet"].get("shortDescription")).lower(): _modules(r)
        for r in (data.get("LCIAResults") or {}).get("LCIAResult", [])
    }
    flows = {_text(e["referenceToFlowDataSet"].get("shortDescription")).lower(): _modules(e) for e in exchanges}

    def pick(table: dict, *fragments: str) -> dict[str, float]:
        for fragment in fragments:
            for name, modules in table.items():
                if fragment in name:
                    return modules
        return {}

    gwp = pick(results, "gwp-total", "(gwp)", "global warming potential total", "climate change - total", "global warming")
    wdp = pick(results, "(wdp)", "water (user) deprivation", "water deprivation")
    method = (data.get("modellingAndValidation") or {}).get("LCIMethodAndAllocation") or {}

    flow_ref = (reference.get("referenceToFlowDataSet") or {}).get("refObjectId")
    material = material_properties(fetch_json(f"{base}/resource/flows/{flow_ref}?format=JSON&view=extended")) if flow_ref else {}

    amount = declared.get("meanValue", reference.get("meanAmount"))
    unit = UNIT_NAMES.get(declared.get("referenceUnit", ""), declared.get("referenceUnit", ""))
    return {
        "pcr": "; ".join(_text(r.get("shortDescription")) for r in method.get("referenceToLCAMethodDetails", [])),
        "declared_amount": amount,
        "declared_unit": unit,
        "mass_kg": declared_mass if declared_mass is not None else mass_from(unit, amount, material),
        **material,
        "gwp_total_a1a3": _sum(gwp, ("A1-A3",)) if "A1-A3" in gwp else _sum(gwp, ("A1", "A2", "A3")),
        "gwp_total_c3c4": _sum(gwp, ("C3", "C4")),
        "gwp_total_d": gwp.get("D"),
        "wdp_a1a3": wdp.get("A1-A3"),
        "fw_a1a3": pick(flows, "(fw)").get("A1-A3"),
        "sm_a1a3": pick(flows, "(sm)").get("A1-A3"),
        "mfr_c": _sum(pick(flows, "(mfr)"), ("C3", "C4")),
        "cru_c": _sum(pick(flows, "(cru)"), ("C3", "C4")),
        "mer_c": _sum(pick(flows, "(mer)"), ("C3", "C4")),
        "hwd_c": _sum(pick(flows, "(hwd)"), ("C3", "C4")),
    }


def mass_from(unit: str, amount, material: dict[str, float]) -> float | None:
    """Mass per declared unit from the material properties, where no mass is declared."""
    if amount is None:
        return None
    if unit == "kg":
        return amount
    if unit == "t":
        return amount * 1000
    if unit == "m2" and "grammage_kg_m2" in material:
        return amount * material["grammage_kg_m2"]
    if unit == "m2" and {"thickness_m", "density_kg_m3"} <= material.keys():
        return amount * material["thickness_m"] * material["density_kg_m3"]
    if unit == "m3" and "density_kg_m3" in material:
        return amount * material["density_kg_m3"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/facade/epd"))
    parser.add_argument("--valid-from", type=int, default=2026, help="drop EPDs expiring before this year")
    args = parser.parse_args()

    candidates: dict[tuple[str, str], dict] = {}
    for node, base in NODES.items():
        for typology, terms in TERMS.items():
            for term in terms:
                for hit in search(base, term):
                    if (hit.get("validUntil") or 9999) < args.valid_from:
                        continue
                    if not matches(term, _text(hit.get("name"))):
                        continue
                    if not any("15804+A2" in c.get("name", "").replace(" ", "") for c in hit.get("compliance", [])):
                        continue
                    if not european(hit.get("geo", "")):
                        continue
                    candidates.setdefault((node, hit["uuid"]), {"typology": typology, "node": node, "hit": hit})
        print(f"{node}: {sum(1 for key in candidates if key[0] == node)} candidates")

    def row(entry: dict) -> dict:
        base, hit = NODES[entry["node"]], entry["hit"]
        record = {
            "typology": entry["typology"],
            "node": entry["node"],
            "name": _text(hit.get("name")).strip(),
            "uuid": hit["uuid"],
            "geo": hit.get("geo", ""),
            "reg_no": hit.get("regNo", ""),
            "ref_year": hit.get("refYear", ""),
            "valid_until": hit.get("validUntil", ""),
            "standard": "; ".join(c.get("name", "") for c in hit.get("compliance", []) if "15804" in c.get("name", "")),
            "url": f"{base}/datasetdetail/process.xhtml?uuid={hit['uuid']}",
        }
        try:
            record.update(details(base, hit["uuid"]))
        except Exception as error:
            record["declared_unit"] = f"unreadable: {error.__class__.__name__}"
        return record

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = sorted(pool.map(row, candidates.values()), key=lambda r: (r["typology"], r["node"], r["name"]))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} EPDs written to {args.out / 'candidates.csv'}")


if __name__ == "__main__":
    main()
