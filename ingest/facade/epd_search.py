"""Find EN 15804+A2 facade EPDs on open soda4LCA registry nodes and tabulate what the facade indicators need."""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ingest.epd.soda4lca import FIELDS, details, european, search, text_of

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

def matches(term: str, name: str) -> bool:
    lowered = name.lower()
    if any(word in lowered for word in EXCLUDED):
        return False
    return re.search(rf"(?<!\w){re.escape(term)}(?:s|es)?(?!\w)", lowered) is not None


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
                    if not matches(term, text_of(hit.get("name"))):
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
            "name": text_of(hit.get("name")).strip(),
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
