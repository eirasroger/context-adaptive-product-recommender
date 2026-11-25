"""Find EN 15804+A2 EPDs for indoor partition layers and systems on open soda4LCA registry nodes, across European languages."""

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
    "Oekobaudat": "https://oekobaudat.de/OEKOBAU.DAT",
    "EPD Italy": "https://epditaly.lca-data.com",
    "ITB": "https://itb.lca-data.com",
    "Kiwa": "https://kiwa.lca-data.com",
    "GlobalEPD": "https://globalepd.lca-data.com",
    "EPD Ireland": "https://epdireland.lca-data.com",
}

#: Name fragments per layer, in English, German, French, Italian, Spanish, Polish, Dutch and the Nordic languages.
#: Order matters: a dataset goes to the first layer whose term it matches, so specific layers come first.
TERMS = {
    "gypsum_fibreboard": ["gypsum fibre board", "gypsum fiber board", "gypsum fibreboard", "gypsum fiberboard",
                          "gipsfaserplatte", "fermacell", "plaque de plâtre fibrée", "lastra in gessofibra"],
    "gypsum_block": ["gypsum block", "plaster block", "gipswandbauplatte", "gips-wandbauplatte", "carreau de plâtre",
                     "carreaux de plâtre", "gipsblok", "gipsblokken", "blocco in gesso"],
    "plasterboard": ["plasterboard", "gypsum board", "gypsum plasterboard", "gypsum wallboard", "drywall board",
                     "gipsplatte", "gipskartonplatte", "gipskarton", "plaque de plâtre", "plaques de plâtre",
                     "placa de yeso", "placa de yeso laminado", "cartongesso", "lastra in gesso", "lastra di gesso",
                     "płyta gipsowo-kartonowa", "płyta gipsowa", "gipsplaat", "gipsskiva", "gipsplade", "gipsplate",
                     "gyproc", "plaster board"],
    "mineral_wool": ["glass wool", "glasswool", "stone wool", "rock wool", "rockwool", "mineral wool", "glaswolle",
                     "steinwolle", "mineralwolle", "laine de verre", "laine de roche", "laine minérale",
                     "lana di vetro", "lana di roccia", "lana minerale", "lana mineral", "lana de vidrio",
                     "lana de roca", "wełna mineralna", "wełna szklana", "wełna skalna", "glaswol", "steenwol",
                     "glasuld", "stenuld", "glassull", "steinull", "mineralull", "mineraalivilla", "lasivilla"],
    "steel_profile": ["drywall profile", "metal profile", "metal profiles", "steel profile", "steel profiles",
                      "metal stud", "steel stud", "cw profile", "uw profile", "light gauge steel",
                      "metallprofil", "metallprofile", "ständerprofil", "trockenbauprofil", "profilé métallique",
                      "profilés métalliques", "fourrure", "profilo metallico", "profili metallici",
                      "perfil metálico", "perfiles metálicos", "profil stalowy", "profile stalowe",
                      "metalen profiel", "stålprofil", "stålregel"],
    "calcium_silicate": ["calcium silicate", "sand-lime", "sand lime", "kalksandstein", "kalksandsteine",
                         "kalkzandsteen", "silikat", "silka", "brique silico-calcaire", "cegła wapienno-piaskowa",
                         "bloczek wapienno-piaskowy"],
    "aerated_concrete": ["aerated concrete", "autoclaved aerated", "aircrete", "porenbeton", "gasbeton", "ytong",
                         "béton cellulaire", "calcestruzzo cellulare", "hormigón celular", "beton komórkowy",
                         "cellenbeton", "lättbetong", "siporex"],
    "clay_unit": ["clay block", "clay blocks", "clay brick", "clay bricks", "hollow brick", "ceramic block",
                  "masonry brick", "hochlochziegel", "mauerziegel", "ziegel", "brique", "briques",
                  "brique de terre cuite", "mattone", "mattoni", "laterizio", "laterizi", "blocco in laterizio",
                  "ladrillo", "ladrillos", "bloque cerámico", "pustak ceramiczny", "cegła", "baksteen",
                  "tegelsten", "murstein", "poroton", "porotherm"],
    "concrete_block": ["concrete block", "concrete blocks", "concrete masonry unit", "masonry unit",
                       "lightweight concrete block", "betonstein", "betonsteine", "leichtbetonstein",
                       "mauerstein", "mauersteine", "bloc béton", "blocs béton", "parpaing", "blocco in calcestruzzo",
                       "bloque de hormigón", "pustak betonowy", "betonblok", "lecablokk", "leca block"],
    "gypsum_plaster": ["gypsum plaster", "plaster", "gipsputz", "maschinenputz", "putz", "enduit plâtre",
                       "plâtre", "intonaco", "intonaco a base di gesso", "yeso", "tynk gipsowy", "tynk",
                       "pleister", "gipspleister", "skim coat"],
    "mortar": ["masonry mortar", "mortar", "mauermörtel", "dünnbettmörtel", "mörtel", "mortier", "malta",
               "mortero", "zaprawa", "metselmortel", "mørtel", "murbruk"],
    "joint_compound": ["joint compound", "jointing compound", "joint filler", "ready mixed joint", "fugenspachtel",
                       "spachtelmasse", "enduit de jointoiement", "enduit joint", "stucco in pasta",
                       "pasta de juntas", "masa szpachlowa", "voegenmiddel", "fugemasse"],
    "resilient_strip": ["acoustic strip", "isolation strip", "resilient strip", "sealing strip", "dichtungsband",
                        "trennwandband", "anschlussdichtung", "bande résiliente", "nastro isolante"],
    "partition_system": ["partition wall", "partition walls", "partition system", "internal wall", "interior wall",
                         "drywall system", "dry wall system", "stud wall", "metal stud partition",
                         "trennwand", "trennwände", "metallständerwand", "innenwand", "cloison", "cloisons",
                         "tabique", "parete divisoria", "pareti divisorie", "ściana działowa", "scheidingswand",
                         "binnenwand", "innervegg", "mellanvägg", "skillevæg"],
}

#: Names that match a partition term but describe another product.
EXCLUDED = (
    "window", "door", "floor", "roof", "pipe", "duct", "furniture", "table", "desk", "worktop", "facade", "façade",
    "fassade", "exterior", "external", "outdoor", "cladding", "ceiling tile", "suspended ceiling", "screed",
    "tile adhesive", "fliesenkleber", "render", "außenputz", "paving", "pavement", "chimney", "sewer", "road",
    "bridge", "precast", "fertigteil", "sandwich", "cold room", "kitchen", "bathroom furniture",
)


#: Second pass, per layer: a dataset is kept when its name holds one of `require` (if given) and none of `drop`.
CURATION: dict[str, dict[str, tuple[str, ...]]] = {
    "all": {"drop": (
        "paint", "färg", "primer", "voranstrich", "glass", "glazed", "glazing", "armering", "reinforced", "lintel",
        "slab", "paving", "massello", "wall tie", "dach", "waterproof", "impermeab", "pasivación", "passivation",
        "estrich", "verkleidungspanel", "rockpanel", "adhesive", "kleber", "adesiv", "isolflex", "basiflex",
        "restauro", "restoration", "consolidation", "ripara", "repair", "wdvs", "composite system", "pipe",
    )},
    "steel_profile": {
        "require": ("drywall", "stud", "cw", "uw", "profilé", "profil", "light gauge", "metal profile",
                    "steel profiles and accessories", "sheet steel"),
        "drop": ("stainless", "hot rolled", "structural", "welded", "cold drawn", "grc", "gfrc", "eaf", "angle",
                 "cable tray", "mining", "tunnel", "schlüter", "stålprofil"),
    },
    "calcium_silicate": {"require": ("kalksandstein", "calcium silicate", "silka", "sand-lime", "sand lime",
                                     "kalkzandsteen", "silico-calcaire", "wapienno"),
                         "drop": ("board",)},
    "partition_system": {"drop": ("frameless", "mullion", "loft", "skillevæg 20cm", "ceramics", "coverings",
                                   "house elements", "ceiling", "dichtungssystem")},
    "mortar": {"drop": ("joint", "juntas", "thin-coat", "screed", "tile", "grout", "fliesen", "concrete element",
                        "anchoring", "injection", "self-levelling", "levelling")},
    "gypsum_plaster": {"drop": ("cementitious adhesives", "acrylic", "thermoelastic", "texture")},
    "mineral_wool": {"drop": ("roof", "facade", "flat", "pipe section", "lamella mat", "technical insulation")},
    "concrete_block": {"drop": ("fence", "interlocking", "kerb", "curb")},
    "clay_unit": {"drop": ("facing", "cara vista", "vormauer", "klinker", "polystyrol", "roof", "tile", "tegola")},
}


def curated(layer: str, name: str) -> bool:
    """Whether a matched dataset describes a layer an indoor partition can use."""
    lowered = name.lower()
    rules = CURATION.get(layer, {})
    if any(word in lowered for word in CURATION["all"]["drop"] + rules.get("drop", ())):
        return False
    required = rules.get("require")
    return not required or any(word in lowered for word in required)


def matches(term: str, name: str) -> bool:
    lowered = name.lower()
    if any(word in lowered for word in EXCLUDED):
        return False
    return re.search(rf"(?<!\w){re.escape(term)}(?:s|es|n|e|i)?(?!\w)", lowered) is not None


def layer_of(name: str) -> str | None:
    """The first layer whose terms the dataset name matches."""
    return next((layer for layer, terms in TERMS.items() if any(matches(t, name) for t in terms)), None)


def _qualifies(hit: dict, valid_from: int) -> bool:
    if (hit.get("validUntil") or 9999) < valid_from:
        return False
    if not any("15804+A2" in c.get("name", "").replace(" ", "") for c in hit.get("compliance", [])):
        return False
    return european(hit.get("geo", ""))


def _safe_search(job: tuple[str, str]) -> tuple[str, list[dict]]:
    node, term = job
    try:
        return node, search(NODES[node], term)
    except Exception:
        return node, []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/partition/epd"))
    parser.add_argument("--valid-from", type=int, default=2026, help="drop EPDs expiring before this year")
    args = parser.parse_args()

    jobs = [(node, term) for node in NODES for terms in TERMS.values() for term in terms]
    candidates: dict[tuple[str, str], dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for node, hits in pool.map(_safe_search, jobs):
            for hit in hits:
                name = text_of(hit.get("name"))
                layer = layer_of(name)
                if layer is None or not _qualifies(hit, args.valid_from):
                    continue
                candidates.setdefault((node, hit["uuid"]), {"typology": layer, "node": node, "hit": hit})
    for node in NODES:
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

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = sorted(pool.map(row, candidates.values()), key=lambda r: (r["typology"], r["node"], r["name"]))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} EPDs written to {args.out / 'candidates.csv'}")
    write_curated(args.out)


def write_curated(out: Path) -> None:
    """Keep the candidates a partition can use, from candidates.csv, into curated.csv."""
    with (out / "candidates.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    kept = [r for r in rows if curated(r["typology"], r["name"])]
    with (out / "curated.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(kept)
    print(f"{len(kept)} of {len(rows)} EPDs kept in {out / 'curated.csv'}")


if __name__ == "__main__":
    main()
