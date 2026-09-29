"""Named callers with a rate limit of their own, identified by a bearer key.

    python -m serve.partners <name>

prints a new key for the partner and the entry to add to RECOMMENDER_PARTNER_KEYS.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import secrets

KEYS_ENV = "RECOMMENDER_PARTNER_KEYS"
NAME = re.compile(r"[a-z0-9_-]+")
DIGEST = re.compile(r"[0-9a-f]{64}")


def digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def parse(raw: str) -> dict[str, str]:
    """Partner name by key digest, from comma-separated ``name:sha256`` entries."""
    partners = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        name, _, hexdigest = entry.partition(":")
        if not NAME.fullmatch(name) or not DIGEST.fullmatch(hexdigest):
            raise ValueError(f"{KEYS_ENV} entry {entry!r} is not name:sha256-hex")
        partners[hexdigest] = name
    return partners


def from_env() -> dict[str, str]:
    return parse(os.environ.get(KEYS_ENV, ""))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Issue a key to a named partner.")
    parser.add_argument("name", help="lower case letters, digits, - and _")
    name = parser.parse_args(argv).name
    if not NAME.fullmatch(name):
        parser.error(f"{name!r} is not a valid partner name")

    key = secrets.token_urlsafe(32)
    print(f"key, for {name} only:      {key}")
    print(f"entry for {KEYS_ENV}:  {name}:{digest(key)}")


if __name__ == "__main__":
    main()
