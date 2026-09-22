"""Replace the registry a checkpoint carries, without retraining.

A checkpoint stores the registry blob it was trained under, so a change to a
display name or a definition never reaches a deployed model on its own. Such a
change touches nothing the model reads. This tool swaps the blob in place and
refuses whenever any other column moved, so the weights keep meaning what they
meant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from core.registry import from_blob

#: Columns a restamp is allowed to change. Everything here is prose or a
#: declaration the API echoes; none of it reaches the model or a label.
PROSE_COLUMNS = frozenset(
    {
        "display_name",
        "definition_text",
        "domain_scope_text",
        "nominal_justification",
        "control_note",
        "eligibility_precondition_text",
        "is_disqualifying",
        "created_at",
        "notes",
        "version",
        "content_hash",
        "yaml_blob",
    }
)


class Drift(Exception):
    pass


def _structural(document: dict) -> dict:
    return {
        table: [
            {k: v for k, v in row.items() if k not in PROSE_COLUMNS} for row in rows
        ]
        for table, rows in document.items()
    }


def drift(before: str, after: str) -> list[str]:
    """Tables whose model-facing columns differ between two registry blobs."""
    a = _structural(yaml.safe_load(before))
    b = _structural(yaml.safe_load(after))
    return sorted(
        set(a) ^ set(b) | {table for table in set(a) & set(b) if a[table] != b[table]}
    )


def restamp(
    checkpoint: Path, blob: str, version: str, content_hash: str
) -> dict[str, str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    moved = drift(payload["registry_blob"], blob)
    if moved:
        raise Drift(
            f"these tables changed something the model reads: {moved}. "
            "Retrain instead of restamping."
        )

    was = dict(payload["meta"])
    payload["registry_blob"] = blob
    payload["table_sizes"] = dict(from_blob(blob).table_sizes)
    payload["meta"]["registry_version"] = version
    payload["meta"]["registry_content_hash"] = content_hash
    torch.save(payload, checkpoint)
    return {
        "from_version": was.get("registry_version") or "",
        "from_hash": was.get("registry_content_hash") or "",
    }


def refresh_manifest(checkpoint: Path, version: str, content_hash: str) -> Path | None:
    """Re-stamp the served release manifest sitting beside a checkpoint."""
    import json

    from serve import release

    path = checkpoint.parent / release.MANIFEST_FILE
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["registry_version"] = version
    manifest["registry_content_hash"] = content_hash
    manifest["model_sha256"] = release.digest(checkpoint)
    manifest["model_bytes"] = checkpoint.stat().st_size
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("version", help="the registry release to stamp in")
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    from db.models import RegistryRelease
    from db.session import create_db_engine, session_scope

    with session_scope(create_db_engine(args.db)) as session:
        release = session.get(RegistryRelease, args.version)
        if release is None:
            raise SystemExit(f"no registry release {args.version}")
        blob, digest = release.yaml_blob, release.content_hash

    was = restamp(args.checkpoint, blob, args.version, digest)
    print(f"{args.checkpoint}: {was['from_version']} -> {args.version}")
    print(f"  {was['from_hash']}\n  {digest}")

    manifest = refresh_manifest(args.checkpoint, args.version, digest)
    if manifest is not None:
        print(f"  refreshed {manifest}")


if __name__ == "__main__":
    main()
