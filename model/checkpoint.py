"""Save and load checkpoints with their registry, growing them onto a larger one."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from core.registry import Registry, from_blob
from model.recommender import GROWABLE, ModelConfig, Recommender

FORMAT_VERSION = 2


@dataclass
class CheckpointMeta:

    registry_version: str | None
    registry_content_hash: str | None
    snapshot_hash: str | None
    split_key: str | None
    created_at: str
    metrics: dict[str, Any]
    notes: str = ""


def save(
    path: Path | str,
    model: Recommender,
    registry: Registry,
    registry_blob: str,
    meta: CheckpointMeta,
    optimizer_state: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "optimizer_state": optimizer_state,
            "registry_blob": registry_blob,
            "table_sizes": dict(registry.table_sizes),
            "meta": meta.__dict__,
        },
        path,
    )
    return path


def load_registry(path: Path | str) -> Registry:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    meta = payload["meta"]
    return from_blob(
        payload["registry_blob"],
        version=meta.get("registry_version"),
        content_hash=meta.get("registry_content_hash"),
    )


def load(
    path: Path | str,
    registry: Registry | None = None,
    strict_sizes: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[Recommender, CheckpointMeta, dict]:
    """Load a checkpoint, sized for ``registry`` when given, else exactly as saved."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    saved_config = ModelConfig(**payload["config"])
    state = payload["state_dict"]

    if registry is None:
        config = saved_config
    else:
        config = ModelConfig.for_registry(
            registry,
            **{
                name: getattr(saved_config, name)
                for name in (
                    "dim",
                    "encoder_heads",
                    "encoder_blocks",
                    "comparator_heads",
                    "comparator_blocks",
                    "ffn_factor",
                    "dropout",
                    "use_family_prior",
                    "use_context_residual",
                )
            },
        )
        state, report = grow_state_dict(state, saved_config, config)
        if report and strict_sizes:
            raise ValueError(
                "checkpoint was trained under a smaller registry and "
                f"strict_sizes was requested: {report}"
            )

    model = Recommender(config)
    model.load_state_dict(state)
    model.to(device)

    meta = CheckpointMeta(**payload["meta"])
    return model, meta, payload


def grow_state_dict(
    state: dict, old: ModelConfig, new: ModelConfig
) -> tuple[dict, dict[str, tuple[int, int]]]:
    """Widen every embedding table the registry has outgrown, with zeros for the new rows."""
    grown = dict(state)
    report: dict[str, tuple[int, int]] = {}

    for module_path, size_attr in GROWABLE.items():
        key = f"{module_path}.weight"
        if key not in grown:
            continue
        old_rows = getattr(old, size_attr)
        new_rows = getattr(new, size_attr)
        if module_path == "tokens.level":
            # Row zero of the level table stands for "no level".
            old_rows += 1
            new_rows += 1
        if new_rows == old_rows:
            continue
        if new_rows < old_rows:
            raise ValueError(
                f"{module_path} would shrink from {old_rows} to {new_rows} rows; "
                "slots are append-only, so this means the registry is not a "
                "descendant of the one this checkpoint was trained under"
            )

        weight = grown[key]
        widened = torch.zeros(
            (new_rows, weight.shape[1]), dtype=weight.dtype, device=weight.device
        )
        widened[:old_rows] = weight
        grown[key] = widened
        report[module_path] = (old_rows, new_rows)

    return grown, report


def describe(path: Path | str) -> dict:
    """Summarise a checkpoint without building the model."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {
        "format_version": payload.get("format_version"),
        "config": payload["config"],
        "table_sizes": payload.get("table_sizes", {}),
        "meta": payload["meta"],
    }


def now() -> str:
    return datetime.now(timezone.utc).isoformat()
