"""Training configuration.

Configuration is data, not constants in a module: a run is defined by a YAML
file that is committed alongside its result, so "what produced this number" has
an answer that does not involve reading a diff of somebody's editor session.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OptimConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 128
    epochs: int = 40
    warmup_steps: int = 300
    grad_clip: float = 1.0
    patience: int = 6
    min_delta: float = 1e-5
    seed: int = 0
    #: Staging batches in pinned memory only helps when there is a device to
    #: copy them to; on CPU it is pure overhead, so it follows the device.
    pin_memory: bool = True
    num_workers: int = 0


@dataclass
class LossConfig:
    pointwise: float = 1.0
    pairwise: float = 1.0
    listwise: float = 0.0


@dataclass
class ArchConfig:
    dim: int = 96
    encoder_heads: int = 4
    encoder_blocks: int = 2
    comparator_heads: int = 4
    comparator_blocks: int = 2
    ffn_factor: float = 4.0
    dropout: float = 0.1
    use_family_prior: bool = True
    use_context_residual: bool = True


@dataclass
class TrainConfig:
    name: str = "default"
    db: str | None = None
    registry_version: str = "0.1.0"
    snapshot: str | None = None
    split_key: str = "default"
    device: str = "auto"
    output_dir: str = "runs"
    categories: list[str] = field(default_factory=list)
    provenances: list[str] = field(default_factory=list)

    arch: ArchConfig = field(default_factory=ArchConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    notes: str = ""

    @classmethod
    def load(cls, path: Path | str) -> "TrainConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        nested = {
            "arch": ArchConfig(**(raw.get("arch") or {})),
            "optim": OptimConfig(**(raw.get("optim") or {})),
            "loss": LossConfig(**(raw.get("loss") or {})),
        }
        plain = {
            key: value
            for key, value in raw.items()
            if key not in ("arch", "optim", "loss")
        }
        return cls(**plain, **nested)

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )
        return path

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
