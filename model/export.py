"""Export a checkpoint to the ONNX model the service runs, and score with torch."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from core import scoring
from core.dataset import Batch
from core.encoding import N_CHANNELS, NO_LEVEL
from model import checkpoint as checkpoint_module
from model.recommender import Recommender
from serve import release


class TorchScorer:
    """The `core.scoring.Scorer` interface over a torch model."""

    def __init__(self, model: Recommender, device: str | torch.device = "cpu"):
        self.model = model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        inputs = {
            name: torch.from_numpy(arrays[name]).to(self.device)
            for name in scoring.MODEL_INPUTS
        }
        return self.model(_as_batch(inputs)).cpu().numpy()


class _Positional(torch.nn.Module):
    """The model with its inputs named and in order, as ONNX needs them."""

    def __init__(self, model: Recommender):
        super().__init__()
        self.model = model

    def forward(
        self,
        indicator_slots: torch.Tensor,
        family_slots: torch.Tensor,
        level_slots: torch.Tensor,
        channels: torch.Tensor,
        token_mask: torch.Tensor,
        alternative_mask: torch.Tensor,
        category_slots: torch.Tensor,
        stakeholder_slots: torch.Tensor,
        stakeholder_mask: torch.Tensor,
        context_slots: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs = dict(locals())
        return self.model(_as_batch({name: inputs[name] for name in scoring.MODEL_INPUTS}))


def _as_batch(inputs: Mapping[str, torch.Tensor]) -> Batch:
    return Batch(
        **inputs, pref=None, conf=None, provenance=(), category_keys=(), set_indices=None
    )


def _example() -> tuple[torch.Tensor, ...]:
    """Two of everything: torch.export fixes any dimension that is 1 in the example."""
    batch, alternatives, tokens, stakeholders, contexts = 2, 3, 4, 2, 2
    arrays = {
        "indicator_slots": np.zeros((batch, tokens), dtype=np.int64),
        "family_slots": np.zeros((batch, tokens), dtype=np.int64),
        "level_slots": np.full((batch, alternatives, tokens), NO_LEVEL, dtype=np.int64),
        "channels": np.zeros((batch, alternatives, tokens, N_CHANNELS), dtype=np.float32),
        "token_mask": np.ones((batch, tokens), dtype=bool),
        "alternative_mask": np.ones((batch, alternatives), dtype=bool),
        "category_slots": np.zeros(batch, dtype=np.int64),
        "stakeholder_slots": np.zeros((batch, stakeholders), dtype=np.int64),
        "stakeholder_mask": np.ones((batch, stakeholders), dtype=bool),
        "context_slots": np.zeros((batch, contexts), dtype=np.int64),
        "context_mask": np.ones((batch, contexts), dtype=bool),
    }
    return tuple(torch.from_numpy(arrays[name]) for name in scoring.MODEL_INPUTS)


def _dynamic_shapes() -> tuple[dict, ...]:
    batch = torch.export.Dim("shortlists")
    alternatives = torch.export.Dim("alternatives")
    tokens = torch.export.Dim("tokens")
    stakeholders = torch.export.Dim("stakeholders")
    contexts = torch.export.Dim("contexts")
    axes = {
        "indicator_slots": {0: batch, 1: tokens},
        "family_slots": {0: batch, 1: tokens},
        "level_slots": {0: batch, 1: alternatives, 2: tokens},
        "channels": {0: batch, 1: alternatives, 2: tokens},
        "token_mask": {0: batch, 1: tokens},
        "alternative_mask": {0: batch, 1: alternatives},
        "category_slots": {0: batch},
        "stakeholder_slots": {0: batch, 1: stakeholders},
        "stakeholder_mask": {0: batch, 1: stakeholders},
        "context_slots": {0: batch, 1: contexts},
        "context_mask": {0: batch, 1: contexts},
    }
    return tuple(axes[name] for name in scoring.MODEL_INPUTS)


def export(checkpoint: Path, out: Path) -> Path:
    """Write the ONNX model for a checkpoint, carrying its registry and provenance."""
    import onnx

    checkpoint, out = Path(checkpoint), Path(out)
    model, meta, payload = checkpoint_module.load(checkpoint, device="cpu")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        program = torch.onnx.export(
            _Positional(model.eval()),
            _example(),
            dynamo=True,
            input_names=list(scoring.MODEL_INPUTS),
            output_names=[scoring.SCORES],
            dynamic_shapes=_dynamic_shapes(),
            verbose=False,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(out))

    exported = onnx.load(str(out))
    onnx.helper.set_model_props(
        exported,
        {
            scoring.REGISTRY_BLOB: payload["registry_blob"],
            scoring.REGISTRY_VERSION: meta.registry_version or "",
            scoring.REGISTRY_CONTENT_HASH: meta.registry_content_hash or "",
            scoring.SNAPSHOT_HASH: meta.snapshot_hash or "",
            scoring.SOURCE_SHA256: release.digest(checkpoint),
        },
    )
    onnx.save(exported, str(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path, nargs="?", default=release.RELEASE_DIR / release.MODEL_FILE
    )
    parser.add_argument("out", type=Path, nargs="?", default=None)
    args = parser.parse_args()
    out = args.out or args.checkpoint.with_name(release.SERVED_MODEL_FILE)
    export(args.checkpoint, out)
    print(f"exported {args.checkpoint} -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
