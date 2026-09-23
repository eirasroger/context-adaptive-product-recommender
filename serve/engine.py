"""The served model: ONNX Runtime, and the registry the model file carries."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import onnxruntime

from core import scoring
from core.registry import Registry, from_blob


class OnnxScorer:
    """The `core.scoring.Scorer` interface over an exported model."""

    def __init__(self, session: onnxruntime.InferenceSession):
        self.session = session

    def __call__(self, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
        feeds = {name: arrays[name] for name in scoring.MODEL_INPUTS}
        return self.session.run([scoring.SCORES], feeds)[0]


class ServedModel:
    def __init__(self, path: Path | str):
        session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        carried = session.get_modelmeta().custom_metadata_map
        self.scorer = OnnxScorer(session)
        self.registry_version = carried[scoring.REGISTRY_VERSION] or None
        self.snapshot_hash = carried[scoring.SNAPSHOT_HASH] or None
        self.source_sha256 = carried[scoring.SOURCE_SHA256]
        self.registry: Registry = from_blob(
            carried[scoring.REGISTRY_BLOB],
            version=self.registry_version,
            content_hash=carried[scoring.REGISTRY_CONTENT_HASH] or None,
        )
