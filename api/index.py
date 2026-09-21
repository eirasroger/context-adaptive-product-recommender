"""Entry point for a serverless deployment.

Vercel looks for an ASGI application named ``app`` under ``api/``. Everything it
needs travels in the repository: the checkpoint and the SHAP background sample
live in ``serve/release``, so no database and no external storage is involved at
request time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.api import create_app  # noqa: E402

app = create_app(device=os.environ.get("RECOMMENDER_DEVICE", "cpu"))
