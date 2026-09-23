"""Deployment entry point.

Vercel's Python runtime looks for a top-level `app` in a file named `app.py`,
`index.py`, `server.py`, `main.py` or `asgi.py` at the project root, and routes
every request to it. Keeping the name and the variable is the whole
configuration; no rewrite rules are involved, and adding any would replace the
request path before the application sees it.

The checkpoint travels in serve/release, so no database is involved at request
time. The build command in vercel.json builds the frontend into frontend/dist
before this module is imported, and create_app fails if the build is missing.
"""

from __future__ import annotations

import os

from serve.api import create_app

app = create_app(device=os.environ.get("RECOMMENDER_DEVICE", "cpu"))
