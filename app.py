"""Deployment entry point. Vercel routes every request to the top-level ``app`` in ``app.py``.

A rewrite rule in vercel.json would change the request path before the app sees it.
"""

from __future__ import annotations

from serve.api import create_app

app = create_app()
