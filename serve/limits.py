"""Caps on how much work the service will do for one caller.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict, deque
from threading import Lock

from fastapi import HTTPException, Request

WINDOW_SECONDS = 60.0
PER_CLIENT = 10
TOTAL = 40
MAX_TRACKED_CLIENTS = 256
MAX_ALTERNATIVES = 5


class Window:
    """A sliding count of recent hits against a limit."""

    def __init__(self, limit: int, seconds: float) -> None:
        self.limit = limit
        self.seconds = seconds
        self.hits: deque[float] = deque()

    def retry_after(self, now: float) -> float:
        while self.hits and now - self.hits[0] >= self.seconds:
            self.hits.popleft()
        if len(self.hits) < self.limit:
            return 0.0
        return self.seconds - (now - self.hits[0])

    def record(self, now: float) -> None:
        self.hits.append(now)


class RateLimiter:
    def __init__(
        self,
        per_client: int = PER_CLIENT,
        total: int = TOTAL,
        seconds: float = WINDOW_SECONDS,
        max_clients: int = MAX_TRACKED_CLIENTS,
    ) -> None:
        self.per_client = per_client
        self.total = total
        self.seconds = seconds
        self.max_clients = max_clients
        self._clients: OrderedDict[str, Window] = OrderedDict()
        self._total = Window(total, seconds) if total > 0 else None
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.per_client > 0 or self.total > 0

    def retry_after(self, client: str, now: float | None = None) -> float:
        """Zero when the call may proceed, else the seconds until it may."""
        now = time.monotonic() if now is None else now
        with self._lock:
            waits = []
            window = None
            if self.per_client > 0:
                window = self._clients.get(client) or Window(
                    self.per_client, self.seconds
                )
                self._clients[client] = window
                self._clients.move_to_end(client)
                waits.append(window.retry_after(now))
            if self._total is not None:
                waits.append(self._total.retry_after(now))

            wait = max(waits) if waits else 0.0
            if wait > 0.0:
                return wait

            if window is not None:
                window.record(now)
            if self._total is not None:
                self._total.record(now)
            while len(self._clients) > self.max_clients:
                self._clients.popitem(last=False)
            return 0.0


def _env_int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    return fallback if raw is None or not raw.strip() else int(raw)


def _env_float(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    return fallback if raw is None or not raw.strip() else float(raw)


def from_env() -> RateLimiter:
    """Build the limiter from the environment. Zero on either limit lifts it."""
    return RateLimiter(
        per_client=_env_int("RECOMMENDER_RATE_LIMIT", PER_CLIENT),
        total=_env_int("RECOMMENDER_RATE_LIMIT_TOTAL", TOTAL),
        seconds=_env_float("RECOMMENDER_RATE_WINDOW", WINDOW_SECONDS),
    )


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def gate(limiter: RateLimiter):
    """A dependency that refuses a caller who is asking for too much compute."""

    def dependency(request: Request) -> None:
        if not limiter.enabled:
            return
        wait = limiter.retry_after(client_key(request))
        if wait > 0.0:
            raise HTTPException(
                status_code=429,
                detail=(
                    "This deployment serves a limited number of scoring calls "
                    "per minute. Run it yourself for unmetered use; the "
                    "repository carries the model."
                ),
                headers={"Retry-After": str(max(1, int(wait) + 1))},
            )

    return dependency
