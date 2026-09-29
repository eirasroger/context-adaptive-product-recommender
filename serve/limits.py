"""Per-process rate limits, per caller and in total; each serverless instance keeps its own.

A caller with a partner key is counted against its own limit, apart from the public one.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict, deque
from threading import Lock
from typing import Mapping

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from serve import partners

WINDOW_SECONDS = 60.0
PER_CLIENT = 10
TOTAL = 40
PER_PARTNER = 60
MAX_TRACKED_CLIENTS = 256
MIN_ALTERNATIVES = 2
MAX_ALTERNATIVES = 5


class Window:

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
    """Build the public limiter from the environment; zero lifts a limit."""
    return RateLimiter(
        per_client=_env_int("RECOMMENDER_RATE_LIMIT", PER_CLIENT),
        total=_env_int("RECOMMENDER_RATE_LIMIT_TOTAL", TOTAL),
        seconds=_env_float("RECOMMENDER_RATE_WINDOW", WINDOW_SECONDS),
    )


def partner_limiter_from_env() -> RateLimiter:
    return RateLimiter(
        per_client=_env_int("RECOMMENDER_PARTNER_RATE_LIMIT", PER_PARTNER),
        total=0,
        seconds=_env_float("RECOMMENDER_RATE_WINDOW", WINDOW_SECONDS),
    )


def client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


bearer = HTTPBearer(
    auto_error=False,
    description="A partner key. Calls without one share the public limit.",
)

PUBLIC_REFUSAL = (
    "This deployment serves a limited number of scoring calls per minute. "
    "Run it yourself for unmetered use; the repository carries the model."
)
PARTNER_REFUSAL = "This key has used its scoring calls for the current minute."


def gate(public: RateLimiter, partner: RateLimiter, partner_by_digest: Mapping[str, str]):

    def dependency(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Security(bearer),
    ) -> None:
        if credentials is None:
            _admit(public, client_key(request), PUBLIC_REFUSAL)
            return
        name = partner_by_digest.get(partners.digest(credentials.credentials))
        if name is None:
            raise HTTPException(
                status_code=401,
                detail="unrecognised key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        _admit(partner, name, PARTNER_REFUSAL)

    return dependency


def _admit(limiter: RateLimiter, caller: str, refusal: str) -> None:
    if not limiter.enabled:
        return
    wait = limiter.retry_after(caller)
    if wait > 0.0:
        raise HTTPException(
            status_code=429,
            detail=refusal,
            headers={"Retry-After": str(max(1, int(wait) + 1))},
        )
