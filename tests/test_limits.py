"""A public deployment refuses to burn unbounded compute for one caller."""

from __future__ import annotations

import pytest

from serve import limits

pytest.importorskip("fastapi")


def test_calls_up_to_the_limit_are_served():
    limiter = limits.RateLimiter(per_client=3, total=0, seconds=60.0)
    assert [limiter.retry_after("a", now=0.0) for _ in range(3)] == [0.0, 0.0, 0.0]


def test_the_call_after_the_limit_is_refused():
    limiter = limits.RateLimiter(per_client=2, total=0, seconds=60.0)
    limiter.retry_after("a", now=0.0)
    limiter.retry_after("a", now=1.0)
    assert limiter.retry_after("a", now=2.0) == pytest.approx(58.0)


def test_the_window_reopens_once_the_oldest_call_ages_out():
    limiter = limits.RateLimiter(per_client=1, total=0, seconds=60.0)
    limiter.retry_after("a", now=0.0)
    assert limiter.retry_after("a", now=59.0) > 0.0
    assert limiter.retry_after("a", now=60.0) == 0.0


def test_one_caller_exhausting_the_limit_does_not_block_another():
    limiter = limits.RateLimiter(per_client=1, total=0, seconds=60.0)
    limiter.retry_after("a", now=0.0)
    assert limiter.retry_after("a", now=1.0) > 0.0
    assert limiter.retry_after("b", now=1.0) == 0.0


def test_the_total_cap_holds_however_many_callers_there_are():
    limiter = limits.RateLimiter(per_client=10, total=2, seconds=60.0)
    assert limiter.retry_after("a", now=0.0) == 0.0
    assert limiter.retry_after("b", now=0.0) == 0.0
    assert limiter.retry_after("c", now=0.0) > 0.0


def test_a_refused_call_is_not_counted_against_the_caller():
    limiter = limits.RateLimiter(per_client=1, total=0, seconds=10.0)
    limiter.retry_after("a", now=0.0)
    limiter.retry_after("a", now=5.0)
    assert limiter.retry_after("a", now=10.0) == 0.0


def test_tracking_many_callers_does_not_grow_without_bound():
    limiter = limits.RateLimiter(per_client=5, total=0, seconds=60.0, max_clients=4)
    for i in range(50):
        limiter.retry_after(f"client-{i}", now=float(i))
    assert len(limiter._clients) <= 4


def test_zero_on_both_limits_lifts_the_gate():
    assert not limits.RateLimiter(per_client=0, total=0).enabled


def test_the_forwarded_address_identifies_the_caller_behind_a_proxy():
    class Request:
        headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}
        client = None

    assert limits.client_key(Request()) == "203.0.113.7"
