"""Check that a live deployment serves the page and scores sensibly.

    python tests/smoke.py https://example.vercel.app [--commit SHA]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

COMMIT_WAIT_SECONDS = 300
BETTER = 1


def fetch(base: str, path: str, body: dict | None = None) -> tuple[int, str, str]:
    request = urllib.request.Request(
        base.rstrip("/") + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/html,application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read().decode()


def fetch_json(base: str, path: str, body: dict | None = None) -> dict:
    status, _, text = fetch(base, path, body)
    assert status == 200, f"{path} answered {status}"
    return json.loads(text)


def wait_for_commit(base: str, commit: str) -> dict:
    deadline = time.monotonic() + COMMIT_WAIT_SECONDS
    while True:
        health = fetch_json(base, "/api/health")
        if (health.get("commit") or "").startswith(commit):
            return health
        assert time.monotonic() < deadline, (
            f"after {COMMIT_WAIT_SECONDS} s the site still runs {health.get('commit')}, not {commit}"
        )
        time.sleep(10)


def opposed_pair(form: dict) -> dict:
    """Two alternatives equal except for one directed indicator; the one at BETTER should lead."""
    category = form["categories"][0]
    numeric = [f for f in category["fields"] if not f["levels"] and f["range"]]
    field = next(f for f in numeric if f["direction"] != 0)
    middle = {f["key"]: (f["range"]["low"] + f["range"]["high"]) / 2 for f in numeric}
    low, high = field["range"]["low"], field["range"]["high"]
    better, worse = (low, high) if field["direction"] < 0 else (high, low)
    body = {
        "category": category["key"],
        "context": [category["default_context"]],
        "alternatives": [
            {"id": "worse", "values": {**middle, field["key"]: worse}},
            {"id": "better", "values": {**middle, field["key"]: better}},
        ],
    }
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--commit", help="wait until the site reports this commit")
    args = parser.parse_args()

    status, kind, page = fetch(args.url, "/")
    assert status == 200 and kind.startswith("text/html") and 'id="root"' in page, "the page did not load"
    print("page loads")

    health = wait_for_commit(args.url, args.commit) if args.commit else fetch_json(args.url, "/api/health")
    assert health["status"] == "ok" and health["categories"], f"unhealthy: {health}"
    print(f"health ok, registry {health['registry_version']}, commit {health.get('commit')}")

    body = opposed_pair(fetch_json(args.url, "/api/explore/form"))
    results = fetch_json(args.url, "/api/score", body)["results"]
    assert all(0.0 <= r["score"] <= 1.0 for r in results), f"scores out of range: {results}"
    assert results[BETTER]["rank"] == 1, f"the better alternative did not lead: {results}"
    print(f"score ranks the better alternative first: {[(r['id'], r['score']) for r in results]}")

    compared = fetch_json(args.url, "/api/explore/compare", body)
    assert compared["leader_index"] == BETTER, f"compare disagrees: {compared['leader_index']}"
    print("compare agrees")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as failure:
        print(f"SMOKE TEST FAILED: {failure}", file=sys.stderr)
        sys.exit(1)
