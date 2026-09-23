"""Evaluate a checkpoint under the registry it carries, optionally against a baseline run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.prepare import load_or_prepare
from eval import behavioural, report as report_module
from model import checkpoint as checkpoint_module
from snapshot.build import SNAPSHOT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--snapshot", default=None, help="snapshot hash to score on")
    parser.add_argument("--fold", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--baseline", default=None, help="a previous metrics.json")
    parser.add_argument("--out", default=None, help="where to write metrics.json")
    parser.add_argument("--max-gap-fidelity", type=float, default=None)
    parser.add_argument("--max-band-placement", type=float, default=None)
    parser.add_argument("--max-stratum-regression", type=float, default=None)
    parser.add_argument("--skip-behavioural", action="store_true")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    model, meta, _ = checkpoint_module.load(path, device=args.device)
    registry = checkpoint_module.load_registry(path)

    snapshot_hash = args.snapshot or meta.snapshot_hash
    if snapshot_hash is None:
        raise SystemExit(
            "this checkpoint records no snapshot; pass --snapshot to say what to "
            "evaluate it on"
        )
    snapshot_dir = SNAPSHOT_ROOT / snapshot_hash
    if not snapshot_dir.exists():
        raise SystemExit(f"no snapshot at {snapshot_dir}")

    prepared = load_or_prepare(registry, snapshot_dir)
    results = report_module.evaluate_all(
        model, registry, prepared, device=args.device, fold=args.fold
    )

    if not args.skip_behavioural:
        suite = behavioural.run(model, registry, device=args.device)
        results["behavioural"] = {
            "passed": suite.passed,
            "summary": suite.summary(),
            "failures": [str(assertion) for assertion in suite.failures],
            "no_response": [str(assertion) for assertion in suite.flat],
        }

    print(report_module.render(results, path.parent.name))

    thresholds = report_module.Thresholds(
        max_gap_fidelity=args.max_gap_fidelity,
        max_band_placement=args.max_band_placement,
        max_stratum_regression=args.max_stratum_regression,
        require_behavioural=not args.skip_behavioural,
    )
    baseline = report_module.load(args.baseline) if args.baseline else None
    verdict = report_module.gate(results, thresholds, baseline)
    print(verdict)

    if args.out:
        Path(args.out).write_text(
            json.dumps(results, indent=2, sort_keys=True, default=float),
            encoding="utf-8",
        )

    if not verdict.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
