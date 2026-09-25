from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from model import checkpoint as checkpoint_module
from model.recommender import ModelConfig, Recommender
from serve import release as release_module


@pytest.fixture(autouse=True)
def quick_export(request, monkeypatch):
    """Stands in for the ONNX export, which takes seconds and is covered by test_export.py."""
    if request.node.get_closest_marker("real_export"):
        return
    from model import export as export_module

    def stand_in(checkpoint, out):
        Path(out).write_bytes(b"stand-in")
        return Path(out)

    monkeypatch.setattr(export_module, "export", stand_in)


def _run(tmp_path, registry, passed: bool | None, gap: float = 0.05):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    torch.manual_seed(0)
    model = Recommender(
        ModelConfig.for_registry(registry, dim=16, encoder_blocks=1, comparator_blocks=1)
    )
    checkpoint_module.save(
        run_dir / "model.pt",
        model,
        registry,
        registry_blob="indicator: []\n",
        meta=checkpoint_module.CheckpointMeta(
            registry_version="test",
            registry_content_hash=registry.content_hash,
            snapshot_hash="abc123",
            split_key="default",
            created_at=checkpoint_module.now(),
            metrics={},
        ),
    )
    if passed is not None:
        (run_dir / "metrics.json").write_text(
            json.dumps(_metrics(passed, gap)), encoding="utf-8"
        )
    return run_dir


def _root() -> Path:
    """The repository root, for subprocesses that would inherit pytest's working directory."""
    return Path(__file__).resolve().parents[1]


def _snapshot(root, digest="abc123"):
    """A snapshot small enough to read, holding one set in each fold."""
    import pandas as pd

    directory = root / digest
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "set_id": [1, 2], "external_id": ["a", "b"], "category_key": ["c", "c"],
        "provenance_key": ["control", "control"], "generator_key": [None, None],
        "fold": ["train", "test"], "stakeholder_keys": ["s", "s"],
        "context_keys": ["x", "x"],
    }).to_parquet(directory / "sets.parquet", index=False)
    pd.DataFrame({
        "set_id": [1, 2], "position": [0, 0], "product_id": [10, 20],
        "local_key": ["p", "q"], "pref": [0.5, 0.5], "conf": [1.0, 1.0],
        "scale_semantics": ["absolute_reference"] * 2,
    }).to_parquet(directory / "members.parquet", index=False)
    pd.DataFrame({
        "product_id": [10, 20], "indicator_key": ["gwp", "gwp"],
        "present": [1, 1], "value_num": [0.1, 0.2], "level_key": [None, None],
    }).to_parquet(directory / "values.parquet", index=False)
    (directory / "manifest.json").write_text(
        json.dumps({"content_hash": digest, "filter_spec": {}, "row_counts": {}}),
        encoding="utf-8",
    )
    return directory


def _promote(run_dir, tmp_path, **kwargs):
    """Promote with every artefact directory pointed away from the repository."""
    kwargs.setdefault("release_dir", tmp_path / "release")
    kwargs.setdefault("fixture_dir", tmp_path / "fixture")
    kwargs.setdefault("snapshot_root", _snapshot(tmp_path / "snapshots").parent)
    kwargs.setdefault("readme", _readme(tmp_path))
    return release_module.promote(run_dir, **kwargs)


def _readme(tmp_path):
    path = tmp_path / "README.md"
    if not path.exists():
        path.write_text(
            f"# Title\n\n{release_module.RESULTS_START}\nold\n"
            f"{release_module.RESULTS_END}\n\nAfter.\n",
            encoding="utf-8",
        )
    return path


def _metrics(passed: bool = True, gap: float = 0.05) -> dict:
    return {
        "overall": {
            "gap_fidelity": gap, "band_placement": 0.02, "top1_agreement": 0.9,
            "tie_tolerant_tau": 0.9, "n_sets": 1234,
        },
        "stratified": {"context": {"standard": {"gap_fidelity": gap}}},
        "behavioural": {"passed": passed, "failures": [] if passed else ["x"]},
    }


def test_a_failing_run_is_refused(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=False)
    with pytest.raises(release_module.GateFailed, match="behavioural gate"):
        _promote(run_dir, tmp_path)
    assert not (tmp_path / "release" / "model.pt").exists()


def test_an_unevaluated_run_is_refused(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=None)
    with pytest.raises(release_module.GateFailed, match="no evaluation"):
        _promote(run_dir, tmp_path)


def test_forcing_records_that_it_was_forced(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=False)
    manifest = _promote(
        run_dir, tmp_path, force=True, notes="known bad, for a reproduction",
    )
    assert manifest["evaluation"]["behavioural_passed"] is False
    assert manifest["notes"]


def test_a_passing_run_ships_with_its_provenance(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True)
    release_dir = tmp_path / "release"
    manifest = _promote(run_dir, tmp_path, release_dir=release_dir)

    assert (release_dir / "model.pt").exists()
    assert manifest["snapshot_hash"] == "abc123"
    assert manifest["registry_version"] == "test"
    assert len(manifest["model_sha256"]) == 64
    assert manifest["model_sha256"] == release_module.digest(release_dir / "model.pt")

    on_disk = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


@pytest.mark.real_export
def test_a_promotion_ships_the_served_model_exported_from_its_checkpoint(tmp_path, registry):
    import onnxruntime

    from core import scoring

    run_dir = _run(tmp_path, registry, passed=True)
    release_dir = tmp_path / "release"
    _promote(run_dir, tmp_path, release_dir=release_dir)

    served = release_dir / release_module.SERVED_MODEL_FILE
    carried = (
        onnxruntime.InferenceSession(str(served), providers=["CPUExecutionProvider"])
        .get_modelmeta()
        .custom_metadata_map
    )
    assert carried[scoring.SOURCE_SHA256] == release_module.digest(release_dir / "model.pt")


def test_the_released_artefacts_stay_small():
    release_dir = release_module.RELEASE_DIR
    if not (release_dir / "model.pt").exists():
        pytest.skip("nothing has been promoted yet")

    total = sum(f.stat().st_size for f in release_dir.iterdir() if f.is_file())
    assert total < release_module.SIZE_WARNING_MB * 1e6, (
        f"release is {total / 1e6:.1f} MB"
    )


def test_serving_does_not_import_the_data_stack():
    """pandas, pyarrow and SQLAlchemy would add about 175 MB to a size-limited deployment."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, serve.api;"
         "heavy = {'pandas', 'pyarrow', 'sqlalchemy', 'alembic'};"
         "found = sorted(m for m in sys.modules if m.split('.')[0] in heavy);"
         "print(','.join(found))"],
        capture_output=True, text=True, timeout=180, cwd=_root(),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"serving imports {result.stdout.strip()}"


def _modules_serving_imports() -> set[str]:
    """Top-level modules loaded by importing the deployed app, in a fresh process."""
    import subprocess
    import sys

    # Same imports as app.py, without needing a frontend build.
    result = subprocess.run(
        [sys.executable, "-c",
         "import functools, sys, serve.api;"
         "serve.api.create_app = functools.partial(serve.api.create_app, frontend=None);"
         "import app;"
         "print(','.join(sorted({m.split('.')[0] for m in sys.modules})))"],
        capture_output=True, text=True, timeout=180, cwd=_root(),
    )
    assert result.returncode == 0, result.stderr
    return set(result.stdout.strip().split(","))


def test_serving_never_imports_the_explanation_stack():
    """SHAP and its dependencies would add about 330 MB."""
    found = _modules_serving_imports() & {"shap", "numba", "llvmlite", "sklearn", "scipy"}
    assert not found, f"serving imports {sorted(found)}"


def test_serving_never_imports_torch():
    """Torch was 702 MB of an 815 MB deployment."""
    found = _modules_serving_imports() & {"torch", "onnx", "onnxscript"}
    assert not found, f"serving imports {sorted(found)}"


def _requirement_names(path: Path) -> set[str]:
    """Package names a requirements file asks for, ignoring comments and options."""
    import re

    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line and not line.startswith("-"):
            names.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower())
    return names


def test_heavy_packages_are_absent_from_serving_requirements():
    serving = _requirement_names(_root() / "requirements.txt")
    heavy = {"shap", "numba", "llvmlite", "scikit-learn", "scipy", "torch", "onnx", "onnxscript"}
    assert not serving & heavy, f"serving requires {sorted(serving & heavy)}"

    development = _requirement_names(_root() / "requirements-dev.txt")
    assert {"shap", "torch", "onnx", "onnxscript"} <= development


def test_serving_requirements_cover_what_serving_imports():
    """A missing requirement passes the build and crashes on import."""
    declared = _requirement_names(_root() / "requirements.txt")

    required = {
        "onnxruntime": "onnxruntime",
        "numpy": "numpy",
        "fastapi": "fastapi",
        "pydantic": "pydantic",
        "yaml": "pyyaml",
    }
    missing = [
        module for module, package in required.items() if package not in declared
    ]
    assert not missing, f"requirements.txt is missing {missing}"


def test_every_third_party_import_is_declared():
    """A package installed for another reason hides a missing requirement until a clean build."""
    import ast
    import re
    import sys
    from importlib.metadata import packages_distributions

    root = _root()
    first_party = {
        "app", "core", "db", "eval", "experiments", "ingest", "model",
        "registry", "serve", "snapshot", "tests", "train", "wip",
    }

    def _normalise(name: str) -> str:
        return name.strip().lower().replace("_", ".").replace("-", ".")

    declared = set()
    for name in ("requirements.txt", "requirements-dev.txt"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            requirement = line.split("#")[0].strip()
            if not requirement or requirement.startswith("-"):
                continue
            declared.add(_normalise(re.split(r"[\[<>=!~;\s]", requirement)[0]))

    distributions = packages_distributions()

    undeclared: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module.split(".")[0]]
            else:
                continue
            for module in modules:
                if module in sys.stdlib_module_names or module in first_party:
                    continue
                for name in distributions.get(module, [module]):
                    if _normalise(name) not in declared:
                        undeclared[name] = str(path.relative_to(root))

    assert not undeclared, (
        "imported but absent from the requirements files: "
        + ", ".join(f"{name} ({where})" for name, where in sorted(undeclared.items()))
    )


def _released(release_dir, gap: float):
    """A release directory holding only the metrics a promotion compares against."""
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / release_module.METRICS_FILE).write_text(
        json.dumps(_metrics(gap=gap)), encoding="utf-8"
    )
    return release_dir


def test_a_promotion_ships_the_metrics_it_was_gated_on(tmp_path, registry):
    """Without them the next promotion's regression gate compares against nothing."""
    run_dir = _run(tmp_path, registry, passed=True)
    release_dir = tmp_path / "release"

    _promote(run_dir, tmp_path, release_dir=release_dir)

    shipped = json.loads(
        (release_dir / release_module.METRICS_FILE).read_text(encoding="utf-8")
    )
    assert shipped == json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def test_a_run_worse_than_the_shipped_release_is_refused(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True, gap=0.20)
    release_dir = _released(tmp_path / "release", gap=0.05)

    with pytest.raises(release_module.GateFailed, match="worse than the release"):
        _promote(run_dir, tmp_path, release_dir=release_dir)

    assert not (release_dir / "model.pt").exists()


def test_a_run_that_improves_a_stratum_ships(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True, gap=0.02)
    release_dir = _released(tmp_path / "release", gap=0.05)

    _promote(run_dir, tmp_path, release_dir=release_dir)

    assert (release_dir / "model.pt").exists()


def test_the_first_promotion_has_nothing_to_regress_against(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True, gap=0.99)

    _promote(run_dir, tmp_path)

    assert (tmp_path / "release" / "model.pt").exists()


def test_a_regression_can_be_forced_and_says_so(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True, gap=0.20)
    release_dir = _released(tmp_path / "release", gap=0.05)

    manifest = _promote(
        run_dir, tmp_path, release_dir=release_dir, force=True,
        notes="shipped knowingly",
    )

    assert manifest["notes"] == "shipped knowingly"


def test_promoting_refreshes_the_evaluation_fixture(tmp_path, registry):
    import pandas as pd

    run_dir = _run(tmp_path, registry, passed=True)
    fixture = tmp_path / "fixture"

    _promote(run_dir, tmp_path, fixture_dir=fixture)

    manifest = json.loads((fixture / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["derived_from"] == "abc123"
    assert manifest["filter_spec"]["folds"] == ["test"]

    sets = pd.read_parquet(fixture / "sets.parquet")
    assert list(sets["fold"].unique()) == ["test"], "a non-test fold came through"
    assert list(pd.read_parquet(fixture / "members.parquet")["product_id"]) == [20]
    assert list(pd.read_parquet(fixture / "values.parquet")["product_id"]) == [20]


def test_a_refresh_drops_the_cache_the_old_contents_produced(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True)
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    stale = fixture / "deadbeef-prepared.pkl"
    stale.write_bytes(b"old")

    _promote(run_dir, tmp_path, fixture_dir=fixture)

    assert not stale.exists()


def test_a_promotion_that_cannot_refresh_the_fixture_is_refused(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True)

    with pytest.raises(release_module.GateFailed, match="no snapshot at"):
        _promote(run_dir, tmp_path, snapshot_root=tmp_path / "nowhere")

    assert not (tmp_path / "release" / "manifest.json").exists()


def test_promoting_rewrites_the_readme_results(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True, gap=0.0421)
    readme = _readme(tmp_path)

    _promote(run_dir, tmp_path, readme=readme)

    text = readme.read_text(encoding="utf-8")
    assert "| 0.042 |" in text
    assert "1,234 test shortlists" in text
    assert "old" not in text
    assert text.startswith("# Title") and text.endswith("After.\n")


def test_a_readme_without_a_results_block_is_refused(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True)
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n", encoding="utf-8")

    with pytest.raises(release_module.GateFailed, match="results:start"):
        _promote(run_dir, tmp_path, readme=readme)


def test_the_pyproject_declares_no_dependencies():
    """A [project] table would give Vercel's builder a second list of what to install."""
    import tomllib

    root = _root()
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert set(config) == {"tool"}, (
        f"pyproject.toml declares {sorted(set(config) - {'tool'})}; keep "
        "dependencies in requirements.txt, or teach the deployment about them"
    )
    assert "pyproject.toml" in (root / ".vercelignore").read_text(encoding="utf-8")
