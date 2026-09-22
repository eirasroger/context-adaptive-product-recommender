"""What gets shipped, and what refuses to ship."""

from __future__ import annotations

import json

import pytest
import torch

from model import checkpoint as checkpoint_module
from model.recommender import ModelConfig, Recommender
from serve import release as release_module
from serve.background import Background


def _run(tmp_path, registry, passed: bool | None):
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
            json.dumps({
                "overall": {"gap_fidelity": 0.05},
                "behavioural": {"passed": passed, "failures": [] if passed else ["x"]},
            }),
            encoding="utf-8",
        )
    return run_dir


def test_a_failing_run_is_refused(tmp_path, registry):
    """The gate has to bite at the point where a model becomes deployable.

    Somewhere to record that a build failed is worth nothing if the failing
    build ships anyway.
    """
    run_dir = _run(tmp_path, registry, passed=False)
    with pytest.raises(release_module.GateFailed, match="behavioural gate"):
        release_module.promote(run_dir, database=None, release_dir=tmp_path / "release")
    assert not (tmp_path / "release" / "model.pt").exists()


def test_an_unevaluated_run_is_refused(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=None)
    with pytest.raises(release_module.GateFailed, match="no evaluation"):
        release_module.promote(run_dir, database=None, release_dir=tmp_path / "release")


def test_forcing_records_that_it_was_forced(tmp_path, registry):
    """Shipping a failing model stays possible and stays on the record."""
    run_dir = _run(tmp_path, registry, passed=False)
    manifest = release_module.promote(
        run_dir, database=None, release_dir=tmp_path / "release",
        force=True, notes="known bad, for a reproduction",
    )
    assert manifest["evaluation"]["behavioural_passed"] is False
    assert manifest["notes"]


def test_a_passing_run_ships_with_its_provenance(tmp_path, registry):
    run_dir = _run(tmp_path, registry, passed=True)
    release_dir = tmp_path / "release"
    manifest = release_module.promote(run_dir, database=None, release_dir=release_dir)

    assert (release_dir / "model.pt").exists()
    assert manifest["snapshot_hash"] == "abc123"
    assert manifest["registry_version"] == "test"
    assert len(manifest["model_sha256"]) == 64
    assert manifest["model_sha256"] == release_module.digest(release_dir / "model.pt")

    on_disk = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest


def test_background_falls_back_when_nothing_is_attached():
    """A deployment with no database still answers, and says what it used."""
    background = Background({}, "reference ranges")
    assert not background.available
    assert background.size("anything") == 0
    assert background.rows("anything", ["gwp"]) == []


def test_background_round_trips_through_a_file(tmp_path):
    path = tmp_path / "background.json"
    path.write_text(json.dumps({
        "origin": "corpus",
        "categories": {"concrete": [{"gwp": 0.2, "health": "h4", "wdp": None}]},
    }), encoding="utf-8")

    background = Background.from_file(path)
    assert background.origin == "corpus"
    assert background.size("concrete") == 1

    rows = background.rows("concrete", ["gwp", "health"])
    assert rows == [{"gwp": 0.2, "health": "h4"}]


def test_the_released_artefacts_stay_small():
    """A checkpoint belongs in the repository. A corpus does not."""
    from serve.background import RELEASE_DIR

    if not (RELEASE_DIR / "model.pt").exists():
        pytest.skip("nothing has been promoted yet")

    total = sum(f.stat().st_size for f in RELEASE_DIR.iterdir() if f.is_file())
    assert total < release_module.SIZE_WARNING_MB * 1e6, (
        f"release is {total / 1e6:.1f} MB"
    )


def test_serving_does_not_import_the_data_stack():
    """The serving bundle has a size limit, and these are not needed to score.

    pandas, pyarrow and SQLAlchemy belong to building data and training. Letting
    them back into the import path silently adds about 175 MB to a deployment.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, serve.api;"
         "heavy = {'pandas', 'pyarrow', 'sqlalchemy', 'alembic'};"
         "found = sorted(m for m in sys.modules if m.split('.')[0] in heavy);"
         "print(','.join(found))"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"serving imports {result.stdout.strip()}"


def test_serving_never_imports_the_explanation_stack():
    """SHAP is offline analysis for the paper and stays out of the deployment.

    It carries numba, llvmlite, scipy, scikit-learn and pandas behind it, about
    330 MB, which is the difference between a bundle that fits a standard
    function limit and one that does not.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, serve.api, app;"
         "heavy = {'shap', 'numba', 'llvmlite', 'sklearn', 'scipy'};"
         "found = sorted(m for m in sys.modules if m.split('.')[0] in heavy);"
         "print(','.join(found))"],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"serving imports {result.stdout.strip()}"


def test_the_explanation_stack_is_absent_from_serving_requirements():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    serving = (root / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("shap", "numba", "llvmlite", "scikit-learn", "scipy"):
        assert package not in serving, f"{package} is in the serving requirements"

    development = (root / "requirements-dev.txt").read_text(encoding="utf-8").lower()
    assert "shap" in development


def test_serving_requirements_cover_what_serving_imports():
    """Every third-party module the serving path imports has to be installed.

    PyYAML went missing from this list once and the deployment would have
    crashed on import, after a successful build.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "requirements.txt").read_text(encoding="utf-8").lower()

    required = {
        "torch": "torch",
        "numpy": "numpy",
        "fastapi": "fastapi",
        "pydantic": "pydantic",
        "yaml": "pyyaml",
    }
    missing = [
        module for module, package in required.items() if package not in text
    ]
    assert not missing, f"requirements.txt is missing {missing}"


def test_every_third_party_import_is_declared():
    """Nothing may be imported that the requirements files do not ask for.

    A package installed for an unrelated reason makes a missing requirement
    invisible on the machine that wrote the code, and the build that finds it is
    the one on a clean checkout. This is the direct-import half of that; an
    optional dependency of a declared package, such as the test client's HTTP
    library, still surfaces only on a clean install.
    """
    import ast
    import re
    import sys
    from importlib.metadata import packages_distributions
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    first_party = {
        "app", "core", "db", "eval", "experiments", "ingest", "model",
        "registry", "serve", "snapshot", "tests", "train",
    }

    def _normalise(name: str) -> str:
        return name.strip().lower().replace("_", ".").replace("-", ".")

    declared = set()
    for name in ("requirements.txt", "requirements-dev.txt"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            # A comment naming a package is not a declaration of it, and the
            # first version of this test read the whole file as one string and
            # so could not tell the difference.
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
