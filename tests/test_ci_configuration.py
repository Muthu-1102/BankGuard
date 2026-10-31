"""Tests for CI workflow configuration and repository delivery integrity (Task 7.2).

Verifies:
1. GitHub Actions CI workflow syntax and triggers.
2. Dependency manifest completeness in requirements.txt and pyproject.toml.
3. Test suite coverage and pyproject.toml pytest configuration.
4. Clean importability of all core BankGuard packages.
"""

from pathlib import Path
import yaml
import pytest


def test_ci_workflow_file_exists_and_valid_yaml():
    """Verifies that .github/workflows/ci.yml exists and parses as valid YAML."""
    ci_path = Path(".github/workflows/ci.yml")
    assert ci_path.is_file(), f"CI workflow file not found at: {ci_path}"

    with open(ci_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "name" in cfg
    assert "on" in cfg or True in cfg
    assert "jobs" in cfg
    assert "test" in cfg["jobs"]

    # Verify triggers
    triggers = cfg.get("on") or cfg.get(True)
    assert "push" in triggers
    assert "pull_request" in triggers

    # Verify steps execute pytest and monitoring verification
    steps = cfg["jobs"]["test"]["steps"]
    step_runs = [s.get("run", "") for s in steps]
    any_pytest = any("pytest" in r for r in step_runs)
    any_monitoring = any("run_monitoring_batch.py" in r for r in step_runs)

    assert any_pytest, "CI workflow must include a step running pytest"
    assert any_monitoring, "CI workflow must include operational monitoring verification"


def test_requirements_file_has_core_dependencies():
    """Verifies that requirements.txt declares all production libraries."""
    req_path = Path("requirements.txt")
    assert req_path.is_file()

    with open(req_path, "r", encoding="utf-8") as f:
        lines = [line.strip().lower() for line in f if line.strip() and not line.startswith("#")]

    required_pkgs = [
        "pandas",
        "numpy",
        "scikit-learn",
        "lightgbm",
        "shap",
        "fastapi",
        "uvicorn",
        "pydantic",
        "streamlit",
        "plotly",
        "pyyaml",
        "pytest",
    ]

    for pkg in required_pkgs:
        assert any(pkg in line for line in lines), f"Missing core dependency in requirements.txt: {pkg}"


def test_pyproject_toml_configuration():
    """Verifies that pyproject.toml configures project metadata and pytest options."""
    toml_path = Path("pyproject.toml")
    assert toml_path.is_file()

    content = toml_path.read_text(encoding="utf-8")
    assert "[project]" in content
    assert "name = \"bankguard\"" in content
    assert "[tool.pytest.ini_options]" in content
    assert "pythonpath = [\"src\"]" in content
    assert "testpaths = [\"tests\"]" in content


def test_all_bankguard_modules_importable():
    """Verifies that all primary package modules import cleanly without side-effect errors."""
    import bankguard
    import bankguard.api
    import bankguard.dashboard
    import bankguard.data
    import bankguard.decisioning
    import bankguard.features
    import bankguard.models
    import bankguard.monitoring

    assert bankguard is not None
    assert bankguard.api is not None
    assert bankguard.dashboard is not None
    assert bankguard.data is not None
    assert bankguard.decisioning is not None
    assert bankguard.features is not None
    assert bankguard.models is not None
    assert bankguard.monitoring is not None
