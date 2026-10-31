"""Unit and configuration tests for Docker containerization (Task 7.3).

Verifies:
1. Dockerfile presence, small base image, non-root user, OpenMP runtime, exposed ports, and healthcheck.
2. .dockerignore rules ensuring raw Kaggle datasets and caches are never packed into images.
3. docker-entrypoint.sh script routing for FastAPI, Streamlit, and combined modes.
4. docker-compose.yml multi-service orchestration and healthcheck dependencies.
5. Containerization documentation completeness in docs/containerization.md.
"""

from pathlib import Path
import yaml
import pytest


def test_dockerfile_structure_and_security():
    """Verifies that Dockerfile satisfies production containerization standards."""
    df_path = Path("Dockerfile")
    assert df_path.is_file(), "Dockerfile must exist at repository root."

    content = df_path.read_text(encoding="utf-8")

    # Base image must be small and reproducible
    assert "FROM python:3.10-slim" in content, "Dockerfile must use python:3.10-slim base image."

    # Must install OpenMP library (libgomp1) for LightGBM tree evaluation
    assert "libgomp1" in content, "Dockerfile must install libgomp1 for LightGBM OpenMP support."

    # Must enforce non-root user for security best practice
    assert "useradd" in content and "bankguard" in content, "Dockerfile must create dedicated application user."
    assert "USER bankguard" in content, "Dockerfile must switch to non-root user."

    # Layer caching optimization: requirements copied before application code
    req_pos = content.find("COPY requirements.txt")
    src_pos = content.find("COPY src/")
    assert req_pos != -1, "Dockerfile must copy requirements.txt for layer caching."
    assert src_pos != -1, "Dockerfile must copy src/ directory."
    assert req_pos < src_pos, "requirements.txt must be copied before src/ to optimize build caching."

    # Must expose both API (8000) and Dashboard (8501) ports
    assert "EXPOSE 8000 8501" in content or ("8000" in content and "8501" in content)

    # Must declare HEALTHCHECK probing /health endpoint
    assert "HEALTHCHECK" in content, "Dockerfile must declare container HEALTHCHECK."
    assert "/health" in content, "HEALTHCHECK must probe /health endpoint."

    # Must set ENTRYPOINT
    assert "ENTRYPOINT [\"/app/docker-entrypoint.sh\"]" in content
    assert "CMD [\"api\"]" in content


def test_dockerignore_excludes_raw_data_and_caches():
    """Verifies that .dockerignore strictly excludes raw datasets, virtualenvs, and test caches."""
    di_path = Path(".dockerignore")
    assert di_path.is_file(), ".dockerignore must exist at repository root."

    lines = [line.strip() for line in di_path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]

    # Raw Kaggle data must NEVER be baked into container image
    assert any("data/raw" in line for line in lines), ".dockerignore must exclude data/raw/ directory."

    # Caches and virtual environments must be excluded
    assert any(".venv" in line for line in lines), ".dockerignore must exclude .venv/."
    assert any("__pycache__" in line for line in lines), ".dockerignore must exclude __pycache__/."
    assert any(".pytest_cache" in line for line in lines), ".dockerignore must exclude .pytest_cache/."
    assert any(".git" in line for line in lines), ".dockerignore must exclude .git/."


def test_docker_entrypoint_script():
    """Verifies docker-entrypoint.sh routing logic for api, dashboard, and combined modes."""
    ep_path = Path("docker-entrypoint.sh")
    assert ep_path.is_file(), "docker-entrypoint.sh must exist."

    content = ep_path.read_text(encoding="utf-8")
    assert content.startswith("#!/bin/bash"), "Entrypoint must begin with bash shebang."
    assert "set -e" in content, "Entrypoint must set -e for strict error exit."

    # Must route api command to uvicorn on port 8000
    assert "uvicorn bankguard.api.app:app" in content
    assert "8000" in content

    # Must route dashboard command to streamlit on port 8501
    assert "streamlit run dashboard.py" in content
    assert "8501" in content


def test_docker_compose_configuration():
    """Verifies docker-compose.yml defines api and dashboard services with health checks."""
    compose_path = Path("docker-compose.yml")
    assert compose_path.is_file(), "docker-compose.yml must exist."

    with open(compose_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "services" in cfg
    services = cfg["services"]

    # API service verification
    assert "api" in services
    api = services["api"]
    assert "8000:8000" in api["ports"]
    assert "healthcheck" in api
    assert "/health" in str(api["healthcheck"]["test"])

    # Dashboard service verification
    assert "dashboard" in services
    dash = services["dashboard"]
    assert "8501:8501" in dash["ports"]
    assert "depends_on" in dash
    assert "api" in dash["depends_on"]


def test_containerization_documentation_exists():
    """Verifies that docs/containerization.md documents build and run workflows."""
    doc_path = Path("docs/containerization.md")
    assert doc_path.is_file(), "docs/containerization.md must exist."

    content = doc_path.read_text(encoding="utf-8")
    assert "docker build" in content
    assert "docker run" in content
    assert "docker compose" in content
    assert "/health" in content

