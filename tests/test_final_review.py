"""Automated verification tests for Task 7.4 (Final Review).

Verifies that:
1. README.md exists, is comprehensive, and contains all essential quickstart sections.
2. All documentation files in docs/ exist and have no unfulfilled placeholders.
3. Metric claims in README.md and docs/ strictly match the saved evaluation artifacts.
4. Key referenced paths and scripts in documentation actually exist on disk.
5. Definition of done criteria are complete.
"""

from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_readme_file_exists_and_comprehensive():
    """Verify that README.md exists and contains all required architectural and quickstart sections."""
    readme_path = PROJECT_ROOT / "README.md"
    assert readme_path.exists(), "README.md must exist in the project root."

    content = readme_path.read_text(encoding="utf-8")
    assert len(content) > 2000, "README.md must be comprehensive."

    required_sections = [
        "Executive Summary",
        "Empirical Results",
        "Three-Tier Operational Decision Policy",
        "System Architecture",
        "Repository Structure",
        "Quickstart & Installation",
        "End-to-End Pipeline Reproduction",
        "FastAPI Serving Microservice",
        "Streamlit FinTech Dashboard",
        "Docker & Container Deployment",
        "Automated Testing",
        "Continuous Integration",
        "Explainability",
        "Governance",
    ]

    for section in required_sections:
        assert section.lower() in content.lower(), f"README.md missing required section: '{section}'"


def test_readme_referenced_paths_exist():
    """Verify that file and directory paths referenced in README.md exist on disk."""
    referenced_paths = [
        "Dockerfile",
        "docker-compose.yml",
        "docker-entrypoint.sh",
        ".dockerignore",
        "pyproject.toml",
        "requirements.txt",
        ".github/workflows/ci.yml",
        "artifacts/improved_model.joblib",
        "artifacts/baseline_model.joblib",
        "artifacts/preprocessor.joblib",
        "artifacts/improved_predictions.npz",
        "report/model_comparison.json",
        "report/decision_policy_report.md",
        "report/monitoring_summary.md",
        "docs/containerization.md",
        "docs/model_evaluation.md",
        "docs/Project_Overview.md",
        "src/bankguard/api/service.py",
        "src/bankguard/dashboard/app.py",
        "src/bankguard/decisioning/policy.py",
        "src/bankguard/monitoring/tracker.py",
    ]

    for rel_path in referenced_paths:
        full_path = PROJECT_ROOT / rel_path
        assert full_path.exists(), f"Path referenced in README does not exist on disk: {rel_path}"


def test_readme_metrics_match_evaluation_artifacts():
    """Verify that quantitative metrics in README match report/model_comparison.json exactly."""
    comparison_path = PROJECT_ROOT / "report" / "model_comparison.json"
    assert comparison_path.exists()

    with open(comparison_path, "r", encoding="utf-8") as f:
        comp = json.load(f)

    readme_text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    # Baseline and candidate PR-AUC
    baseline_prauc = f"{comp['metrics']['pr_auc']['baseline']:.4f}"
    candidate_prauc = f"{comp['metrics']['pr_auc']['candidate']:.4f}"
    assert baseline_prauc in readme_text, f"Baseline PR-AUC {baseline_prauc} not found in README"
    assert candidate_prauc in readme_text, f"Candidate PR-AUC {candidate_prauc} not found in README"

    # Baseline and candidate ROC-AUC
    baseline_rocauc = f"{comp['metrics']['roc_auc']['baseline']:.4f}"
    candidate_rocauc = f"{comp['metrics']['roc_auc']['candidate']:.4f}"
    assert baseline_rocauc in readme_text, f"Baseline ROC-AUC {baseline_rocauc} not found in README"
    assert candidate_rocauc in readme_text, f"Candidate ROC-AUC {candidate_rocauc} not found in README"


def test_docs_directory_completeness():
    """Verify all core documentation files exist and have no unfulfilled placeholders."""
    docs_dir = PROJECT_ROOT / "docs"
    assert docs_dir.exists()

    expected_docs = [
        "Architecture_and_Data_Flow.md",
        "containerization.md",
        "data_card.md",
        "data_dictionary.md",
        "Functionality_Specification.md",
        "implementation_plan.md",
        "leakage_policy.md",
        "model_evaluation.md",
        "progress.md",
        "Project_Overview.md",
    ]

    for doc_name in expected_docs:
        doc_path = docs_dir / doc_name
        assert doc_path.exists(), f"Expected documentation file missing: {doc_name}"
        content = doc_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 100, f"Documentation file {doc_name} is unexpectedly empty."
        assert "Fill from run" not in content, f"Unfilled placeholder in {doc_name}"


def test_project_overview_definition_of_done_completed():
    """Verify that Project_Overview.md definition of done items are all checked."""
    overview_path = PROJECT_ROOT / "docs" / "Project_Overview.md"
    content = overview_path.read_text(encoding="utf-8")

    # Check for unfulfilled checkboxes in Definition of Done
    dod_section = re.search(r"## 7\. Definition of done(.*?)(?=##|\Z)", content, re.DOTALL)
    assert dod_section is not None, "Definition of done section not found in Project_Overview.md"

    unchecked = re.findall(r"- \[ \]", dod_section.group(1))
    assert len(unchecked) == 0, f"Found {len(unchecked)} unchecked items in Definition of Done"
