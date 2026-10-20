"""Regression tests for the legacy EpilogSlurmctld entry point."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _clean_env() -> dict:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["MPLCONFIGDIR"] = os.environ.get("MPLCONFIGDIR", "/tmp")
    return env


def test_legacy_entry_point_imports_when_run_by_path(tmp_path: Path):
    # The installed wrapper runs `python <repo>/src/services/slurm_power_query.py`
    # from an unrelated directory and without PYTHONPATH.
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "src" / "services" / "slurm_power_query.py"), "--help"],
        cwd=tmp_path,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "No module named 'oob'" not in result.stderr
    if result.returncode == 0:  # the full dependency set is installed
        assert "usage" in result.stdout


def test_resolving_job_context_does_not_need_requests():
    # Only the slurmrestd path needs `requests`; job-end hooks must not import it.
    result = subprocess.run(
        [sys.executable, "-c", "import sys, shared.slurm.resolver; print('requests' in sys.modules)"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
