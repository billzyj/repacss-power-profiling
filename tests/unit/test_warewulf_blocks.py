"""Tests for the compute-node hook blocks and node settings distributed with Warewulf."""

from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WAREWULF = ROOT / "shared" / "slurm" / "warewulf"
NODE_ENV = "/etc/repacss-power/node.env"
RUN_STATE = "/run/repacss-power/"


def test_the_deployment_doc_shows_the_distributed_files():
    doc = (ROOT / "docs" / "SLURM_PROFILING_MODES.md").read_text(encoding="utf-8")
    blocks = [textwrap.dedent(block) for block in re.findall(r"```(?:bash|text)\n(.*?)```", doc, re.S)]
    for name in ("slurm.prolog.block", "slurm.epilog.block", "node.env.ww"):
        assert (WAREWULF / name).read_text(encoding="utf-8") in blocks, name


@pytest.fixture
def node(tmp_path: Path):
    """A fake node: settings file, run-state directory, and stub hook wrappers that fail."""
    home = tmp_path / "home"
    (home / "shared" / "slurm").mkdir(parents=True)
    for hook in ("prolog", "epilog"):
        stub = home / "shared" / "slurm" / f"repacss-power.{hook}.sh"
        stub.write_text(f'echo "{hook} ran store=$MONSTER_POWER_IB_STORE_ROOT"\nexit 7\n', encoding="utf-8")
    (tmp_path / "run").mkdir()

    def settings(enabled: int) -> None:
        (tmp_path / "node.env").write_text(
            f"REPACSS_POWER_IB_ENABLED={enabled}\nREPACSS_POWER_HOME={home}\nMONSTER_POWER_IB_STORE_ROOT=/shared/staging\n",
            encoding="utf-8",
        )

    def run(block: str, comment: str) -> str:
        text = (WAREWULF / block).read_text(encoding="utf-8")
        text = text.replace(NODE_ENV, str(tmp_path / "node.env")).replace(RUN_STATE, f"{tmp_path / 'run'}/")
        script = f"{text}\necho site-continues status=$? store=${{MONSTER_POWER_IB_STORE_ROOT:-unset}}\n"
        result = subprocess.run(
            ["sh", "-c", script],
            env={"PATH": "/usr/bin:/bin", "SLURM_JOB_ID": "42", "SLURM_JOB_COMMENT": comment},
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    run.settings = settings
    run.tmp_path = tmp_path
    return run


def test_prolog_block_runs_the_hook_only_for_an_enabled_node_and_an_in_band_comment(node):
    node.settings(1)
    assert node("slurm.prolog.block", "power:both;interval_ms=1000").splitlines() == [
        "prolog ran store=/shared/staging",
        "site-continues status=0 store=unset",  # a failing hook never fails the site script or leaks settings
    ]
    assert "ran" not in node("slurm.prolog.block", "power:oob")
    assert "ran" not in node("slurm.prolog.block", "an unrelated comment")

    node.settings(0)
    assert "ran" not in node("slurm.prolog.block", "power:inband")

    (node.tmp_path / "node.env").unlink()
    assert node("slurm.prolog.block", "power:inband") == "site-continues status=0 store=unset\n"


def test_epilog_block_finishes_a_started_run_even_after_the_node_was_disabled(node):
    node.settings(0)
    assert "ran" not in node("slurm.epilog.block", "power:inband")  # nothing was started for job 42

    (node.tmp_path / "run" / "42.env").write_text("STORAGE_KEY=k\n", encoding="utf-8")
    assert node("slurm.epilog.block", "power:inband").splitlines() == [
        "epilog ran store=/shared/staging",
        "site-continues status=0 store=unset",
    ]
