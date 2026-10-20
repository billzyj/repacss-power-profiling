"""Unit tests for in-band Slurm hook helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from inband import hooks
from inband.storage import job_state_file_path, load_env_file, read_json_file, unit_env_file_path, write_env_file


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _base_env(tmp_path: Path) -> dict[str, str]:
    return {
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_USER": "alice",
        "SLURM_CLUSTER_NAME": "repacss",
        "SLURMD_NODENAME": "rpg-93-1",
        "SLURM_JOB_PARTITION": "h100",
        "SLURM_JOB_START_TIME": "1775808000",
        "SLURM_JOB_EXCLUSIVE": "NODE",
        "MONSTER_POWER_STATE_DIR": str(tmp_path / "run"),
        "MONSTER_POWER_IB_STORE_ROOT": str(tmp_path / "stage root"),
        "REPACSS_POWER_IB_LOCAL_SPOOL_ROOT": str(tmp_path / "spool"),
        "MONSTER_POWER_PYTHON": "python3",
    }


def _record_systemctl(monkeypatch) -> list:
    calls = []

    def fake_systemctl(args, check=True):
        calls.append(args)
        return _Result(0)

    monkeypatch.setattr(hooks, "_run_systemctl", fake_systemctl)
    return calls


def test_prolog_writes_local_state_and_starts_the_runner(tmp_path: Path, monkeypatch):
    env = _base_env(tmp_path)
    env["SLURM_JOB_COMMENT"] = "power:inband;collectors=rapl;interval_ms=500"
    calls = _record_systemctl(monkeypatch)

    assert hooks.run_prolog(env) == 0

    key = "repacss-12345-1775808000"
    state = load_env_file(job_state_file_path("12345", state_dir=tmp_path / "run"))
    assert state["STORAGE_KEY"] == key
    assert state["INTERVAL_MS"] == "500"
    assert state["PARTITION"] == "h100"
    assert state["REPACSS_POWER_OUTPUT_DIR"] == str(tmp_path / "spool" / key / "rpg-93-1")
    assert state["REPACSS_POWER_STAGING_DIR"] == str(tmp_path / "stage root" / key / "rpg-93-1")

    unit_env = unit_env_file_path(key, state_dir=tmp_path / "run").read_text(encoding="utf-8")
    assert "REPACSS_POWER_PARTITION=h100" in unit_env
    assert f"REPACSS_POWER_RUNNER_UNIT=repacss-power-ib@{key}.service" in unit_env

    attempt = read_json_file(tmp_path / "spool" / key / "rpg-93-1" / "attempt.json")
    assert attempt["partition"] == "h100"
    assert attempt["collectors"] == ["rapl"]

    # Prolog never touches shared storage.
    assert not (tmp_path / "stage root").exists()
    assert calls == [["start", f"repacss-power-ib@{key}.service"]]


def test_prolog_without_a_start_time_still_derives_one_key_per_job(tmp_path: Path, monkeypatch):
    env = _base_env(tmp_path)
    del env["SLURM_JOB_START_TIME"]
    env["SLURM_JOB_COMMENT"] = "power:inband"
    _record_systemctl(monkeypatch)

    assert hooks.run_prolog(env) == 0
    state = load_env_file(job_state_file_path("12345", state_dir=tmp_path / "run"))
    assert state["STORAGE_KEY"] == "repacss-12345-12345"


@pytest.mark.parametrize("comment", ["", "power:oob", "power:inband;interval_ms=1", "power:bogus"])
def test_prolog_skips_without_a_valid_inband_comment(tmp_path: Path, monkeypatch, comment):
    env = _base_env(tmp_path)
    env["SLURM_JOB_COMMENT"] = comment
    calls = _record_systemctl(monkeypatch)

    assert hooks.run_prolog(env) == 0
    assert calls == []
    assert not job_state_file_path("12345", state_dir=tmp_path / "run").exists()


def test_prolog_rejects_a_node_name_that_could_escape_the_spool(tmp_path: Path, monkeypatch):
    env = _base_env(tmp_path)
    env["SLURM_JOB_COMMENT"] = "power:inband"
    env["SLURMD_NODENAME"] = "../rpg-93-1"
    calls = _record_systemctl(monkeypatch)

    with pytest.raises(ValueError, match="safe path token"):
        hooks.run_prolog(env)
    assert calls == []


def test_epilog_requests_runner_stop_and_stager_without_touching_staging(tmp_path: Path, monkeypatch):
    env = _base_env(tmp_path)
    write_env_file(
        job_state_file_path("12345", state_dir=tmp_path / "run"),
        {
            "STORAGE_KEY": "repacss-12345-1775808000",
            "SYSTEMD_UNIT": "repacss-power-ib@repacss-12345-1775808000.service",
            "STAGE_UNIT": "repacss-power-stage@repacss-12345-1775808000.service",
        },
    )
    calls = _record_systemctl(monkeypatch)

    assert hooks.run_epilog(env) == 0

    assert calls == [
        ["stop", "--no-block", "repacss-power-ib@repacss-12345-1775808000.service"],
        ["start", "--no-block", "repacss-power-stage@repacss-12345-1775808000.service"],
    ]
    assert not (tmp_path / "stage root").exists()


def test_epilog_without_state_does_nothing(tmp_path: Path, monkeypatch):
    calls = _record_systemctl(monkeypatch)
    assert hooks.run_epilog(_base_env(tmp_path)) == 0
    assert calls == []


@pytest.mark.parametrize("exclusive", [None, "NO", "USER"])
def test_prolog_skips_a_job_without_an_exclusive_node(tmp_path: Path, monkeypatch, exclusive):
    env = _base_env(tmp_path)
    env["SLURM_JOB_COMMENT"] = "power:inband"
    if exclusive is None:
        del env["SLURM_JOB_EXCLUSIVE"]
    else:
        env["SLURM_JOB_EXCLUSIVE"] = exclusive
    calls = _record_systemctl(monkeypatch)

    assert hooks.run_prolog(env) == 0
    assert calls == []
    assert not job_state_file_path("12345", state_dir=tmp_path / "run").exists()
