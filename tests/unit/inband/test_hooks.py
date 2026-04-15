"""Unit tests for in-band Slurm hook helpers."""

from __future__ import annotations

from pathlib import Path

from inband import hooks
from inband.storage import job_state_file_path, load_env_file, read_json_file, unit_env_file_path


class _Result:
    def __init__(self, returncode: int = 0, stdout: str = ""):
        self.returncode = returncode
        self.stdout = stdout


def _base_env(tmp_path: Path) -> dict[str, str]:
    return {
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_USER": "alice",
        "SLURM_CLUSTER_NAME": "repacss",
        "SLURMD_NODENAME": "rpg-93-1",
        "MONSTER_POWER_STATE_DIR": str(tmp_path / "state"),
        "MONSTER_POWER_IB_STORE_ROOT": str(tmp_path / "stage root"),
        "MONSTER_POWER_PYTHON": "python3",
    }


def test_run_prolog_writes_state_and_systemd_env_files(tmp_path: Path, monkeypatch):
    env = _base_env(tmp_path)
    monkeypatch.setattr(
        hooks,
        "fetch_job_metadata",
        lambda job_id, env=None: {
            "Comment": "power:inband;collectors=rapl;interval_ms=500",
            "StartTime": "1775808000",
        },
    )

    systemctl_calls = []

    def fake_systemctl(args, check=True):
        systemctl_calls.append(args)
        return _Result(0, "")

    monkeypatch.setattr(hooks, "_run_systemctl", fake_systemctl)

    assert hooks.run_prolog(env) == 0

    state_file = job_state_file_path("12345", state_dir=Path(env["MONSTER_POWER_STATE_DIR"]))
    shell_state = load_env_file(state_file)
    assert shell_state["STORAGE_KEY"] == "repacss-12345-1775808000"
    assert shell_state["INTERVAL_MS"] == "500"

    systemd_env = unit_env_file_path("repacss-12345-1775808000", state_dir=Path(env["MONSTER_POWER_STATE_DIR"]))
    systemd_text = systemd_env.read_text(encoding="utf-8")
    assert "REPACSS_POWER_OUTPUT_DIR=" in systemd_text
    assert "'stage root'" not in systemd_text
    assert systemctl_calls == [["start", "repacss-power-ib@repacss-12345-1775808000.service"]]


def test_run_prolog_reuses_cached_epoch_when_metadata_missing(tmp_path: Path, monkeypatch):
    env = _base_env(tmp_path)
    stage_root = Path(env["MONSTER_POWER_IB_STORE_ROOT"])
    cache = stage_root / ".state" / "repacss-12345.epoch"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("1775808123\n", encoding="utf-8")

    monkeypatch.setattr(hooks, "fetch_job_metadata", lambda job_id, env=None: {"Comment": "power:inband"})
    monkeypatch.setattr(hooks, "_run_systemctl", lambda args, check=True: _Result(0, ""))

    assert hooks.run_prolog(env) == 0
    shell_state = load_env_file(job_state_file_path("12345", state_dir=Path(env["MONSTER_POWER_STATE_DIR"])))
    assert shell_state["STORAGE_KEY"] == "repacss-12345-1775808123"


def test_run_epilog_marks_failed_when_runner_status_is_missing(tmp_path: Path, monkeypatch):
    env = _base_env(tmp_path)
    stage_root = Path(env["MONSTER_POWER_IB_STORE_ROOT"])
    state_dir = Path(env["MONSTER_POWER_STATE_DIR"])
    state_dir.mkdir(parents=True, exist_ok=True)
    stage_root.mkdir(parents=True, exist_ok=True)

    state = {
        "JOB_ID": "12345",
        "JOB_USER": "alice",
        "CLUSTER_NAME": "repacss",
        "HOSTNAME": "rpg-93-1",
        "JOB_START_EPOCH": "1775808000",
        "STORAGE_KEY": "repacss-12345-1775808000",
        "IB_STORE_ROOT": str(stage_root),
        "NODE_OUTPUT_DIR": str(stage_root / "repacss-12345-1775808000" / "rpg-93-1"),
        "INTERVAL_MS": "1000",
        "COLLECTORS": "rapl",
        "SYSTEMD_UNIT": "repacss-power-ib@repacss-12345-1775808000.service",
    }
    from inband.storage import write_env_file

    write_env_file(job_state_file_path("12345", state_dir=state_dir), state)

    responses = iter(
        [
            _Result(0, ""),
            _Result(0, "Result=success\nExecMainStatus=0\nSubState=dead\n"),
        ]
    )
    monkeypatch.setattr(hooks, "_run_systemctl", lambda args, check=True: next(responses))

    assert hooks.run_epilog(env) == 0

    node_status = read_json_file(stage_root / "repacss-12345-1775808000" / "rpg-93-1" / "node_status.json")
    assert node_status["state"] == "failed"
    assert "runner_status.json" in node_status["message"]
    assert (stage_root / "repacss-12345-1775808000" / "rpg-93-1" / ".done").exists()
