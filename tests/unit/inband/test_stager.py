"""Tests for publishing a sealed spool to shared staging."""

from __future__ import annotations

from pathlib import Path

import pytest

from inband import stager
from inband.spool import SEAL_FILENAME, SampleWriter, sample_path, write_seal
from inband.stager import (
    DONE_FILENAME,
    NODE_STATUS_FILENAME,
    StageConfig,
    StageError,
    copy_verified,
    run_stager,
)
from inband.storage import load_env_file, read_json_file, write_env_file, write_json_file

def _spool(tmp_path: Path, samples: int = 2) -> Path:
    spool = tmp_path / "spool" / "repacss-12345-100" / "rpc-91-1"
    writer = SampleWriter(sample_path(spool, "rapl"), ["intel-rapl:0"])
    for number in range(samples):
        writer.write(100.0 + number, [[number]])
    writer.close()
    write_json_file(
        spool / "attempt.json",
        {"job_id": "12345", "storage_key": "repacss-12345-100", "hostname": "rpc-91-1"},
    )
    write_json_file(spool / "runner_status.json", {"state": "complete"})
    return spool


def _seal(spool: Path):
    return write_seal(
        spool, job_id="12345", storage_key="repacss-12345-100", hostname="rpc-91-1", runner_state="complete"
    )


def _config(tmp_path: Path, spool: Path, **overrides) -> StageConfig:
    values = dict(spool_dir=spool, staging_dir=tmp_path / "staging" / "rpc-91-1", wait_seconds=0)
    values.update(overrides)
    return StageConfig(**values)


def test_sealed_spool_is_published_with_done_last_and_local_files_removed(tmp_path: Path):
    spool = _spool(tmp_path)
    seal = _seal(spool)
    config = _config(tmp_path, spool)

    assert run_stager(config) == 0

    staging = config.staging_dir
    for entry in seal["files"]:
        assert (staging / entry["path"]).read_bytes() != b""
    assert (staging / SEAL_FILENAME).exists()
    status = read_json_file(staging / NODE_STATUS_FILENAME)
    assert status["state"] == "complete"
    assert status["collectors"]["rapl"] == {"rows": 2, "first_t": 100.0, "last_t": 101.0}
    assert (staging / DONE_FILENAME).exists()
    assert not list(staging.rglob("*.partial"))
    assert not spool.exists() and not spool.parent.exists()


def test_failed_verification_leaves_no_done_and_keeps_local_spool(tmp_path: Path, monkeypatch):
    spool = _spool(tmp_path)
    _seal(spool)
    sample_path(spool, "rapl").write_text("changed after sealing\n", encoding="utf-8")
    config = _config(tmp_path, spool)

    assert run_stager(config) == 1

    assert not (config.staging_dir / DONE_FILENAME).exists()
    assert not sample_path(config.staging_dir, "rapl").exists()
    assert (spool / SEAL_FILENAME).exists()


def test_rerun_after_publication_is_a_no_op(tmp_path: Path):
    spool = _spool(tmp_path)
    _seal(spool)
    config = _config(tmp_path, spool, keep_local=True)
    assert run_stager(config) == 0
    assert run_stager(_config(tmp_path, spool)) == 0
    assert not spool.exists()


def test_existing_destination_with_other_content_is_never_overwritten(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("a\n", encoding="utf-8")
    destination = tmp_path / "staging" / "source.csv"
    destination.parent.mkdir()
    destination.write_text("b\n", encoding="utf-8")
    with pytest.raises(StageError, match="different content"):
        copy_verified(source, destination)
    assert destination.read_text(encoding="utf-8") == "b\n"


def test_runner_that_exits_without_a_seal_gets_a_forced_seal(tmp_path: Path, monkeypatch):
    spool = _spool(tmp_path, samples=0)
    # Killed mid-write: the last line is incomplete.
    partial = sample_path(spool, "rapl").with_name("rapl.csv.partial")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"t,intel-rapl:0\n108.000,8\n109.000,9\n110.0")
    monkeypatch.setattr(stager, "runner_alive", lambda unit: False)
    config = _config(tmp_path, spool, runner_unit="repacss-power-ib@repacss-12345-100.service")

    assert run_stager(config) == 0

    status = read_json_file(config.staging_dir / NODE_STATUS_FILENAME)
    assert status["state"] == "forced_partial"
    assert status["collectors"]["rapl"]["rows"] == 2
    recovered = sample_path(config.staging_dir, "rapl")
    assert recovered.read_bytes() == b"t,intel-rapl:0\n108.000,8\n109.000,9\n"


def test_without_a_seal_or_runner_unit_the_stager_gives_up(tmp_path: Path):
    spool = _spool(tmp_path)
    config = _config(tmp_path, spool)
    assert run_stager(config) == 1
    assert not (config.staging_dir / DONE_FILENAME).exists()


def test_runtime_state_of_a_newer_attempt_is_kept(tmp_path: Path):
    spool = _spool(tmp_path)
    _seal(spool)
    state_dir = tmp_path / "run"
    write_env_file(state_dir / "repacss-12345-100.env", {"STORAGE_KEY": "repacss-12345-100"})
    write_env_file(state_dir / "12345.env", {"STORAGE_KEY": "repacss-12345-200"})

    assert run_stager(_config(tmp_path, spool, state_dir=state_dir)) == 0

    assert not (state_dir / "repacss-12345-100.env").exists()
    assert load_env_file(state_dir / "12345.env") == {"STORAGE_KEY": "repacss-12345-200"}
