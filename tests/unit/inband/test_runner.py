"""Tests for the per-node in-band runner."""

from __future__ import annotations

import signal
import threading
from pathlib import Path

from inband import runner
from inband.spool import SEAL_FILENAME
from inband.collectors.base import CollectorProbe
from inband.storage import read_json_file
from tests.unit.inband.test_collectors_base import DummyCollector


def _args(tmp_path: Path, *extra: str):
    return runner.build_parser().parse_args(
        [
            "--job-id",
            "12345",
            "--storage-key",
            "repacss-12345-1775808000",
            "--output-dir",
            str(tmp_path),
            "--hostname",
            "rpg-93-1",
            "--interval-ms",
            "10",
            *extra,
        ]
    )


def test_runner_records_why_rapl_was_skipped_and_seals(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner.signal, "signal", lambda *_args: None)

    assert runner.run_runner(_args(tmp_path, "--partition", "zen4", "--collector", "rapl")) == 0

    manifest = read_json_file(tmp_path / "rapl_manifest.json")
    assert manifest["status"] == "skipped"
    assert manifest["reason"] == "unverified_node"
    status = read_json_file(tmp_path / "runner_status.json")
    assert status["state"] == "partial"
    assert status["skipped_collectors"] == [{"name": "rapl", "reason": "skipped: unverified_node"}]
    seal = read_json_file(tmp_path / SEAL_FILENAME)
    assert seal["runner_state"] == "partial"
    assert [entry["path"] for entry in seal["files"]] == ["rapl_manifest.json", "runner_status.json"]


def test_runner_writes_one_sample_file_per_collector_and_seals_it_after_sigterm(tmp_path: Path, monkeypatch):
    handlers = {}
    monkeypatch.setattr(runner.signal, "signal", lambda signum, handler: handlers.setdefault(signum, handler))
    monkeypatch.setattr(runner, "build_collectors", lambda names, partition, node: [DummyCollector()])
    stopper = threading.Timer(0.15, lambda: handlers[signal.SIGTERM](signal.SIGTERM, None))
    stopper.start()

    assert runner.run_runner(_args(tmp_path)) == 0

    status = read_json_file(tmp_path / "runner_status.json")
    assert status["state"] == "complete"
    seal = read_json_file(tmp_path / SEAL_FILENAME)
    (samples,) = [entry for entry in seal["files"] if entry.get("collector") == "dummy"]
    assert samples["path"] == "samples/dummy.csv"
    assert samples["rows"] == status["collector_results"]["dummy"]["sample_count"] >= 2
    assert not list(tmp_path.rglob("*.partial"))


class AbsentCollector(DummyCollector):
    name = "absent"

    def probe(self) -> CollectorProbe:
        return CollectorProbe(name=self.name, available=False, reason="not on this hardware")


def _run_until_sigterm(tmp_path: Path, monkeypatch, *extra: str):
    handlers = {}
    monkeypatch.setattr(runner.signal, "signal", lambda signum, handler: handlers.setdefault(signum, handler))
    monkeypatch.setattr(runner, "build_collectors", lambda names, partition, node: [DummyCollector(), AbsentCollector()])
    threading.Timer(0.05, lambda: handlers[signal.SIGTERM](signal.SIGTERM, None)).start()
    assert runner.run_runner(_args(tmp_path, *extra)) == 0
    return read_json_file(tmp_path / "runner_status.json")


def test_a_default_collector_missing_from_this_node_does_not_make_it_partial(tmp_path: Path, monkeypatch):
    assert _run_until_sigterm(tmp_path, monkeypatch)["state"] == "complete"


def test_a_requested_collector_that_cannot_start_makes_the_node_partial(tmp_path: Path, monkeypatch):
    status = _run_until_sigterm(tmp_path, monkeypatch, "--collector", "dummy", "--collector", "absent")
    assert status["state"] == "partial"
