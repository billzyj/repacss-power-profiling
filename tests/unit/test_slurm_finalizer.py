"""Tests for queueing and finalizing in-band job attempts on the controller."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from inband.hooks import storage_key_from_env
from inband.storage import read_json_file
from shared.slurm import dispatcher, finalizer
from shared.slurm.finalize_queue import COMPLETE, PENDING, enqueue
from shared.slurm.finalizer import FinalizerConfig, build_record, export_dir, process_queue
from tests.unit.inband.staging_fixtures import T0, stage_node

START = int(T0.timestamp())


def _config(tmp_path: Path) -> FinalizerConfig:
    return FinalizerConfig(
        queue_root=tmp_path / "queue", staging_root=tmp_path / "staging", report_root=tmp_path / "reports"
    )


def _queue(
    config: FinalizerConfig, nodes, now: float, timeout: float = 600.0, exclusive: str = "NODE", attempt_id: str = "repacss-777-100"
):
    record = build_record(
        job_id="777",
        attempt_id=attempt_id,
        cluster="repacss",
        user="alice",
        nodes=nodes,
        start_epoch=START,
        end_epoch=START + 10,
        now=now,
        timeout_seconds=timeout,
        exclusive=exclusive,
    )
    return enqueue(config.queue_root, record), record


def test_finalizer_waits_for_nodes_then_exports_a_partial_report(tmp_path: Path):
    config = _config(tmp_path)
    stage_node(config.staging_root / "repacss-777-100", tmp_path / "spool", "rpg-93-1")
    _, record = _queue(config, ["rpg-93-1", "rpg-93-2"], now=1000.0)

    assert process_queue(config, now=1100.0) == 0  # rpg-93-2 may still publish
    assert (config.queue_root / PENDING / "repacss-777-100.json").exists()

    assert process_queue(config, now=1700.0) == 1  # deadline passed
    summary = read_json_file(export_dir(config, record) / "summary.json")
    assert summary["status"] == "partial"
    assert summary["nodes"]["missing"] == ["rpg-93-2"]
    settled = read_json_file(config.queue_root / COMPLETE / "repacss-777-100.json")
    assert settled["outcome"]["status"] == "partial"


def test_finalizer_exports_once_into_the_documented_layout(tmp_path: Path):
    config = _config(tmp_path)
    stage_node(config.staging_root / "repacss-777-100", tmp_path / "spool", "rpg-93-1")
    _, record = _queue(config, ["rpg-93-1"], now=1000.0)

    assert process_queue(config, now=1000.0) == 1

    target = tmp_path / "reports" / "alice" / "777" / "ib"
    assert target == export_dir(config, record)
    assert sorted(path.name for path in target.iterdir()) == ["provenance.json", "raw_power.csv", "summary.json"]
    summary = read_json_file(target / "summary.json")
    assert summary["status"] == "complete"
    assert summary["energy_j"]["cpu_package"] == 2200.0
    provenance = read_json_file(target / "provenance.json")
    assert provenance["nodes"]["rpg-93-1"]["node_status"]["state"] == "complete"
    assert not list(target.parent.glob(".*partial"))

    delivery = read_json_file(config.queue_root / "deliver" / PENDING / "repacss-777-ib.json")
    assert (delivery["user"], delivery["job_id"], delivery["name"]) == ("alice", "777", "ib")

    # A repeated record for the same attempt never rewrites the export.
    _queue(config, ["rpg-93-1"], now=2000.0)
    assert process_queue(config, now=2000.0) == 0


def test_a_requeued_attempt_exports_beside_the_first(tmp_path: Path):
    config = _config(tmp_path)
    for attempt in ("repacss-777-100", "repacss-777-200"):
        stage_node(config.staging_root / attempt, tmp_path / "spool" / attempt, "rpg-93-1", attempt_id=attempt)
        _queue(config, ["rpg-93-1"], now=1000.0, attempt_id=attempt)
        assert process_queue(config, now=1000.0) == 1

    job_dir = tmp_path / "reports" / "alice" / "777"
    assert sorted(path.name for path in job_dir.iterdir()) == ["ib", "ib-200"]
    assert read_json_file(job_dir / "ib-200" / "summary.json")["attempt_id"] == "repacss-777-200"
    assert (config.queue_root / "deliver" / PENDING / "repacss-777-ib-200.json").exists()


def test_dispatcher_queues_inband_jobs_under_the_compute_node_key(tmp_path: Path, monkeypatch):
    env = {
        "SLURM_JOB_ID": "777",
        "SLURM_JOB_USER": "alice",
        "SLURM_CLUSTER_NAME": "repacss",
        "SLURM_JOB_NODELIST": "rpg-93-[1-2]",
        "SLURM_JOB_START_TIME": str(START),
        "SLURM_JOB_END_TIME": str(START + 10),
        "SLURM_JOB_COMMENT": "power:both;collectors=rapl",
        "SLURM_JOB_OVERSUBSCRIBE": "NO",  # what Slurm 25.11 gives EpilogSlurmctld for --exclusive
        "SLURM_JOB_WORK_DIR": "/home/alice/run",
        "REPACSS_POWER_IB_QUEUE_ROOT": str(tmp_path / "queue"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(finalizer, "DEFAULT_QUEUE_ROOT", tmp_path / "queue")
    oob_jobs = []
    fake_oob = types.ModuleType("oob.slurm.epilog_handler")
    fake_oob.handle_oob_job = lambda context: oob_jobs.append(context) or tmp_path / "reports" / "alice" / "777" / "oob"
    monkeypatch.setitem(sys.modules, "oob.slurm.epilog_handler", fake_oob)

    assert dispatcher.dispatch_from_env() == 0

    key = storage_key_from_env(env)
    record = read_json_file(tmp_path / "queue" / PENDING / f"{key}.json")
    assert record["nodes"] == ["rpg-93-1", "rpg-93-2"]
    assert record["end_epoch"] == START + 10
    assert record["exclusive"] == "NODE"
    assert record["work_dir"] == "/home/alice/run"
    assert [context.job_id for context in oob_jobs] == ["777"]  # hybrid keeps the OOB path
    oob_delivery = read_json_file(tmp_path / "queue" / "deliver" / PENDING / "repacss-777-oob.json")
    assert (oob_delivery["name"], oob_delivery["work_dir"]) == ("oob", "/home/alice/run")


def test_a_job_without_exclusive_nodes_gets_an_immediate_explanation(tmp_path: Path):
    config = _config(tmp_path)
    _, record = _queue(config, ["rpg-93-1"], now=1000.0, exclusive="OVERSUBSCRIBE=OK")

    assert process_queue(config, now=1000.0) == 1  # no waiting for nodes that never start

    target = export_dir(config, record)
    assert sorted(path.name for path in target.iterdir()) == ["summary.json"]
    summary = read_json_file(target / "summary.json")
    assert summary["status"] == "skipped"
    assert "--exclusive" in summary["reason"] and "OVERSUBSCRIBE=OK" in summary["reason"]


def test_inband_only_dispatch_leaves_the_oob_report_to_the_wrapper(tmp_path: Path, monkeypatch):
    env = {
        "SLURM_JOB_ID": "778",
        "SLURM_JOB_USER": "alice",
        "SLURM_CLUSTER_NAME": "repacss",
        "SLURM_JOB_NODELIST": "rpc-97-[1-4]",
        "SLURM_JOB_START_TIME": str(START),
        "SLURM_JOB_END_TIME": str(START + 10),
        "SLURM_JOB_COMMENT": "power:both",
        "SLURM_JOB_OVERSUBSCRIBE": "NO",
        "REPACSS_POWER_IB_QUEUE_ROOT": str(tmp_path / "queue"),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(finalizer, "DEFAULT_QUEUE_ROOT", tmp_path / "queue")
    fake_oob = types.ModuleType("oob.slurm.epilog_handler")
    fake_oob.handle_oob_job = lambda context: (_ for _ in ()).throw(AssertionError("the wrapper runs OOB"))
    monkeypatch.setitem(sys.modules, "oob.slurm.epilog_handler", fake_oob)

    assert dispatcher.main(["--inband-only"]) == 0

    record = read_json_file(tmp_path / "queue" / PENDING / f"{storage_key_from_env(env)}.json")
    assert record["nodes"] == ["rpc-97-1", "rpc-97-2", "rpc-97-3", "rpc-97-4"]
    assert record["mode"] == "both"
    assert not (tmp_path / "queue" / "deliver").exists()
