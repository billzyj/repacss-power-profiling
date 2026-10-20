"""Tests for delivering finished reports into the job owner's directory."""

from __future__ import annotations

import os
import pwd
from pathlib import Path

from inband.storage import read_json_file
from shared.slurm import delivery
from shared.slurm.delivery import copy_report, deliver, enqueue_delivery, enqueue_oob_delivery, process_queue, run_as_user
from shared.slurm.finalize_queue import COMPLETE, FAILED, PENDING

ME = pwd.getpwuid(os.geteuid()).pw_name


def _report(root: Path, user: str = "alice", job_id: str = "777", name: str = "ib") -> Path:
    source = root / user / job_id / name
    source.mkdir(parents=True)
    (source / "summary.json").write_text('{"status": "complete"}\n', encoding="utf-8")
    (source / "raw_power.csv").write_text("timestamp,hostname,metric,group,energy_j,power_w\n", encoding="utf-8")
    return source


def test_report_is_copied_into_the_work_directory(tmp_path: Path):
    source = _report(tmp_path / "reports")
    work = tmp_path / "work"
    work.mkdir()

    result = copy_report(source, [str(work), str(tmp_path / "home")], "777", "ib")

    target = work / "power_reports" / "777" / "ib"
    assert result == {"status": "delivered", "path": str(target)}
    assert sorted(path.name for path in target.iterdir()) == ["raw_power.csv", "summary.json"]
    assert not list(target.parent.glob(".*partial"))


def test_an_unusable_work_directory_falls_back_to_home(tmp_path: Path):
    source = _report(tmp_path / "reports")
    home = tmp_path / "home"
    home.mkdir()

    result = copy_report(source, [str(tmp_path / "deleted"), "", "relative/dir", str(home)], "777", "ib")

    assert result["status"] == "delivered"
    assert result["path"] == str(home / "power_reports" / "777" / "ib")


def test_an_existing_delivery_is_never_overwritten(tmp_path: Path):
    source = _report(tmp_path / "reports")
    target = tmp_path / "work" / "power_reports" / "777" / "ib"
    target.mkdir(parents=True)
    (target / "notes.txt").write_text("the user's own file\n", encoding="utf-8")

    result = copy_report(source, [str(tmp_path / "work")], "777", "ib")

    assert result["status"] == "already_delivered"
    assert sorted(path.name for path in target.iterdir()) == ["notes.txt"]


def test_without_a_usable_directory_the_reasons_are_reported(tmp_path: Path):
    source = _report(tmp_path / "reports")

    result = copy_report(source, [str(tmp_path / "missing")], "777", "ib")

    assert result["status"] == "not_delivered"
    assert result["attempts"] == [{"root": str(tmp_path / "missing"), "reason": "not a directory on this host"}]


def test_the_source_comes_from_the_report_layout_not_from_the_record(tmp_path: Path):
    calls = []
    record = {"user": ME, "job_id": "777", "name": "oob", "work_dir": "/work/run", "source": "/etc"}

    deliver(record, tmp_path / "reports", runner=lambda user, function, *args: calls.append((user, args)) or {})

    ((user, (source, roots, job_id, name)),) = calls
    assert user == ME
    assert source == tmp_path / "reports" / ME / "777" / "oob"
    assert roots == ["/work/run", pwd.getpwnam(ME).pw_dir]
    assert (job_id, name) == ("777", "oob")


def test_unsafe_names_are_rejected_before_any_identity_switch(tmp_path: Path):
    def runner(*_args):
        raise AssertionError("must not run")

    for record in (
        {"user": ME, "job_id": "..", "name": "ib"},
        {"user": ME, "job_id": "777", "name": "../../etc"},
        {"user": "a/b", "job_id": "777", "name": "ib"},
    ):
        assert deliver(record, tmp_path, runner=runner)["status"] == "not_delivered"


def test_identity_rules_without_root(monkeypatch, tmp_path: Path):
    assert run_as_user(ME, lambda: {"status": "delivered"}) == {"status": "delivered"}  # already the user
    assert run_as_user("root", lambda: {})["reason"] == "refusing to deliver as root"

    other = pwd.struct_passwd(("other", "x", os.geteuid() + 1, os.getegid(), "", str(tmp_path), "/bin/sh"))
    monkeypatch.setattr(delivery.pwd, "getpwnam", lambda name: other)
    result = run_as_user("other", lambda: {"status": "delivered"})
    assert result["status"] == "not_delivered" and "must run as root" in result["reason"]


def test_queue_records_settle_with_their_outcome(tmp_path: Path):
    queue = tmp_path / "queue"
    reports = tmp_path / "reports"
    _report(reports, user=ME)
    work = tmp_path / "work"
    work.mkdir()
    enqueue_delivery(queue, cluster="repacss", job_id="777", user=ME, name="ib", work_dir=str(work))
    enqueue_delivery(queue, cluster="repacss", job_id="777", user=ME, name="ib", work_dir=str(work))  # no-op
    enqueue_delivery(queue, cluster="repacss", job_id="778", user=ME, name="oob", work_dir=str(work))

    assert process_queue(queue, reports) == 2

    done = read_json_file(queue / "deliver" / COMPLETE / "repacss-777-ib.json")
    assert done["outcome"] == {"status": "delivered", "path": str(work / "power_reports" / "777" / "ib")}
    failed = read_json_file(queue / "deliver" / FAILED / "repacss-778-oob.json")
    assert failed["outcome"]["status"] == "not_delivered"  # job 778 has no report to copy
    assert not list((queue / "deliver" / PENDING).iterdir())


def test_only_oob_reports_in_the_report_layout_are_queued(tmp_path: Path):
    env = {"REPACSS_POWER_IB_QUEUE_ROOT": str(tmp_path / "queue"), "SLURM_JOB_WORK_DIR": "/work/run"}

    enqueue_oob_delivery("repacss", "777", "alice", tmp_path / "output" / "tmp", env)
    assert not (tmp_path / "queue").exists()

    enqueue_oob_delivery("repacss", "777", "alice", tmp_path / "reports" / "alice" / "777" / "oob", env)
    record = read_json_file(tmp_path / "queue" / "deliver" / PENDING / "repacss-777-oob.json")
    assert (record["user"], record["name"], record["work_dir"]) == ("alice", "oob", "/work/run")


def test_a_queue_that_cannot_be_written_never_breaks_the_oob_handler(tmp_path: Path):
    blocker = tmp_path / "queue"
    blocker.write_text("not a directory\n", encoding="utf-8")
    env = {"REPACSS_POWER_IB_QUEUE_ROOT": str(blocker)}

    enqueue_oob_delivery("repacss", "777", "alice", tmp_path / "reports" / "alice" / "777" / "oob", env)


def _pretend_root(monkeypatch, tmp_path: Path) -> None:
    """Take the root branch of run_as_user without root: identity switches become no-ops."""
    other = pwd.struct_passwd(("other", "x", os.geteuid() + 1, os.getegid(), "", str(tmp_path), "/bin/sh"))
    monkeypatch.setattr(delivery.pwd, "getpwnam", lambda name: other)
    monkeypatch.setattr(delivery.os, "geteuid", lambda: 0)
    for name in ("setgroups", "setgid", "setuid"):
        monkeypatch.setattr(delivery.os, name, lambda *_args: None)
    monkeypatch.setattr(delivery.os, "getgrouplist", lambda *_args: [])


def test_the_forked_delivery_returns_the_child_result(monkeypatch, tmp_path: Path):
    _pretend_root(monkeypatch, tmp_path)
    source = _report(tmp_path / "reports")
    work = tmp_path / "work"
    work.mkdir()

    result = run_as_user("other", copy_report, source, [str(work)], "777", "ib")

    assert result == {"status": "delivered", "path": str(work / "power_reports" / "777" / "ib")}


def test_a_delivery_that_hangs_is_abandoned_after_the_timeout(monkeypatch, tmp_path: Path):
    _pretend_root(monkeypatch, tmp_path)

    def hang():
        import time

        time.sleep(30)
        return {}

    result = run_as_user("other", hang, timeout=0.3)

    assert result == {"status": "not_delivered", "reason": "delivery did not finish within 0.3 s"}
