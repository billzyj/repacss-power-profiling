"""Deliver finished reports into the job owner's own directory, acting as that user.

The OOB handler and the in-band finalizer, both running as SlurmUser, always write
their report to the shared report root first:

    <report_root>/<user>/<job_id>/oob/      <report_root>/<user>/<job_id>/ib/

That copy stays as the fallback. Each report directory is then queued here, and the
delivery service copies it into the user's space:

    <work_dir>/power_reports/<job_id>/<name>/   the job's working directory, or
    <home>/power_reports/<job_id>/<name>/       when the working directory is unusable

SlurmUser cannot write there, so the service runs as root, only to become the user.
Each delivery forks a child that switches to the user's uid, gid, and groups before it
touches any path, and the child itself reads the shared copy and writes the
destination. Root never opens a report file. The queue is writable by SlurmUser, so a
record supplies only names: the source is always `<report_root>/<user>/<job_id>/<name>`,
and a forged record can at most make a user copy files that user can already read.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pwd
import re
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from shared.slurm.finalize_queue import COMPLETE, DEFAULT_QUEUE_ROOT, FAILED, enqueue, pending, settle

LOGGER = logging.getLogger("repacss_power.slurm.delivery")

DELIVERY_QUEUE_DIRNAME = "deliver"
DELIVERY_DIRNAME = "power_reports"
DEFAULT_REPORT_ROOT = Path(
    os.environ.get("REPACSS_POWER_REPORT_ROOT") or os.environ.get("MONSTER_POWER_BASE") or "/var/lib/repacss-power/reports"
)
DEFAULT_TIMEOUT_SECONDS = 120.0
REPORT_NAME = re.compile(r"(oob|ib)(-[0-9]+)?")
SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.-]+")
DELIVERED = ("delivered", "already_delivered")

Runner = Callable[..., Dict[str, Any]]


def delivery_queue(queue_root: Path) -> Path:
    return queue_root / DELIVERY_QUEUE_DIRNAME


def enqueue_delivery(
    queue_root: Path, *, cluster: str, job_id: str, user: str, name: str, work_dir: str = "", now: Optional[float] = None
) -> Path:
    """Queue one report directory for delivery. Queuing the same report again is a no-op."""
    record = {
        "schema_version": 1,
        "kind": "repacss_delivery_request",
        "cluster": cluster or "cluster",
        "job_id": job_id,
        "user": user,
        "name": name,
        "work_dir": work_dir,
        "enqueued_at": datetime.fromtimestamp(time.time() if now is None else now, tz=timezone.utc).isoformat(),
    }
    return enqueue(delivery_queue(queue_root), record, key=f"{record['cluster']}-{job_id}-{name}")


def enqueue_oob_delivery(cluster: Optional[str], job_id: str, user: str, out_dir: Path, env: Mapping[str, str]) -> None:
    """Queue an OOB report written to `<report_root>/<user>/<job_id>/oob`; never raises."""
    if out_dir.name != "oob":
        return  # written somewhere other than the report layout; nothing to deliver
    queue_root = Path(env.get("REPACSS_POWER_IB_QUEUE_ROOT") or DEFAULT_QUEUE_ROOT)
    try:
        enqueue_delivery(
            queue_root,
            cluster=cluster or env.get("SLURM_CLUSTER_NAME", ""),
            job_id=job_id,
            user=user,
            name="oob",
            work_dir=env.get("SLURM_JOB_WORK_DIR", ""),
        )
    except OSError as exc:
        LOGGER.warning("could not queue delivery of the OOB report for job %s: %s", job_id, exc)


def copy_report(source: Path, roots: Sequence[str], job_id: str, name: str) -> Dict[str, Any]:
    """Copy `source` to `<root>/power_reports/<job_id>/<name>` under the first usable root.

    Runs as the job user. An existing destination is left alone, so a report the user
    already received, moved, or edited is never overwritten.
    """
    if not source.is_dir():
        return {"status": "not_delivered", "reason": f"{source} is not a readable report directory"}
    attempts: List[Dict[str, str]] = []
    for root in roots:
        base = Path(root) if root else None
        if base is None or not base.is_absolute():
            continue
        if not base.is_dir():
            attempts.append({"root": root, "reason": "not a directory on this host"})
            continue
        job_dir = base / DELIVERY_DIRNAME / job_id
        target = job_dir / name
        if target.exists():
            return {"status": "already_delivered", "path": str(target)}
        partial = job_dir / f".{name}.partial"
        try:
            job_dir.mkdir(parents=True, exist_ok=True)
            if partial.exists():
                shutil.rmtree(partial)
            partial.mkdir()
            for path in sorted(source.rglob("*")):
                if path.is_symlink():
                    continue
                destination = partial / path.relative_to(source)
                if path.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif path.is_file():
                    shutil.copyfile(path, destination)
            os.replace(partial, target)
            return {"status": "delivered", "path": str(target)}
        except OSError as exc:
            shutil.rmtree(partial, ignore_errors=True)
            attempts.append({"root": root, "reason": str(exc)})
    return {"status": "not_delivered", "attempts": attempts}


def run_as_user(user: str, function: Callable[..., Dict[str, Any]], *args: Any, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Call `function(*args)` with the identity of `user` and return its result.

    As root, the call runs in a forked child that drops to the user before it starts.
    A process already running as the user calls it directly. Any other identity cannot
    act for the user, and root is never used to write.
    """
    entry = pwd.getpwnam(user)
    if entry.pw_uid == 0:
        return {"status": "not_delivered", "reason": "refusing to deliver as root"}
    if os.geteuid() == entry.pw_uid:
        return function(*args)
    if os.geteuid() != 0:
        return {"status": "not_delivered", "reason": "the delivery service must run as root to act as the job user"}

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: become the user, do the work, report through the pipe
        code = 1
        try:
            os.close(read_fd)
            os.setgroups(os.getgrouplist(user, entry.pw_gid))
            os.setgid(entry.pw_gid)
            os.setuid(entry.pw_uid)
            os.umask(0o022)
            os.chdir("/")
            result = function(*args)
            code = 0
        except BaseException as exc:  # report every failure; never return into the parent's code
            result = {"status": "not_delivered", "reason": f"{type(exc).__name__}: {exc}"}
        try:
            with os.fdopen(write_fd, "wb") as handle:
                handle.write(json.dumps(result).encode("utf-8"))
        finally:
            os._exit(code)

    os.close(write_fd)
    deadline = time.monotonic() + timeout
    while os.waitpid(pid, os.WNOHANG) == (0, 0):
        if time.monotonic() >= deadline:
            # A hung NFS mount can keep the child in uninterruptible sleep; reap it if it
            # dies within a second, and otherwise move on without it.
            os.kill(pid, signal.SIGKILL)
            for _ in range(10):
                if os.waitpid(pid, os.WNOHANG) != (0, 0):
                    break
                time.sleep(0.1)
            os.close(read_fd)
            return {"status": "not_delivered", "reason": f"delivery did not finish within {timeout:g} s"}
        time.sleep(0.1)
    with os.fdopen(read_fd, "rb") as handle:
        payload = handle.read()
    return json.loads(payload) if payload else {"status": "not_delivered", "reason": "delivery process ended without a result"}


def deliver(record: Mapping[str, Any], report_root: Path, runner: Runner = run_as_user) -> Dict[str, Any]:
    """Deliver one queued report directory, acting as its owner."""
    user, job_id, name = str(record.get("user", "")), str(record.get("job_id", "")), str(record.get("name", ""))
    if not (SAFE_TOKEN.fullmatch(user) and SAFE_TOKEN.fullmatch(job_id) and REPORT_NAME.fullmatch(name)):
        return {"status": "not_delivered", "reason": "invalid user, job id, or report name"}
    if job_id in {".", ".."} or user in {".", ".."}:
        return {"status": "not_delivered", "reason": "invalid user or job id"}
    try:
        home = pwd.getpwnam(user).pw_dir
    except KeyError:
        return {"status": "not_delivered", "reason": f"unknown user {user}"}
    source = report_root / user / job_id / name
    roots = [str(record.get("work_dir") or ""), home]
    return runner(user, copy_report, source, roots, job_id, name)


def process_queue(queue_root: Path, report_root: Path, runner: Runner = run_as_user) -> int:
    """Deliver every pending report. Returns how many records were settled."""
    queue = delivery_queue(queue_root)
    settled = 0
    for path in pending(queue):
        record: Dict[str, Any] = {}
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            outcome = deliver(record, report_root, runner)
        except Exception as exc:  # one bad record must not stop the others
            LOGGER.exception("delivery of %s failed", path.name)
            outcome = {"status": "error", "reason": str(exc)}
        state = COMPLETE if outcome.get("status") in DELIVERED else FAILED
        settle(queue, path, state, {**record, "outcome": outcome})
        LOGGER.info("delivery %s: %s %s", path.stem, outcome.get("status"), outcome.get("path") or outcome)
        settled += 1
    return settled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deliver finished power reports into each job owner's directory.")
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="Process the delivery queue (the default command).")
    run.add_argument("--once", action="store_true", help="Process the queue once and exit.")
    run.add_argument("--poll-seconds", type=float, default=15.0)

    manual = commands.add_parser("enqueue", help="Queue one report by hand, for testing.")
    manual.add_argument("--job-id", required=True)
    manual.add_argument("--user", required=True)
    manual.add_argument("--name", required=True, help="Report directory under <report_root>/<user>/<job_id>/: oob or ib.")
    manual.add_argument("--work-dir", default="", help="The job's working directory; omit to deliver to the home directory.")
    manual.add_argument("--cluster", default="cluster")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    level = logging.DEBUG if os.environ.get("MONSTER_POWER_DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "enqueue":
        print(
            enqueue_delivery(
                args.queue_root, cluster=args.cluster, job_id=args.job_id, user=args.user, name=args.name, work_dir=args.work_dir
            )
        )
        return 0
    while True:
        process_queue(args.queue_root, args.report_root)
        if getattr(args, "once", False):
            return 0
        time.sleep(getattr(args, "poll_seconds", 15.0))


if __name__ == "__main__":
    raise SystemExit(main())
