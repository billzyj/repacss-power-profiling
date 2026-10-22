"""Controller-side finalization of in-band (IB) job attempts.

`enqueue_inband_job` runs inside EpilogSlurmctld through the dispatcher and only
writes a queue record. The finalizer (`run`, as a service or with `--once`) checks
each pending record against shared staging. Once every expected node has published
`.done`, or the record's deadline has passed, it aggregates the staged raw counter
readings and exports:

    <report_root>/<user>/<job_id>/ib/
        raw_power.csv     one row per sample interval and domain: energy (J) and power (W)
        summary.json      energy, average power, node coverage, status, and notes
        provenance.json   request, node publication records, RAPL manifests

The OOB report of the same job goes to `<report_root>/<user>/<job_id>/oob/`. A later
attempt of a requeued job exports to `ib-<attempt start epoch>/` beside the first. The
export is written to a sibling `.partial` directory and renamed into place. An existing
export is never rewritten. Each export is then queued for delivery into the user's own
directory (see `shared.slurm.delivery`); the copy under the report root stays.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from inband.aggregation import COUNTER_COLLECTORS, ENERGY_METHOD, EXPORT_COLUMNS, Window, aggregate_attempt
from inband.hooks import EXCLUSIVE_NODE, job_exclusive, storage_key_from_env
from inband.stager import DONE_FILENAME, NODE_STATUS_FILENAME
from inband.storage import DEFAULT_IB_STORE_ROOT, read_json_file, resolve_storage_key_for_job, write_json_file
from shared.models import ParsedPowerComment, SlurmJobContext
from shared.slurm.delivery import enqueue_delivery
from shared.slurm.finalize_queue import COMPLETE, DEFAULT_QUEUE_ROOT, FAILED, enqueue, pending, settle
from shared.slurm.resolver import expand_nodelist, parse_job_start_epoch

LOGGER = logging.getLogger("repacss_power.slurm.finalizer")

DEFAULT_REPORT_ROOT = Path(
    os.environ.get("REPACSS_POWER_REPORT_ROOT") or os.environ.get("MONSTER_POWER_BASE") or "/var/lib/repacss-power/reports"
)
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("REPACSS_POWER_IB_FINALIZE_TIMEOUT_SECONDS", "600"))
IB_DIRNAME = "ib"
# Optional Slurm hostlist of the nodes whose hooks collect in-band data, such as
# "rpc-97-[1-4]". Unset means every node. Read by EpilogSlurmctld when it queues a job.
IB_NODES_ENV = "REPACSS_POWER_IB_NODES"


@dataclass(frozen=True)
class FinalizerConfig:
    queue_root: Path = DEFAULT_QUEUE_ROOT
    staging_root: Path = DEFAULT_IB_STORE_ROOT
    report_root: Path = DEFAULT_REPORT_ROOT


def _iso(epoch: Optional[float]) -> Optional[str]:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat() if epoch is not None else None


def build_record(
    *,
    job_id: str,
    attempt_id: str,
    cluster: str,
    user: str,
    nodes: Sequence[str],
    start_epoch: Optional[int],
    end_epoch: Optional[int],
    now: float,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    comment: str = "",
    mode: str = "inband",
    collectors: Sequence[str] = (),
    interval_ms: int = 1000,
    exclusive: str = "",
    work_dir: str = "",
    unsupported_nodes: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "repacss_finalize_request",
        "job_id": job_id,
        "attempt_id": attempt_id,
        "cluster": cluster,
        "user": user,
        "nodes": list(nodes),
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        "comment": comment,
        "mode": mode,
        "collectors": list(collectors),
        "interval_ms": interval_ms,
        "exclusive": exclusive,
        "work_dir": work_dir,
        "unsupported_nodes": list(unsupported_nodes),
        "enqueued_at": _iso(now),
        "deadline_epoch": now + timeout_seconds,
    }


def unsupported_nodes(nodes: Sequence[str], env: Mapping[str, str]) -> List[str]:
    """Nodes outside the site's in-band host list; with no list, every node collects."""
    enabled = set(expand_nodelist(env.get(IB_NODES_ENV, "")))
    return [node for node in nodes if node not in enabled] if enabled else []


def enqueue_inband_job(
    context: SlurmJobContext,
    parsed: ParsedPowerComment,
    env: Optional[Mapping[str, str]] = None,
    queue_root: Optional[Path] = None,
    now: Optional[float] = None,
) -> Path:
    """Queue one attempt for finalization. Called from EpilogSlurmctld; does no other work."""
    env = dict(os.environ if env is None else env)
    record = build_record(
        job_id=context.job_id,
        attempt_id=storage_key_from_env(env),
        cluster=context.cluster_name or "cluster",
        user=context.user,
        nodes=context.nodes,
        start_epoch=parse_job_start_epoch(env.get("SLURM_JOB_START_TIME")),
        end_epoch=parse_job_start_epoch(env.get("SLURM_JOB_END_TIME")),
        now=time.time() if now is None else now,
        comment=context.comment,
        mode=parsed.mode or "inband",
        collectors=parsed.collectors,
        interval_ms=parsed.interval_ms,
        exclusive=job_exclusive(env),
        work_dir=env.get("SLURM_JOB_WORK_DIR", ""),
        unsupported_nodes=unsupported_nodes(context.nodes, env),
    )
    return enqueue(queue_root or DEFAULT_QUEUE_ROOT, record)


def export_dir(config: FinalizerConfig, record: Mapping[str, Any]) -> Path:
    """`ib/` for the job's first exported attempt, `ib-<start epoch>/` for any later one."""
    job_dir = config.report_root / record["user"] / record["job_id"]
    first = job_dir / IB_DIRNAME
    exported = read_json_file(first / "summary.json").get("attempt_id")
    if exported in (None, record["attempt_id"]):
        return first
    return job_dir / f"{IB_DIRNAME}-{record['attempt_id'].rsplit('-', 1)[-1]}"


def _locate_job_dir(config: FinalizerConfig, record: Mapping[str, Any]) -> Tuple[Path, List[str]]:
    job_dir = config.staging_root / record["attempt_id"]
    if job_dir.exists():
        return job_dir, []
    # The controller and the compute nodes derive the key from the same Slurm start
    # time; fall back to the newest key for this job if they ever disagree. The listing
    # can also find the expected key itself, when an NFS client still caches the
    # directory as missing; that is no disagreement and needs no note.
    found = resolve_storage_key_for_job(record["job_id"], ib_store_root=config.staging_root, cluster_name=record["cluster"])
    if found == record["attempt_id"]:
        return config.staging_root / found, []
    if found:
        return config.staging_root / found, [f"Staging directory {record['attempt_id']} not found; used {found}."]
    return job_dir, []


def _unaggregated_collector_notes(job_dir: Path, nodes: Sequence[str]) -> List[str]:
    collectors = set()
    for node in nodes:
        collectors.update(read_json_file(job_dir / node / NODE_STATUS_FILENAME).get("collectors", {}))
    collectors.difference_update(COUNTER_COLLECTORS)
    return [f"{name} samples were published but are not aggregated yet." for name in sorted(collectors)]


def _write_export(config: FinalizerConfig, record: Mapping[str, Any], job_dir: Path, notes: List[str]) -> Dict[str, Any]:
    target = export_dir(config, record)
    partial = target.with_name(f".{target.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)  # left by an interrupted export of this same attempt
    partial.mkdir(parents=True)

    window = Window(
        start=datetime.fromtimestamp(record["start_epoch"], tz=timezone.utc) if record.get("start_epoch") else None,
        end=datetime.fromtimestamp(record["end_epoch"], tz=timezone.utc) if record.get("end_epoch") else None,
    )
    with (partial / "raw_power.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        result = aggregate_attempt(job_dir, expected_nodes=record["nodes"], window=window, emit=writer.writerow)

    finalized_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": 2,
        "kind": "repacss_inband_summary",
        "job_id": record["job_id"],
        "attempt_id": record["attempt_id"],
        "user": record["user"],
        "status": result["status"],
        "window": {"start": _iso(record.get("start_epoch")), "end": _iso(record.get("end_epoch")), "seconds": window.seconds},
        "nodes": {"expected": record["nodes"], "missing": result["missing_nodes"]},
        "energy_j": result["energy_j"],
        "average_power_w": result["average_power_w"],
        "per_node": result["nodes"],
        "method": ENERGY_METHOD,
        "attribution": "Whole-node RAPL and all-GPU NVML readings; job energy assumes exclusive node allocation.",
        "notes": notes + result["notes"] + _unaggregated_collector_notes(job_dir, record["nodes"]),
        "finalized_at": finalized_at,
    }
    provenance = {
        "schema_version": 1,
        "kind": "repacss_inband_provenance",
        "request": {key: record.get(key) for key in ("comment", "mode", "collectors", "interval_ms")},
        "queue_record": dict(record),
        "staging_dir": str(job_dir),
        "nodes": {
            node: {
                "node_status": read_json_file(job_dir / node / NODE_STATUS_FILENAME),
                **{
                    f"{collector}_manifest": read_json_file(job_dir / node / f"{collector}_manifest.json")
                    for collector in COUNTER_COLLECTORS
                },
            }
            for node in record["nodes"]
        },
        "finalizer_host": socket.gethostname(),
        "exported_at": finalized_at,
    }
    write_json_file(partial / "summary.json", summary)
    write_json_file(partial / "provenance.json", provenance)
    os.replace(partial, target)
    return {"status": summary["status"], "export_dir": str(target), "finalized_at": finalized_at}


def _write_skipped_export(config: FinalizerConfig, record: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    """Tell the user why no in-band data exists, instead of waiting for nodes that never start."""
    target = export_dir(config, record)
    partial = target.with_name(f".{target.name}.partial")
    if partial.exists():
        shutil.rmtree(partial)
    finalized_at = datetime.now(timezone.utc).isoformat()
    write_json_file(
        partial / "summary.json",
        {
            "schema_version": 2,
            "kind": "repacss_inband_summary",
            "job_id": record["job_id"],
            "attempt_id": record["attempt_id"],
            "user": record["user"],
            "status": "skipped",
            "reason": reason,
            "finalized_at": finalized_at,
        },
    )
    os.replace(partial, target)
    return {"status": "skipped", "export_dir": str(target), "finalized_at": finalized_at}


def try_finalize(config: FinalizerConfig, record: Mapping[str, Any], now: float) -> Optional[Dict[str, Any]]:
    """Export the attempt, or return None while nodes are still publishing before the deadline."""
    target = export_dir(config, record)
    if target.exists():
        return {"status": "already_exported", "export_dir": str(target)}
    if record.get("exclusive") != EXCLUSIVE_NODE:
        exclusive = record.get("exclusive") or "no exclusivity setting"
        return _write_skipped_export(
            config,
            record,
            f"In-band collection requires --exclusive; Slurm reported {exclusive} for this job.",
        )
    if record.get("unsupported_nodes"):
        return _write_skipped_export(
            config,
            record,
            f"In-band collection is not enabled on {', '.join(record['unsupported_nodes'])}, so this job has no in-band data.",
        )
    job_dir, notes = _locate_job_dir(config, record)
    waiting = [node for node in record["nodes"] if not (job_dir / node / DONE_FILENAME).exists()]
    if waiting and now < record["deadline_epoch"]:
        return None
    return _write_export(config, record, job_dir, notes)


def process_queue(config: FinalizerConfig, now: Optional[float] = None) -> int:
    """Finalize every pending record that is ready. Returns how many were settled."""
    now = time.time() if now is None else now
    settled = 0
    for path in pending(config.queue_root):
        record: Dict[str, Any] = {}
        try:
            record = read_json_file(path)
            outcome = try_finalize(config, record, now)
        except Exception as exc:  # one bad record must not stop the others
            LOGGER.exception("finalization of %s failed", path.name)
            settle(config.queue_root, path, FAILED, {**record, "outcome": {"status": "error", "error": str(exc)}})
            settled += 1
            continue
        if outcome is None:
            continue
        settle(config.queue_root, path, COMPLETE, {**record, "outcome": outcome})
        LOGGER.info("job %s attempt %s: %s -> %s", record["job_id"], record["attempt_id"], outcome["status"], outcome["export_dir"])
        settled += 1
        try:
            enqueue_delivery(
                config.queue_root,
                cluster=record["cluster"],
                job_id=record["job_id"],
                user=record["user"],
                name=Path(outcome["export_dir"]).name,
                work_dir=record.get("work_dir", ""),
            )
        except OSError as exc:
            LOGGER.warning("could not queue delivery of job %s: %s", record["job_id"], exc)
    return settled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize in-band job attempts into user reports.")
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_IB_STORE_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    commands = parser.add_subparsers(dest="command")

    run = commands.add_parser("run", help="Process the queue (the default command).")
    run.add_argument("--once", action="store_true", help="Process the queue once and exit.")
    run.add_argument("--poll-seconds", type=float, default=15.0)

    manual = commands.add_parser("enqueue", help="Queue an attempt by hand, for testing without Slurm.")
    manual.add_argument("--job-id", required=True)
    manual.add_argument("--attempt-id", required=True, help="Storage key of the staged attempt.")
    manual.add_argument("--user", required=True)
    manual.add_argument("--node", action="append", required=True, help="Expected node; repeat for each node.")
    manual.add_argument("--start", default="", help="Job start as epoch seconds or ISO time; omit to skip clipping.")
    manual.add_argument("--end", default="", help="Job end as epoch seconds or ISO time; omit to skip clipping.")
    manual.add_argument("--timeout-seconds", type=float, default=0.0, help="Wait this long for missing nodes.")
    manual.add_argument("--exclusive", default=EXCLUSIVE_NODE, help="Job exclusivity as hooks.job_exclusive reports it.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    level = logging.DEBUG if os.environ.get("MONSTER_POWER_DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    config = FinalizerConfig(queue_root=args.queue_root, staging_root=args.staging_root, report_root=args.report_root)

    if args.command == "enqueue":
        record = build_record(
            job_id=args.job_id,
            attempt_id=args.attempt_id,
            cluster="cluster",
            user=args.user,
            nodes=args.node,
            start_epoch=parse_job_start_epoch(args.start),
            end_epoch=parse_job_start_epoch(args.end),
            now=time.time(),
            timeout_seconds=args.timeout_seconds,
            exclusive=args.exclusive,
        )
        print(enqueue(config.queue_root, record))
        return 0

    once = getattr(args, "once", False)
    poll_seconds = getattr(args, "poll_seconds", 15.0)
    while True:
        process_queue(config)
        if once:
            return 0
        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
