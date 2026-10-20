"""Long-running per-node in-band collector runner for Slurm/systemd."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from inband.spool import sample_path, write_seal
from inband.collectors import build_collectors
from inband.collectors.base import CollectorResult
from inband.storage import write_json_file
from shared.errors import InbandCollectorError

LOGGER = logging.getLogger("repacss_power.inband.runner")
RUNNER_STATUS_FILENAME = "runner_status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_collectors(raw: str) -> List[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _configure_logging() -> None:
    level = logging.DEBUG if os.environ.get("MONSTER_POWER_DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _status_payload(
    *,
    state: str,
    job_id: str,
    storage_key: str,
    hostname: str,
    requested_collectors: List[str],
    started_collectors: List[str],
    skipped_collectors: List[Dict[str, str]],
    collector_results: Optional[Dict[str, Dict[str, object]]] = None,
    message: str = "",
) -> Dict[str, object]:
    return {
        "state": state,
        "job_id": job_id,
        "storage_key": storage_key,
        "hostname": hostname,
        "requested_collectors": requested_collectors,
        "started_collectors": started_collectors,
        "skipped_collectors": skipped_collectors,
        "collector_results": collector_results or {},
        "message": message,
        "updated_at": _now(),
        "pid": os.getpid(),
    }


def _collector_result_payload(result: CollectorResult) -> Dict[str, object]:
    return {
        "status": result.status,
        "sample_count": result.sample_count,
        "output_path": str(result.output_path) if result.output_path else "",
        "message": result.message,
    }


def _coerce_args_from_env(args: argparse.Namespace) -> argparse.Namespace:
    args.job_id = args.job_id or os.environ.get("REPACSS_POWER_JOB_ID", "")
    args.storage_key = args.storage_key or os.environ.get("REPACSS_POWER_STORAGE_KEY", "")
    args.output_dir = args.output_dir or os.environ.get("REPACSS_POWER_OUTPUT_DIR", "")
    args.hostname = args.hostname or os.environ.get("REPACSS_POWER_HOSTNAME") or socket.gethostname()
    args.partition = args.partition or os.environ.get("REPACSS_POWER_PARTITION", "")
    args.interval_ms = args.interval_ms or int(os.environ.get("REPACSS_POWER_INTERVAL_MS", "1000"))
    if not args.collector:
        args.collector = tuple(_split_collectors(os.environ.get("REPACSS_POWER_COLLECTORS", "")))
    return args


def run_runner(args: argparse.Namespace) -> int:
    """Sample until SIGTERM or SIGINT, then seal the spool directory.

    Every exit path writes `runner_status.json` and then `node_sealed.json`, so the
    stager can publish the node without guessing whether the runner is finished.
    """
    args = _coerce_args_from_env(args)
    if not args.job_id:
        raise InbandCollectorError("job id is required for the in-band runner")
    if not args.storage_key:
        raise InbandCollectorError("storage_key is required for the in-band runner")
    if not args.output_dir:
        raise InbandCollectorError("output directory is required for the in-band runner")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / RUNNER_STATUS_FILENAME
    requested = list(args.collector) if args.collector else []
    collector_instances = build_collectors(requested or None, partition=args.partition, node=args.hostname)
    requested_names = requested or [collector.name for collector in collector_instances]

    skipped_collectors: List[Dict[str, str]] = []
    started_collectors: List[str] = []
    active = []
    stop_event = threading.Event()

    def write_status(
        state: str,
        collector_results: Optional[Dict[str, Dict[str, object]]] = None,
        message: str = "",
    ) -> None:
        write_json_file(
            status_path,
            _status_payload(
                state=state,
                job_id=args.job_id,
                storage_key=args.storage_key,
                hostname=args.hostname,
                requested_collectors=requested_names,
                started_collectors=started_collectors,
                skipped_collectors=skipped_collectors,
                collector_results=collector_results,
                message=message,
            ),
        )

    def finish(state: str, collector_results: Optional[Dict[str, Dict[str, object]]] = None, message: str = "") -> None:
        write_status(state, collector_results, message)
        write_seal(
            output_dir,
            job_id=args.job_id,
            storage_key=args.storage_key,
            hostname=args.hostname,
            runner_state=state,
        )

    def stop_active() -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        for active_collector, handle in active:
            result = active_collector.stop(handle)
            results[result.collector_name] = _collector_result_payload(result)
        return results

    def _signal_handler(signum, _frame) -> None:
        LOGGER.info("runner received signal %s, beginning graceful shutdown", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    write_status("starting")

    try:
        for collector in collector_instances:
            # Write startup provenance before the availability check so a skipped
            # collector still records why it was skipped.
            manifest = collector.manifest_payload()
            if manifest is not None:
                write_json_file(output_dir / f"{collector.name}_manifest.json", manifest)
            probe = collector.probe()
            if not probe.available:
                skipped_collectors.append({"name": collector.name, "reason": probe.reason})
                continue
            handle = collector.start(interval_ms=args.interval_ms, output_path=sample_path(output_dir, collector.name))
            active.append((collector, handle))
            started_collectors.append(collector.name)
    except Exception as exc:
        finish("failed", stop_active(), str(exc))
        raise

    for skipped in skipped_collectors:
        LOGGER.warning("collector %s not started (%s)", skipped["name"], skipped["reason"])

    if not active:
        LOGGER.warning("no requested collector started on %s; exiting without sampling", args.hostname)
        finish("partial", message="no requested collectors were available on this node")
        return 0

    write_status("running")
    LOGGER.info("sampling %s into %s until SIGTERM or SIGINT", ",".join(started_collectors), output_dir)

    try:
        while not stop_event.wait(timeout=1.0):
            pass
    finally:
        collector_results = stop_active()
        final_states = {str(result["status"]) for result in collector_results.values()}
        final_state = "complete"
        if "failed" in final_states:
            final_state = "failed"
        elif "partial" in final_states or (requested and skipped_collectors):
            # A default collector absent on this hardware (for example nvml on Zen4) is
            # expected; only a collector the job asked for makes the node partial.
            final_state = "partial"
        finish(final_state, collector_results)
        LOGGER.info("sealed %s with state %s", output_dir, final_state)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="REPACSS in-band runner")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--storage-key", default="")
    parser.add_argument("--output-dir", default="", help="Node-local spool directory for this job.")
    parser.add_argument("--hostname", default="")
    parser.add_argument("--partition", default="")
    parser.add_argument("--interval-ms", type=int, default=0)
    parser.add_argument("--collector", action="append", default=[])
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_runner(args)
    except InbandCollectorError as exc:
        LOGGER.error("runner failed: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive runner guard
        LOGGER.exception("unexpected runner failure: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
