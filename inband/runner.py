"""Long-running per-node in-band collector runner for Slurm/systemd."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from inband.collectors import build_collectors
from inband.collectors.base import CollectorResult
from inband.storage import write_json_file
from shared.errors import InbandCollectorError

LOGGER = logging.getLogger("repacss_power.inband.runner")


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
        "output_path": str(result.output_path),
        "message": result.message,
    }


def _coerce_args_from_env(args: argparse.Namespace) -> argparse.Namespace:
    args.job_id = args.job_id or os.environ.get("REPACSS_POWER_JOB_ID", "")
    args.storage_key = args.storage_key or os.environ.get("REPACSS_POWER_STORAGE_KEY", "")
    args.output_dir = args.output_dir or os.environ.get("REPACSS_POWER_OUTPUT_DIR", "")
    args.hostname = args.hostname or os.environ.get("REPACSS_POWER_HOSTNAME") or socket.gethostname()
    args.interval_ms = args.interval_ms or int(os.environ.get("REPACSS_POWER_INTERVAL_MS", "1000"))
    if not args.collector:
        args.collector = tuple(_split_collectors(os.environ.get("REPACSS_POWER_COLLECTORS", "")))
    if not args.status_path:
        args.status_path = os.environ.get("REPACSS_POWER_RUNNER_STATUS_PATH", "")
    return args


def run_runner(args: argparse.Namespace) -> int:
    args = _coerce_args_from_env(args)
    if not args.job_id:
        raise InbandCollectorError("job id is required for the in-band runner")
    if not args.storage_key:
        raise InbandCollectorError("storage_key is required for the in-band runner")
    if not args.output_dir:
        raise InbandCollectorError("output directory is required for the in-band runner")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(args.status_path) if args.status_path else output_dir / "runner_status.json"
    requested = list(args.collector) if args.collector else []
    collector_instances = build_collectors(requested or None)

    skipped_collectors: List[Dict[str, str]] = []
    started_collectors: List[str] = []
    active = []
    stop_event = threading.Event()

    def _signal_handler(signum, _frame) -> None:
        LOGGER.info("runner received signal %s, beginning graceful shutdown", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    write_json_file(
        status_path,
        _status_payload(
            state="starting",
            job_id=args.job_id,
            storage_key=args.storage_key,
            hostname=args.hostname,
            requested_collectors=requested or [collector.name for collector in collector_instances],
            started_collectors=[],
            skipped_collectors=[],
        ),
    )

    try:
        for collector in collector_instances:
            probe = collector.probe()
            if not probe.available:
                skipped_collectors.append({"name": collector.name, "reason": probe.reason})
                continue
            output_path = output_dir / f"{collector.name}.csv"
            handle = collector.start(interval_ms=args.interval_ms, output_path=output_path)
            active.append((collector, handle))
            started_collectors.append(collector.name)
    except Exception as exc:
        collector_results: Dict[str, Dict[str, object]] = {}
        for active_collector, handle in active:
            result = active_collector.stop(handle)
            collector_results[result.collector_name] = _collector_result_payload(result)
        write_json_file(
            status_path,
            _status_payload(
                state="failed",
                job_id=args.job_id,
                storage_key=args.storage_key,
                hostname=args.hostname,
                requested_collectors=requested or [collector.name for collector in collector_instances],
                started_collectors=started_collectors,
                skipped_collectors=skipped_collectors,
                collector_results=collector_results,
                message=str(exc),
            ),
        )
        raise

    if not active:
        write_json_file(
            status_path,
            _status_payload(
                state="partial",
                job_id=args.job_id,
                storage_key=args.storage_key,
                hostname=args.hostname,
                requested_collectors=requested or [collector.name for collector in collector_instances],
                started_collectors=[],
                skipped_collectors=skipped_collectors,
                message="no requested collectors were available on this node",
            ),
        )
        return 0

    write_json_file(
        status_path,
        _status_payload(
            state="running",
            job_id=args.job_id,
            storage_key=args.storage_key,
            hostname=args.hostname,
            requested_collectors=requested or [collector.name for collector in collector_instances],
            started_collectors=started_collectors,
            skipped_collectors=skipped_collectors,
        ),
    )

    try:
        while not stop_event.wait(timeout=1.0):
            pass
    finally:
        collector_results: Dict[str, Dict[str, object]] = {}
        final_states = set()
        for collector, handle in active:
            result = collector.stop(handle)
            collector_results[result.collector_name] = _collector_result_payload(result)
            final_states.add(result.status)

        final_state = "complete"
        if "failed" in final_states:
            final_state = "failed"
        elif "partial" in final_states or skipped_collectors:
            final_state = "partial"

        write_json_file(
            status_path,
            _status_payload(
                state=final_state,
                job_id=args.job_id,
                storage_key=args.storage_key,
                hostname=args.hostname,
                requested_collectors=requested or [collector.name for collector in collector_instances],
                started_collectors=started_collectors,
                skipped_collectors=skipped_collectors,
                collector_results=collector_results,
            ),
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="REPACSS in-band runner")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--storage-key", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--hostname", default="")
    parser.add_argument("--interval-ms", type=int, default=0)
    parser.add_argument("--status-path", default="")
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
