"""Slurm-facing helpers for starting and stopping in-band collection."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
from pathlib import Path
from typing import Dict, Optional

from inband.storage import (
    DEFAULT_IB_STORE_ROOT,
    DEFAULT_STATE_DIR,
    build_storage_key,
    done_marker_path,
    epoch_cache_path,
    ensure_node_staging,
    job_state_file_path,
    load_env_file,
    node_status_path,
    read_json_file,
    runner_status_path,
    unit_env_file_path,
    write_env_file,
    write_systemd_env_file,
    write_json_file,
)
from shared.slurm.comments import parse_power_comment
from shared.slurm.resolver import fetch_job_metadata, parse_job_start_epoch

LOGGER = logging.getLogger("repacss_power.inband.hooks")

def _configure_logging() -> None:
    level = logging.DEBUG if os.environ.get("MONSTER_POWER_DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _required_env(name: str, env: Dict[str, str]) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _read_cached_epoch(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _resolve_job_start_epoch(
    *,
    env: Dict[str, str],
    metadata: Dict[str, str],
    job_id: str,
    cluster_name: str,
    ib_store_root: Path,
) -> int:
    cache_path = epoch_cache_path(job_id, cluster_name=cluster_name, ib_store_root=ib_store_root)
    cached = _read_cached_epoch(cache_path)
    if cached is not None:
        return cached

    raw_candidates = [
        (env.get("SLURM_JOB_START_TIME") or "").strip(),
        metadata.get("StartTime", ""),
    ]
    candidate: Optional[int] = None
    for raw in raw_candidates:
        parsed = parse_job_start_epoch(raw)
        if parsed is not None:
            candidate = parsed
            break

    if candidate is None:
        candidate = int(job_id)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("x", encoding="utf-8") as handle:
            handle.write(f"{candidate}\n")
    except FileExistsError:
        cached = _read_cached_epoch(cache_path)
        if cached is not None:
            return cached
    return candidate


def _hostname(env: Dict[str, str]) -> str:
    return (env.get("SLURMD_NODENAME") or env.get("HOSTNAME") or socket.gethostname()).strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_executable(env: Dict[str, str]) -> str:
    return (env.get("MONSTER_POWER_PYTHON") or os.environ.get("MONSTER_POWER_PYTHON") or "python3").strip()


def _run_systemctl(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    systemctl = os.environ.get("MONSTER_POWER_SYSTEMCTL", "systemctl")
    return subprocess.run([systemctl, *args], check=check, capture_output=True, text=True)


def _build_base_state(env: Dict[str, str]) -> Optional[Dict[str, str]]:
    job_id = _required_env("SLURM_JOB_ID", env)
    job_user = (env.get("SLURM_JOB_USER") or "").strip()
    metadata = fetch_job_metadata(job_id, env)
    comment = metadata.get("Comment", "")
    parsed = parse_power_comment(comment)
    if not parsed.includes_inband:
        LOGGER.info("job %s has no in-band mode in comment; skipping hook", job_id)
        return None

    cluster_name = (env.get("SLURM_CLUSTER_NAME") or "cluster").strip() or "cluster"
    hostname = _hostname(env)
    ib_store_root = Path(env.get("MONSTER_POWER_IB_STORE_ROOT") or DEFAULT_IB_STORE_ROOT)
    start_epoch = _resolve_job_start_epoch(
        env=env,
        metadata=metadata,
        job_id=job_id,
        cluster_name=cluster_name,
        ib_store_root=ib_store_root,
    )
    storage_key = build_storage_key(job_id, start_epoch, cluster_name=cluster_name)
    node_root = ensure_node_staging(storage_key, hostname, ib_store_root=ib_store_root)
    collectors = ",".join(parsed.collectors)
    state = {
        "JOB_ID": job_id,
        "JOB_USER": job_user,
        "CLUSTER_NAME": cluster_name,
        "HOSTNAME": hostname,
        "JOB_START_EPOCH": str(start_epoch),
        "STORAGE_KEY": storage_key,
        "IB_STORE_ROOT": str(ib_store_root),
        "NODE_OUTPUT_DIR": str(node_root),
        "INTERVAL_MS": str(parsed.interval_ms),
        "COLLECTORS": collectors,
        "SYSTEMD_UNIT": f"repacss-power-ib@{storage_key}.service",
        "PYTHONPATH": env.get("PYTHONPATH") or str(_repo_root()),
        "REPACSS_POWER_PYTHON": _python_executable(env),
        "REPACSS_POWER_JOB_ID": job_id,
        "REPACSS_POWER_STORAGE_KEY": storage_key,
        "REPACSS_POWER_OUTPUT_DIR": str(node_root),
        "REPACSS_POWER_HOSTNAME": hostname,
        "REPACSS_POWER_INTERVAL_MS": str(parsed.interval_ms),
        "REPACSS_POWER_COLLECTORS": collectors,
        "REPACSS_POWER_RUNNER_STATUS_PATH": str(runner_status_path(storage_key, hostname, ib_store_root=ib_store_root)),
        "REPACSS_POWER_NODE_STATUS_PATH": str(node_status_path(storage_key, hostname, ib_store_root=ib_store_root)),
    }
    return state


def run_prolog(env: Optional[Dict[str, str]] = None) -> int:
    env = dict(os.environ if env is None else env)
    state = _build_base_state(env)
    if state is None:
        return 0

    state_dir = Path(env.get("MONSTER_POWER_STATE_DIR") or DEFAULT_STATE_DIR)
    write_env_file(job_state_file_path(state["JOB_ID"], state_dir=state_dir), state)
    write_systemd_env_file(unit_env_file_path(state["STORAGE_KEY"], state_dir=state_dir), state)

    LOGGER.info(
        "starting in-band unit %s for job %s on host %s",
        state["SYSTEMD_UNIT"],
        state["JOB_ID"],
        state["HOSTNAME"],
    )
    _run_systemctl(["start", state["SYSTEMD_UNIT"]], check=True)
    return 0


def _load_state_for_epilog(env: Dict[str, str]) -> Dict[str, str]:
    job_id = _required_env("SLURM_JOB_ID", env)
    state_dir = Path(env.get("MONSTER_POWER_STATE_DIR") or DEFAULT_STATE_DIR)
    state = load_env_file(job_state_file_path(job_id, state_dir=state_dir))
    if state:
        return state

    LOGGER.warning("state file for job %s is missing; falling back to derived values", job_id)
    fallback = _build_base_state(env)
    if fallback is None:
        return {}
    return fallback


def _systemd_unit_status(unit: str) -> Dict[str, str]:
    result = _run_systemctl(["show", unit, "--property=Result,ExecMainStatus,SubState"], check=False)
    payload = {"result": "unknown", "exec_main_status": "", "sub_state": ""}
    if result.returncode != 0:
        payload["result"] = "systemctl-show-failed"
        return payload
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "Result":
            payload["result"] = value
        elif key == "ExecMainStatus":
            payload["exec_main_status"] = value
        elif key == "SubState":
            payload["sub_state"] = value
    return payload


def run_epilog(env: Optional[Dict[str, str]] = None) -> int:
    env = dict(os.environ if env is None else env)
    state = _load_state_for_epilog(env)
    if not state:
        return 0

    ib_store_root = Path(state["IB_STORE_ROOT"])
    storage_key = state["STORAGE_KEY"]
    hostname = state["HOSTNAME"]
    unit = state["SYSTEMD_UNIT"]

    stop_result = _run_systemctl(["stop", unit], check=False)
    unit_status = _systemd_unit_status(unit)
    runner_status = read_json_file(runner_status_path(storage_key, hostname, ib_store_root=ib_store_root))

    node_state = runner_status.get("state", "failed")
    message = runner_status.get("message", "")
    if not runner_status:
        node_state = "failed"
        message = "runner_status.json was not written before epilog finalization"
    elif stop_result.returncode != 0 or unit_status.get("exec_main_status") not in {"", "0"}:
        node_state = "failed"
        message = message or f"systemd unit {unit} exited abnormally"

    node_status = {
        "job_id": state["JOB_ID"],
        "storage_key": storage_key,
        "hostname": hostname,
        "state": node_state,
        "message": message,
        "requested_collectors": _split_csv(state.get("COLLECTORS", "")),
        "runner_status": runner_status,
        "systemd": {
            "stop_returncode": stop_result.returncode,
            "result": unit_status.get("result"),
            "exec_main_status": unit_status.get("exec_main_status"),
            "sub_state": unit_status.get("sub_state"),
        },
    }
    write_json_file(node_status_path(storage_key, hostname, ib_store_root=ib_store_root), node_status)
    done_marker = done_marker_path(storage_key, hostname, ib_store_root=ib_store_root)
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.touch()
    return 0


def _split_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="REPACSS in-band Slurm hooks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prolog")
    subparsers.add_parser("epilog")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    _configure_logging()
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prolog":
            return run_prolog()
        if args.command == "epilog":
            return run_epilog()
    except Exception as exc:  # pragma: no cover - shell-facing defensive wrapper
        LOGGER.exception("in-band hook failure: %s", exc)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
