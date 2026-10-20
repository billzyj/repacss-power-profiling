"""Slurm-facing helpers for starting and stopping in-band collection.

Neither hook touches shared storage. Prolog writes node-local state and starts the
runner, which samples into node-local spool. Epilog sends non-blocking requests to
stop the runner and to start the stager, then returns. The runner seals its spool
after SIGTERM, and the stager publishes the sealed inventory and `.done`.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from inband.chunks import DEFAULT_CHUNK_SECONDS
from inband.profiles import PROFILE_PATH_ENV
from inband.stager import ATTEMPT_FILENAME
from inband.storage import (
    DEFAULT_IB_STORE_ROOT,
    DEFAULT_LOCAL_SPOOL_ROOT,
    DEFAULT_STATE_DIR,
    build_storage_key,
    job_state_file_path,
    load_env_file,
    node_root_path,
    spool_node_path,
    unit_env_file_path,
    write_env_file,
    write_json_file,
    write_systemd_env_file,
)
from shared.errors import CommentParseError
from shared.slurm.comments import parse_power_comment
from shared.slurm.resolver import parse_job_start_epoch

LOGGER = logging.getLogger("repacss_power.inband.hooks")
SAFE_TOKEN = re.compile(r"[A-Za-z0-9_.-]+")
# RAPL and NVML measure the whole node, so collection requires `--exclusive`, which
# Slurm reports as SLURM_JOB_EXCLUSIVE=NODE.
EXCLUSIVE_NODE = "NODE"


def job_exclusive(env: Dict[str, str]) -> str:
    return (env.get("SLURM_JOB_EXCLUSIVE") or "").strip().upper()


def _configure_logging() -> None:
    level = logging.DEBUG if os.environ.get("MONSTER_POWER_DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _required_env(name: str, env: Dict[str, str]) -> str:
    value = (env.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _safe_token(name: str, value: str) -> str:
    """Reject values that could escape a directory or a systemd instance name."""
    if not SAFE_TOKEN.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"{name} {value!r} is not a safe path token")
    return value


def _job_start_epoch(env: Dict[str, str], job_id: str) -> int:
    """Every node of a job sees the same start time, so all nodes derive the same key."""
    parsed = parse_job_start_epoch((env.get("SLURM_JOB_START_TIME") or "").strip())
    return parsed if parsed is not None else int(job_id)


def storage_key_from_env(env: Dict[str, str]) -> str:
    """Derive the attempt's storage key from Slurm's environment.

    Compute Prolog and the controller's EpilogSlurmctld both call this, so the
    finalizer finds the directories that the compute nodes published.
    """
    job_id = _required_env("SLURM_JOB_ID", env)
    cluster_name = (env.get("SLURM_CLUSTER_NAME") or "cluster").strip() or "cluster"
    return _safe_token("storage key", build_storage_key(job_id, _job_start_epoch(env, job_id), cluster_name=cluster_name))


def _hostname(env: Dict[str, str]) -> str:
    return (env.get("SLURMD_NODENAME") or env.get("HOSTNAME") or socket.gethostname()).strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _python_executable(env: Dict[str, str]) -> str:
    return (env.get("MONSTER_POWER_PYTHON") or os.environ.get("MONSTER_POWER_PYTHON") or "python3").strip()


def _state_dir(env: Dict[str, str]) -> Path:
    return Path(env.get("MONSTER_POWER_STATE_DIR") or DEFAULT_STATE_DIR)


def _run_systemctl(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    systemctl = os.environ.get("MONSTER_POWER_SYSTEMCTL", "systemctl")
    return subprocess.run([systemctl, *args], check=check, capture_output=True, text=True)


def _build_base_state(env: Dict[str, str]) -> Optional[Dict[str, str]]:
    job_id = _required_env("SLURM_JOB_ID", env)
    job_user = (env.get("SLURM_JOB_USER") or "").strip()
    # Hooks read only the Slurm-provided environment; they never query slurmctld.
    comment = (env.get("SLURM_JOB_COMMENT") or "").strip()
    try:
        parsed = parse_power_comment(comment)
    except CommentParseError as exc:
        LOGGER.warning("job %s has an invalid power comment (%s); skipping hook", job_id, exc)
        return None
    if not parsed.includes_inband:
        LOGGER.info("job %s has no in-band mode in comment; skipping hook", job_id)
        return None
    if job_exclusive(env) != EXCLUSIVE_NODE:
        LOGGER.warning(
            "job %s requested in-band collection without --exclusive (SLURM_JOB_EXCLUSIVE=%s); skipping",
            job_id,
            job_exclusive(env) or "unset",
        )
        return None

    cluster_name = (env.get("SLURM_CLUSTER_NAME") or "cluster").strip() or "cluster"
    hostname = _safe_token("node name", _hostname(env))
    partition = (env.get("SLURM_JOB_PARTITION") or "").strip()
    start_epoch = _job_start_epoch(env, job_id)
    storage_key = storage_key_from_env(env)
    ib_store_root = Path(env.get("MONSTER_POWER_IB_STORE_ROOT") or DEFAULT_IB_STORE_ROOT)
    spool_root = Path(env.get("REPACSS_POWER_IB_LOCAL_SPOOL_ROOT") or DEFAULT_LOCAL_SPOOL_ROOT)
    spool_dir = spool_node_path(storage_key, hostname, spool_root=spool_root)
    staging_dir = node_root_path(storage_key, hostname, ib_store_root=ib_store_root)
    chunk_seconds = float(env.get("REPACSS_POWER_IB_CHUNK_SECONDS") or DEFAULT_CHUNK_SECONDS)
    runner_unit = f"repacss-power-ib@{storage_key}.service"
    collectors = ",".join(parsed.collectors)
    state = {
        "JOB_ID": job_id,
        "JOB_USER": job_user,
        "CLUSTER_NAME": cluster_name,
        "HOSTNAME": hostname,
        "PARTITION": partition,
        "JOB_START_EPOCH": str(start_epoch),
        "STORAGE_KEY": storage_key,
        "IB_STORE_ROOT": str(ib_store_root),
        "SPOOL_DIR": str(spool_dir),
        "STAGING_DIR": str(staging_dir),
        "INTERVAL_MS": str(parsed.interval_ms),
        "COLLECTORS": collectors,
        "SYSTEMD_UNIT": runner_unit,
        "STAGE_UNIT": f"repacss-power-stage@{storage_key}.service",
        "PYTHONPATH": env.get("PYTHONPATH") or str(_repo_root()),
        "REPACSS_POWER_PYTHON": _python_executable(env),
        "REPACSS_POWER_JOB_ID": job_id,
        "REPACSS_POWER_STORAGE_KEY": storage_key,
        "REPACSS_POWER_OUTPUT_DIR": str(spool_dir),
        "REPACSS_POWER_STAGING_DIR": str(staging_dir),
        "REPACSS_POWER_HOSTNAME": hostname,
        "REPACSS_POWER_PARTITION": partition,
        "REPACSS_POWER_INTERVAL_MS": str(parsed.interval_ms),
        "REPACSS_POWER_CHUNK_SECONDS": str(chunk_seconds),
        "REPACSS_POWER_COLLECTORS": collectors,
        "REPACSS_POWER_RUNNER_UNIT": runner_unit,
        "REPACSS_POWER_STATE_DIR": str(_state_dir(env)),
    }
    profile_path = (env.get(PROFILE_PATH_ENV) or "").strip()
    if profile_path:
        state[PROFILE_PATH_ENV] = profile_path
    return state


def _attempt_record(state: Dict[str, str]) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "repacss_attempt",
        "job_id": state["JOB_ID"],
        "job_user": state["JOB_USER"],
        "cluster": state["CLUSTER_NAME"],
        "hostname": state["HOSTNAME"],
        "partition": state["PARTITION"],
        "storage_key": state["STORAGE_KEY"],
        "job_start_epoch": int(state["JOB_START_EPOCH"]),
        "collectors": _split_csv(state["COLLECTORS"]),
        "interval_ms": int(state["INTERVAL_MS"]),
        "chunk_seconds": float(state["REPACSS_POWER_CHUNK_SECONDS"]),
        "staging_dir": state["STAGING_DIR"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run_prolog(env: Optional[Dict[str, str]] = None) -> int:
    env = dict(os.environ if env is None else env)
    state = _build_base_state(env)
    if state is None:
        return 0

    state_dir = _state_dir(env)
    write_json_file(Path(state["SPOOL_DIR"]) / ATTEMPT_FILENAME, _attempt_record(state))
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


def run_epilog(env: Optional[Dict[str, str]] = None) -> int:
    env = dict(os.environ if env is None else env)
    job_id = _required_env("SLURM_JOB_ID", env)
    state = load_env_file(job_state_file_path(job_id, state_dir=_state_dir(env)))
    if not state:
        # The job did not request in-band collection here. Never guess a unit name.
        LOGGER.info("no in-band state for job %s on this node; nothing to stop", job_id)
        return 0

    # Both requests return immediately, so Epilog cannot time out waiting on them.
    for args in (["stop", "--no-block", state["SYSTEMD_UNIT"]], ["start", "--no-block", state.get("STAGE_UNIT", "")]):
        if not args[-1]:
            continue
        result = _run_systemctl(args, check=False)
        if result.returncode != 0:
            LOGGER.warning("systemctl %s failed: %s", " ".join(args), result.stderr.strip())
    LOGGER.info("requested stop of %s for job %s", state["SYSTEMD_UNIT"], job_id)
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
