"""Publish one node's sealed in-band spool to shared staging.

Runs as `repacss-power-stage@<storage_key>.service`, started by Epilog right after
it asks the runner to stop. The runner writes only to node-local spool; this process
is the only one that touches NFS. It waits for `node_sealed.json`, copies every file
in the sealed inventory, verifies each SHA-256, writes `node_status.json`, and writes
`.done` last. `.done` means the node's sealed inventory is fully published, not that
every collector succeeded.

Every copy goes to a sibling `.partial` on the destination filesystem and is renamed
only after its digest matches, so readers never see a partial file. Rerunning the
stager after any failure is safe.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from inband.chunks import (
    CHUNK_SUFFIX,
    CHUNKS_DIRNAME,
    PARTIAL_SUFFIX,
    SEAL_FILENAME,
    SEAL_FORCED_PARTIAL,
    SEAL_NORMAL,
    fsync_directory,
    sha256_file,
    write_bytes_atomic,
    write_seal,
)
from inband.storage import DEFAULT_STATE_DIR, load_env_file, read_json_file, write_json_file
from shared.errors import InbandCollectorError

LOGGER = logging.getLogger("repacss_power.inband.stager")

ATTEMPT_FILENAME = "attempt.json"
NODE_STATUS_FILENAME = "node_status.json"
DONE_FILENAME = ".done"
RUNNER_STATUS_FILENAME = "runner_status.json"

# systemctl is-active states in which the runner may still write chunks or its seal.
RUNNER_ALIVE_STATES = {"active", "activating", "deactivating", "reloading"}
SEAL_CHECK_SECONDS = 2.0


class StageError(InbandCollectorError):
    """Raised when a copy cannot be verified or would overwrite different content."""


@dataclass
class StageConfig:
    spool_dir: Path
    staging_dir: Path
    runner_unit: str = ""
    wait_seconds: float = 120.0
    keep_local: bool = False
    state_dir: Optional[Path] = None


def copy_verified(source: Path, destination: Path, expected_sha256: Optional[str] = None) -> str:
    """Copy through a destination-side `.partial`, verify the digest, then rename.

    An existing destination with the expected digest is left alone. One with a
    different digest is an error: published files are immutable.
    """
    expected = expected_sha256 or sha256_file(source)
    if destination.exists():
        if sha256_file(destination) == expected:
            return expected
        raise StageError(f"{destination} already exists with different content")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
    with source.open("rb") as reader, partial.open("wb") as writer:
        shutil.copyfileobj(reader, writer, 1 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    actual = sha256_file(partial)
    if actual != expected:
        partial.unlink()
        raise StageError(f"checksum mismatch copying {source}: expected {expected}, got {actual}")
    os.replace(partial, destination)
    return expected


def runner_alive(unit: str) -> Optional[bool]:
    """Return whether the runner unit may still write, or None when no unit is known."""
    if not unit:
        return None
    systemctl = os.environ.get("MONSTER_POWER_SYSTEMCTL", "systemctl")
    result = subprocess.run([systemctl, "is-active", unit], capture_output=True, text=True, check=False)
    return result.stdout.strip() in RUNNER_ALIVE_STATES


def _identity(config: StageConfig) -> Dict[str, str]:
    attempt = read_json_file(config.spool_dir / ATTEMPT_FILENAME)
    return {
        "job_id": str(attempt.get("job_id") or os.environ.get("REPACSS_POWER_JOB_ID", "")),
        "storage_key": str(attempt.get("storage_key") or os.environ.get("REPACSS_POWER_STORAGE_KEY", "")),
        "hostname": str(attempt.get("hostname") or os.environ.get("REPACSS_POWER_HOSTNAME", "")),
    }


def write_forced_seal(config: StageConfig) -> Dict[str, Any]:
    """Seal a spool whose runner exited without sealing it.

    Call only after confirming the runner is no longer running. Open chunks are closed
    here, keeping only complete lines, because no process will write to them again.
    """
    for partial in sorted(config.spool_dir.glob(f"{CHUNKS_DIRNAME}/*/*{CHUNK_SUFFIX}{PARTIAL_SUFFIX}")):
        data = partial.read_bytes()
        complete_size = data.rfind(b"\n") + 1
        if data[:complete_size].count(b"\n") < 2:  # no row after the header
            partial.unlink()
            continue
        # Drop a line the runner was killed while writing, then close the chunk.
        with partial.open("r+b") as handle:
            handle.truncate(complete_size)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, partial.with_name(partial.name[: -len(PARTIAL_SUFFIX)]))
        fsync_directory(partial.parent)
    runner_status = read_json_file(config.spool_dir / RUNNER_STATUS_FILENAME)
    return write_seal(
        config.spool_dir,
        runner_state=str(runner_status.get("state") or "unknown"),
        seal_type=SEAL_FORCED_PARTIAL,
        **_identity(config),
    )


def wait_for_seal(config: StageConfig) -> Optional[Dict[str, Any]]:
    """Return the runner's seal, a forced seal if the runner died, or None on timeout."""
    deadline = time.monotonic() + config.wait_seconds
    while True:
        # Check the runner before the seal: a runner that exits after sealing must be
        # seen with its seal, never as exited without one.
        alive = runner_alive(config.runner_unit)
        seal = read_json_file(config.spool_dir / SEAL_FILENAME)
        if seal:
            return seal
        if alive is False:
            LOGGER.warning("runner %s exited without a seal; writing a forced seal", config.runner_unit)
            return write_forced_seal(config)
        if time.monotonic() >= deadline:
            return None
        time.sleep(SEAL_CHECK_SECONDS)


def node_status(seal: Dict[str, Any]) -> Dict[str, Any]:
    collectors: Dict[str, Dict[str, Any]] = {}
    for entry in seal["files"]:
        if "collector" not in entry:
            continue
        summary = collectors.setdefault(
            entry["collector"], {"chunks": 0, "rows": 0, "first_timestamp": "", "last_timestamp": ""}
        )
        summary["chunks"] += 1
        summary["rows"] += entry["rows"]
        summary["first_timestamp"] = summary["first_timestamp"] or entry["first_timestamp"]
        summary["last_timestamp"] = entry["last_timestamp"] or summary["last_timestamp"]
    state = seal["runner_state"] if seal["seal_type"] == SEAL_NORMAL else SEAL_FORCED_PARTIAL
    return {
        "schema_version": 1,
        "kind": "repacss_node_status",
        "job_id": seal["job_id"],
        "storage_key": seal["storage_key"],
        "hostname": seal["hostname"],
        "state": state,
        "seal_type": seal["seal_type"],
        "runner_state": seal["runner_state"],
        "published_at": datetime.now(timezone.utc).isoformat(),
        "collectors": collectors,
        "files": seal["files"],
    }


def publish(config: StageConfig, seal: Dict[str, Any]) -> None:
    """Copy and verify the sealed inventory, then write node status and `.done` last."""
    for entry in seal["files"]:
        copy_verified(config.spool_dir / entry["path"], config.staging_dir / entry["path"], entry["sha256"])
    copy_verified(config.spool_dir / SEAL_FILENAME, config.staging_dir / SEAL_FILENAME)
    write_json_file(config.staging_dir / NODE_STATUS_FILENAME, node_status(seal))
    write_bytes_atomic(config.staging_dir / DONE_FILENAME, b"")


def remove_local(config: StageConfig, seal: Dict[str, Any]) -> None:
    """Delete only the published files, then any directories they leave empty."""
    for entry in seal["files"]:
        (config.spool_dir / entry["path"]).unlink(missing_ok=True)
    (config.spool_dir / SEAL_FILENAME).unlink(missing_ok=True)
    directories = sorted((path for path in config.spool_dir.rglob("*") if path.is_dir()), reverse=True)
    for directory in [*directories, config.spool_dir, config.spool_dir.parent]:
        try:
            directory.rmdir()
        except OSError:
            pass  # not empty: something unpublished is left for an operator


def remove_runtime_state(config: StageConfig, seal: Dict[str, Any]) -> None:
    """Remove this attempt's files under /run, leaving a newer attempt's state alone."""
    if config.state_dir is None:
        return
    storage_key = seal["storage_key"]
    (config.state_dir / f"{storage_key}.env").unlink(missing_ok=True)
    job_state = config.state_dir / f"{seal['job_id']}.env"
    if load_env_file(job_state).get("STORAGE_KEY") == storage_key:
        job_state.unlink(missing_ok=True)


def run_stager(config: StageConfig) -> int:
    done_marker = config.staging_dir / DONE_FILENAME
    seal = read_json_file(config.spool_dir / SEAL_FILENAME)
    if not done_marker.exists():
        seal = wait_for_seal(config)
        if seal is None:
            LOGGER.error("no seal in %s after %.0f s; is the runner still running?", config.spool_dir, config.wait_seconds)
            return 1
        try:
            publish(config, seal)
        except (OSError, StageError) as exc:
            # The local spool is untouched; the unit's Restart=on-failure retries.
            LOGGER.error("publication to %s failed: %s", config.staging_dir, exc)
            return 1
        LOGGER.info("published %d files to %s", len(seal["files"]), config.staging_dir)
    if seal:
        if not config.keep_local:
            remove_local(config, seal)
        remove_runtime_state(config, seal)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a sealed in-band spool directory to shared staging.")
    parser.add_argument("--spool-dir", default=os.environ.get("REPACSS_POWER_OUTPUT_DIR", ""))
    parser.add_argument("--staging-dir", default=os.environ.get("REPACSS_POWER_STAGING_DIR", ""))
    parser.add_argument(
        "--runner-unit",
        default=os.environ.get("REPACSS_POWER_RUNNER_UNIT", ""),
        help="Runner systemd unit; when it has stopped without a seal, a forced seal is written.",
    )
    parser.add_argument("--wait-seconds", type=float, default=120.0, help="How long to wait for the seal.")
    parser.add_argument("--keep-local", action="store_true", help="Keep the spool after publication.")
    parser.add_argument(
        "--state-dir",
        default=os.environ.get("REPACSS_POWER_STATE_DIR") or os.environ.get("MONSTER_POWER_STATE_DIR", ""),
        help=f"Runtime state to clean after publication, normally {DEFAULT_STATE_DIR}.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    level = logging.DEBUG if os.environ.get("MONSTER_POWER_DEBUG") == "1" else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if not args.spool_dir or not args.staging_dir:
        LOGGER.error("--spool-dir and --staging-dir are required")
        return 2
    config = StageConfig(
        spool_dir=Path(args.spool_dir),
        staging_dir=Path(args.staging_dir),
        runner_unit=args.runner_unit,
        wait_seconds=args.wait_seconds,
        keep_local=args.keep_local,
        state_dir=Path(args.state_dir) if args.state_dir else None,
    )
    return run_stager(config)


if __name__ == "__main__":
    raise SystemExit(main())
