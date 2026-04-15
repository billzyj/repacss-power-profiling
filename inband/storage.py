"""Storage and state helpers for in-band Slurm integration."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

DEFAULT_IB_STORE_ROOT = Path(os.environ.get("MONSTER_POWER_IB_STORE_ROOT", "/mnt/SHARED-AREA/repacss-power-staging"))
DEFAULT_STATE_DIR = Path(os.environ.get("MONSTER_POWER_STATE_DIR", "/run/repacss-power"))


def build_storage_key(job_id: str, job_start_epoch: int, cluster_name: Optional[str] = None) -> str:
    """Build the internal staging key for one Slurm job run."""
    cluster = (cluster_name or "cluster").strip() or "cluster"
    return f"{cluster}-{job_id}-{int(job_start_epoch)}"


def ensure_state_dir(state_dir: Path = DEFAULT_STATE_DIR) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def job_state_file_path(job_id: str, state_dir: Path = DEFAULT_STATE_DIR) -> Path:
    return state_dir / f"{job_id}.env"


def unit_env_file_path(storage_key: str, state_dir: Path = DEFAULT_STATE_DIR) -> Path:
    return state_dir / f"{storage_key}.env"


def job_root_path(storage_key: str, ib_store_root: Path = DEFAULT_IB_STORE_ROOT) -> Path:
    return ib_store_root / storage_key


def node_root_path(storage_key: str, hostname: str, ib_store_root: Path = DEFAULT_IB_STORE_ROOT) -> Path:
    return job_root_path(storage_key, ib_store_root=ib_store_root) / hostname


def collector_output_path(
    storage_key: str, hostname: str, collector_name: str, ib_store_root: Path = DEFAULT_IB_STORE_ROOT
) -> Path:
    return node_root_path(storage_key, hostname, ib_store_root=ib_store_root) / f"{collector_name}.csv"


def node_status_path(storage_key: str, hostname: str, ib_store_root: Path = DEFAULT_IB_STORE_ROOT) -> Path:
    return node_root_path(storage_key, hostname, ib_store_root=ib_store_root) / "node_status.json"


def runner_status_path(storage_key: str, hostname: str, ib_store_root: Path = DEFAULT_IB_STORE_ROOT) -> Path:
    return node_root_path(storage_key, hostname, ib_store_root=ib_store_root) / "runner_status.json"


def done_marker_path(storage_key: str, hostname: str, ib_store_root: Path = DEFAULT_IB_STORE_ROOT) -> Path:
    return node_root_path(storage_key, hostname, ib_store_root=ib_store_root) / ".done"


def manifest_path(storage_key: str, ib_store_root: Path = DEFAULT_IB_STORE_ROOT) -> Path:
    return job_root_path(storage_key, ib_store_root=ib_store_root) / "manifest.json"


def summary_path(storage_key: str, ib_store_root: Path = DEFAULT_IB_STORE_ROOT) -> Path:
    return job_root_path(storage_key, ib_store_root=ib_store_root) / "summary.json"


def epoch_cache_path(job_id: str, cluster_name: Optional[str], ib_store_root: Path = DEFAULT_IB_STORE_ROOT) -> Path:
    cluster = (cluster_name or "cluster").strip() or "cluster"
    return ib_store_root / ".state" / f"{cluster}-{job_id}.epoch"


def ensure_node_staging(storage_key: str, hostname: str, ib_store_root: Path = DEFAULT_IB_STORE_ROOT) -> Path:
    """Create the shared NFS staging root for one job/node."""
    root = node_root_path(storage_key, hostname, ib_store_root=ib_store_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_json_file(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Expected JSON object at {path}")


def write_env_file(path: Path, values: Mapping[str, Any]) -> Path:
    """Write a shell-friendly env file, overwriting any previous content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(values):
            value = "" if values[key] is None else str(values[key])
            handle.write(f"{key}={shlex.quote(value)}\n")
    return path


def _format_systemd_value(raw_value: str) -> str:
    """Format an EnvironmentFile-compatible value for systemd."""
    if raw_value == "":
        return '""'
    if any(ch.isspace() for ch in raw_value) or any(ch in raw_value for ch in ['"', "\\"]):
        escaped = raw_value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return raw_value


def write_systemd_env_file(path: Path, values: Mapping[str, Any]) -> Path:
    """Write an EnvironmentFile-compatible env file for systemd units."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(values):
            value = "" if values[key] is None else str(values[key])
            handle.write(f"{key}={_format_systemd_value(value)}\n")
    return path


def load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    payload: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            tokens = shlex.split(raw_value, posix=True)
            payload[key] = tokens[0] if tokens else ""
    return payload


def resolve_storage_key_for_job(
    job_id: str, ib_store_root: Path = DEFAULT_IB_STORE_ROOT, cluster_name: Optional[str] = None
) -> Optional[str]:
    """Resolve the latest matching storage key for a Slurm job id."""
    prefix = f"{cluster_name.strip()}-{job_id}-" if cluster_name else f"-{job_id}-"
    candidates = []
    if not ib_store_root.exists():
        return None
    for child in ib_store_root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if cluster_name:
            if not name.startswith(prefix):
                continue
        else:
            if prefix not in name:
                continue
        suffix = name.rsplit("-", 1)[-1]
        try:
            epoch = int(suffix)
        except ValueError:
            continue
        candidates.append((epoch, name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]
