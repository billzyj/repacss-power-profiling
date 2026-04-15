"""Resolve Slurm job context from environment or Slurm commands."""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional

from shared.errors import SlurmResolutionError
from shared.models import SlurmJobContext

try:
    import hostlist
except ImportError:  # pragma: no cover - optional dependency
    hostlist = None


def expand_nodelist(nodelist: str) -> List[str]:
    """Expand a Slurm nodelist into concrete hostnames."""
    cleaned = (nodelist or "").strip()
    if not cleaned:
        return []
    if hostlist is not None:
        try:
            return hostlist.expand_hostlist(cleaned)
        except Exception:
            pass
    return [node.strip() for node in cleaned.split(",") if node.strip()]


def _parse_slurm_epoch(raw: Optional[str]) -> Optional[datetime]:
    if raw is None or raw == "":
        return None
    return datetime.fromtimestamp(int(raw))


def _extract_scontrol_field(output: str, key: str) -> str:
    pattern = rf"\b{re.escape(key)}=(.*?)(?= [A-Za-z][A-Za-z0-9_]*=|$)"
    match = re.search(pattern, output)
    if not match:
        return ""
    return match.group(1).strip()


def fetch_job_metadata(job_id: str, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Resolve Slurm job metadata from env or one scontrol call."""
    env = env or os.environ
    metadata: Dict[str, str] = {}
    if not job_id:
        return metadata

    comment = (env.get("SLURM_JOB_COMMENT") or "").strip()
    if comment:
        metadata["Comment"] = comment

    start_time = (env.get("SLURM_JOB_START_TIME") or "").strip()
    if start_time:
        metadata["StartTime"] = start_time

    if metadata.get("Comment") and metadata.get("StartTime"):
        return metadata

    try:
        output = subprocess.check_output(["scontrol", "show", "job", "-o", job_id], text=True)
    except Exception:
        return metadata

    if "Comment" not in metadata:
        metadata["Comment"] = _extract_scontrol_field(output, "Comment")
    if "StartTime" not in metadata:
        metadata["StartTime"] = _extract_scontrol_field(output, "StartTime")
    return metadata


def parse_job_start_epoch(raw: Optional[str]) -> Optional[int]:
    """Parse a Slurm start time string into an epoch integer."""
    value = (raw or "").strip()
    if not value or value in {"Unknown", "N/A", "None"}:
        return None
    if value.isdigit():
        return int(value)

    # Slurm commonly emits ISO-like timestamps such as:
    #   2026-04-14T08:00:00
    #   2026-04-14T08:00:00Z
    # Parse these explicitly so the behavior stays stable across Python versions.
    formats = [
        ("%Y-%m-%dT%H:%M:%S", None),
        ("%Y-%m-%dT%H:%M:%SZ", timezone.utc),
        ("%Y-%m-%dT%H:%M:%S.%f", None),
        ("%Y-%m-%dT%H:%M:%S.%fZ", timezone.utc),
    ]
    for pattern, tzinfo in formats:
        try:
            parsed = datetime.strptime(value, pattern)
            if tzinfo is not None:
                parsed = parsed.replace(tzinfo=tzinfo)
            return int(parsed.timestamp())
        except ValueError:
            continue

    try:
        return int(datetime.fromisoformat(value).timestamp())
    except ValueError:
        return None


def fetch_job_comment(job_id: str, env: Optional[Dict[str, str]] = None) -> str:
    """Resolve job comment from env or scontrol."""
    return fetch_job_metadata(job_id, env).get("Comment", "")


def resolve_job_context_from_env(env: Optional[Dict[str, str]] = None) -> SlurmJobContext:
    """Build a SlurmJobContext from Slurm-provided environment variables."""
    env = env or os.environ
    job_id = (env.get("SLURM_JOB_ID") or "").strip()
    user = (env.get("SLURM_JOB_USER") or "").strip()
    nodelist = (env.get("SLURM_JOB_NODELIST") or "").strip()
    start_time = _parse_slurm_epoch(env.get("SLURM_JOB_START_TIME"))
    end_time = _parse_slurm_epoch(env.get("SLURM_JOB_END_TIME"))
    nodes = expand_nodelist(nodelist)

    if not job_id:
        raise SlurmResolutionError("SLURM_JOB_ID is required")
    if not user:
        raise SlurmResolutionError("SLURM_JOB_USER is required")
    if not nodelist or not nodes:
        raise SlurmResolutionError("SLURM_JOB_NODELIST is required")
    if start_time is None or end_time is None:
        raise SlurmResolutionError("SLURM_JOB_START_TIME and SLURM_JOB_END_TIME are required")

    return SlurmJobContext(
        job_id=job_id,
        user=user,
        nodelist=nodelist,
        nodes=nodes,
        start_time=start_time,
        end_time=end_time,
        comment=fetch_job_comment(job_id, env),
        stdout_path=env.get("SLURM_JOB_STDOUT"),
        stderr_path=env.get("SLURM_JOB_STDERR"),
        cluster_name=env.get("SLURM_CLUSTER_NAME"),
    )


def resolve_job_context(job_id: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> SlurmJobContext:
    """Resolve Slurm job context, preferring env and falling back to scontrol."""
    env = env or os.environ
    if env.get("SLURM_JOB_ID"):
        return resolve_job_context_from_env(env)
    raise SlurmResolutionError(
        "Offline/manual Slurm resolution is not implemented in P0-P2 without Slurm job env"
    )
