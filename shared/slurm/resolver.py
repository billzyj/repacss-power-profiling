"""Resolve Slurm job context from environment or Slurm commands."""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Dict, List, Optional

from shared.config import config
from shared.errors import SlurmResolutionError
from shared.models import SlurmJobContext
from shared.slurm.rest_client import SlurmRESTClient

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
    """Resolve Slurm job context, preferring env and falling back to Slurm REST."""
    env = env or os.environ
    if env.get("SLURM_JOB_ID"):
        return resolve_job_context_from_env(env)
    if not job_id:
        raise SlurmResolutionError("job_id is required when Slurm job env is unavailable")
    return resolve_job_context_from_rest(job_id)


def resolve_job_context_from_rest(job_id: str, client: Optional[SlurmRESTClient] = None) -> SlurmJobContext:
    """Build a SlurmJobContext from SlurmDBD through slurmrestd."""
    slurm_config = config.get_slurm_rest_config()
    issues = config.validate_slurm_rest_config()
    if issues:
        raise SlurmResolutionError("; ".join(issues))

    rest_client = client or SlurmRESTClient(slurm_config)
    job = rest_client.get_job(job_id, from_database=True)
    if not job:
        raise SlurmResolutionError(f"No Slurm job found for job_id={job_id}")

    nodelist = str(job.get("nodes") or job.get("node_list") or "").strip()
    nodes = expand_nodelist(nodelist)
    start_time = _parse_rest_time(_nested_get(job, "time", "start") or job.get("start_time"))
    end_time = _parse_rest_time(_nested_get(job, "time", "end") or job.get("end_time"))
    user = _extract_rest_user(job)

    if not user:
        raise SlurmResolutionError(f"Slurm REST job {job_id} did not include a user")
    if not nodelist or not nodes:
        raise SlurmResolutionError(f"Slurm REST job {job_id} did not include nodes")
    if start_time is None or end_time is None:
        raise SlurmResolutionError(f"Slurm REST job {job_id} did not include start/end times")

    return SlurmJobContext(
        job_id=str(job.get("job_id") or job.get("id") or job_id),
        user=user,
        nodelist=nodelist,
        nodes=nodes,
        start_time=start_time,
        end_time=end_time,
        comment=str(job.get("comment") or ""),
        cluster_name=str(job.get("cluster") or "") or None,
    )


def _nested_get(data: Dict[str, object], *keys: str):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _parse_rest_time(raw: object) -> Optional[datetime]:
    if raw in (None, "", 0, "0"):
        return None
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(int(raw))
    epoch = parse_job_start_epoch(str(raw))
    return datetime.fromtimestamp(epoch) if epoch is not None else None


def _extract_rest_user(job: Dict[str, object]) -> str:
    raw_user = job.get("user") or job.get("user_name")
    if isinstance(raw_user, dict):
        return str(raw_user.get("name") or raw_user.get("user_name") or "").strip()
    return str(raw_user or "").strip()
