"""File-backed finalization queue on the controller.

EpilogSlurmctld writes one record per attempt to `pending/` and returns. The
finalizer moves each record to `complete/` or `failed/` with its outcome attached.
Records are JSON files written atomically, so a reader sees a whole record or none.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from inband.storage import write_json_file

DEFAULT_QUEUE_ROOT = Path(os.environ.get("REPACSS_POWER_IB_QUEUE_ROOT", "/var/spool/repacss-power/finalize-queue"))
PENDING = "pending"
COMPLETE = "complete"
FAILED = "failed"


def record_path(queue_root: Path, state: str, attempt_id: str) -> Path:
    return queue_root / state / f"{attempt_id}.json"


def enqueue(queue_root: Path, record: Mapping[str, Any], key: Optional[str] = None) -> Path:
    """Add a record once, under `key` or its attempt id. Adding the same key again is a no-op."""
    key = key or str(record["attempt_id"])
    for state in (PENDING, COMPLETE, FAILED):
        existing = record_path(queue_root, state, key)
        if existing.exists():
            return existing
    return write_json_file(record_path(queue_root, PENDING, key), record)


def pending(queue_root: Path) -> List[Path]:
    return sorted((queue_root / PENDING).glob("*.json"))


def settle(queue_root: Path, path: Path, state: str, record: Dict[str, Any]) -> Path:
    """Write the record with its outcome under `state`, then remove it from pending."""
    target = write_json_file(record_path(queue_root, state, path.stem), record)
    path.unlink(missing_ok=True)
    return target
