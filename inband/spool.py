"""Per-collector sample files and the sealed inventory of one node's spool directory.

Each collector appends to one file, `samples/<collector>.csv.partial`, on node-local
storage and flushes after every sample. When collection stops, the file is fsynced and
renamed to `samples/<collector>.csv`, so a closed file never changes again.

A sample file holds raw readings only. The first column, `t`, is the sample time in
epoch seconds; the other columns are what the collector names in its header, one per
counter domain for RAPL and NVML. Everything derived from the readings (deltas, wraps,
power, energy in joules) is computed at finalization, using the collector's manifest.

When the runner stops, it writes `node_sealed.json`: the complete list of closed sample
files and metadata files with their sizes and SHA-256 digests. The stager publishes
exactly that list and nothing else.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SAMPLES_DIRNAME = "samples"
SAMPLE_SUFFIX = ".csv"
PARTIAL_SUFFIX = ".partial"
SEAL_FILENAME = "node_sealed.json"
TIME_COLUMN = "t"

SEAL_NORMAL = "normal"
SEAL_FORCED_PARTIAL = "forced_partial"


def fsync_directory(directory: Path) -> None:
    """Make a rename in `directory` durable."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write through a sibling `.partial` file so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    with partial.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    fsync_directory(path.parent)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_path(spool_dir: Path, collector: str) -> Path:
    return spool_dir / SAMPLES_DIRNAME / f"{collector}{SAMPLE_SUFFIX}"


class SampleWriter:
    """Append samples to one CSV file, visible under its final name only once closed."""

    def __init__(self, path: Path, columns: Sequence[str]):
        self.path = path
        self.columns = [TIME_COLUMN, *columns]
        self._partial = path.with_name(path.name + PARTIAL_SUFFIX)
        self._handle = None
        self.closed = False

    def write(self, t: float, rows: Sequence[Sequence[Any]]) -> None:
        """Append one sample: each row holds the values for the columns after `t`."""
        if not rows:
            return
        if self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._partial.open("xb")
            self._handle.write(self._encode([self.columns]))
        stamp = f"{t:.3f}"
        self._handle.write(self._encode([[stamp, *("" if value is None else value for value in row)] for row in rows]))
        self._handle.flush()

    def close(self) -> Optional[Path]:
        """Close and publish the file; return its path, or None if nothing was written."""
        handle, self._handle = self._handle, None
        if handle is None:
            return self.path if self.closed else None
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(self._partial, self.path)
        fsync_directory(self.path.parent)
        self.closed = True
        return self.path

    @staticmethod
    def _encode(records: Sequence[Sequence[Any]]) -> bytes:
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerows(records)
        return buffer.getvalue().encode("utf-8")


def _sample_entry(node_dir: Path, path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    lines = data.decode("utf-8").splitlines()
    body = lines[1:]
    first = body[0].split(",", 1)[0] if body else ""
    last = body[-1].split(",", 1)[0] if body else ""
    return {
        "path": path.relative_to(node_dir).as_posix(),
        "collector": path.name[: -len(SAMPLE_SUFFIX)],
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(body),
        "first_t": float(first) if first else None,
        "last_t": float(last) if last else None,
    }


def inventory(node_dir: Path) -> List[Dict[str, Any]]:
    """List closed sample files and top-level metadata files. Open `.partial` files are excluded."""
    entries = [_sample_entry(node_dir, path) for path in sorted(node_dir.glob(f"{SAMPLES_DIRNAME}/*{SAMPLE_SUFFIX}"))]
    for path in sorted(node_dir.glob("*.json")):
        if path.name == SEAL_FILENAME:
            continue
        entries.append(
            {
                "path": path.relative_to(node_dir).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


def write_seal(
    node_dir: Path,
    *,
    job_id: str,
    storage_key: str,
    hostname: str,
    runner_state: str,
    seal_type: str = SEAL_NORMAL,
) -> Dict[str, Any]:
    """Record the final inventory. Write it only after the producer has stopped writing."""
    seal = {
        "schema_version": 2,
        "kind": "repacss_node_seal",
        "seal_type": seal_type,
        "job_id": job_id,
        "storage_key": storage_key,
        "hostname": hostname,
        "runner_state": runner_state,
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "files": inventory(node_dir),
    }
    write_bytes_atomic(node_dir / SEAL_FILENAME, (json.dumps(seal, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return seal
