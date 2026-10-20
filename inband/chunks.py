"""Rotating CSV chunks and the sealed inventory of one node's spool directory.

A collector appends rows to `chunks/<collector>/<seq>.csv.partial` on node-local
storage and flushes after every sample. When the chunk is older than the rotation
interval, or when collection stops, the file is fsynced and renamed to `<seq>.csv`.
A closed chunk never changes again, so the stager can copy it while sampling
continues. Each chunk carries its own CSV header.

When the runner stops, it writes `node_sealed.json`: the complete list of closed
chunks and metadata files with their sizes and SHA-256 digests. The stager publishes
exactly that list and nothing else.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

CHUNKS_DIRNAME = "chunks"
CHUNK_SUFFIX = ".csv"
PARTIAL_SUFFIX = ".partial"
SEAL_FILENAME = "node_sealed.json"
DEFAULT_CHUNK_SECONDS = 30.0

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


class ChunkWriter:
    """Append rows to rotating chunk files in one directory."""

    def __init__(
        self,
        directory: Path,
        columns: Sequence[str],
        chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")
        self.directory = directory
        self.columns = list(columns)
        self.chunk_seconds = chunk_seconds
        self.closed: List[Path] = []
        self._clock = clock
        self._sequence = self._last_sequence()
        self._handle = None
        self._partial: Optional[Path] = None
        self._opened_at = 0.0

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Append one sample's rows. A sample never spans two chunks."""
        if not rows:
            return
        if self._handle is not None and self._clock() - self._opened_at >= self.chunk_seconds:
            self._close_current()
        if self._handle is None:
            self._open_next()
        self._handle.write(self._encode(rows, header=False))
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._close_current()

    def _last_sequence(self) -> int:
        """Continue after any chunks already present, so a name is never reused."""
        if not self.directory.exists():
            return 0
        numbers = [int(path.name[:8]) for path in self.directory.iterdir() if path.name[:8].isdigit()]
        return max(numbers, default=0)

    def _encode(self, rows: Sequence[Mapping[str, Any]], header: bool) -> bytes:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=self.columns, lineterminator="\n")
        if header:
            writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue().encode("utf-8")

    def _open_next(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._sequence += 1
        self._partial = self.directory / f"{self._sequence:08d}{CHUNK_SUFFIX}{PARTIAL_SUFFIX}"
        self._handle = self._partial.open("xb")
        self._handle.write(self._encode([], header=True))
        self._opened_at = self._clock()

    def _close_current(self) -> None:
        handle, self._handle = self._handle, None
        assert self._partial is not None
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        final = self._partial.with_name(self._partial.name[: -len(PARTIAL_SUFFIX)])
        os.replace(self._partial, final)
        fsync_directory(self.directory)
        self.closed.append(final)


def _chunk_entry(node_dir: Path, path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    records = list(csv.reader(data.decode("utf-8").splitlines()))
    header, body = (records[0], records[1:]) if records else ([], [])
    column = header.index("timestamp") if "timestamp" in header else None
    first = body[0][column] if body and column is not None else ""
    last = body[-1][column] if body and column is not None else ""
    return {
        "path": path.relative_to(node_dir).as_posix(),
        "collector": path.parent.name,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "rows": len(body),
        "first_timestamp": first,
        "last_timestamp": last,
    }


def inventory(node_dir: Path) -> List[Dict[str, Any]]:
    """List closed chunks and top-level metadata files. Open `.partial` files are excluded."""
    entries = [_chunk_entry(node_dir, path) for path in sorted(node_dir.glob(f"{CHUNKS_DIRNAME}/*/*{CHUNK_SUFFIX}"))]
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
        "schema_version": 1,
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
