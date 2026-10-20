"""Shared collector abstractions for in-band sampling."""

from __future__ import annotations

import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from inband.chunks import DEFAULT_CHUNK_SECONDS, ChunkWriter
from shared.errors import InbandCollectorError


@dataclass(frozen=True)
class CollectorProbe:
    """Availability and environment details for one collector."""

    name: str
    available: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectorHandle:
    """Handle returned by a running collector."""

    collector_name: str
    hostname: str
    output_dir: Path
    interval_ms: int
    chunk_seconds: float
    started_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _stop_event: Optional[threading.Event] = field(default=None, repr=False)
    _runtime: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class CollectorResult:
    """Result returned after a collector is stopped."""

    collector_name: str
    hostname: str
    status: Literal["complete", "partial", "failed"]
    output_dir: Path
    sample_count: int
    message: str = ""
    chunks: Tuple[Path, ...] = ()


class Collector(ABC):
    """Base collector interface for in-band sampling."""

    name: str

    @abstractmethod
    def probe(self) -> CollectorProbe:
        """Inspect whether this collector is currently usable."""

    def is_available(self) -> bool:
        return self.probe().available

    def manifest_payload(self) -> Optional[Dict[str, Any]]:
        """Startup provenance the runner writes next to the collector output, if any."""
        return None

    @abstractmethod
    def start(self, interval_ms: int, output_dir: Path, chunk_seconds: float = DEFAULT_CHUNK_SECONDS) -> CollectorHandle:
        """Start collecting samples into rotating chunks in `output_dir`."""

    @abstractmethod
    def stop(self, handle: CollectorHandle) -> CollectorResult:
        """Stop a running collector and return the captured data."""


class PollingCSVCollector(Collector):
    """Common implementation for collectors that write CSV rows on a timer.

    Rows go to rotating chunks (see `inband.chunks`). After a stop request the loop
    takes one more sample, so the data covers the time up to the stop.
    """

    def hostname(self) -> str:
        return socket.gethostname()

    @abstractmethod
    def csv_columns(self) -> Sequence[str]:
        """CSV columns emitted by this collector."""

    @abstractmethod
    def prepare_runtime(self) -> Dict[str, Any]:
        """Build runtime state before the background sampler starts."""

    @abstractmethod
    def collect_rows(self, runtime: Dict[str, Any], elapsed_seconds: float) -> List[Dict[str, Any]]:
        """Collect one interval worth of rows."""

    def start(self, interval_ms: int, output_dir: Path, chunk_seconds: float = DEFAULT_CHUNK_SECONDS) -> CollectorHandle:
        probe = self.probe()
        if not probe.available:
            raise InbandCollectorError(f"{self.name} collector is unavailable: {probe.reason}")

        output_dir.mkdir(parents=True, exist_ok=True)
        runtime = self.prepare_runtime()
        runtime.setdefault("status", "running")
        runtime.setdefault("message", "")
        runtime.setdefault("sample_count", 0)
        runtime["started_monotonic"] = time.monotonic()

        stop_event = threading.Event()
        started_at = datetime.now(timezone.utc)

        handle = CollectorHandle(
            collector_name=self.name,
            hostname=self.hostname(),
            output_dir=output_dir,
            interval_ms=interval_ms,
            chunk_seconds=chunk_seconds,
            started_at=started_at,
            metadata=probe.details.copy(),
            _stop_event=stop_event,
            _runtime=runtime,
        )

        thread = threading.Thread(target=self._run_loop, args=(handle,), name=f"{self.name}-collector", daemon=True)
        handle._thread = thread
        thread.start()
        return handle

    def stop(self, handle: CollectorHandle) -> CollectorResult:
        if handle._stop_event is not None:
            handle._stop_event.set()
        if handle._thread is not None:
            handle._thread.join(timeout=max(handle.interval_ms / 1000.0 * 3, 2.0))

        runtime = handle._runtime
        status = runtime.get("status", "complete")
        message = runtime.get("message", "")
        if runtime.get("status") == "running":
            status = "partial"
            message = message or "collector thread did not report a terminal state"
        return CollectorResult(
            collector_name=handle.collector_name,
            hostname=handle.hostname,
            status=status,
            output_dir=handle.output_dir,
            sample_count=int(runtime.get("sample_count", 0)),
            message=message,
            chunks=tuple(runtime.get("chunks", ())),
        )

    def _run_loop(self, handle: CollectorHandle) -> None:
        runtime = handle._runtime
        writer = ChunkWriter(handle.output_dir, self.csv_columns(), handle.chunk_seconds)
        try:
            while handle._stop_event is not None and not handle._stop_event.is_set():
                loop_start = time.monotonic()
                self._sample(runtime, writer, loop_start)
                sleep_for = max(0.0, (handle.interval_ms / 1000.0) - (time.monotonic() - loop_start))
                if sleep_for > 0:
                    handle._stop_event.wait(timeout=sleep_for)
            # Cover the time between the last sample and the stop request.
            self._sample(runtime, writer, time.monotonic())
            runtime["status"] = "complete" if runtime.get("sample_count", 0) > 0 else "partial"
        except Exception as exc:  # pragma: no cover - defensive sampler guard
            runtime["status"] = "failed"
            runtime["message"] = str(exc)
        finally:
            try:
                writer.close()
            except OSError as exc:  # pragma: no cover - defensive sampler guard
                runtime["status"] = "failed"
                runtime["message"] = runtime.get("message") or f"cannot close chunk: {exc}"
            runtime["chunks"] = tuple(writer.closed)

    def _sample(self, runtime: Dict[str, Any], writer: ChunkWriter, now: float) -> None:
        rows = self.collect_rows(runtime, now - runtime["started_monotonic"])
        if rows:
            writer.write_rows(rows)
            runtime["sample_count"] += len(rows)
