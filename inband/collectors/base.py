"""Shared collector abstractions for in-band sampling."""

from __future__ import annotations

import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

from inband.spool import SampleWriter
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
    output_path: Path
    interval_ms: int
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
    output_path: Optional[Path]
    sample_count: int
    message: str = ""


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
    def start(self, interval_ms: int, output_path: Path) -> CollectorHandle:
        """Start appending samples to the CSV file `output_path`."""

    @abstractmethod
    def stop(self, handle: CollectorHandle) -> CollectorResult:
        """Stop a running collector and return the captured data."""


class PollingCSVCollector(Collector):
    """Common implementation for collectors that append raw readings to a CSV file on a timer.

    The writer adds the sample time `t` as the first column. A sample's time comes from
    the monotonic clock, anchored once to the wall clock at start, so intervals stay
    exact even if the wall clock is stepped during the job. After a stop request the
    loop takes one more sample, so the data covers the time up to the stop.
    """

    def hostname(self) -> str:
        return socket.gethostname()

    @abstractmethod
    def csv_columns(self) -> Sequence[str]:
        """CSV columns emitted by this collector, after the time column."""

    @abstractmethod
    def prepare_runtime(self) -> Dict[str, Any]:
        """Build runtime state before the background sampler starts."""

    @abstractmethod
    def collect_rows(self, runtime: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Read one sample: rows keyed by `csv_columns()`. A missing value is None."""

    def start(self, interval_ms: int, output_path: Path) -> CollectorHandle:
        probe = self.probe()
        if not probe.available:
            raise InbandCollectorError(f"{self.name} collector is unavailable: {probe.reason}")

        runtime = self.prepare_runtime()
        runtime.setdefault("status", "running")
        runtime.setdefault("message", "")
        runtime.setdefault("sample_count", 0)
        runtime["clock_origin"] = (time.time(), time.monotonic())

        stop_event = threading.Event()
        handle = CollectorHandle(
            collector_name=self.name,
            hostname=self.hostname(),
            output_path=output_path,
            interval_ms=interval_ms,
            started_at=datetime.now(timezone.utc),
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
            output_path=runtime.get("output_path"),
            sample_count=int(runtime.get("sample_count", 0)),
            message=message,
        )

    def _run_loop(self, handle: CollectorHandle) -> None:
        runtime = handle._runtime
        columns = list(self.csv_columns())
        writer = SampleWriter(handle.output_path, columns)
        try:
            while handle._stop_event is not None and not handle._stop_event.is_set():
                loop_start = time.monotonic()
                self._sample(runtime, writer, columns)
                sleep_for = max(0.0, (handle.interval_ms / 1000.0) - (time.monotonic() - loop_start))
                if sleep_for > 0:
                    handle._stop_event.wait(timeout=sleep_for)
            # Cover the time between the last sample and the stop request.
            self._sample(runtime, writer, columns)
            runtime["status"] = "complete" if runtime.get("sample_count", 0) > 0 else "partial"
        except Exception as exc:  # pragma: no cover - defensive sampler guard
            runtime["status"] = "failed"
            runtime["message"] = str(exc)
        finally:
            try:
                runtime["output_path"] = writer.close()
            except OSError as exc:  # pragma: no cover - defensive sampler guard
                runtime["status"] = "failed"
                runtime["message"] = runtime.get("message") or f"cannot close sample file: {exc}"

    def _sample(self, runtime: Dict[str, Any], writer: SampleWriter, columns: Sequence[str]) -> None:
        wall_origin, monotonic_origin = runtime["clock_origin"]
        t = wall_origin + (time.monotonic() - monotonic_origin)
        rows = self.collect_rows(runtime)
        if rows:
            writer.write(t, [[row.get(column) for column in columns] for row in rows])
            runtime["sample_count"] += 1
