"""Shared collector abstractions for in-band sampling."""

from __future__ import annotations

import csv
import socket
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import pandas as pd

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
    samples: pd.DataFrame
    status: Literal["complete", "partial", "failed"]
    output_path: Path
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

    @abstractmethod
    def start(self, interval_ms: int, output_path: Path) -> CollectorHandle:
        """Start collecting samples to the given output path."""

    @abstractmethod
    def stop(self, handle: CollectorHandle) -> CollectorResult:
        """Stop a running collector and return the captured data."""


class PollingCSVCollector(Collector):
    """Common implementation for collectors that write CSV rows on a timer."""

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

    def start(self, interval_ms: int, output_path: Path) -> CollectorHandle:
        probe = self.probe()
        if not probe.available:
            raise InbandCollectorError(f"{self.name} collector is unavailable: {probe.reason}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
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
            output_path=output_path,
            interval_ms=interval_ms,
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

        samples = pd.read_csv(handle.output_path) if handle.output_path.exists() else pd.DataFrame(columns=self.csv_columns())
        runtime = handle._runtime
        status = runtime.get("status", "complete")
        message = runtime.get("message", "")
        if runtime.get("status") == "running":
            status = "partial"
            message = message or "collector thread did not report a terminal state"
        return CollectorResult(
            collector_name=handle.collector_name,
            hostname=handle.hostname,
            samples=samples,
            status=status,
            output_path=handle.output_path,
            sample_count=int(runtime.get("sample_count", 0)),
            message=message,
        )

    def _run_loop(self, handle: CollectorHandle) -> None:
        runtime = handle._runtime
        try:
            with handle.output_path.open("w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(self.csv_columns()))
                writer.writeheader()
                csvfile.flush()

                while handle._stop_event is not None and not handle._stop_event.is_set():
                    loop_start = time.monotonic()
                    elapsed_seconds = loop_start - runtime["started_monotonic"]
                    rows = self.collect_rows(runtime, elapsed_seconds)
                    if rows:
                        writer.writerows(rows)
                        csvfile.flush()
                        runtime["sample_count"] += len(rows)
                    sleep_for = max(0.0, (handle.interval_ms / 1000.0) - (time.monotonic() - loop_start))
                    if sleep_for > 0:
                        handle._stop_event.wait(timeout=sleep_for)
            runtime["status"] = "complete" if runtime.get("sample_count", 0) > 0 else "partial"
            runtime["message"] = runtime.get("message", "")
        except Exception as exc:  # pragma: no cover - defensive sampler guard
            runtime["status"] = "failed"
            runtime["message"] = str(exc)
