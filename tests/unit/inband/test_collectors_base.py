"""Tests for the shared in-band collector framework."""

from __future__ import annotations

import time
from pathlib import Path

from inband.collectors.base import CollectorProbe, PollingCSVCollector


class DummyCollector(PollingCSVCollector):
    name = "dummy"

    def probe(self) -> CollectorProbe:
        return CollectorProbe(name=self.name, available=True, details={"kind": "test"})

    def csv_columns(self):
        return ["timestamp", "elapsed_seconds", "hostname", "value"]

    def prepare_runtime(self):
        return {"counter": 0}

    def collect_rows(self, runtime, elapsed_seconds):
        runtime["counter"] += 1
        return [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "elapsed_seconds": round(elapsed_seconds, 6),
                "hostname": self.hostname(),
                "value": runtime["counter"],
            }
        ]


def test_polling_csv_collector_start_and_stop(tmp_path: Path):
    collector = DummyCollector()
    output_path = tmp_path / "dummy.csv"
    handle = collector.start(interval_ms=10, output_path=output_path)
    time.sleep(0.05)
    result = collector.stop(handle)

    assert result.status in {"complete", "partial"}
    assert result.sample_count >= 1
    assert output_path.exists()
    assert "value" in result.samples.columns
