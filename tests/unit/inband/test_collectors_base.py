"""Tests for the shared in-band collector framework."""

from __future__ import annotations

import csv
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
                "timestamp": f"2026-01-01T00:00:{runtime['counter']:02d}+00:00",
                "elapsed_seconds": round(elapsed_seconds, 6),
                "hostname": self.hostname(),
                "value": runtime["counter"],
            }
        ]


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_collector_writes_rotating_chunks_and_samples_once_more_on_stop(tmp_path: Path):
    collector = DummyCollector()
    handle = collector.start(interval_ms=10, output_dir=tmp_path, chunk_seconds=0.03)
    time.sleep(0.12)
    result = collector.stop(handle)

    assert result.status == "complete"
    assert len(result.chunks) >= 2
    assert sorted(tmp_path.iterdir()) == sorted(result.chunks)  # no .partial left behind
    values = [int(row["value"]) for chunk in result.chunks for row in _rows(chunk)]
    # Every sample, including the one taken after the stop request, is in exactly one chunk.
    assert values == list(range(1, result.sample_count + 1))
