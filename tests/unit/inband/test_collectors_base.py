"""Tests for the shared in-band collector framework."""

from __future__ import annotations

import csv
import time
from pathlib import Path

from inband.collectors.base import CollectorProbe, PollingCSVCollector
from inband.spool import sample_path


class DummyCollector(PollingCSVCollector):
    name = "dummy"

    def probe(self) -> CollectorProbe:
        return CollectorProbe(name=self.name, available=True, details={"kind": "test"})

    def csv_columns(self):
        return ["value"]

    def prepare_runtime(self):
        return {"counter": 0}

    def collect_rows(self, runtime):
        runtime["counter"] += 1
        return [{"value": runtime["counter"]}]


def test_collector_writes_one_file_and_samples_once_more_on_stop(tmp_path: Path):
    collector = DummyCollector()
    path = sample_path(tmp_path, collector.name)
    before = time.time()
    handle = collector.start(interval_ms=10, output_path=path)
    time.sleep(0.12)
    result = collector.stop(handle)

    assert result.status == "complete"
    assert result.output_path == path
    assert [entry.name for entry in path.parent.iterdir()] == ["dummy.csv"]  # no .partial left behind
    with path.open(newline="", encoding="utf-8") as handle_:
        rows = list(csv.DictReader(handle_))
    # Every sample, including the one taken after the stop request, is in the file once.
    assert [int(row["value"]) for row in rows] == list(range(1, result.sample_count + 1))
    times = [float(row["t"]) for row in rows]
    assert before - 1 < times[0] and times == sorted(times)
