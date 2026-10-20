"""Tests for per-collector sample files and the sealed spool inventory."""

from __future__ import annotations

import json
from pathlib import Path

from inband.spool import SEAL_FILENAME, SampleWriter, sample_path, sha256_file, write_seal


def test_samples_append_to_one_file_that_appears_only_when_closed(tmp_path: Path):
    path = sample_path(tmp_path, "rapl")
    writer = SampleWriter(path, ["intel-rapl:0", "intel-rapl:1"])
    for second in range(3):
        writer.write(1_800_000_000 + second + 0.0004, [[1_000 * second, None if second == 1 else 7]])
        # The open file is visible only under its .partial name.
        assert [entry.name for entry in path.parent.iterdir()] == ["rapl.csv.partial"]

    assert writer.close() == path
    assert [entry.name for entry in path.parent.iterdir()] == ["rapl.csv"]
    assert path.read_text(encoding="utf-8").splitlines() == [
        "t,intel-rapl:0,intel-rapl:1",
        "1800000000.000,0,7",
        "1800000001.000,1000,",  # a failed read is an empty cell
        "1800000002.000,2000,7",
    ]


def test_writer_without_samples_creates_no_file(tmp_path: Path):
    writer = SampleWriter(sample_path(tmp_path, "nvml"), ["GPU-0"])
    assert writer.close() is None
    assert not (tmp_path / "samples").exists()


def test_seal_lists_closed_sample_files_and_metadata_but_not_open_files(tmp_path: Path):
    writer = SampleWriter(sample_path(tmp_path, "rapl"), ["intel-rapl:0"])
    writer.write(1_800_000_001.0, [[5]])
    writer.write(1_800_000_002.5, [[9]])
    writer.close()
    (tmp_path / "samples" / "nvml.csv.partial").write_text("t,GPU-0\n", encoding="utf-8")
    (tmp_path / "runner_status.json").write_text('{"state": "complete"}\n', encoding="utf-8")

    seal = write_seal(tmp_path, job_id="12345", storage_key="k", hostname="rpc-91-1", runner_state="complete")

    assert json.loads((tmp_path / SEAL_FILENAME).read_text(encoding="utf-8")) == seal
    samples, status = seal["files"]
    assert samples["path"] == "samples/rapl.csv"
    assert samples["collector"] == "rapl"
    assert (samples["rows"], samples["first_t"], samples["last_t"]) == (2, 1_800_000_001.0, 1_800_000_002.5)
    assert samples["sha256"] == sha256_file(tmp_path / samples["path"])
    assert status["path"] == "runner_status.json"
