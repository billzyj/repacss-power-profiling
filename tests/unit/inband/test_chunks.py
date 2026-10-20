"""Tests for rotating chunks and the sealed spool inventory."""

from __future__ import annotations

import json
from pathlib import Path

from inband.chunks import SEAL_FILENAME, ChunkWriter, sha256_file, write_seal

COLUMNS = ["timestamp", "value"]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def _row(second: int):
    return {"timestamp": f"2026-10-08T22:00:{second:02d}+00:00", "value": second}


def test_chunks_rotate_by_age_and_keep_each_sample_whole(tmp_path: Path):
    clock = FakeClock()
    writer = ChunkWriter(tmp_path, COLUMNS, chunk_seconds=30, clock=clock)
    for second in range(0, 61, 10):
        clock.now = second
        writer.write_rows([_row(second), _row(second)])
        if second == 20:
            # The open chunk is visible only under its .partial name.
            assert [path.name for path in tmp_path.iterdir()] == ["00000001.csv.partial"]
    writer.close()

    assert [path.name for path in writer.closed] == ["00000001.csv", "00000002.csv", "00000003.csv"]
    assert sorted(tmp_path.iterdir()) == writer.closed
    first = writer.closed[0].read_text(encoding="utf-8").splitlines()
    assert first[0] == "timestamp,value"
    assert len(first) == 1 + 3 * 2  # header, then samples at 0, 10, and 20 s


def test_writer_continues_after_existing_chunks(tmp_path: Path):
    (tmp_path / "00000007.csv").write_text("timestamp,value\n", encoding="utf-8")
    writer = ChunkWriter(tmp_path, COLUMNS, chunk_seconds=30)
    writer.write_rows([_row(1)])
    writer.close()
    assert writer.closed == [tmp_path / "00000008.csv"]


def test_writer_without_rows_creates_no_chunk(tmp_path: Path):
    writer = ChunkWriter(tmp_path / "rapl", COLUMNS)
    writer.close()
    assert writer.closed == []
    assert not (tmp_path / "rapl").exists()


def test_seal_lists_closed_chunks_and_metadata_but_not_open_chunks(tmp_path: Path):
    writer = ChunkWriter(tmp_path / "chunks" / "rapl", COLUMNS)
    writer.write_rows([_row(1), _row(2)])
    writer.close()
    (tmp_path / "chunks" / "rapl" / "00000002.csv.partial").write_text("timestamp,value\n", encoding="utf-8")
    (tmp_path / "runner_status.json").write_text('{"state": "complete"}\n', encoding="utf-8")

    seal = write_seal(tmp_path, job_id="12345", storage_key="k", hostname="rpc-91-1", runner_state="complete")

    assert json.loads((tmp_path / SEAL_FILENAME).read_text(encoding="utf-8")) == seal
    chunk, status = seal["files"]
    assert chunk["path"] == "chunks/rapl/00000001.csv"
    assert chunk["collector"] == "rapl"
    assert chunk["rows"] == 2
    assert chunk["first_timestamp"] == "2026-10-08T22:00:01+00:00"
    assert chunk["last_timestamp"] == "2026-10-08T22:00:02+00:00"
    assert chunk["sha256"] == sha256_file(tmp_path / chunk["path"])
    assert status["path"] == "runner_status.json"
