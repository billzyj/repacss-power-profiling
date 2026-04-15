from __future__ import annotations

from datetime import datetime, timezone

from shared.slurm.resolver import parse_job_start_epoch


def test_parse_job_start_epoch_accepts_naive_iso_timestamp() -> None:
    raw = "2026-04-14T08:00:00"
    expected = int(datetime(2026, 4, 14, 8, 0, 0).timestamp())
    assert parse_job_start_epoch(raw) == expected


def test_parse_job_start_epoch_accepts_utc_z_suffix() -> None:
    raw = "2026-04-14T08:00:00Z"
    expected = int(datetime(2026, 4, 14, 8, 0, 0, tzinfo=timezone.utc).timestamp())
    assert parse_job_start_epoch(raw) == expected
