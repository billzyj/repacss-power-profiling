from __future__ import annotations

from datetime import date, datetime, timezone

from click.testing import CliRunner

from cli import eguage as eguage_cli
from cli.eguage import eguage_group
from eguage.history import (
    BackfillRunResult,
    DayFetchResult,
    build_day_window,
    fetch_history_day,
    run_backfill_command,
    run_history_backfill,
    summarize_history_payload,
)
from shared.errors import EGaugeError


def _payload(row_count: int) -> dict:
    return {
        "registers": [{"idx": 0, "name": "MAIN_W"}],
        "ranges": [{"delta": 60.0, "rows": [[str(i)] for i in range(row_count)]}],
    }


def test_build_day_window_uses_utc_midnight_epoch_range() -> None:
    window = build_day_window(date(2026, 6, 29), step_seconds=60)
    expected_start = int(datetime(2026, 6, 29, tzinfo=timezone.utc).timestamp())
    expected_end = int(datetime(2026, 6, 30, tzinfo=timezone.utc).timestamp())

    assert window.day == "2026-06-29"
    assert window.start_epoch == expected_start
    assert window.end_epoch == expected_end
    assert window.time_range == f"{expected_start}:60:{expected_end}"
    assert window.max_rows == 1440


def test_summarize_history_payload_counts_rows_across_ranges() -> None:
    summary = summarize_history_payload(
        {
            "ranges": [
                {"delta": 1.0, "rows": [["a"], ["b"]]},
                {"delta": 60.0, "rows": [["c"]]},
            ]
        }
    )

    assert summary.has_data is True
    assert summary.range_count == 2
    assert summary.row_count == 3
    assert summary.deltas == [1.0, 60.0]
    assert summary.row_widths == [1, 1]


def test_summarize_history_payload_treats_empty_ranges_as_no_data() -> None:
    summary = summarize_history_payload({"ranges": [{"delta": 60.0, "rows": []}]})

    assert summary.has_data is False
    assert summary.range_count == 1
    assert summary.row_count == 0


def test_fetch_history_day_retries_transient_errors_and_writes_success(tmp_path) -> None:
    attempts = []

    def fetch_payload(_window):
        attempts.append("called")
        if len(attempts) < 3:
            raise EGaugeError("temporary failure")
        return _payload(2)

    result = fetch_history_day(
        day=date(2026, 6, 29),
        output_dir=tmp_path,
        step_seconds=60,
        fetch_payload=fetch_payload,
        max_attempts=3,
        sleep_func=lambda _seconds: None,
    )

    assert result.status == "success"
    assert result.attempts == 3
    assert result.row_count == 2
    assert (tmp_path / "days" / "2026-06-29.json").exists()
    assert not (tmp_path / "days" / "2026-06-29.error.json").exists()


def test_fetch_history_day_writes_error_without_marking_no_data(tmp_path) -> None:
    def fetch_payload(_window):
        raise EGaugeError("meter failed")

    result = fetch_history_day(
        day=date(2026, 6, 29),
        output_dir=tmp_path,
        step_seconds=60,
        fetch_payload=fetch_payload,
        max_attempts=2,
        sleep_func=lambda _seconds: None,
    )

    assert result.status == "error"
    assert result.error == "meter failed"
    assert (tmp_path / "days" / "2026-06-29.error.json").exists()
    assert not (tmp_path / "days" / "2026-06-29.no_data.json").exists()


def test_fetch_history_day_resumes_existing_success_unless_overwrite(tmp_path) -> None:
    days_dir = tmp_path / "days"
    days_dir.mkdir()
    (days_dir / "2026-06-29.json").write_text(
        '{"ranges": [{"delta": 60.0, "rows": [["saved"]]}]}\n',
        encoding="utf-8",
    )
    calls = []

    def fetch_payload(_window):
        calls.append("called")
        return _payload(3)

    skipped = fetch_history_day(
        day=date(2026, 6, 29),
        output_dir=tmp_path,
        step_seconds=60,
        fetch_payload=fetch_payload,
        overwrite=False,
    )
    overwritten = fetch_history_day(
        day=date(2026, 6, 29),
        output_dir=tmp_path,
        step_seconds=60,
        fetch_payload=fetch_payload,
        overwrite=True,
    )

    assert skipped.status == "skipped"
    assert skipped.row_count == 1
    assert overwritten.status == "success"
    assert overwritten.row_count == 3
    assert calls == ["called"]


def test_run_history_backfill_stops_on_first_no_data_day(tmp_path) -> None:
    calls: list[str] = []

    def fetch_day(day, **_kwargs):
        day_text = day.isoformat()
        calls.append(day_text)
        has_data = day_text in {"2026-06-29", "2026-06-28"}
        return DayFetchResult(
            day=day_text,
            status="success" if has_data else "no_data",
            has_data=has_data,
            row_count=1 if has_data else 0,
            range_count=1 if has_data else 0,
            attempts=1,
            elapsed_seconds=0.0,
            time_range="",
            output_path=None,
        )

    result = run_history_backfill(
        start_day=date(2026, 6, 29),
        output_dir=tmp_path,
        max_workers=2,
        stop_empty_days=1,
        max_days=10,
        fetch_day_func=fetch_day,
    )

    assert result.stop_day == "2026-06-27"
    assert [item.day for item in result.days[:3]] == ["2026-06-29", "2026-06-28", "2026-06-27"]
    assert "2026-06-25" not in calls


def test_run_history_backfill_can_delay_between_serial_submissions(tmp_path) -> None:
    sleeps: list[float] = []

    def fetch_day(day, **_kwargs):
        return DayFetchResult(
            day=day.isoformat(),
            status="success",
            has_data=True,
            row_count=1,
            range_count=1,
            attempts=1,
            elapsed_seconds=0.0,
            time_range="",
            output_path=None,
        )

    result = run_history_backfill(
        start_day=date(2026, 6, 29),
        output_dir=tmp_path,
        max_workers=1,
        max_days=3,
        fetch_day_func=fetch_day,
        submit_delay_seconds=2.5,
        sleep_func=sleeps.append,
    )

    assert [item.day for item in result.days] == ["2026-06-29", "2026-06-28", "2026-06-27"]
    assert sleeps == [2.5, 2.5]


def test_cli_backfill_history_parses_options(monkeypatch, tmp_path) -> None:
    captured = {}

    def fake_run_backfill_command(**kwargs):
        captured.update(kwargs)
        return BackfillRunResult(
            output_dir=str(tmp_path),
            selected_workers=None,
            stop_day=None,
            days=[],
            calibration=None,
            dry_run=True,
        )

    monkeypatch.setattr(eguage_cli, "run_backfill_command", fake_run_backfill_command)

    result = CliRunner().invoke(
        eguage_group,
        [
            "backfill-history",
            "--start-day",
            "2026-06-29",
            "--output-dir",
            str(tmp_path),
            "--candidate-workers",
            "1,2,4",
            "--max-workers",
            "auto",
            "--dry-run",
            "--raw",
            "--delta",
            "--no-rate",
            "--submit-delay-seconds",
            "2.5",
        ],
    )

    assert result.exit_code == 0
    assert captured["start_day"] == date(2026, 6, 29)
    assert captured["output_dir"] == tmp_path
    assert captured["candidate_workers"] == [1, 2, 4]
    assert captured["max_workers"] == "auto"
    assert captured["dry_run"] is True
    assert captured["include_raw"] is True
    assert captured["include_delta"] is True
    assert captured["include_rate"] is False
    assert captured["submit_delay_seconds"] == 2.5
