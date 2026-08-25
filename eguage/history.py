"""Historical eGauge `/register` backfill helpers."""

from __future__ import annotations

import csv
import json
import math
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .client import EGaugeClient


FetchPayload = Callable[["HistoryWindow"], dict[str, Any]]
SleepFunc = Callable[[float], None]


@dataclass(frozen=True)
class HistoryWindow:
    """One UTC day expressed in eGauge `/register` time selector form."""

    day: str
    start_epoch: int
    end_epoch: int
    step_seconds: int
    time_range: str
    max_rows: int


@dataclass(frozen=True)
class HistoryPayloadSummary:
    """Small summary of historical rows returned by `/register`."""

    has_data: bool
    range_count: int
    row_count: int
    deltas: list[float]
    row_widths: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DayFetchResult:
    """Result of fetching one day from eGauge history."""

    day: str
    status: str
    has_data: bool | None
    row_count: int
    range_count: int
    attempts: int
    elapsed_seconds: float
    time_range: str
    output_path: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationCandidateResult:
    """Observed result for one concurrency candidate."""

    workers: int
    failures: int
    elapsed_seconds: float
    request_seconds: list[float]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationReport:
    """Concurrency calibration outcome."""

    selected_workers: int
    candidates: list[CalibrationCandidateResult]
    calibration_minutes: int
    step_seconds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_workers": self.selected_workers,
            "calibration_minutes": self.calibration_minutes,
            "step_seconds": self.step_seconds,
            "candidates": [item.to_dict() for item in self.candidates],
        }


@dataclass(frozen=True)
class BackfillRunResult:
    """Overall history backfill result."""

    output_dir: str
    selected_workers: int | None
    stop_day: str | None
    days: list[DayFetchResult]
    calibration: CalibrationReport | None
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "selected_workers": self.selected_workers,
            "stop_day": self.stop_day,
            "dry_run": self.dry_run,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "days": [item.to_dict() for item in self.days],
        }


def build_day_window(day: date, *, step_seconds: int) -> HistoryWindow:
    """Build a UTC midnight-to-midnight eGauge time selector for one day."""

    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    start_epoch = int(start.timestamp())
    end_epoch = int(end.timestamp())
    max_rows = math.ceil((end_epoch - start_epoch) / step_seconds)
    return HistoryWindow(
        day=day.isoformat(),
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        step_seconds=step_seconds,
        time_range=f"{start_epoch}:{step_seconds}:{end_epoch}",
        max_rows=max_rows,
    )


def summarize_history_payload(payload: dict[str, Any]) -> HistoryPayloadSummary:
    """Return row counts and range metadata without transforming raw values."""

    ranges = payload.get("ranges") or []
    row_count = 0
    deltas: list[float] = []
    row_widths: list[int] = []
    has_data = False

    for range_payload in ranges:
        if not isinstance(range_payload, dict):
            continue
        rows = range_payload.get("rows") or []
        if not isinstance(rows, list):
            rows = []
        row_count += len(rows)
        if rows:
            has_data = True
            first_row = rows[0]
            if isinstance(first_row, list):
                row_widths.append(len(first_row))
        delta = range_payload.get("delta")
        if isinstance(delta, (int, float)):
            deltas.append(float(delta))

    return HistoryPayloadSummary(
        has_data=has_data,
        range_count=len(ranges) if isinstance(ranges, list) else 0,
        row_count=row_count,
        deltas=deltas,
        row_widths=row_widths,
    )


def parse_worker_candidates(raw: str) -> list[int]:
    """Parse a comma-separated worker candidate list."""

    candidates: list[int] = []
    for item in raw.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        workers = int(cleaned)
        if workers <= 0:
            raise ValueError("worker candidates must be positive integers")
        candidates.append(workers)
    if not candidates:
        raise ValueError("at least one worker candidate is required")
    return candidates


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def fetch_register_payload(
    window: HistoryWindow,
    *,
    register_range: str,
    virtual: str,
    include_rate: bool,
    include_raw: bool,
    include_delta: bool,
    client_factory: Callable[[], EGaugeClient] = EGaugeClient.from_env,
) -> dict[str, Any]:
    """Fetch one historical `/register` window with a fresh client."""

    with client_factory() as client:
        return client.get_registers(
            register_range=register_range,
            time_range=window.time_range,
            include_rate=include_rate,
            include_raw=include_raw,
            include_delta=include_delta,
            virtual=virtual,
            max_rows=window.max_rows,
        )


def fetch_history_day(
    *,
    day: date,
    output_dir: Path,
    step_seconds: int,
    register_range: str = "all",
    virtual: str = "value",
    include_rate: bool = True,
    include_raw: bool = False,
    include_delta: bool = False,
    overwrite: bool = False,
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    sleep_func: SleepFunc = time.sleep,
    fetch_payload: FetchPayload | None = None,
) -> DayFetchResult:
    """Fetch and persist one day, retrying failures without treating them as no-data."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    window = build_day_window(day, step_seconds=step_seconds)
    days_dir = output_dir / "days"
    success_path = days_dir / f"{window.day}.json"
    no_data_path = days_dir / f"{window.day}.no_data.json"
    error_path = days_dir / f"{window.day}.error.json"

    if not overwrite and success_path.exists():
        payload = _load_json(success_path)
        summary = summarize_history_payload(payload)
        return DayFetchResult(
            day=window.day,
            status="skipped",
            has_data=summary.has_data,
            row_count=summary.row_count,
            range_count=summary.range_count,
            attempts=0,
            elapsed_seconds=0.0,
            time_range=window.time_range,
            output_path=str(success_path),
        )

    if not overwrite and no_data_path.exists():
        marker = _load_json(no_data_path)
        summary = marker.get("summary") or {}
        return DayFetchResult(
            day=window.day,
            status="skipped",
            has_data=False,
            row_count=int(summary.get("row_count", 0)),
            range_count=int(summary.get("range_count", 0)),
            attempts=0,
            elapsed_seconds=0.0,
            time_range=window.time_range,
            output_path=str(no_data_path),
        )

    if fetch_payload is None:
        fetch_payload = lambda current_window: fetch_register_payload(
            current_window,
            register_range=register_range,
            virtual=virtual,
            include_rate=include_rate,
            include_raw=include_raw,
            include_delta=include_delta,
        )

    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            payload = fetch_payload(window)
            summary = summarize_history_payload(payload)
            if summary.has_data:
                _unlink_if_exists(no_data_path)
                _unlink_if_exists(error_path)
                _write_json(success_path, payload)
                output_path = success_path
                status = "success"
            else:
                _unlink_if_exists(success_path)
                _unlink_if_exists(error_path)
                _write_json(
                    no_data_path,
                    {
                        "day": window.day,
                        "status": "no_data",
                        "time_range": window.time_range,
                        "summary": summary.to_dict(),
                    },
                )
                output_path = no_data_path
                status = "no_data"

            return DayFetchResult(
                day=window.day,
                status=status,
                has_data=summary.has_data,
                row_count=summary.row_count,
                range_count=summary.range_count,
                attempts=attempt,
                elapsed_seconds=time.monotonic() - started,
                time_range=window.time_range,
                output_path=str(output_path),
            )
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                sleep_func(backoff_seconds * attempt)

    assert last_error is not None
    _unlink_if_exists(no_data_path)
    _write_json(
        error_path,
        {
            "day": window.day,
            "status": "error",
            "time_range": window.time_range,
            "attempts": max_attempts,
            "error": str(last_error),
        },
    )
    return DayFetchResult(
        day=window.day,
        status="error",
        has_data=None,
        row_count=0,
        range_count=0,
        attempts=max_attempts,
        elapsed_seconds=time.monotonic() - started,
        time_range=window.time_range,
        output_path=str(error_path),
        error=str(last_error),
    )


def run_history_backfill(
    *,
    start_day: date,
    output_dir: Path,
    max_workers: int,
    stop_empty_days: int = 1,
    max_days: int | None = None,
    submit_delay_seconds: float = 0.0,
    sleep_func: SleepFunc = time.sleep,
    fetch_day_func: Callable[..., DayFetchResult] = fetch_history_day,
    **fetch_kwargs: Any,
) -> BackfillRunResult:
    """Walk backward by day and process results newest-to-oldest."""

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if stop_empty_days <= 0:
        raise ValueError("stop_empty_days must be positive")
    if max_days is not None and max_days <= 0:
        raise ValueError("max_days must be positive when provided")
    if submit_delay_seconds < 0:
        raise ValueError("submit_delay_seconds must be non-negative")

    executor = ThreadPoolExecutor(max_workers=max_workers)
    pending: dict[int, Future[DayFetchResult]] = {}
    completed: dict[int, DayFetchResult] = {}
    days: list[DayFetchResult] = []
    next_submit = 0
    next_process = 0
    empty_streak = 0
    stop_day: str | None = None

    def can_submit() -> bool:
        return max_days is None or next_submit < max_days

    def submit_one() -> None:
        nonlocal next_submit
        if submit_delay_seconds and next_submit > 0:
            sleep_func(submit_delay_seconds)
        day = start_day - timedelta(days=next_submit)
        pending[next_submit] = executor.submit(
            fetch_day_func,
            day=day,
            output_dir=output_dir,
            **fetch_kwargs,
        )
        next_submit += 1

    try:
        while can_submit() and len(pending) < max_workers:
            submit_one()

        while pending:
            done, _not_done = wait(pending.values(), return_when=FIRST_COMPLETED)
            done_by_future = {future: index for index, future in pending.items()}
            for future in done:
                index = done_by_future[future]
                del pending[index]
                completed[index] = future.result()

            should_stop = False
            while next_process in completed:
                result = completed.pop(next_process)
                days.append(result)
                if result.status == "error":
                    should_stop = True
                    break
                if result.has_data is False:
                    empty_streak += 1
                    if empty_streak >= stop_empty_days:
                        stop_day = result.day
                        should_stop = True
                        break
                else:
                    empty_streak = 0
                next_process += 1

                while can_submit() and len(pending) + len(completed) < max_workers:
                    submit_one()

            if should_stop:
                for future in pending.values():
                    future.cancel()
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    return BackfillRunResult(
        output_dir=str(output_dir),
        selected_workers=max_workers,
        stop_day=stop_day,
        days=days,
        calibration=None,
    )


def _fetch_recent_payload(
    *,
    calibration_minutes: int,
    step_seconds: int,
    register_range: str,
    virtual: str,
    include_rate: bool,
    include_raw: bool,
    include_delta: bool,
    client_factory: Callable[[], EGaugeClient] = EGaugeClient.from_env,
) -> dict[str, Any]:
    request_seconds = calibration_minutes * 60
    max_rows = math.ceil(request_seconds / step_seconds)
    with client_factory() as client:
        return client.get_registers(
            register_range=register_range,
            time_range=f"now-{request_seconds}:{step_seconds}:now",
            include_rate=include_rate,
            include_raw=include_raw,
            include_delta=include_delta,
            virtual=virtual,
            max_rows=max_rows,
        )


def calibrate_concurrency(
    *,
    candidate_workers: list[int],
    calibration_minutes: int,
    step_seconds: int,
    register_range: str,
    virtual: str,
    include_rate: bool,
    include_raw: bool,
    include_delta: bool,
) -> CalibrationReport:
    """Probe a few bounded worker counts with short recent-history requests."""

    if calibration_minutes <= 0:
        raise ValueError("calibration_minutes must be positive")

    calibration_step = min(step_seconds, 60)
    candidates: list[CalibrationCandidateResult] = []
    selected_workers = 1

    for workers in candidate_workers:
        errors: list[str] = []
        request_seconds: list[float] = []
        started = time.monotonic()

        def request_once() -> None:
            request_started = time.monotonic()
            _fetch_recent_payload(
                calibration_minutes=calibration_minutes,
                step_seconds=calibration_step,
                register_range=register_range,
                virtual=virtual,
                include_rate=include_rate,
                include_raw=include_raw,
                include_delta=include_delta,
            )
            request_seconds.append(time.monotonic() - request_started)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(request_once) for _ in range(workers)]
            for future in futures:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(str(exc))

        candidate = CalibrationCandidateResult(
            workers=workers,
            failures=len(errors),
            elapsed_seconds=time.monotonic() - started,
            request_seconds=request_seconds,
            errors=errors,
        )
        candidates.append(candidate)
        if candidate.failures == 0:
            selected_workers = workers

    return CalibrationReport(
        selected_workers=selected_workers,
        candidates=candidates,
        calibration_minutes=calibration_minutes,
        step_seconds=calibration_step,
    )


def write_summary_csv(path: Path, days: list[DayFetchResult]) -> None:
    """Write one summary row per processed day."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "day",
        "status",
        "has_data",
        "row_count",
        "range_count",
        "attempts",
        "elapsed_seconds",
        "time_range",
        "output_path",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in days:
            writer.writerow(result.to_dict())


def run_backfill_command(
    *,
    start_day: date,
    output_dir: Path,
    register_range: str,
    virtual: str,
    step_seconds: int,
    max_workers: str | int,
    candidate_workers: list[int],
    calibration_minutes: int,
    stop_empty_days: int,
    max_days: int | None,
    submit_delay_seconds: float,
    overwrite: bool,
    dry_run: bool,
    include_raw: bool,
    include_delta: bool,
    include_rate: bool,
) -> BackfillRunResult:
    """Run the CLI-facing calibration and daily historical backfill workflow."""

    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")
    if submit_delay_seconds < 0:
        raise ValueError("submit_delay_seconds must be non-negative")
    if calibration_minutes <= 0:
        raise ValueError("calibration_minutes must be positive")
    if stop_empty_days <= 0:
        raise ValueError("stop_empty_days must be positive")
    if max_days is not None and max_days <= 0:
        raise ValueError("max_days must be positive when provided")

    if dry_run:
        return BackfillRunResult(
            output_dir=str(output_dir),
            selected_workers=None,
            stop_day=None,
            days=[],
            calibration=None,
            dry_run=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    calibration: CalibrationReport | None = None
    if max_workers == "auto":
        calibration = calibrate_concurrency(
            candidate_workers=candidate_workers,
            calibration_minutes=calibration_minutes,
            step_seconds=step_seconds,
            register_range=register_range,
            virtual=virtual,
            include_rate=include_rate,
            include_raw=include_raw,
            include_delta=include_delta,
        )
        selected_workers = calibration.selected_workers
        _write_json(output_dir / "calibration.json", calibration.to_dict())
    else:
        selected_workers = int(max_workers)
        if selected_workers <= 0:
            raise ValueError("max_workers must be 'auto' or a positive integer")

    result = run_history_backfill(
        start_day=start_day,
        output_dir=output_dir,
        max_workers=selected_workers,
        stop_empty_days=stop_empty_days,
        max_days=max_days,
        submit_delay_seconds=submit_delay_seconds,
        step_seconds=step_seconds,
        register_range=register_range,
        virtual=virtual,
        include_rate=include_rate,
        include_raw=include_raw,
        include_delta=include_delta,
        overwrite=overwrite,
    )

    final_result = BackfillRunResult(
        output_dir=str(output_dir),
        selected_workers=selected_workers,
        stop_day=result.stop_day,
        days=result.days,
        calibration=calibration,
    )
    write_summary_csv(output_dir / "summary.csv", final_result.days)
    _write_json(
        output_dir / "manifest.json",
        {
            "arguments": {
                "start_day": start_day.isoformat(),
                "register_range": register_range,
                "virtual": virtual,
                "step_seconds": step_seconds,
                "max_workers": max_workers,
                "candidate_workers": candidate_workers,
                "calibration_minutes": calibration_minutes,
                "stop_empty_days": stop_empty_days,
                "max_days": max_days,
                "submit_delay_seconds": submit_delay_seconds,
                "overwrite": overwrite,
                "include_raw": include_raw,
                "include_delta": include_delta,
                "include_rate": include_rate,
            },
            "selected_workers": selected_workers,
            "stop_day": final_result.stop_day,
            "failures": [item.to_dict() for item in final_result.days if item.status == "error"],
            "result": final_result.to_dict(),
        },
    )
    return final_result
