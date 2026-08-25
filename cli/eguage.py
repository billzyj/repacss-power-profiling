"""eGauge API CLI commands."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Optional

import click

from eguage.client import EGaugeClient
from eguage.history import parse_worker_candidates, run_backfill_command
from shared.errors import EGaugeError


def _write_json(payload: dict, output: Optional[Path]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output is None:
        click.echo(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    click.echo(f"Wrote JSON output to {output}")


@click.group(name="eguage")
def eguage_group():
    """eGauge API workflows."""


@eguage_group.command("probe")
def probe_eguage() -> None:
    """Verify SSH tunnel setup and JWT authentication."""

    try:
        with EGaugeClient.from_env() as client:
            rights = client.get_token_rights()
            snapshot = client.get_registers(register_range="all", include_rate=True, virtual="formula")
    except EGaugeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("Connected to eGauge API.")
    click.echo(f"Authenticated user: {rights.get('usr', '<unknown>')}")
    click.echo(f"Rights: {', '.join(rights.get('rights', [])) or '<none reported>'}")
    registers = snapshot.get("registers", [])
    click.echo(f"Registers returned: {len(registers)}")
    for register in registers[:10]:
        rate = register.get("rate")
        rate_text = f", rate={rate}" if rate is not None else ""
        click.echo(
            f"- {register.get('name', '<unnamed>')} "
            f"(idx={register.get('idx', '?')}, type={register.get('type', '?')}{rate_text})"
        )


@eguage_group.command("registers")
@click.option("--view", help="View selector, for example '=default' or '+generation'.")
@click.option("--reg", "register_range", help="Register selector, for example 'all' or '0:10'.")
@click.option("--time-range", help="Time selector, for example 'now-3600:60:now'.")
@click.option("--max-rows", type=int, help="Maximum number of rows to request when time ranges are used.")
@click.option("--virtual", "virtual_mode", type=click.Choice(["formula", "value"]), default="formula", show_default=True)
@click.option("--rate/--no-rate", default=True, show_default=True, help="Include current register rate values.")
@click.option("--raw", "include_raw", is_flag=True, help="Request raw cumulative values instead of epoch-relative values.")
@click.option("--delta", "include_delta", is_flag=True, help="Request delta-encoded historical rows.")
@click.option("--output", type=click.Path(path_type=Path), help="Optional JSON output file.")
def read_registers(
    view: Optional[str],
    register_range: Optional[str],
    time_range: Optional[str],
    max_rows: Optional[int],
    virtual_mode: str,
    rate: bool,
    include_raw: bool,
    include_delta: bool,
    output: Optional[Path],
) -> None:
    """Fetch register data from the eGauge meter."""

    try:
        with EGaugeClient.from_env() as client:
            payload = client.get_registers(
                view=view,
                register_range=register_range,
                time_range=time_range,
                include_rate=rate,
                include_raw=include_raw,
                include_delta=include_delta,
                virtual=virtual_mode,
                max_rows=max_rows,
            )
    except EGaugeError as exc:
        raise click.ClickException(str(exc)) from exc

    _write_json(payload, output)


def _parse_iso_day(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise click.BadParameter("Use YYYY-MM-DD format.") from exc


@eguage_group.command("backfill-history")
@click.option("--start-day", default="2026-06-29", show_default=True, help="UTC day to start walking backward from.")
@click.option("--output-dir", type=click.Path(path_type=Path), help="Output directory for manifest, summaries, and daily JSON.")
@click.option("--reg", "register_range", default="all", show_default=True, help="Register selector, for example 'all' or '0:10'.")
@click.option("--virtual", "virtual_mode", type=click.Choice(["formula", "value"]), default="value", show_default=True)
@click.option("--step-seconds", type=int, default=60, show_default=True, help="Historical stride for daily windows.")
@click.option("--max-workers", default="auto", show_default=True, help="Worker count or 'auto' for calibration.")
@click.option("--candidate-workers", default="1,2,4", show_default=True, help="Comma-separated worker counts for auto calibration.")
@click.option("--calibration-minutes", type=int, default=5, show_default=True, help="Recent-history minutes per calibration request.")
@click.option("--stop-empty-days", type=int, default=1, show_default=True, help="Stop after this many newest-to-oldest no-data days.")
@click.option("--max-days", type=int, help="Optional safety cap on days to query.")
@click.option("--submit-delay-seconds", type=float, default=0.0, show_default=True, help="Delay before submitting each additional day request.")
@click.option("--overwrite", is_flag=True, help="Refetch days even when a completed day JSON or no-data marker exists.")
@click.option("--dry-run", is_flag=True, help="Parse options and print the planned output path without making eGauge requests.")
@click.option("--raw", "include_raw", is_flag=True, help="Request raw cumulative values instead of epoch-relative values.")
@click.option("--delta", "include_delta", is_flag=True, help="Request delta-encoded historical rows.")
@click.option("--rate/--no-rate", default=True, show_default=True, help="Include current register rate values.")
def backfill_history(
    start_day: str,
    output_dir: Optional[Path],
    register_range: str,
    virtual_mode: str,
    step_seconds: int,
    max_workers: str,
    candidate_workers: str,
    calibration_minutes: int,
    stop_empty_days: int,
    max_days: Optional[int],
    submit_delay_seconds: float,
    overwrite: bool,
    dry_run: bool,
    include_raw: bool,
    include_delta: bool,
    rate: bool,
) -> None:
    """Walk backward by UTC day and persist raw historical `/register` payloads."""

    parsed_start_day = _parse_iso_day(start_day)
    target_dir = output_dir or Path("output") / "eguage-history" / f"{parsed_start_day.isoformat()}-backfill"
    normalized_max_workers: str | int = max_workers
    if max_workers != "auto":
        try:
            normalized_max_workers = int(max_workers)
        except ValueError as exc:
            raise click.BadParameter("--max-workers must be 'auto' or a positive integer.") from exc

    try:
        result = run_backfill_command(
            start_day=parsed_start_day,
            output_dir=target_dir,
            register_range=register_range,
            virtual=virtual_mode,
            step_seconds=step_seconds,
            max_workers=normalized_max_workers,
            candidate_workers=parse_worker_candidates(candidate_workers),
            calibration_minutes=calibration_minutes,
            stop_empty_days=stop_empty_days,
            max_days=max_days,
            submit_delay_seconds=submit_delay_seconds,
            overwrite=overwrite,
            dry_run=dry_run,
            include_raw=include_raw,
            include_delta=include_delta,
            include_rate=rate,
        )
    except (EGaugeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if result.dry_run:
        click.echo("Dry run: no eGauge requests were made.")
        click.echo(f"Start day: {parsed_start_day.isoformat()}")
        click.echo(f"Output directory: {target_dir}")
        return

    click.echo(f"Selected workers: {result.selected_workers}")
    click.echo(f"Days recorded: {len(result.days)}")
    if result.stop_day:
        click.echo(f"Stopped at no-data day: {result.stop_day}")
    click.echo(f"Output directory: {result.output_dir}")
