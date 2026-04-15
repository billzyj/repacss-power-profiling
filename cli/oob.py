"""Out-of-band query and export CLI commands."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from oob.backends.monster_db.backend import MonsterDBBackend
from oob.slurm.epilog_handler import summarize_job_power
from shared.errors import OOBBackendError
from shared.models import SlurmJobContext
from shared.slurm.resolver import expand_nodelist, resolve_job_context_from_env


def _parse_datetime(raw: Optional[str]) -> Optional[datetime]:
    if raw is None:
        return None
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")


def _infer_database(hostname: str) -> str:
    cleaned = hostname.lower()
    if cleaned.startswith("rpg"):
        return "h100"
    if cleaned.startswith("rpc"):
        return "zen4"
    if cleaned.startswith("irc") or cleaned.startswith("pdu"):
        return "infra"
    raise click.ClickException("Could not infer database from hostname; pass --database explicitly.")


def _print_dataframe(df: pd.DataFrame, max_rows: int = 50):
    if df.empty:
        click.echo("No rows returned.")
        return
    preview = df.head(max_rows)
    click.echo(preview.to_string(index=False))
    if len(df) > max_rows:
        click.echo(f"\n... truncated to first {max_rows} rows ({len(df)} total).")


def _write_dataframe(df: pd.DataFrame, output: Path, format_name: str):
    output.parent.mkdir(parents=True, exist_ok=True)
    if format_name == "csv":
        df.to_csv(output, index=False)
    elif format_name == "json":
        output.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        raise click.ClickException(f"Unsupported output format: {format_name}")


def _print_job_summary(context: SlurmJobContext, raw_df: pd.DataFrame):
    _, pie_segments, energy_by_metric, _ = summarize_job_power(raw_df, context.nodes, context.start_time, context.end_time)
    click.echo(f"Job ID: {context.job_id}")
    click.echo(f"User: {context.user}")
    click.echo(f"Nodes: {', '.join(context.nodes)}")
    click.echo(f"Window: {context.start_time} -> {context.end_time}")
    click.echo(f"Rows returned: {len(raw_df)}")
    if energy_by_metric:
        click.echo("Energy by metric (kWh):")
        for metric, value in sorted(energy_by_metric.items()):
            click.echo(f"  {metric}: {value:.6f}")
    if pie_segments:
        click.echo("Summary segments (kWh):")
        for label, value in sorted(pie_segments.items()):
            click.echo(f"  {label}: {value:.6f}")
    if not raw_df.empty:
        click.echo("")
        _print_dataframe(raw_df)


def _build_manual_context(job_id: str, user: str, nodelist: str, start: str, end: str) -> SlurmJobContext:
    nodes = expand_nodelist(nodelist)
    if not nodes:
        raise click.ClickException("Could not expand --nodelist into any nodes.")
    return SlurmJobContext(
        job_id=job_id,
        user=user,
        nodelist=nodelist,
        nodes=nodes,
        start_time=_parse_datetime(start),
        end_time=_parse_datetime(end),
        comment="power:oob",
    )


@click.group(name="oob")
def oob_group():
    """Out-of-band query and export workflows."""


@oob_group.command("query")
@click.option("--hostname", required=True, help="Target hostname, for example rpg-93-1.")
@click.option("--database", type=click.Choice(["h100", "zen4", "infra"]), help="Database override.")
@click.option("--start", "start_time", help="Start time in 'YYYY-MM-DD HH:MM:SS'.")
@click.option("--end", "end_time", help="End time in 'YYYY-MM-DD HH:MM:SS'.")
@click.option("--limit", type=int, default=100, show_default=True, help="Maximum rows to fetch.")
@click.option("--output", type=click.Path(path_type=Path), help="Optional output path.")
@click.option("--format", "format_name", type=click.Choice(["csv", "json"]), default="csv", show_default=True, help="File format used with --output.")
def query_metrics(hostname: str, database: Optional[str], start_time: Optional[str], end_time: Optional[str], limit: int, output: Optional[Path], format_name: str):
    """Run a general OOB metrics query."""
    database_name = database or _infer_database(hostname)
    backend = MonsterDBBackend(database_name)
    try:
        df = backend.query_metrics(
            hostname,
            start_time=_parse_datetime(start_time),
            end_time=_parse_datetime(end_time),
            limit=limit,
        )
    except OOBBackendError as exc:
        raise click.ClickException(str(exc)) from exc
    if output is not None:
        _write_dataframe(df, output, format_name)
        click.echo(f"Wrote {len(df)} rows to {output}")
        return
    _print_dataframe(df)


@oob_group.command("job")
@click.option("--job-id", required=False, help="Slurm job id.")
@click.option("--user", default="unknown", show_default=True, help="Job owner for manual mode.")
@click.option("--nodelist", help="Slurm nodelist for manual mode, for example 'rpg-93-[1-2]'.")
@click.option("--start", "start_time", help="Start time in 'YYYY-MM-DD HH:MM:SS' for manual mode.")
@click.option("--end", "end_time", help="End time in 'YYYY-MM-DD HH:MM:SS' for manual mode.")
@click.option("--from-env", is_flag=True, help="Resolve the job context from Slurm environment variables.")
def export_job(job_id: Optional[str], user: str, nodelist: Optional[str], start_time: Optional[str], end_time: Optional[str], from_env: bool):
    """Run the job-scoped OOB query and print a summary preview."""
    if from_env:
        context = resolve_job_context_from_env()
    else:
        missing = [name for name, value in (("job-id", job_id), ("nodelist", nodelist), ("start", start_time), ("end", end_time)) if not value]
        if missing:
            raise click.ClickException(
                "Manual job query requires --job-id, --nodelist, --start, and --end; "
                f"missing: {', '.join(missing)}."
            )
        context = _build_manual_context(job_id, user, nodelist, start_time, end_time)

    backend = MonsterDBBackend()
    try:
        raw_df = backend.query_job(context)
    except OOBBackendError as exc:
        raise click.ClickException(str(exc)) from exc
    _print_job_summary(context, raw_df)
