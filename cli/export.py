"""Unified export command surface for the current migration stage."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from cli.oob import _build_manual_context
from oob.slurm.epilog_handler import handle_oob_job
from shared.slurm.resolver import resolve_job_context, resolve_job_context_from_env


@click.command(name="export")
@click.option("--job", "job_id", required=False, help="Slurm job id.")
@click.option("--user", default="unknown", show_default=True, help="Job owner for manual mode.")
@click.option("--nodelist", help="Slurm nodelist for manual mode.")
@click.option("--start", required=False, help="Start time in 'YYYY-MM-DD HH:MM:SS' for manual mode.")
@click.option("--end", required=False, help="End time in 'YYYY-MM-DD HH:MM:SS' for manual mode.")
@click.option("--from-env", is_flag=True, help="Resolve the job context from Slurm environment variables.")
@click.option("--output", "outdir", type=click.Path(path_type=Path), help="Output directory.")
def export_command(job_id: Optional[str], user: str, nodelist: Optional[str], start: Optional[str], end: Optional[str], from_env: bool, outdir: Optional[Path]):
    """Produce file exports for the currently implemented OOB workflow."""
    if from_env:
        context = resolve_job_context_from_env()
    else:
        missing = [name for name, value in (("job", job_id), ("nodelist", nodelist), ("start", start), ("end", end)) if not value]
        if job_id and set(missing) == {"nodelist", "start", "end"}:
            context = resolve_job_context(job_id)
        elif missing:
            raise click.ClickException(
                "Current export support is OOB-only and requires either --job alone for Slurm REST lookup, "
                "or --job, --nodelist, --start, and --end; "
                f"missing: {', '.join(missing)}."
            )
        else:
            context = _build_manual_context(job_id, user, nodelist, start, end)

    output_dir = handle_oob_job(context, outdir)
    click.echo(f"Export written under {output_dir}")
