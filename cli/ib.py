"""Debug-oriented CLI helpers for in-band collectors."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import click

from inband.collectors import build_collectors, probe_collectors, select_collectors
from shared.errors import InbandCollectorError


def _print_probe(collector_probe):
    status = "available" if collector_probe.available else "unavailable"
    click.echo(f"{collector_probe.name}: {status}")
    if collector_probe.reason:
        click.echo(f"  reason: {collector_probe.reason}")
    for key, value in collector_probe.details.items():
        click.echo(f"  {key}: {value}")


@click.group(name="ib")
def ib_group():
    """Debug and validation helpers for in-band collectors."""


@ib_group.command("probe")
@click.option(
    "--collector",
    "collectors",
    multiple=True,
    type=click.Choice(["rapl", "nvidia_smi", "rocm_smi"]),
    help="Optionally probe only selected collectors.",
)
def probe_command(collectors):
    """Probe in-band collector availability without starting sampling."""
    requested = collectors or None
    for collector_probe in probe_collectors(requested):
        _print_probe(collector_probe)


@ib_group.command("sample")
@click.option(
    "--collector",
    "collectors",
    multiple=True,
    type=click.Choice(["rapl", "nvidia_smi", "rocm_smi"]),
    help="Collectors to run. Defaults to auto-detected available collectors.",
)
@click.option("--interval-ms", default=1000, show_default=True, type=int, help="Sampling interval in milliseconds.")
@click.option("--duration-s", default=3.0, show_default=True, type=float, help="Sampling duration in seconds.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("output") / "inband-debug",
    show_default=True,
    help="Directory used for collector CSV outputs.",
)
def sample_command(collectors, interval_ms: int, duration_s: float, output_dir: Path):
    """Run a short local sampling session for collector validation.

    This is a debug surface only. Production in-band collection remains Slurm-driven.
    """

    requested = collectors or None
    instances = build_collectors(requested) if collectors else select_collectors()
    if not instances:
        raise click.ClickException("No available in-band collectors were selected.")

    output_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for collector in instances:
            output_path = output_dir / f"{collector.name}.csv"
            handles.append(collector.start(interval_ms=interval_ms, output_path=output_path))
            click.echo(f"started {collector.name} -> {output_path}")
        time.sleep(duration_s)
    except InbandCollectorError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        for collector, handle in zip(instances, handles):
            result = collector.stop(handle)
            click.echo(
                f"{result.collector_name}: status={result.status} samples={result.sample_count} output={result.output_path}"
            )
            if result.message:
                click.echo(f"  message: {result.message}")
