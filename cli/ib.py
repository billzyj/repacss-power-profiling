"""Debug-oriented CLI helpers for in-band collectors."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import click

from inband.collectors import build_collectors, probe_collectors, select_collectors
from inband.storage import (
    DEFAULT_IB_STORE_ROOT,
    done_marker_path,
    node_status_path,
    read_json_file,
    resolve_storage_key_for_job,
    runner_status_path,
)
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


@ib_group.command("status")
@click.option("--job", "job_id", help="Resolve the latest internal storage key for this Slurm job id.")
@click.option("--storage-key", help="Inspect one exact internal staging key.")
@click.option("--hostname", default=None, help="Hostname subtree to inspect. Defaults to the local hostname.")
@click.option(
    "--store-root",
    type=click.Path(path_type=Path),
    default=Path(os.environ.get("MONSTER_POWER_IB_STORE_ROOT", str(DEFAULT_IB_STORE_ROOT))),
    show_default=True,
    help="Shared in-band staging root.",
)
def status_command(job_id: Optional[str], storage_key: Optional[str], hostname: Optional[str], store_root: Path):
    """Inspect IB staging and runner status for one node."""

    hostname = hostname or os.uname().nodename
    if not storage_key and not job_id:
        raise click.ClickException("Pass --job or --storage-key.")

    resolved_storage_key = storage_key or resolve_storage_key_for_job(job_id=job_id or "", ib_store_root=store_root)
    if not resolved_storage_key:
        raise click.ClickException("No matching in-band staging path was found.")

    runner_status = read_json_file(runner_status_path(resolved_storage_key, hostname, ib_store_root=store_root))
    node_status = read_json_file(node_status_path(resolved_storage_key, hostname, ib_store_root=store_root))
    done_path = done_marker_path(resolved_storage_key, hostname, ib_store_root=store_root)

    click.echo(f"storage_key: {resolved_storage_key}")
    click.echo(f"hostname: {hostname}")
    click.echo(f"done: {'yes' if done_path.exists() else 'no'}")
    click.echo("runner_status:")
    click.echo(json.dumps(runner_status or {"state": "missing"}, indent=2, sort_keys=True))
    click.echo("node_status:")
    click.echo(json.dumps(node_status or {"state": "missing"}, indent=2, sort_keys=True))


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
