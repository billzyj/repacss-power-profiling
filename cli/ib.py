"""Debug-oriented CLI helpers for in-band collectors."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Optional

import click

from inband.collectors import build_collectors, probe_collectors, select_collectors
from inband.spool import sample_path
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


def _node_options(command):
    """Add the partition and node RAPL uses to select its audited profile."""
    command = click.option(
        "--node",
        default=None,
        help="Slurm node name for RAPL profile selection. Defaults to the short hostname.",
    )(command)
    return click.option(
        "--partition",
        default=os.environ.get("SLURM_JOB_PARTITION", ""),
        help="Slurm partition for RAPL profile selection. Defaults to SLURM_JOB_PARTITION.",
    )(command)


def _local_node(node: Optional[str]) -> str:
    return node or socket.gethostname().split(".", 1)[0]


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
    type=click.Choice(["rapl", "nvml", "nvidia_smi", "rocm_smi"]),
    help="Optionally probe only selected collectors.",
)
@_node_options
def probe_command(collectors, partition: str, node: Optional[str]):
    """Probe in-band collector availability without starting sampling."""
    requested = collectors or None
    for collector_probe in probe_collectors(requested, partition=partition, node=_local_node(node)):
        _print_probe(collector_probe)


@ib_group.command("sample")
@click.option(
    "--collector",
    "collectors",
    multiple=True,
    type=click.Choice(["rapl", "nvml", "nvidia_smi", "rocm_smi"]),
    help="Collectors to run. Defaults to auto-detected available collectors.",
)
@click.option("--interval-ms", default=1000, show_default=True, type=int, help="Sampling interval in milliseconds.")
@click.option("--duration-s", default=3.0, show_default=True, type=float, help="Sampling duration in seconds.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("output") / "inband-debug",
    show_default=True,
    help="Directory for collector samples, written as samples/<collector>.csv.",
)
@_node_options
def sample_command(
    collectors, interval_ms: int, duration_s: float, output_dir: Path, partition: str, node: Optional[str]
):
    """Run a short local sampling session for collector validation.

    This is a debug surface only. Production in-band collection remains Slurm-driven.
    """

    requested = collectors or None
    context = {"partition": partition, "node": _local_node(node)}
    instances = build_collectors(requested, **context) if collectors else select_collectors(**context)
    if not instances:
        raise click.ClickException("No available in-band collectors were selected.")

    output_dir.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for collector in instances:
            path = sample_path(output_dir, collector.name)
            handles.append(collector.start(interval_ms=interval_ms, output_path=path))
            click.echo(f"started {collector.name} -> {path}")
        time.sleep(duration_s)
    except InbandCollectorError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        for collector, handle in zip(instances, handles):
            result = collector.stop(handle)
            click.echo(
                f"{result.collector_name}: status={result.status} samples={result.sample_count} "
                f"output={result.output_path}"
            )
            if result.message:
                click.echo(f"  message: {result.message}")
