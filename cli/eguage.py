"""eGauge API CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from eguage.client import EGaugeClient
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
