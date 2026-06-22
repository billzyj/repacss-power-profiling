"""Configuration-oriented CLI commands."""

from __future__ import annotations

import click

from shared.connection_policy import get_configured_access_mode, get_probe_timeout
from shared.config import config


@click.group(name="config")
def config_group():
    """Inspect and validate REPACSS configuration."""


@config_group.command("show")
def show_config():
    """Show non-secret configuration values."""
    click.echo("REPACSS configuration")
    click.echo("====================")
    click.echo(f"db_host: {config.db_host}")
    click.echo(f"db_port: {config.db_port}")
    click.echo(f"db_default_name: {config.db_default_name}")
    click.echo(f"db_user: {config.db_user}")
    click.echo(f"db_ssl_mode: {config.db_ssl_mode}")
    click.echo(f"ssh_hostname: {config.ssh_hostname}")
    click.echo(f"ssh_port: {config.ssh_port}")
    click.echo(f"ssh_username: {config.ssh_username}")
    click.echo(f"ssh_key_path: {config.ssh_private_key_path or '<system default>'}")
    click.echo(f"db_access_mode: {get_configured_access_mode('db')}")
    click.echo(f"eguage_access_mode: {get_configured_access_mode('eguage')}")
    click.echo(f"db_probe_timeout: {get_probe_timeout('db')}")
    click.echo(f"eguage_probe_timeout: {get_probe_timeout('eguage')}")
    click.echo(f"slurm_rest_host: {config.slurm_rest_host or '<not configured>'}")
    click.echo(f"slurm_rest_port: {config.slurm_rest_port}")
    click.echo(f"slurm_rest_user: {config.slurm_rest_user or '<not configured>'}")
    click.echo(f"slurm_rest_headnode: {config.slurm_rest_headnode or '<not configured>'}")
    click.echo(f"available_databases: {', '.join(config.databases)}")


@config_group.command("test")
def test_config():
    """Validate local configuration."""
    issues = config.validate_config()
    try:
        get_configured_access_mode("db")
        get_configured_access_mode("eguage")
        get_probe_timeout("db")
        get_probe_timeout("eguage")
    except ValueError as exc:
        issues.append(str(exc))
    if issues:
        click.echo("Configuration issues found:")
        for issue in issues:
            click.echo(f"- {issue}")
        raise SystemExit(1)
    click.echo("Configuration validation passed.")
