"""Configuration-oriented CLI commands."""

from __future__ import annotations

import click

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
    click.echo(f"available_databases: {', '.join(config.databases)}")


@config_group.command("test")
def test_config():
    """Validate local configuration."""
    issues = config.validate_config()
    if issues:
        click.echo("Configuration issues found:")
        for issue in issues:
            click.echo(f"- {issue}")
        raise SystemExit(1)
    click.echo("Configuration validation passed.")

