#!/usr/bin/env python3
"""Main CLI entrypoint for the refactored architecture."""

import click

from .config import config_group
from .export import export_command
from .ib import ib_group
from .oob import oob_group


@click.group()
@click.version_option()
def cli():
    """REPACSS power profiling CLI."""


cli.add_command(config_group)
cli.add_command(ib_group)
cli.add_command(oob_group)
cli.add_command(export_command)
