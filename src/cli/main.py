#!/usr/bin/env python3
"""
Main CLI interface for REPACSS Power Measurement
"""

import click
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .commands.analyze import analyze, energy, rack, pue
from .commands.report import excel, rack_report, custom, pue_daily
from .commands.test import connection, databases, config
from .commands.visualize import visualize


@click.group()
@click.version_option()
def cli():
    """
    REPACSS Power Measurement CLI
    
    A comprehensive tool for analyzing power consumption data from the REPACSS cluster.
    Supports H100, ZEN4, and infrastructure monitoring with Excel reporting capabilities.
    """
    pass


# Add subcommands
cli.add_command(analyze)
cli.add_command(energy)
cli.add_command(rack)
cli.add_command(pue)
cli.add_command(excel)
cli.add_command(rack_report)
cli.add_command(custom)
cli.add_command(pue_daily)
cli.add_command(connection)
cli.add_command(databases)
cli.add_command(config)
cli.add_command(visualize)


if __name__ == '__main__':
    cli()
