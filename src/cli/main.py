#!/usr/bin/env python3
"""Legacy compatibility entrypoint that delegates to the refactored CLI."""

from cli.main import cli


if __name__ == "__main__":
    cli()
