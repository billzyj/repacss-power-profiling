#!/usr/bin/env python3
"""Legacy compatibility wrapper for the refactored OOB Slurm handler."""

from oob.slurm.epilog_handler import *  # noqa: F401,F403


if __name__ == "__main__":
    main()
