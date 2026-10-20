#!/usr/bin/env python3
"""Legacy compatibility wrapper for the refactored OOB Slurm handler.

The installed EpilogSlurmctld wrapper runs this file by path, so Python puts only
`src/services/` on sys.path. Add the repository root so the refactored packages import.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from oob.slurm.epilog_handler import *  # noqa: E402,F401,F403


if __name__ == "__main__":
    main()
