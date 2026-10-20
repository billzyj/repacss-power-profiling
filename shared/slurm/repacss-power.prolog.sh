#!/bin/bash
# Compute-node Prolog helper for REPACSS in-band profiling.
# Telemetry must never fail the hook: a non-zero Prolog drains the node and requeues the job,
# so this wrapper logs any helper failure and exits 0.
set -u

# Slurm runs hooks without a search path.
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

PYTHON="${MONSTER_POWER_PYTHON:-/usr/bin/python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# No trailing colon: an empty entry would put slurmd's working directory on sys.path.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if ! "$PYTHON" -m inband.hooks prolog "$@"; then
    logger -t repacss-power "in-band prolog helper failed for job ${SLURM_JOB_ID:-unknown}; continuing without telemetry"
fi
exit 0
