#!/usr/bin/env bash
set -eu

PYTHON="${MONSTER_POWER_PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec "$PYTHON" -m inband.hooks prolog "$@"
