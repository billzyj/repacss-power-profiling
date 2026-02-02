#!/bin/bash
# EpilogSlurmctld script: run at job end on the head node (slurmctld).
# Only runs MonSter API call when the job was submitted with power profiling
# enabled (see README_slurm_plugin.md). Uses SLURM_JOB_COMMENT to detect.
#
# Optional env (set where slurmctld is started so the epilog sees them):
#   MONSTER_POWER_PYTHON  - full path to Python (e.g. your venv: /home/you/repo/.venv/bin/python)
#   MONSTER_POWER_SCRIPT  - full path to slurm_power_query.py (e.g. /home/you/repo/src/services/slurm_power_query.py)
#   MONSTER_POWER_OUTDIR  - fallback output dir if SLURM_JOB_STDOUT dir is unavailable
#   MONSTER_POWER_COMMENT_KEYWORD (default: power_profiling)

# Only run if this job requested power profiling (sbatch --comment=power_profiling)
KEYWORD="${MONSTER_POWER_COMMENT_KEYWORD:-power_profiling}"
case "${SLURM_JOB_COMMENT:-}" in
  *"${KEYWORD}"*) ;;
  *) exit 0 ;;
esac

# Output next to job .out/.err (same dir as SLURM_JOB_STDOUT), in power_{job_id}/
OUTDIR=""
if [ -n "${SLURM_JOB_STDOUT}" ]; then
  OUTDIR="$(dirname "${SLURM_JOB_STDOUT}")"
fi
if [ -z "${OUTDIR}" ] || [ ! -d "${OUTDIR}" ]; then
  OUTDIR="${MONSTER_POWER_OUTDIR:-}"
fi
export MONSTER_POWER_OUTDIR="${OUTDIR}"

# Slurm does not set PATH for epilog; use full paths. Prefer venv Python if set.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MONSTER_POWER_PYTHON:-${PYTHON:-/usr/bin/python3}}"
SCRIPT="${MONSTER_POWER_SCRIPT:-${SCRIPT_DIR}/slurm_power_query.py}"

# Run as job user so the output file is in the user's .out/.err dir with correct ownership
RUN_AS="${SLURM_JOB_USER:-}"
_EXEC() {
  env SLURM_JOB_ID="${SLURM_JOB_ID}" SLURM_JOB_START_TIME="${SLURM_JOB_START_TIME}" \
      SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME}" SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST}" \
      MONSTER_POWER_OUTDIR="${OUTDIR}" "${PYTHON}" "${SCRIPT}" || true
}
if [ -f "${SCRIPT}" ] && [ -n "${OUTDIR}" ]; then
  if [ -n "${RUN_AS}" ] && [ "${RUN_AS}" != "root" ]; then
    if runuser --version &>/dev/null 2>&1; then
      runuser -u "${RUN_AS}" -- env SLURM_JOB_ID="${SLURM_JOB_ID}" SLURM_JOB_START_TIME="${SLURM_JOB_START_TIME}" SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME}" SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST}" MONSTER_POWER_OUTDIR="${OUTDIR}" "${PYTHON}" "${SCRIPT}" || true
    elif sudo -n -u "${RUN_AS}" true 2>/dev/null; then
      sudo -u "${RUN_AS}" env SLURM_JOB_ID="${SLURM_JOB_ID}" SLURM_JOB_START_TIME="${SLURM_JOB_START_TIME}" SLURM_JOB_END_TIME="${SLURM_JOB_END_TIME}" SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST}" MONSTER_POWER_OUTDIR="${OUTDIR}" "${PYTHON}" "${SCRIPT}" || true
    else
      _EXEC
    fi
  else
    _EXEC
  fi
fi
exit 0
