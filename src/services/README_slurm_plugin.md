# Slurm job power profiling (EpilogSlurmctld)

When a job ends, Slurm runs an epilog that uses **this repo’s DB** (no external API): it reads job start/end and nodelist from env, queries power per node, writes raw CSV and plots.

## Behavior

- **EpilogSlurmctld** runs on the head node at job end. Slurm provides `SLURM_JOB_START_TIME`, `SLURM_JOB_END_TIME`, `SLURM_JOB_NODELIST` (no Slurm REST API needed).
- Power profiling runs **only** when the job was submitted with the keyword in comment (e.g. `sbatch --comment=power_profiling myjob.sh`).
- **Node → DB and metrics:**
  - Node id starting with **rpc** → database **zen4**, metrics **ZEN4_METRICS**.
  - Node id starting with **rpg** → database **h100**, metrics **H100_METRICS**.
- For each such node, the script queries raw power in the job time range, then writes under **`power_<job_id>/`** (same directory as the job’s `.out`/`.err`):
  - **`raw_power.csv`**
  - **`timeseries.png`**
  - **`energy_pie.png`** (and `energy_pie.pdf`)

Output directory is the same as the job’s `.out`/`.err` (from `SLURM_JOB_STDOUT`); if that path is unavailable, `MONSTER_POWER_OUTDIR` is used.

## Install

1. **Epilog script:** Deploy this repo on the head node. Set `EpilogSlurmctld` in `slurm.conf` to the path of `monster_epilog_slurmctld.sh` (e.g. `$REPO/src/services/monster_epilog_slurmctld.sh`).
2. **Python and script paths:** The epilog runs with a minimal environment (no user `PATH`). To use **your user venv** and repo:
   - Set **`MONSTER_POWER_PYTHON`** to the full path of the Python to run (e.g. your venv: `/home/you/repacss-power-profiling/.venv/bin/python`).
   - Set **`MONSTER_POWER_SCRIPT`** to the full path of `slurm_power_query.py` (e.g. `/home/you/repacss-power-profiling/src/services/slurm_power_query.py`).
   - These env vars must be visible **where slurmctld is started** (epilog inherits slurmctld’s environment), not in your shell. Options:
     - **systemd:** If slurmctld is started by systemd, add to the service unit (e.g. `systemctl edit slurmctld` or drop-in):
       ```ini
       [Service]
       Environment="MONSTER_POWER_PYTHON=/home/you/repacss-power-profiling/.venv/bin/python"
       Environment="MONSTER_POWER_SCRIPT=/home/you/repacss-power-profiling/src/services/slurm_power_query.py"
       ```
     - **/etc/default/slurmctld:** If your distro sources this before starting slurmctld, add:
       ```bash
       export MONSTER_POWER_PYTHON=/home/you/repacss-power-profiling/.venv/bin/python
       export MONSTER_POWER_SCRIPT=/home/you/repacss-power-profiling/src/services/slurm_power_query.py
       ```
     - **Wrapper script:** Start slurmctld from a script that exports these vars, then `exec slurmctld ...`.
3. **Environment:** The Python (venv) must have repo deps: `pandas`, `matplotlib`, `psycopg2-binary`, `sqlalchemy`, plus SSH/tunnel deps if DB is reached via SSH (see `requirements.txt`). Optional: `hostlist` for nodelist expansion. DB/SSH config (e.g. `src/database/config/.env`) must be readable when the script runs; if it runs as the job user via `runuser`, that user must be able to read the repo and config (or use a shared config path).
4. Restart `slurmctld` after changing `slurm.conf` or env.

## Enable per job

Submit with comment (keyword configurable via `MONSTER_POWER_COMMENT_KEYWORD`, default `power_profiling`):

```bash
sbatch --comment=power_profiling myjob.sh
```

## Notes

- The epilog runs the Python script as the **job user** (via `runuser` or `sudo -u`) when possible, so output files are owned by the user. Ensure `runuser` or passwordless `sudo -u <user>` is available.
- The script uses `|| true` so DB/plot failures do not affect Slurm.
- Epilog scripts must **not** call Slurm commands (e.g. `squeue`, `scontrol`); this plugin only uses env vars and the local DB.
