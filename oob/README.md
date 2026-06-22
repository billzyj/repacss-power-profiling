# Out-of-Band Profiling

This directory contains the refactored out-of-band (OOB) power-query path for REPACSS.

OOB is the database-backed side of the project: it reads power telemetry that was already collected by the cluster infrastructure, then exposes that data through a small Python API, CLI commands, and the Slurm job-end export flow.

## Current Scope

Implemented now:

- direct Monster DB access through `MonsterDBBackend`
- generic hostname/time-range queries through `QueryManager`
- job-scoped queries for H100 and ZEN4 Slurm jobs
- CSV and figure export for completed jobs
- Slurm epilog integration helpers

Reserved for later:

- `MonsterAPIBackend` exists only as a placeholder and raises `NotImplementedError`
- offline Slurm job lookup without Slurm-provided environment variables is not implemented yet

## Directory Layout

```text
oob/
├── query_manager.py              # generic hostname/time-range query API
├── backends/
│   ├── base.py                   # backend interface
│   ├── monster_db/               # implemented direct DB backend
│   └── monster_api/              # planned API backend placeholder
└── slurm/
    └── epilog_handler.py         # job export, summaries, plots, epilog entrypoint
```

## Hostname Routing

The generic query API infers the backing database from the hostname prefix:

| Prefix | Node type | Database | Schema |
|---|---|---|---|
| `rpg` | H100 compute node | `h100` | `idrac` |
| `rpc` | ZEN4 compute node | `zen4` | `idrac` |
| `irc` | cooling infrastructure | `infra` | `irc` |
| `pdu` | power distribution unit | `infra` | `pdu` |

Unknown prefixes raise an error.

## Database Access

OOB reads Monster-backed TimescaleDB data through the shared REPACSS connection policy:

- `REPACSS_DB_ACCESS_MODE=auto` probes `REPACSS_DB_HOST:REPACSS_DB_PORT` first
- if the probe succeeds, OOB connects directly to the database
- if the probe fails, OOB opens an SSH local-forward through `REPACSS_SSH_HOSTNAME`
- `REPACSS_DB_ACCESS_MODE=direct` always skips SSH
- `REPACSS_DB_ACCESS_MODE=tunnel` always uses SSH

The default jump host in `env.template` is `narumuu.ttu.edu`. Keep real database credentials and SSH settings in the local root `.env`; that file is ignored by git.

## Quick Start

Run a recent-data query from the CLI:

```bash
python3 -m cli oob query --hostname rpg-93-1 --limit 20
```

Run a fixed-window query and save the rows:

```bash
python3 -m cli oob query \
  --hostname rpc-91-2 \
  --start "2025-01-01 00:00:00" \
  --end "2025-01-01 01:00:00" \
  --output output/rpc-91-2.csv \
  --format csv
```

Preview one job-scoped export:

```bash
python3 -m cli oob job \
  --job-id 96597 \
  --user kalebuch \
  --nodelist "rpg-93-[1-2]" \
  --start "2026-04-13 00:00:00" \
  --end "2026-04-13 01:00:00"
```

If Slurm REST is configured, the CLI can also resolve the time window and node list from SlurmDBD using only the job id:

```bash
python3 -m cli oob job --job-id 96597
```

Write the full OOB export bundle for a job:

```bash
python3 -m cli export \
  --job 96597 \
  --user kalebuch \
  --nodelist "rpg-93-[1-2]" \
  --start "2026-04-13 00:00:00" \
  --end "2026-04-13 01:00:00" \
  --output output/export-96597
```

With Slurm REST configured, this shorter form is also supported:

```bash
python3 -m cli export --job 96597 --output output/export-96597
```

That export path writes:

- `raw_power.csv`
- `power_timeseries.pdf`
- `energy_ring.pdf`

## Python API Examples

### 1. Generic telemetry query with `MonsterDBBackend`

Use this when callers should not need to know the lower-level query manager.

```python
from datetime import datetime

from oob.backends.monster_db.backend import MonsterDBBackend

backend = MonsterDBBackend(database="h100")

recent = backend.query_metrics("rpg-93-1", limit=20)

window = backend.query_metrics(
    "rpg-93-1",
    start_time=datetime(2025, 1, 1, 0, 0, 0),
    end_time=datetime(2025, 1, 1, 1, 0, 0),
)

print(recent[["timestamp", "hostname", "metric", "value", "units"]].tail())
print(window["metric"].value_counts())
```

Notes:

- without `start_time` and `end_time`, `limit` is applied globally to the combined recent rows
- with a time window, all matching rows are returned; `limit` is still passed to query builders but is not used to trim the final window result

### 2. Lower-level generic query with `QueryManager`

Use this when a caller wants direct access to metadata helpers in addition to telemetry rows.

```python
from oob.query_manager import QueryManager

manager = QueryManager(database="infra")

pdu_rows = manager.get_power_metrics("pdu-91-1", limit=10)
irc_rows = manager.get_power_metrics("irc-91-1", limit=10)
definitions = manager.get_power_metrics_definition()
database_info = manager.get_database_info()

print(pdu_rows.head())
print(irc_rows["metric"].unique())
print(definitions[["metric_id", "units"]].head())
print(database_info["schemas"])
```

The current public methods are:

| Method | Purpose |
|---|---|
| `get_power_metrics(hostname, start_time=None, end_time=None, limit=100)` | query compute, IRC, or PDU rows for one hostname |
| `get_metrics_definition(database=None, schema=None)` | return all metric definitions |
| `get_power_metrics_definition(database=None, schema=None)` | return power-oriented metric definitions |
| `get_database_info(database=None)` | return DB version, table count, and visible schemas |

### 3. Job-scoped query with `SlurmJobContext`

Use this when the query window and node set belong to one completed Slurm job.

```python
from datetime import datetime

from oob.backends.monster_db.backend import MonsterDBBackend
from shared.models import SlurmJobContext

context = SlurmJobContext(
    job_id="96597",
    user="kalebuch",
    nodelist="rpg-93-[1-2]",
    nodes=["rpg-93-1", "rpg-93-2"],
    start_time=datetime(2026, 4, 13, 0, 0, 0),
    end_time=datetime(2026, 4, 13, 1, 0, 0),
    comment="power:oob",
)

backend = MonsterDBBackend()
raw_df = backend.query_job(context)

print(raw_df[["timestamp", "hostname", "metric", "value", "units"]].head())
```

`query_job()` currently supports compute nodes whose hostnames begin with:

- `rpg` for H100 nodes
- `rpc` for ZEN4 nodes

Other node prefixes are skipped by the job-scoped backend.

### 4. Summarize one job in memory

```python
from oob.slurm.epilog_handler import summarize_job_power

raw_df, pie_segments, energy_by_metric, energy_gpu_per_fqdd = summarize_job_power(
    raw_df,
    context.nodes,
    context.start_time,
    context.end_time,
)

print(energy_by_metric)
print(pie_segments)
print(energy_gpu_per_fqdd)
```

The summary path:

- computes energy in kWh per metric
- separates GPU energy by FQDD for H100 data when available
- derives ring-chart segments such as CPU, Memory, Fan, PSU loss, and Others

### 5. Write the same artifacts used by the Slurm epilog

```python
from pathlib import Path

from oob.slurm.epilog_handler import handle_oob_job

output_dir = handle_oob_job(context, Path("output/export-96597"))
print(output_dir)
```

For lower-level control, the export helpers are also public:

```python
from pathlib import Path

from oob.slurm.epilog_handler import plot_pie, plot_time_series, save_csv

save_csv(
    raw_df,
    Path("output/raw_power.csv"),
    energy_by_metric=energy_by_metric,
    pie_segments=pie_segments,
    energy_gpu_per_fqdd=energy_gpu_per_fqdd,
    is_h100=True,
)
plot_time_series(raw_df, Path("output/power_timeseries.pdf"))
plot_pie(pie_segments, Path("output/energy_ring.pdf"), job_id=context.job_id)
```

## Slurm Integration

The job-end flow can resolve `SlurmJobContext` directly from environment variables:

```python
from oob.slurm.epilog_handler import handle_oob_job
from shared.slurm.resolver import resolve_job_context_from_env

context = resolve_job_context_from_env()
handle_oob_job(context)
```

Relevant environment variables include:

- `SLURM_JOB_ID`
- `SLURM_JOB_USER`
- `SLURM_JOB_NODELIST`
- `SLURM_JOB_START_TIME`
- `SLURM_JOB_END_TIME`
- `SLURM_JOB_COMMENT`

By default, `handle_oob_job()` writes under:

```text
/mnt/SHARED-AREA/power_reports/<user>/<job_id>/
```

Set `MONSTER_POWER_BASE` to change the base directory, or pass `out_dir` explicitly.

### Offline lookup through Slurm REST

When no Slurm job environment is available, OOB can resolve a completed job from SlurmDBD:

```python
from shared.slurm.resolver import resolve_job_context

context = resolve_job_context("96597")
```

This path requests a short-lived token by running:

```bash
ssh <headnode> 'scontrol token lifespan=3600'
```

Then it calls:

```text
http://<host>:<port><db_job_path><job_id>
```

For the MonSTer-compatible values below, that becomes:

```text
http://nuetu:6820/slurmdb/v0.0.42/job/96597
```

`slurm_rest_user` is sent as `X-SLURM-USER-NAME` in the HTTP request. It is not the SSH login user; SSH still uses the current shell identity or the matching entry in `~/.ssh/config`.

## Configuration

OOB uses the same `.env` file as the database and SSH settings. Add the Slurm REST settings beside the existing DB configuration:

```bash
REPACSS_SLURM_REST_HOST=nuetu
REPACSS_SLURM_REST_PORT=6820
REPACSS_SLURM_REST_USER=yongzhao
REPACSS_SLURM_REST_HEADNODE=repacss
REPACSS_SLURM_REST_JOBS_PATH=/slurm/v0.0.42/jobs/
REPACSS_SLURM_REST_NODES_PATH=/slurm/v0.0.42/nodes/
REPACSS_SLURM_REST_JOB_PATH=/slurm/v0.0.42/job/
REPACSS_SLURM_REST_DB_JOB_PATH=/slurmdb/v0.0.42/job/
REPACSS_SLURM_REST_OPENAPI_PATH=/openapi/v3
```

This is the same logical configuration as MonSTer's YAML block, expressed in the refactored repository's existing env-based config system.

## Backend Status

| Backend | Status | Notes |
|---|---|---|
| `monster-db` | implemented | direct DB access through SSH-backed DB clients |
| `monster-api` | planned | package path reserved; methods intentionally raise `NotImplementedError` |

The backend-neutral contract is defined in `oob/backends/base.py`:

```python
class OOBBackend(ABC):
    def query_job(self, context: SlurmJobContext) -> pd.DataFrame:
        ...

    def query_metrics(
        self,
        hostname: str,
        start_time=None,
        end_time=None,
        limit: int = 100,
    ) -> pd.DataFrame:
        ...
```

## Practical Notes

- OOB queries return `pandas.DataFrame` objects.
- Recent queries are intended for quick inspection; fixed-window queries are the better default for reproducible analysis.
- Compute job summaries use the configured metric sets from `src/constants/metrics.py` through the temporary shared bridge.
- The refactor is still in progress, so some OOB code currently depends on legacy query builders under `src/queries/`.
- The CLI is the stable user-facing surface today; the Python API is already useful, but still evolving with the migration.
