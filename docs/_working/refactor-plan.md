# repacss-power-profiling Refactor Plan (P0-P4 Snapshot)

Status: active working artifact

This file is a temporary collaboration artifact for refactor planning.
It may use mixed Chinese and English so the user, Codex, and Claude Code can review the same plan during implementation.

Do not treat this file as stable product documentation.
Before the refactor is finalized:

1. Move durable decisions into English documentation such as `README.md` or `docs/*`.
2. Remove this file from the repository.

## Summary

This major refactor rebuilds `repacss-power-profiling` around two first-class domains:

- `Out-of-Band (OOB)`
- `In-Band (IB)`

This is a major-version redesign with no backward-compatibility requirement. All work happens on a dedicated feature branch, is validated there, and replaces the current implementation only after end-to-end testing succeeds.

The Slurm lifecycle split is explicit:

- `OOB` remains a headnode, job-level workflow centered on `EpilogSlurmctld`
- `IB` becomes a compute-node runtime workflow centered on `Prolog + Epilog`
- `TaskEpilog` is not implemented in v1, but remains a documented future extension point

The final artifact policy is:

- primary result = user-facing export delivered to a per-user destination
- optional secondary result = restricted system copy for monitoring / visualization
- no everyone-readable shared output as the default final behavior

### Implementation phases

| Phase | Content | Depends on |
|-------|---------|------------|
| P0 | `shared/` foundation: models, config, errors, slurm helpers, comment parser | — |
| P1 | OOB refactor: backend interface + monster-db migration | P0 |
| P2 | OOB Slurm integration: `EpilogSlurmctld` OOB path | P0, P1 |
| P3 | IB collector framework + `rapl` / `nvidia_smi` / `rocm_smi` implementations | P0 |
| P4 | IB Slurm integration: `Prolog` / `Epilog` + node-level finalization | P0, P3 |
| P5 | `EpilogSlurmctld` dispatcher + IB aggregation + `both` mode orchestration | P2, P4 |
| P6 | Unified export: merge OOB + IB outputs into one artifact | P5 |
| P7 | CLI redesign | P1-P6 |
| P8 | `monster-api` backend | P1 |
| P9 | E2E validation + documentation cleanup | all |

P1 and P3 can run in parallel.
P5 is the integration point where `EpilogSlurmctld` becomes the shared dispatcher for both OOB and IB.
P8 can proceed independently after P1.

## Key Changes

### 1. Architecture split

Refactor the codebase into three implementation areas:

- `oob/`
  OOB query semantics, Slurm job resolution, DB/API adapters, OOB normalization
- `inband/`
  runtime collectors, sampling lifecycle, per-node local artifacts, aggregation staging
- `shared/`
  cross-domain contracts, export builders, comment parsing, filesystem policy, Slurm helpers, shared config/errors

Suggested structure:

```text
shared/
  config/
  models/
  slurm/
  export/
  analysis/
  constants/
  utils/
  errors.py

oob/
  backends/
    base.py
    monster_db/
    monster_api/
  analyzers/
  slurm/

inband/
  collectors/
  runner.py
  hooks.py
  slurm/
  aggregator.py
  storage.py

cli/
  main.py
  oob.py
  ib.py
  export.py
  config.py
```

Design rule:

- only OOB connects to external telemetry services
- IB does not connect to DB/API in v1
- shared owns config keys and contracts, not backend-specific clients

### 2. OOB model

OOB exposes two public query surfaces:

- `job-scoped query`
- `general metric query`

`job-scoped query`:

- primary input is `job_id`
- always uses a unified Slurm resolver
- `EpilogSlurmctld` path resolves from Slurm env on headnode
- offline/manual path resolves through Slurm API
- output is normalized into the shared export model

`general metric query`:

- node
- time window
- selected metrics
- optional aggregation / scope controls
- `job_id` semantics may internally map here, but the public API keeps job-query separate

v1 scope constraints for `general metric query`:

- v1 supports node-level and rack-level granularity only
- cross-rack aggregation is out of v1 scope
- this is a query surface, not a job export surface
- library/API return type is normalized tabular data
- CLI default behavior is stdout table preview
- CLI optional file output is allowed via `--output` in `csv` or `json`
- `general metric query` does not produce the unified job export in v1

OOB telemetry backends:

- `monster-db`
- `monster-api`

Defaults:

- `monster-db` is default
- `monster-api` is first-class
- normalization is backend-neutral

OOB backend adapter interface:

```python
class OOBBackend(ABC):
    @abstractmethod
    def query_job(self, job_id, ...) -> DataFrame: ...
    @abstractmethod
    def query_metrics(self, node, time_range, metrics, ...) -> DataFrame: ...
```

Backend selection:

- config-level default
- CLI per-query override
- comment-grammar override only when mode includes OOB

### 3. Slurm role split

#### Lifecycle overview

`EpilogSlurmctld` is the shared dispatcher on the headnode. It runs once per job, reads the comment grammar, and dispatches accordingly:

```text
Job ends
  ├─ compute nodes: Epilog fires per node
  │    └─ if mode includes IB: stop collectors -> write per-node data -> write node status -> write .done
  │
  └─ headnode: EpilogSlurmctld fires once
       ├─ parse comment grammar -> determine mode
       ├─ if mode includes IB: wait for .done markers -> merge per-node data -> write IB summary
       ├─ if mode includes OOB: query TimescaleDB or API -> write OOB result
       └─ if mode is both: unified export combines OOB + IB into one artifact
```

Compute-node `Epilog` and headnode `EpilogSlurmctld` are treated as parallel job-end triggers.
Synchronization is achieved by `.done` polling rather than assuming strict hook ordering.

`Prolog` on compute nodes:

- fires per node at job start
- if mode includes IB: starts node-local collectors

#### OOB path

- headnode `EpilogSlurmctld` queries TimescaleDB or API after job ends
- this is a one-shot read path
- OOB produces normalized result data for export assembly

#### IB path

- compute-node `Epilog` stops collectors and writes per-node artifacts into the shared IB staging area
- headnode `EpilogSlurmctld` waits for node completion markers and merges per-node data
- IB aggregation is part of `EpilogSlurmctld`, not a separate Slurm hook

#### Staging path creation

The shared job-run root must exist before any compute-node `Epilog` writes data.

Creation rule in v1:

- compute-node `Prolog` creates `{ib_store_root}/{storage_key}/` and its local `{hostname}/` subdirectory with `mkdir -p`
- headnode `EpilogSlurmctld` must treat the root as pre-existing and must not rely on being the first creator
- if `Prolog` did not run or the root is missing, compute-node `Epilog` must also `mkdir -p` the full path as a safety fallback

This guarantees that per-node finalization never depends on headnode-side directory creation timing.

#### `both` mode

- `EpilogSlurmctld` runs OOB query and IB aggregation in sequence
- unified export merges both results
- if IB aggregation times out or is partial, OOB result is still produced
- IB incompleteness is recorded in provenance and summary

`TaskEpilog`:

- not implemented in v1
- documented as a future path for per-step or per-task granularity

### 4. In-band subsystem

IB is a runtime sampling system, not another OOB query backend.

Collectors in v1:

- `rapl`
- `nvidia_smi`
- `rocm_smi`

Defaults:

- all three are first-class in v1
- auto-detect collectors by hardware
- skip unavailable collectors without hard failure
- record skip or missing state in provenance and summary
- default `interval_ms=1000`

Collector interface:

```python
class Collector(ABC):
    name: str
    @abstractmethod
    def is_available(self) -> bool: ...
    @abstractmethod
    def start(self, interval_ms: int, output_path: Path) -> CollectorHandle: ...
    @abstractmethod
    def stop(self, handle: CollectorHandle) -> CollectorResult: ...

@dataclass
class CollectorResult:
    collector_name: str
    hostname: str
    samples: DataFrame
    status: Literal["complete", "partial", "failed"]
```

### 5. IB storage contract

IB storage in v1 is file-based staging on a shared filesystem.

Architectural rule:

- IB does not write to a central DB in v1
- OOB is a job-end read path against pre-existing telemetry storage
- a central DB would become a write hotspot at scale; shared-filesystem staging distributes writes across per-node subdirectories
- v1 therefore uses shared-filesystem staging plus job-end aggregation

#### Storage root and visibility

`ib_store_root` must satisfy all of the following:

- visible from both compute nodes and headnode
- writable from compute-node `Epilog`
- readable and writable from headnode `EpilogSlurmctld`
- treated as restricted system staging, not as the final user-facing destination

This staging area is system-owned.
It is not the primary user artifact location.

#### Unique job-run path

The staging path must not use bare `job_id` alone.

Use a unique job-run key:

```text
{ib_store_root}/{storage_key}/
```

Where `storage_key` is:

```text
{cluster_name}-{job_id}-{job_start_epoch}
```

Users query by Slurm `job_id`; `storage_key` is an internal filesystem identifier only.

#### Staging layout

```text
{ib_store_root}/
  {storage_key}/
    manifest.json
    {hostname}/
      rapl.csv
      nvidia_smi.csv
      rocm_smi.csv
      node_status.json
      .done
    summary.json
```

`manifest.json` contains job-level metadata:

```json
{
  "storage_key": "repacss-12345-1775808000",
  "job_id": "12345",
  "cluster": "repacss",
  "user": "alice",
  "start_time": "2026-04-10T08:00:00Z",
  "end_time": "2026-04-10T14:00:00Z",
  "nodes": ["rpg-93-1", "rpg-93-2"],
  "collectors": ["rapl", "nvidia_smi"],
  "interval_ms": 1000,
  "mode": "both",
  "backend": "db",
  "node_status": {
    "rpg-93-1": "complete",
    "rpg-93-2": "partial"
  }
}
```

#### Ownership and write responsibilities

Ownership is split explicitly:

- headnode `EpilogSlurmctld`
  - creates `{ib_store_root}/{storage_key}/manifest.json`
  - records job-level metadata
  - waits for node completion
  - updates manifest-level node status
  - writes aggregated `summary.json`
- compute-node `Epilog`
  - creates `{hostname}/` subdirectory if missing
  - writes per-collector raw files
  - writes `node_status.json`
  - writes `.done` only after node finalization is complete

This avoids cross-node write contention.
Each node writes only to its own subdirectory.

#### `.done` semantics

`.done` means:

- node-level finalization finished
- raw files and `node_status.json` are in place
- the node may still contain collector-level failures

`.done` does not mean all collectors succeeded.

Collector success and failure are recorded in `node_status.json` and summarized into `manifest.json`.

#### Aggregation contract

- headnode waits for `.done` markers with configurable timeout
- config key: `epilog_done_timeout_s`
- default timeout is `60s`
- missing node data is recorded as incomplete
- nodes missing `.done` after timeout are marked as `timeout` and aggregation continues
- aggregation is idempotent and safe to re-run
- `summary.json` is an intermediate IB artifact, not the final user-facing export

#### File formats

- v1 raw collector files use CSV
- v1 aggregated IB summary uses JSON
- design the storage and export layer so Parquet can be added later without changing collector semantics

#### Retention

- v1 does not auto-delete IB staging data
- docs must describe administrator cleanup policy
- future retention automation is out of v1 scope

### 6. Comment grammar

v1 comment grammar is:

```text
power:<mode>;backend=<db|api>;collectors=<csv>;interval_ms=<int>
```

Required modes:

- `power:oob`
- `power:inband`
- `power:both`

Examples:

- `power:oob`
- `power:oob;backend=api`
- `power:inband`
- `power:inband;collectors=rapl,nvidia_smi;interval_ms=1000`
- `power:both;backend=api;collectors=rapl,nvidia_smi;interval_ms=500`

Defaults:

- omitted `backend` => `db`
- omitted `collectors` for `inband` or `both` => auto-detect
- omitted `interval_ms` => `1000`

Validation rules:

- `backend` is valid only when mode includes OOB
- `backend` in pure `power:inband` is treated as invalid input
- `collectors` and `interval_ms` are valid only when mode includes IB
- invalid grammar causes the job to skip power workflow and emit an explicit parse error record

Forward-compatibility note:

- v1 remains flat key-value grammar
- domain-scoped dotted keys are a future extension, not part of v1

Namespace coexistence rule:

- the Slurm comment field may contain multiple independent workflow blocks
- `power` parsing only consumes the token that starts with `power:`
- ECHO-DVFS parsing only consumes `ECHO=` and its related key-value tokens
- the two parsers must not rewrite or reinterpret each other’s tokens
- recommended mixed example:
  - `ECHO=1;R=0.80 power:both;collectors=rapl;interval_ms=1000`

### 7. Unified export and delivery policy

`power:both` produces one unified export, not two unrelated result trees.

Relationship between IB `summary.json` and the final export:

- `{ib_store_root}/{storage_key}/summary.json` is an intermediate artifact
- final unified export is the user-facing deliverable
- `power:inband` export is derived from IB summary only
- `power:both` export merges OOB data with IB summary
- intermediate staging data is retained for debugging and operator inspection

Required unified export behavior:

- one export directory
- one summary
- provenance explicitly distinguishes OOB vs IB
- raw rows preserve:
  - `source_kind`
  - `source_id`
  - `collector_family`
  - `retrieval_mode`

#### Delivery model

There are two output classes:

- system staging copy
- user-facing delivered export

Rules:

- `ib_store_root` and any headnode assembly path are system staging only
- final user-facing export must be delivered into a per-user destination
- optional secondary system copy may be retained for monitoring or visualization
- system copy must live in a restricted directory

Delivery flow:

1. headnode assembles the final export in a system-controlled temporary path
2. delivery requires privileges sufficient to create, copy, and ownership-fix the user-facing artifact:
   - if `EpilogSlurmctld` has the required privileges, it may perform delivery directly
   - otherwise it must invoke a dedicated privileged helper
   - target file ownership is set to the job user
   - target directory permissions are `0750`
3. the delivered artifact becomes the primary result

The delivery behavior must be consistent across OOB-only, IB-only, and `both`.

### 8. CLI redesign

This is a major-version redesign, so CLI may be restructured to match the architecture.

Target public surfaces:

- OOB job query commands
- OOB general metric query commands
- unified export or report commands
- OOB backend selection controls
- optional IB inspection or debug commands where useful

Suggested v1 CLI:

```text
repacss-power oob job <job_id> [--backend db|api]
repacss-power oob query --node NODE --start START --end END [--metrics METRICS] [--output PATH] [--format csv|json] [--backend db|api]
repacss-power ib status --job <job_id> [--storage-key STORAGE_KEY]
repacss-power export --job <job_id> [--output PATH] [--format csv|json|excel]
repacss-power config test
repacss-power config show
```

Rules:

- `oob job` triggers OOB query for a job and displays a summary preview to stdout; it does not produce file exports
- `oob query` is a general metrics query surface; defaults to stdout table preview, supports `--output` for file export
- `ib status` inspects IB staging and aggregation state; `--job` is the standard user-facing lookup and `--storage-key` is a debug-only exact lookup for internal staging paths
- `export` is the only command that produces file exports:
  - for OOB-only, it runs the OOB job query and writes the export artifact
  - for IB-only, it consumes IB staging artifacts
  - for `both`, it merges OOB query results with IB staging artifacts

### 9. P4 — In-Band Slurm Integration

P4 makes in-band collection a Slurm-driven, per-node systemd service workflow modeled after the existing `echo.prolog.sh` pattern, while keeping shared/NFS staging as the authoritative storage path.

Fixed decisions for P4:

- `Prolog` starts one job-scoped IB collector service per compute node when comment mode includes IB.
- `Epilog` on each compute node stops that service and finalizes per-node artifacts.
- `EpilogSlurmctld` on the headnode remains the single job-level aggregator and orchestrator.
- authoritative IB storage is a shared filesystem path:
  - `{ib_store_root}/{storage_key}/{hostname}/...`
- user-facing delivery is optional in P4:
  - P4 must produce correct system/NFS staging and headnode aggregation
  - delivery into user directories is an enhancement path, not a hard blocker for P4 completion
- local spool is optional, not default:
  - default path is direct write to NFS staging
  - local scratch may be added behind config as an optimization, but must not be the source of truth

#### Runtime model

Use the existing `echo.prolog.sh` idea as the operational template.

`Prolog`:

- parse the job comment
- if mode is `inband` or `both`, compute `storage_key`
- create shared staging root and node subdirectory
- persist node-local runtime state in a file such as `/run/repacss-power/${SLURM_JOB_ID}.env`
- create/start a systemd unit such as:
  - `repacss-power-ib@<storage_key>.service`
- pass:
  - `job_id`
  - `storage_key`
  - `hostname`
  - `interval_ms`
  - selected collectors
  - shared output directory

Integration rule:

- power collection must integrate as its own Slurm helper scripts, not by merging into `echo.prolog.sh`
- expected script layout:
  - `shared/slurm/repacss-power.prolog.sh`
  - `shared/slurm/repacss-power.epilog.sh`
- the site `slurm.prolog` wrapper may call the power prolog with `|| true` so power startup failures do not block job launch
- the compute-node epilog hook must likewise call the power epilog independently of ECHO-DVFS teardown

`Epilog` on compute nodes:

- stop the matching systemd unit
- read the node-local state file created by `Prolog`
- collect stop results from each collector
- write:
  - collector CSV files
  - `node_status.json`
  - `.done`

`storage_key` lookup rule:

- `Prolog` is the authoritative creator of `storage_key`
- `Epilog` must not recompute `storage_key` from Slurm timestamps unless the state file is missing
- primary lookup path is the node-local state file written by `Prolog`
- fallback lookup may use the systemd unit instance name or Slurm job metadata only when the state file is unavailable

`EpilogSlurmctld` on headnode:

- unchanged role: one run per job
- if mode includes IB:
  - wait for expected node `.done` markers
  - read per-node artifacts from NFS
  - build aggregated IB summary
- if mode includes OOB:
  - run the existing OOB query path
- if mode is `both`:
  - merge OOB and IB into one unified export

#### Systemd service shape

Add a small runner that owns the lifetime of the selected P3 collectors.

The service is responsible for:

- reading collector selection and interval from environment or args
- starting all requested/auto-detected collectors
- keeping the process alive until stopped by `Epilog`
- stopping collectors cleanly on SIGTERM
- writing collector outputs to the node staging directory
- writing a service-level status file such as `runner_status.json` if startup fails before collector output exists

P4 should define one node-local runner process per job/node, not one service per collector.

Recommended unit structure:

- template unit:
  - `repacss-power-ib@.service`
- runner:
  - Python module at `inband/runner.py`

The systemd unit must run as a system service on the compute node, matching the existing Slurm Prolog operational model.

Runner signal contract:

- the runner must install a SIGTERM handler
- SIGTERM sets a process-level stop flag and triggers orderly collector shutdown
- the runner must stop all active collectors before exiting
- the runner must flush any buffered state before exit
- normal runner shutdown writes `runner_status.json` with a terminal state such as `complete` or `partial`
- if `Epilog` cannot find `runner_status.json`, or systemd reports a non-zero unit exit, it must mark the node as `failed`
- a forced or abnormal termination must still leave enough state for `Epilog` to write a failed or partial `node_status.json`

#### P4 storage policy

P4 uses shared/NFS staging as the only authoritative IB storage for live sampling and aggregation.

Rules:

- `ib_store_root` must be visible on compute nodes and headnode
- compute-node service writes only its own `{hostname}/` subtree
- headnode writes only job-level files such as:
  - `manifest.json`
  - `summary.json`
- no compute node writes directly into another node’s subtree
- no user directory is used as the primary live sampling destination

Local spool policy:

- default: collectors write directly to NFS staging
- optional config later may allow:
  - write to node-local temp
  - `Epilog` copies to NFS before `.done`
- even if local spool is enabled later, NFS remains the source of truth for aggregation

#### P4 acceptance criteria

Validate:

- `Prolog` with `power:inband` starts the systemd unit
- the service stays alive during job runtime
- `Epilog` stops the unit cleanly
- expected collector CSVs appear under the node staging directory
- `node_status.json` and `.done` are written
- `EpilogSlurmctld` reads staged artifacts and writes `summary.json`
- aggregation still completes when some nodes never write `.done`, and those nodes are marked `timeout`
- `power:both` runs IB aggregation and OOB query in one job-end flow
- compute nodes do not require direct user-directory writes for live collection
- unavailable collectors are recorded as partial/failed without blocking the whole workflow

## Migration Mapping

Existing code assets that should be refactored rather than rewritten from scratch:

| Current module | Target location | Action |
|----------------|-----------------|--------|
| `src/database/client.py` | `oob/backends/monster_db/client.py` | refactor into OOB-internal DB client |
| `src/database/config/` | `shared/config/` | refactor shared config only |
| `src/database/database.py`, `connection_pool.py` | `oob/backends/monster_db/` | refactor as OOB-only DB layer |
| `src/queries/manager.py` plus query modules | `oob/backends/monster_db/` | refactor as DB query layer |
| `src/analysis/energy.py` | `shared/analysis/` | migrate for shared energy logic |
| `src/analysis/power.py` | `oob/analyzers/` | refactor as OOB-specific analysis |
| `src/services/slurm_power_query.py` | `oob/slurm/epilog_handler.py` | refactor as OOB epilog path |
| `src/services/slurm.epilogslurmctld` | `shared/slurm/dispatcher.py` | refactor as shared dispatcher |
| `src/reporting/*` | `shared/export/` | refactor into unified export builders |
| `src/constants/*` | `shared/constants/` | migrate directly |
| `src/utils/conversions.py` | `shared/utils/` | migrate directly |
| `src/utils/node_detection.py` | `shared/slurm/` or `shared/utils/` | minor refactor |
| `src/cli/` | `cli/` | rewrite to new command tree |

## Test Plan

### 1. Fresh clone smoke test

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/run_tests.py --type unit
```

### 2. OOB validation

Validate:

- OOB job query via DB backend
- OOB job query via API backend
- OOB general metric query via DB backend
- OOB general metric query via API backend
- `power:oob` headnode `EpilogSlurmctld` path
- offline Slurm resolver path using Slurm API
- user-facing export delivery path
- optional restricted system copy path

### 3. IB validation

Validate:

- `power:inband` on Intel CPU host with auto-detected RAPL
- `power:inband` on NVIDIA host with auto-detected `nvidia-smi`
- `power:inband` on AMD host with auto-detected `rocm_smi`
- explicit collector override through comment grammar
- default `interval_ms=1000`
- compute-node `Prolog` start flow
- compute-node `Epilog` stop and finalize flow
- shared staging path correctness
- unique `storage_key` path generation
- `.done` semantics with partial collector failure
- headnode job-level merge correctness
- unsupported collector recorded in provenance without hard failure

### 4. Unified export validation

Validate:

- `power:oob` export shape
- `power:inband` export shape
- `power:both` export shape
- summary and provenance distinguish OOB and IB correctly
- raw rows preserve provenance fields
- final delivered export is user-facing
- optional restricted system copy, if enabled, has correct permissions

### 5. Unit test policy

Required unit test areas:

- comment parser for valid and invalid grammar combinations
- collector auto-detect with mocked hardware detection
- export builder provenance correctness
- Slurm env resolver with missing and partial env vars
- energy calculation edge cases
- OOB backend normalization so DB and API return the same schema
- IB manifest read or write behavior
- `.done` and node-status semantics
- delivery-path and path-generation helpers

### 6. Documentation validation

Before finalization:

- English README reflects final architecture and workflows
- English docs explain:
  - OOB role split
  - IB role split
  - comment grammar
  - staging vs delivered artifact policy
  - restricted system-copy policy
  - remote-machine validation flow
- temporary `docs/_working/refactor-plan.md` is deleted

## Assumptions And Defaults

- No backward compatibility is required.
- Work happens on a new feature branch and replaces the current implementation after validation.
- OOB remains headnode/job-level and centered on `EpilogSlurmctld`.
- IB becomes compute-node runtime collection centered on `Prolog + Epilog`.
- `TaskEpilog` is out of v1 scope but documented as a future extension point.
- OOB supports both `monster-db` and `monster-api`, with DB as default.
- IB v1 is shared-filesystem staging plus headnode aggregation, not central DB persistence.
- IB default collectors are auto-detected.
- IB default interval is `1000 ms`.
- Staging paths are system-owned and restricted.
- Final delivered exports are user-facing and ownership-corrected during delivery.
- Temporary mixed-language planning content is allowed only during implementation and must not remain in the final repo state.
