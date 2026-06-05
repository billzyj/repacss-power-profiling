# In-Band Profiling Design Notes

This directory contains the REPACSS in-band (IB) profiling implementation.

The current design is intentionally **Slurm-orchestrated** rather than a thin wrapper around Slurm's native `acct_gather_energy` plugin family. Slurm still provides the control plane and lifecycle hooks, but REPACSS owns the runner lifecycle, staged artifact model, provenance handling, and hybrid OOB+IB reporting flow.

## Why Not Use Only Slurm `acct_gather_energy`

Slurm's built-in `acct_gather_energy` path is designed for cluster-level configuration and native accounting or profiling integration. That model works well when the site wants one selected energy backend and one standardized data path.

REPACSS IB targets a different use case:

- users select profiling mode per job through Slurm comments
- IB can be enabled only when needed
- one job may request multiple node-local collectors
- IB data remains artifact-backed rather than being forced into the same centralized database path as OOB telemetry
- hybrid OOB+IB reporting needs source provenance and partial-failure handling that are easier to preserve in a custom staged-artifact workflow

Summary comparison:

| Aspect | `acct_gather_energy` | REPACSS IB |
|---|---|---|
| Activation | Cluster-global via `slurm.conf` | Per-job via user comment `power:inband` |
| Collector count | Single backend at a time | Multiple in parallel (rapl + nvidia_smi + rocm_smi) |
| Data granularity | Primarily node/job energy accounting; detailed time series typically requires `acct_gather_profile` | Full time-series CSV per collector |
| Data path | Slurm native accounting/profile path (e.g. `sstat`/`sacct`, HDF5, or InfluxDB depending on site configuration) | NFS staged artifacts per node |
| Storage model | Managed by Slurm's configured accounting/profile plugins | Distributed per-node writes to NFS staging |
| OOB+IB hybrid | Not supported | Native `power:both` mode with unified reporting |

In short, Slurm native energy plugins are a good fit for built-in accounting. REPACSS IB is a job-scoped profiling layer with different flexibility and reporting goals.

## Difference From `acct_gather_energy`

`acct_gather_energy` is typically configured as a single cluster-selected backend. REPACSS IB instead starts a per-job, per-node runner and can orchestrate multiple collectors in one job, such as:

- `rapl`
- `nvidia_smi`
- `rocm_smi`

Key differences:

- Slurm native energy plugins are administrator-configured and primarily cluster-global.
- REPACSS IB is selected at job submission time.
- Slurm native plugins integrate with Slurm's internal accounting/profile flow.
- REPACSS IB writes staged artifacts to shared storage and aggregates them after job completion.
- Slurm native plugins generally expose one energy backend path at a time.
- REPACSS IB can combine multiple collectors in one node-local runner.
- detailed time-series workflows in Slurm usually rely on the native profiling stack (for example HDF5 or InfluxDB), whereas REPACSS IB retains the full time-series per collector in its own staged artifact layout
- REPACSS IB keeps the storage contract explicit and separate from the OOB observability database

## Difference From `acct_gather_profile`

REPACSS IB is conceptually closer to Slurm profiling than to plain energy accounting because it produces time-series artifacts during job execution. However, it still differs from `acct_gather_profile` in several ways:

- REPACSS uses custom CSV/JSON staging rather than Slurm-managed HDF5 or InfluxDB profile outputs.
- REPACSS keeps OOB and IB storage distinct by design.
- REPACSS performs headnode aggregation specifically for hybrid job reporting.
- REPACSS tracks collector-level partial failure and provenance explicitly for reporting.

The staging layout is therefore part of the design, not just an implementation detail.

## Current Architecture

The current P4 architecture is:

- `shared/slurm/repacss-power.prolog.sh`
  starts the Python hook on each compute node
- `inband/hooks.py`
  resolves job state, writes node-local state files, and starts or stops the systemd runner unit
- `shared/slurm/repacss-power-ib@.service`
  runs one per-job, per-node runner instance
- `inband/runner.py`
  owns collector lifecycle and writes `runner_status.json`
- `inband/storage.py`
  manages storage keys, shared staging paths, env files, and status files
- `shared/slurm/repacss-power.epilog.sh`
  stops the runner and finalizes `.done` plus `node_status.json`

Compute-node IB data is staged under shared storage using an internal `storage_key`. Headnode aggregation happens later in the job-end workflow.

## Design Tradeoffs

Benefits of the current design:

- per-job collector choice
- hybrid OOB+IB workflow support
- explicit provenance and partial-failure handling
- no requirement to force IB data into the OOB telemetry database schema

Costs of the current design:

- more custom runtime logic than a pure Slurm plugin approach
- custom aggregation and artifact schema must be maintained
- systemd plus Prolog or Epilog behavior must be validated on the target cluster

## Scope

This directory implements the IB runtime and staging layer only.

It does **not** make the claim that Slurm native energy plugins are wrong or obsolete. The design choice here is narrower: REPACSS needs a job-scoped, multi-collector, hybrid-reporting-friendly IB path, and that is why the current implementation uses Slurm hooks plus a custom runner instead of relying exclusively on `acct_gather_energy`.
