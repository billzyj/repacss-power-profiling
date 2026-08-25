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

Compute-node IB data is staged under shared storage using an internal `storage_key`. The current repository does not yet implement production headnode aggregation or user-facing delivery; those are the main gaps addressed by the implementation plan below.

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

## Concrete Implementation Plan

> Status: proposed implementation plan as of 2026-07-31. The target files and behaviors below are not claims about the current implementation. Progress is accepted milestone by milestone using the exit criteria in this document.

### Goal and user experience

After one-time cluster-administrator deployment, the user-facing workflow must require only a Slurm comment:

```bash
sbatch \
  --comment='power:inband;collectors=rapl,nvidia_smi;interval_ms=1000' \
  job.sbatch
```

The site also supports the shorter form, using configured defaults:

```bash
sbatch --comment='power:inband' job.sbatch
```

The completed attempt is delivered to a deterministic user-readable directory:

```text
<report_root>/<uid>/<job_id>/<attempt_id>/
├── power_profile/
│   ├── raw_power.csv
│   └── summary.json
└── provenance.json
```

This attempt directory is the `POWER_EXPORT_DIR` defined by the REPACSS power export contract. Directories are owned by the job UID/GID with mode `0750`; regular files use mode `0640` or stricter.

The IB path must remain structurally different from OOB:

- OOB performs a post-job query against a central telemetry backend.
- IB samples on the allocated compute nodes while the job is running.
- OOB can produce a result without node-local services.
- IB publishes job-local chunks and statuses, then an asynchronous finalizer assembles files.
- A pure `power:inband` job must work when the OOB database and API are unavailable.

### Target architecture

```text
sbatch --comment='power:inband;...'
  |
  v
site job_submit.lua
  - validate the power namespace
  - reject invalid requests
  - preserve unrelated comment content
  - choose the audited handoff: reserved Extra or canonical Comment
  |
  v
compute-node Prolog on every allocated node
  - read the selected trusted request environment
  - derive the same attempt_id on every node
  - persist allocation-scoped runtime state
  - start one systemd runner
  |
  v
per-job/per-node runner
  - sample only allowed and allocated devices
  - rotate closed chunks
  - hand closed chunks to the node-side stager
  |
  v
compute-node Epilog
  - stop and flush with a bounded timeout
  - request final stage-out without waiting indefinitely
  |
  v
node-side stager
  - copy and checksum closed chunks into shared staging
  - publish node_status.json
  - create .done last
  |
  +---------------- job-end hooks run independently ----------------+
  |
  v
EpilogSlurmctld dispatcher
  - atomically enqueue finalization work
  - return immediately
  |
  v
headnode finalizer service
  - wait for expected .done markers with a deadline
  - validate and normalize chunks
  - classify complete / partial / failed
  - atomically publish the user-facing export
```

The headnode hook does not poll for node completion. It only writes a small queue record. Waiting, aggregation, checksums, export, retry, and delivery run outside the Slurm controller hook.

### 1. Freeze the v1 request contract

The canonical REPACSS token is:

```text
power:<mode>[;backend=<backend>][;collectors=<collector>[,<collector>...]][;interval_ms=<integer>]
```

Initial values:

| Field | Allowed values | Default |
|---|---|---|
| `mode` | `oob`, `inband`, `both` | required |
| `backend` | `db` in v1; valid only for `oob` and `both` | `db` |
| `collectors` | site allowlist drawn from `rapl`, `nvidia_smi`, `rocm_smi` | site-configured auto-detection |
| `interval_ms` | site-configured bounded integer | `1000` |

The initial proposed interval policy is `250 <= interval_ms <= 60000`. The exact bounds are configuration, not user-controlled policy.

Parser rules:

- Extract only a whitespace-delimited token beginning with `power:`.
- Preserve other comment namespaces and free text.
- Reject duplicate `power:` tokens, duplicate keys, duplicate collectors, unknown keys, unsupported collectors, control characters, and values outside the configured bounds.
- Reject `backend` for pure `inband`, and reject `collectors` or `interval_ms` for pure `oob`, instead of silently ignoring mode-incompatible fields.
- Reserve `backend=api` for a later milestone; submission must reject it until the API backend and its tests are implemented.
- Reject any request containing a path, executable, systemd unit, output directory, username, UID, shell text, or environment assignment.
- Reject an invalid REPACSS request at submission or modification time with a clear message.
- Reject changes to the REPACSS token after an allocation has started; a running sidecar must not diverge from the submitted request.
- Leave a job with no `power:` token unchanged.
- Accept the existing `power_profiling` spelling as a warned `power:oob;backend=db` compatibility alias for one transition release.

When the site audit allows REPACSS to own a namespace in Slurm Extra, the canonical trusted payload is:

```text
repacss_power_v1=<base64url-without-padding(JSON)>
```

Decoded example:

```json
{
  "schema_version": 1,
  "mode": "inband",
  "collectors": ["rapl", "nvidia_smi"],
  "interval_ms": 1000
}
```

In this branch, the Lua plugin must remove or reject any user-supplied value in the reserved `repacss_power_v1` namespace before setting its own canonical value. It must not append to an arbitrary pre-existing Extra grammar. The compute hook validates the decoded payload again against the local allowlist; it never treats Extra as a command line.

Implementation:

- Extend `shared/slurm/comments.py` with strict validation and canonical serialization.
- Add `shared/slurm/job_submit_repacss_power.lua` as a site-integration helper. Slurm loads one site `job_submit.lua`, so administrators must merge or call this helper from the existing `slurm_job_submit()` and `slurm_job_modify()` handlers rather than overwrite site policy.
- Add `shared/slurm/power_request.py` for decoding and runtime validation of the site-selected Extra or canonical Comment handoff.
- Add `tests/fixtures/slurm_power_comments.json` as the golden valid/invalid corpus used by both Python and Lua tests.
- Extend `tests/unit/test_slurm_comment_parser.py` and add `tests/unit/test_slurm_power_request.py`.

The Lua handler must be CPU-only and perform no file, network, database, or Slurm CLI access because job-submit callbacks execute in `slurmctld`.

Compatibility gate:

1. Record the target output of `scontrol --version`.
2. Submit a probe job whose Lua handler sets a known Extra token.
3. Capture the canary Prolog/Epilog environment and verify `SLURM_JOB_COMMENT`, `SLURM_JOB_EXTRA`, `SLURM_JOB_SLUID`, `SLURM_JOB_START_TIME`, `SLURM_JOB_RESTART_COUNT`, `SLURM_JOB_NODELIST`, `SLURM_JOB_GPUS`, `CUDA_VISIBLE_DEVICES`, any site-provided `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`, `SLURM_JOB_EXCLUSIVE`, `SLURM_JOB_OVERSUBSCRIBE`, and available array/heterogeneous component identifiers.
4. Check whether the site uses `SchedulerParameters=extra_constraints`, accepts user `--extra`, or has another owner/grammar for the Extra field.

The compatibility decision is a hard deployment gate:

- If `extra_constraints` is disabled and site policy grants REPACSS an unambiguous Extra namespace, use `job_desc.extra -> SLURM_JOB_EXTRA`.
- If Extra already has an owner or grammar, do not modify it. Let the job-submit plugin replace only the REPACSS token in the ordinary comment with its canonical form, and use validated `SLURM_JOB_COMMENT` as the handoff.
- In either branch, `slurm_job_modify()` must apply the same validation and must prevent a running allocation from changing its REPACSS request.
- If the installed Slurm build exposes neither selected environment field, upgrade Slurm or add a version-matched controller/prep plugin that writes the trusted request record.

Do not silently call `scontrol show job` from production Prolog/Epilog.

### 2. Make attempts unambiguous

The current permanent epoch cache can mix requeued executions of the same job. Replace it with a versioned attempt identity:

1. Prefer `SLURM_JOB_SLUID` when the deployed Slurm version supplies it.
2. Otherwise derive a deterministic token from:
   - cluster name
   - element/component job ID
   - `SLURM_JOB_RESTART_COUNT`
   - `SLURM_JOB_START_TIME`
3. Store array job/task IDs and heterogeneous-job component metadata in the manifest.

Every path and systemd instance name must pass a bounded safe-token check such as `[A-Za-z0-9_.-]+`. If the full identity is too long for a unit instance, use a bounded digest and persist the full identity in metadata. The exact `attempt_id` is persisted atomically in:

```text
/run/repacss-power/<attempt_id>.env
```

Prolog must also write a required atomic lookup record keyed by the Slurm run identity:

```text
/run/repacss-power/by-slurm/<job_sluid-or-job-restart-start>.current
```

Epilog derives only this deterministic lookup key from its own Slurm environment, reads the exact Prolog state, and validates job ID, restart count, start time, component, and attempt ID before acting. It must not select the newest directory or derive identity from the current wall clock. If the lookup record is absent, Epilog records a safe skip and asks reconciliation to inspect durable spool metadata; it never guesses a unit name.

Implementation:

- Add `inband/identity.py`.
- Replace the epoch-cache behavior in `inband/storage.py`.
- Update `inband/hooks.py` to use attempt-local state and to clean only that attempt's files.
- Update `cli/ib.py` so `--job` lists attempts and reports ambiguity; `--attempt` performs an exact lookup.
- Add array, requeue, duplicate-hook, collision, unsafe-token, and stale-state tests.

Acceptance:

- Two runs after requeue produce different attempt directories.
- All nodes in one allocation derive the same attempt ID.
- Array elements and heterogeneous components never share runtime state.
- Repeated Prolog or Epilog calls are idempotent.

### 3. Replace flat writes with attempt-safe staging

Use this shared staging layout:

```text
<staging_root>/v2/<cluster>/<job_id>/<attempt_id>/
├── request.json
├── manifest.json
├── nodes/
│   └── <hostname>/
│       ├── chunks/
│       │   ├── rapl/
│       │   │   └── 00000001.csv
│       │   ├── nvidia_smi/
│       │   │   └── 00000001.csv
│       │   └── rocm_smi/
│       │       └── 00000001.csv
│       ├── runner_status.json
│       ├── node_status.json
│       └── .done
└── finalization_status.json
```

The production local spool contains recovery metadata as well as chunks:

```text
<local_spool_root>/<attempt_id>/<hostname>/
├── attempt.json
├── chunks/
├── runner_status.json
└── node_sealed.json
```

Storage rules:

- The shared staging tree is administrator-owned and not user-writable.
- Use a dedicated cluster-wide service group if NFS root squashing is enabled.
- Use `0750` directories, `0640` data/status files, and `umask 0027`.
- Prolog must not require shared storage to start local collection. It first writes durable local `attempt.json`, including the trusted request hash and configured shared target.
- The stager publishes shared `request.json` with create-if-absent semantics. Concurrent nodes may accept a byte-identical record, but any mismatch marks the attempt invalid instead of selecting one writer.
- Write JSON and chunks to a sibling `.partial` file, flush, `fsync`, close, and `os.replace` within the destination filesystem.
- Never rename a local-spool file directly to NFS. Copy it to an NFS-side `.partial`, verify size and SHA-256, then rename within NFS.
- Readers ignore `.partial` files.
- `node_status.json` records chunk names, sizes, checksums, sample counts, collector results, and time bounds.
- After the final chunk is closed, the runner atomically writes `node_sealed.json` containing the complete immutable chunk inventory, sizes, checksums, terminal runner status, and request hash.
- The stager may copy closed chunks early, but it must not publish final `node_status.json` or `.done` until it has read the seal and verified every item in the sealed inventory.
- If the runner is killed before sealing, reconciliation may create an explicit `forced_partial` seal only after proving the producer is no longer active; it inventories closed chunks and records the abnormal termination.
- `.done` is an atomic final marker written last. It means the sealed node inventory has finished publishing, not that every collector succeeded.
- An export is immutable after publication. Late chunks are recorded for operator review. An explicit re-finalize operation writes a new sibling export named `<attempt_id>.revision-<N>` with `supersedes` provenance; it never overwrites the original attempt directory.

The sampling hot path should not flush a long-lived CSV to NFS after every sample. Implement rotating chunks by time or size:

- default chunk duration: site-configured, initially 30 seconds
- default maximum chunk size: site-configured
- production node-local spool: `/var/spool/repacss-power/<attempt_id>/<hostname>/`
- authoritative data: closed, checksum-verified chunks in shared staging

The production default enables local spool because NFS-outage and reboot recovery depend on durable node-local metadata. A site may disable it and write directly in the shared node directory using NFS-side `.partial` files, but that reduced mode must explicitly waive offline collection and reboot-recovery acceptance tests.

Implementation:

- Extend `inband/storage.py` with safe path builders, atomic JSON, chunk publication, checksums, and permissions.
- Add `inband/chunks.py` for rotation and publication.
- Add `inband/stager.py` for retryable local-to-shared publication and `.done`-last semantics.
- Modify `inband/collectors/base.py` to stream rows into chunks instead of reading a long CSV at shutdown.
- Add `shared/slurm/repacss-power-stage@.service` so closed local chunks can finish staging after the allocation is released.
- Retain a read-only resolver for the current flat layout for one transition release; do not dual-write old and new layouts.
- Extend `tests/unit/inband/test_storage.py` and add `tests/unit/inband/test_chunks.py`.

Acceptance:

- A reader never observes a half-written final file.
- An interrupted publication leaves only an ignorable `.partial`.
- A transient shared-filesystem outage does not corrupt already closed local chunks.
- User-controlled strings cannot escape configured roots.

### 4. Scope collection to the allocation

Add an explicit allocation record passed from Prolog to the runner:

```json
{
  "job_id": "12345",
  "attempt_id": "...",
  "node": "node001",
  "exclusive": true,
  "allocated_gpu_ids": ["GPU-..."],
  "requested_collectors": ["rapl", "nvidia_smi"],
  "interval_ms": 1000
}
```

Collector policy:

- NVIDIA and AMD collectors query only devices allocated to this job.
- Resolve Slurm physical IDs to stable GPU UUIDs once at startup and store the mapping in provenance.
- Validate cgroup/GRES, MIG, UUID, and index behavior on the target partition before enabling a GPU collector.
- If device mapping is ambiguous, skip the collector and record `unsupported_allocation_mapping`; never sample every GPU as a fallback.
- RAPL is node/socket scoped. The default `rapl_attribution_policy` is `exclusive_only`.
- Treat `SLURM_JOB_EXCLUSIVE` as its documented enum, not a boolean. The conservative v1 truth table enables job-attributed RAPL only for `NODE`; `NO` is shared, while `USER`, `MCS`, and `TOPO` remain unsupported until site configuration proves they provide full-node isolation.
- On a shared allocation under `exclusive_only`, skip RAPL and record `skipped_nonexclusive`.
- A site may opt into `node_observed`, but the result must be labeled as shared-node observation and must not be reported as job-only energy.
- One collector's failure must not stop healthy collectors.

Implementation:

- Add `inband/allocation.py`.
- Update `inband/hooks.py` and `inband/runner.py` to carry allocation metadata.
- Update `inband/collectors/nvidia_smi.py` and `inband/collectors/rocm_smi.py` to filter allocated devices.
- Update `inband/collectors/rapl.py` to record node/socket scope and enforce the attribution policy.
- Add tests for GPU index/UUID filtering, MIG ambiguity, no-GPU jobs, shared-node RAPL, exclusive RAPL, and partial collector startup.

Acceptance:

- No artifact contains an unallocated GPU.
- A shared-node RAPL sample is never labeled as job-only energy.
- Every metric has explicit source, scope, unit, and timestamp semantics.

### 5. Bound and harden the compute-node lifecycle

Prolog:

- Read only the canonical trusted request on the normal path.
- Build the attempt and allocation records atomically.
- Start exactly one `repacss-power-ib@<attempt_id>.service` instance.
- Wait only for the systemd readiness result, bounded by `TimeoutStartSec`.
- Record `running`, `partial`, `skipped`, or `failed`.
- Return promptly.

Runner:

- Probe requested collectors before reporting readiness.
- With `Type=notify`, send `READY=1` only after the requested collectors have been classified and all usable collectors have started.
- If no requested collector can start, write `failed`, still notify readiness so the fail-open Prolog can return, and let the job proceed without telemetry.
- Run all healthy collectors concurrently.
- Install a SIGTERM handler.
- Stop sampling, close the current chunk, atomically write `node_sealed.json`, publish final local status, request stage-out, and exit within `TimeoutStopSec`.
- Leave enough local/shared status for Epilog to classify a forced termination.

Epilog:

- Load the exact Prolog state.
- Issue a non-blocking systemd stop for the matching unit.
- Optionally observe shutdown for only the configured short Epilog wait budget; systemd owns the remaining graceful-stop timeout after the hook returns.
- Start or notify the matching stage-out unit.
- Remove only the runner's matching runtime state after local durability; stage-out state remains until shared publication succeeds.
- Record failures without changing the user's job exit status.

Stager:

- Run as a system service outside the user's allocation cgroup.
- Copy only closed chunks from the validated attempt spool.
- Retry transient shared-filesystem errors with bounded backoff.
- Publish `node_status.json` and `.done` only after a producer seal exists and every sealed chunk passes checksum verification.
- Retain un-staged chunks at the configured high-water/retention boundary and alert instead of deleting them silently.

Reconciliation:

- Run from a separate timer, never in Prolog/Epilog.
- Recover closed local chunks after reboot and restart interrupted stage-out.
- Treat a runner as orphaned only after a grace period and a positive check that the matching allocation is no longer active; never kill a runner based only on file age.
- Because reconciliation is outside the scheduling-critical hook, it may use a bounded, read-only site-approved Slurm query when no reliable local cgroup/state check exists.
- Preserve all un-staged data when stopping an orphan.

Update these files:

- `inband/hooks.py`
- `inband/runner.py`
- `shared/slurm/repacss-power.prolog.sh`
- `shared/slurm/repacss-power.epilog.sh`
- `shared/slurm/repacss-power-ib@.service`
- `shared/slurm/repacss-power-stage@.service`

The production telemetry policy is fail-open:

- Invalid requests are rejected at submission time.
- Runtime telemetry failure is written to status/logs.
- Both compute-node shell wrappers catch REPACSS failures, log them, and exit `0`.
- The site wrapper also invokes the REPACSS helper independently of other Prolog/Epilog logic.
- A future fail-closed option must be explicit and is out of scope for v1.

This policy is required because a non-zero compute-node Prolog can drain the node and requeue the job, while a non-zero compute-node Epilog can drain the node.

The service must use a fixed administrator-owned interpreter and an installed package. The repository's current root `setup.py` is an interactive setup utility, not Python package metadata, so M0 must add a real `pyproject.toml` and build/install a wheel into `/opt/repacss-power/venv`. Production must not depend on repository working directory or an injected `PYTHONPATH`. A smoke test must import and launch the installed entry points from a directory outside the checkout.

No request field can alter `ExecStart` or add shell arguments. Proposed hardening, subject to validation on the oldest deployed systemd:

```ini
[Service]
Type=notify
NotifyAccess=main
User=repacss-power
Group=repacss-power
SupplementaryGroups=powercap video render
EnvironmentFile=/run/repacss-power/%i.env
ExecStart=/opt/repacss-power/venv/bin/python -m inband.runner
Restart=no
KillMode=control-group
KillSignal=SIGTERM
TimeoutStartSec=5
TimeoutStopSec=30
UMask=0027
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/run/repacss-power /var/spool/repacss-power
```

Prefer udev/group access over running the full runner as root. Keep device isolation compatible with RAPL, NVIDIA, and AMD access; validate every directive instead of copying hardening options blindly. If local spool is disabled and the runner writes shared staging directly, the generated site override must also include the configured staging root in `ReadWritePaths`.

Slurm deployment must evaluate:

```ini
PrologFlags=Alloc,DeferBatch
PrologTimeout=<measured-total-site-prolog-budget-plus-margin>
EpilogTimeout=<measured-total-site-epilog-budget-plus-margin>
```

`Alloc` is required for collection on every allocated node even if no job step starts there. `DeferBatch` is recommended when collection must be ready on every node before the batch workload begins, but its launch-latency effect must be measured in the canary partition. `DeferBatch` becomes meaningful for REPACSS only with the `Type=notify` readiness protocol above.

Do not copy a fixed global timeout into `slurm.conf`. Audit the combined runtime of every site Prolog/Epilog, including ECHO and unrelated hooks. The REPACSS Prolog readiness deadline and Epilog observation deadline must each fit inside their measured Slurm hook budget with explicit margin. `TimeoutStopSec` may be longer only because the non-blocking stop continues under systemd after Epilog has returned. Fault injection must prove that runner shutdown at its worst-case timeout cannot cause a Slurm hook timeout or node drain.

Production hooks must use absolute paths and must not invoke `scontrol`, `squeue`, or `sacct`.

### 6. Add asynchronous finalization and export

The controller-side dispatcher receives job-end context and atomically writes one queue record:

```text
<queue_root>/
├── pending/<attempt_id>.json
├── running/<attempt_id>.json
├── complete/<attempt_id>.json
└── dead/<attempt_id>.json
```

The enqueue record includes an absolute `deadline_at` calculated from job end/enqueue time, not from eventual worker start. It also includes request hash, expected nodes, UID/GID, start/end time, attempt identity, retry count, and schema version.

A persistent `repacss-power-finalizer.service` coordinates a bounded worker pool. It must not let one missing node block all later jobs:

1. Atomically claim a due record from `pending` into `running` and write `lease_owner`, `claimed_at`, and `lease_expires_at`.
2. On startup and periodically, recover expired `running` leases after verifying that no matching worker is active.
3. Check shared staging once; do not sleep or poll while holding a worker slot.
4. If nodes are still pending and `deadline_at` has not passed, set `next_check_at`, release the lease, and return the record to `pending`.
5. If every node is sealed and done, or the absolute deadline has passed, dispatch an attempt-scoped aggregation worker subject to configured concurrency and backpressure limits.
6. The worker acquires an attempt lock, validates every `.done`, status, sealed inventory, chunk size, and checksum, and classifies missing or invalid inputs.
7. Normalize valid rows into long-form `raw_power.csv`.
8. Build `summary.json`, `provenance.json`, and finalization status.
9. Write the export under an administrator-controlled sibling publishing directory on the same filesystem, flush it, apply final numeric UID/GID and modes, then atomically rename it to the final attempt path.
10. If a crash occurs after publication but before queue completion, the retry verifies the immutable export checksums and completes the queue record rather than duplicating data.
11. On transient failure, renew/release the lease with bounded exponential backoff. On permanent failure or exhausted retry budget, move the record to `dead` with the error history intact.

Completion states:

- `complete`: every expected node published valid final status.
- `partial`: at least one valid sample exists, but a node, collector, or chunk is missing or invalid.
- `failed`: no valid samples can be exported.

A missing node Epilog after cancellation or node failure is expected and becomes `partial` after timeout; it must never block forever or cause head-of-line blocking. Retrying finalization must not create duplicate output or change an already published attempt silently.

Implementation:

- Add `inband/aggregation.py`.
- Add `shared/slurm/finalize_queue.py`.
- Add `shared/slurm/finalizer.py` as the cross-source coordinator and bounded worker implementation.
- Update `shared/slurm/dispatcher.py` so `inband` enqueues finalization instead of returning immediately.
- Add `shared/slurm/repacss-power-finalizer.service`.
- Add `shared/slurm/repacss-power-finalize@.service` as the bounded attempt worker launched by the coordinator.
- Update `cli/ib.py` with exact status and an operator-only revision-producing re-finalize command.
- Update `cli/export.py` to consume the same normalized export implementation.
- Add `shared/export/power_export.py`.

For `power:both`, preserve the current OOB backend as a separate source. Cross-source orchestration belongs to `shared/slurm/finalizer.py`; `inband/aggregation.py` must never import or query an OOB backend. Refactor `oob/slurm/epilog_handler.py` to return a normalized, side-effect-free source result that the shared worker can export, instead of directly owning the legacy output directory. The shared worker assembles OOB and IB only at final export time. IB timeout or partial failure must not discard a valid OOB result, and OOB failure must not discard valid IB chunks.

Required normalized `raw_power.csv` columns:

```text
timestamp
elapsed_ms
sequence
job_id
attempt_id
node
scope_type
scope_id
metric_name
value
unit
source_id
source_kind
collector_family
timestamp_mode
retrieval_mode
```

Required semantics:

- IB rows use `source_kind=in_band` and `retrieval_mode=job_local_artifact`; hybrid exports preserve their independent OOB values.
- `source_id` is stable for a collector/node/scope combination.
- `summary.json` contains contract-required time range, node scope, source kinds, power summary, energy values, availability flags, aggregation method, and confidence notes.
- `provenance.json` records request, software/schema versions, expected/received/missing nodes, allocation scope, sampling interval, collector/device mapping, chunk checksums, partial failures, and export method.
- Sampled power is integrated with an explicitly named numerical method.
- RAPL counters handle wraparound.
- Overlapping sources are not blindly summed into one `energy_total`. When no non-overlapping aggregation policy is valid, report source-specific energy and set the combined total to `null` with a confidence note.

### 7. Configuration

Add typed configuration for:

| Setting | Purpose |
|---|---|
| `REPACSS_POWER_IB_STAGING_ROOT` | administrator-owned shared staging |
| `REPACSS_POWER_IB_LOCAL_SPOOL_ROOT` | durable node-local chunk spool; required for outage/reboot recovery |
| `REPACSS_POWER_IB_QUEUE_ROOT` | finalizer queue |
| `REPACSS_POWER_REPORT_ROOT` | user-facing exports |
| `REPACSS_POWER_IB_ALLOWED_COLLECTORS` | submission/runtime allowlist |
| `REPACSS_POWER_IB_DEFAULT_INTERVAL_MS` | default sampling cadence |
| `REPACSS_POWER_IB_MIN_INTERVAL_MS` | lower interval bound |
| `REPACSS_POWER_IB_MAX_INTERVAL_MS` | upper interval bound |
| `REPACSS_POWER_IB_CHUNK_SECONDS` | chunk rotation |
| `REPACSS_POWER_IB_CHUNK_MAX_BYTES` | size rotation |
| `REPACSS_POWER_IB_FINALIZE_TIMEOUT_SECONDS` | missing-node deadline |
| `REPACSS_POWER_IB_FINALIZE_MAX_WORKERS` | bounded aggregation concurrency |
| `REPACSS_POWER_IB_QUEUE_LEASE_SECONDS` | crash-recoverable queue claim |
| `REPACSS_POWER_IB_EPILOG_WAIT_SECONDS` | short observation budget inside Epilog |
| `REPACSS_POWER_IB_RAPL_ATTRIBUTION` | `exclusive_only` or `node_observed` |
| `REPACSS_POWER_IB_SPOOL_HIGH_WATERMARK` | controlled degradation threshold |
| `REPACSS_POWER_IB_RETENTION_DAYS` | staging retention |

New code uses the `REPACSS_POWER_*` namespace. Existing `MONSTER_POWER_*` settings may remain as warned compatibility aliases for one transition release. Secrets are not required for a pure IB run.

### 8. File-by-file work map

| File | Planned change |
|---|---|
| `pyproject.toml` | installable wheel/editable package for `inband`, `shared`, `oob`, and `cli` |
| `shared/config/config.py` | typed roots, policy bounds, chunking, timeout, retention |
| `shared/models/__init__.py` | versioned request, attempt, node-status, and finalization models |
| `shared/slurm/comments.py` | strict grammar, canonical request, allowlist and bounds |
| `shared/slurm/job_submit_repacss_power.lua` | submission/modify validation helper for the site plugin |
| `shared/slurm/power_request.py` | decode and revalidate the selected Extra/Comment handoff |
| `shared/slurm/resolver.py` | use hook environment; remove production Slurm CLI fallback |
| `shared/slurm/dispatcher.py` | enqueue IB/both finalization |
| `shared/slurm/finalize_queue.py` | atomic claim, retry, dead-letter state |
| `shared/slurm/finalizer.py` | bounded cross-source queue coordinator and workers |
| `shared/slurm/repacss-power.prolog.sh` | bounded fail-open compute Prolog wrapper |
| `shared/slurm/repacss-power.epilog.sh` | bounded fail-open compute Epilog wrapper |
| `shared/slurm/repacss-power.epilogslurmctld.sh` | fail-open controller enqueue wrapper |
| `shared/slurm/repacss-power-ib@.service` | fixed execution path and hardening |
| `shared/slurm/repacss-power-stage@.service` | retryable node-to-shared stage-out after job end |
| `shared/slurm/repacss-power-finalizer.service` | persistent non-blocking headnode coordinator |
| `shared/slurm/repacss-power-finalize@.service` | bounded attempt worker template |
| `shared/slurm/repacss-power-reaper.service` and `.timer` | recover orphan runners and un-staged chunks outside Slurm hooks |
| `shared/slurm/slurm.prolog` | compose the REPACSS helper with existing ECHO/site logic without replacement |
| `src/services/slurm.epilogslurmctld` | remove legacy sleep/output creation; retain only a warned compatibility handoff or retire it |
| `inband/identity.py` | attempt and safe-token rules |
| `inband/allocation.py` | GPU/exclusive allocation normalization |
| `inband/chunks.py` | rotation, atomic publication, checksums |
| `inband/stager.py` | local backlog, retry, shared publication, `.done` marker |
| `inband/storage.py` | v2 paths, permissions, atomic JSON, compatibility reader |
| `inband/hooks.py` | trusted request, attempt state, idempotent lifecycle |
| `inband/runner.py` | readiness, concurrent collectors, bounded shutdown |
| `inband/collectors/*.py` | chunk streaming, device filtering, scope metadata |
| `inband/aggregation.py` | wait deadline, validation, normalization, summary |
| `inband/reaper.py` | bounded reconciliation after missed Epilog or node reboot |
| `oob/slurm/epilog_handler.py` | return a normalized side-effect-free OOB source result for shared orchestration |
| `shared/export/power_export.py` | contract-compliant atomic delivery for IB and both |
| `cli/ib.py` | attempt-aware status and operator recovery |
| `cli/export.py` | reuse normalized export writer for IB/both |
| `tests/fixtures/slurm_power_comments.json` | cross-language golden requests |
| `tests/unit/inband/*` | storage, runner, hook, collector, aggregator tests |
| `tests/unit/test_slurm_*.py` | parser, request handoff, dispatcher, queue tests |
| `tests/integration/test_inband_fake_slurm_lifecycle.py` | rootless end-to-end hook/systemd/stager/finalizer smoke |

### 9. Milestones and exit criteria

#### M0: contract and compatibility

- Freeze request, attempt, staging, row, status, and export schemas.
- Complete the target Slurm/Extra/environment compatibility probe.
- Decide interval bounds, RAPL policy, roots, service identity, chunking, retention, and timeout.
- Add installable package metadata and prove imports from outside the checkout.

Exit: one reviewed schema version, no unresolved production handoff, and no runtime behavior change.

#### M1: submission and trusted handoff

- Implement strict Python/Lua parser parity.
- Integrate with the site `job_submit.lua`.
- Read the selected canonical Extra/Comment handoff in hooks.
- Remove the normal-path `scontrol` lookup.

Exit: a valid comment reaches every canary node unchanged; invalid requests are rejected; comment-free jobs are unaffected.

#### M2: attempt-safe storage and allocation scope

- Implement v2 attempt paths, atomic state, chunks, and checksums.
- Enforce GPU allocation filtering and RAPL policy.
- Preserve a read-only v1 compatibility resolver.

Exit: requeues/arrays are isolated, no unallocated GPU is observed, and interrupted writes are never treated as final.

#### M3: bounded compute lifecycle

- Implement `Type=notify` readiness, fail-open wrappers, non-blocking Epilog stop, durable producer seal, node stager retry, final chunk publication, and `.done`-last behavior.
- Harden and validate the systemd unit.
- Validate `PrologFlags=Alloc` and, if selected, `DeferBatch`.

Exit: every allocated canary node starts one runner; all hook durations are bounded; no telemetry failure changes job result or drains a node.

#### M4: asynchronous finalizer and user export

- Implement leased queueing, non-blocking deadline checks, bounded worker concurrency, aggregation, normalization, delivery, retry, and dead-letter handling.
- Publish the required power export and telemetry provenance shapes.
- Make reruns idempotent.

Exit: comment-only submission yields the documented user-owned files for complete, partial, and failed telemetry cases.

#### M5: hybrid mode and operational rollout

- Refactor the OOB handler into a normalized source result and assemble `power:both` in the shared finalizer without coupling IB collection to the OOB database.
- Add health metrics, spool/queue monitoring, retention, kill switch, packaging, orphan reconciliation, and operator recovery.
- Complete canary and concurrency tests before expansion.

Exit: pure IB remains functional with OOB unavailable, hybrid output preserves both provenances, and rollback is proven.

Dependency order:

```text
M0 -> M1 -> M2 -> M3 -> M4 -> M5
              \-> collector work can proceed in parallel ->/
```

### 10. Test and validation matrix

| Scenario | Expected result |
|---|---|
| No `power:` token | no unit, spool, queue item, or output; job unchanged |
| Valid `power:inband` | canonical request reaches all allocated nodes without a Slurm CLI call |
| Invalid mode/key/collector/interval/control character | submission or modification rejected clearly |
| Extra owned by `extra_constraints` or another site policy | REPACSS leaves Extra unchanged and uses the canonical validated Comment branch |
| `backend=api` before API implementation | submission rejected as unsupported |
| One-node exclusive CPU job | valid RAPL chunks, statuses, checksums, summary, and provenance |
| One allocated NVIDIA or AMD GPU | only that device appears |
| Shared-node RAPL with default policy | RAPL skipped and reason recorded |
| Multi-node job | every expected node is complete or explicitly missing/partial |
| User program exits non-zero | telemetry flushes; original job exit code is unchanged |
| Cancel or timeout | bounded cleanup; complete or explicitly partial attempt |
| Duplicate Prolog/Epilog | no duplicate runner, truncation, or duplicate output |
| Requeue | distinct attempts and zero cross-attempt rows |
| Job array | isolated state and output for every element |
| Collector unavailable | healthy collectors continue; telemetry becomes partial |
| Runner crash or SIGKILL | failure recorded; job continues |
| Runner/stager race during final chunk | `.done` is impossible before a valid producer seal and full sealed inventory |
| Worst-case runner stop | Epilog returns inside its audited budget and the node is not drained |
| Shared filesystem unavailable | local closed chunks retained; hook returns `0`; stage-out retries |
| Local spool full | controlled degradation and alert; no arbitrary deletion or node drain |
| Node failure with no Epilog | finalizer times out to partial and lists missing node |
| Missed Epilog or node reboot | reconciliation stops orphan runners and stages closed chunks without deleting evidence |
| Checksum mismatch | affected chunk quarantined and export marked partial |
| Repeated finalizer execution | no duplicate output; same result or explicit immutable conflict |
| Finalizer crash after queue claim | expired lease is recovered and the original absolute deadline is preserved |
| Late chunk plus operator re-finalize | original export remains unchanged and a revisioned sibling is produced |
| OOB database unavailable for pure IB | IB export still succeeds |
| `power:both` with one source failing | valid source preserved; failure present in provenance |
| 100+ short concurrent jobs, including a missing-node attempt | bounded fairness/backpressure; no head-of-line blocking |

Automated gates:

- Golden Python/Lua parser parity.
- Boundary and malformed input tests.
- Path traversal and unsafe systemd-instance tests.
- Atomic-write and checksum failure injection.
- Attempt/requeue/array tests.
- Fake-systemd Prolog/readiness/runner/Epilog/stager integration.
- Producer-seal race and forced-partial-seal tests.
- Hook-timeout failure injection proving no REPACSS-attributable drain.
- Aggregator complete/partial/timeout/idempotency tests.
- Queue lease expiry, coordinator restart, absolute deadline, fairness, and backpressure tests.
- Export contract and ownership/mode tests.
- Installed-package smoke from outside the repository checkout.
- `systemd-analyze verify` on every deployed unit.
- Real RAPL, NVIDIA, and AMD permission probes on their matching canary nodes.
- Fresh-clone unit test path without credentials or database access.

Fresh-clone gate after `pyproject.toml` is implemented:

```bash
python3 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements.txt
.venv-test/bin/python -m pip install -e .
.venv-test/bin/python tests/run_tests.py --type unit

repo_root="$PWD"
(cd /tmp && "$repo_root/.venv-test/bin/python" -c \
  'import inband.runner, shared.slurm.dispatcher, shared.slurm.finalizer')

.venv-test/bin/python -m pytest \
  tests/integration/test_inband_fake_slurm_lifecycle.py -q
```

Initial performance targets, to be confirmed during canary:

- job-submit parsing performs only bounded in-memory work
- compute Prolog and Epilog p99 remain below the site-agreed hook budget
- finalizer enqueue p99 remains below 500 ms
- runner CPU overhead remains below 1% of one core at the default interval
- NFS write frequency is bounded by chunk rotation rather than sample cadence

### 11. Deployment and canary rollout

One-time administrator work:

1. Inventory the actually configured `Prolog`, `Epilog`, `EpilogSlurmctld`, job-submit plugin, and installed wrapper paths; do not assume repository examples are active.
2. Build a wheel and install it into the same fixed root-owned virtual environment on all compute and controller nodes.
3. Create the service account/group and sensor access rules.
4. Create local spool, shared staging, queue, and report roots with reviewed ownership and setgid policy.
5. Install and verify systemd units.
6. Merge the request validator into the existing site `job_submit.lua`; do not replace unrelated policy.
7. Compose REPACSS helpers into existing site Prolog/Epilog wrappers; do not replace ECHO or other site logic.
8. Replace the behavior of `src/services/slurm.epilogslurmctld` with the new enqueue-only controller wrapper, or retire it after changing the installed path. Remove its legacy sleep and pre-created output directory.
9. Ensure exactly one controller entry invokes the asynchronous dispatcher. The legacy synchronous wrapper and new enqueue path must never both be enabled.
10. Configure `PrologFlags=Alloc` and measure whether `DeferBatch` is required.
11. Set hook timeouts only after auditing all site hooks and proving margin under failure injection.
12. Start the finalizer coordinator and reconciliation services.
13. Enable one canary partition or node set.
14. Run the full validation matrix before broader opt-in.

Rollout gates:

1. **Audit only:** validate and log requests but do not start collectors.
2. **Canary:** enable a small partition and all failure-path tests.
3. **Opt-in:** allow selected users/partitions while monitoring hook duration, runner errors, node drain reasons, stage backlog, spool utilization, missing nodes, and finalizer latency.
4. **General availability:** expand only after an agreed normal operating window with no REPACSS-attributable job failure or node drain.

### 12. Rollback and kill switch

Maintain a site kill switch that makes REPACSS hooks log and return `0` without starting new units.

Rollback order:

1. Disable new IB activation.
2. Allow active runners to stop and stage already collected data.
3. Remove or disable only the REPACSS calls in site Prolog/Epilog wrappers.
4. Restore the previous site job-submit handler.
5. Quiesce the finalizer, let active workers finish or return their leased records to `pending`, and verify that no non-expired `running` claim is abandoned.
6. Validate Slurm configuration using the site's version-appropriate procedure.
7. Run a comment-free smoke job and an OOB-only smoke job.
8. Retain local/shared staged data for recovery; rollback must not delete it.

Rollback immediately if REPACSS causes any node drain or job failure, if unallocated-device data is observed, or if hook/queue/spool limits exceed the agreed safety threshold.

### 13. Definition of done

The implementation is complete only when:

- a user can request IB collection using only `--comment`
- no-comment jobs are unchanged
- invalid requests are rejected before allocation
- normal production hooks make no Slurm CLI calls
- exactly one installed controller wrapper enqueues finalization
- all intended allocated nodes start exactly one bounded runner
- readiness means collectors have been classified and usable collectors have started
- GPU data is limited to allocated devices
- RAPL attribution is explicit and safe
- requeues, arrays, and duplicate hooks cannot mix attempts
- all final files are atomically published, user-readable, and contract compliant
- `.done` can be created only from a durable sealed node inventory
- complete, partial, and failed telemetry states are deterministic and visible
- telemetry failure never changes the user's exit code or drains a node
- queue claims and node-local stage-out recover after worker/node restart
- installed services run without repository working-directory or `PYTHONPATH` assumptions
- pure IB works without OOB credentials, database, or API
- `power:both` preserves independent OOB and IB provenance
- the canary rollback procedure has been exercised successfully

### Slurm references

- [SchedMD `sbatch` documentation](https://slurm.schedmd.com/sbatch.html) for `--comment`.
- [SchedMD Prolog and Epilog Guide](https://slurm.schedmd.com/prolog_epilog.html) for lifecycle, exported environment, hook ordering, failure semantics, and the recommendation to avoid Slurm commands in hooks.
- [SchedMD `slurm.conf` documentation](https://slurm.schedmd.com/slurm.conf.html) for `PrologFlags`, `DeferBatch`, and hook timeouts.
- [SchedMD Job Submit Plugin API](https://slurm.schedmd.com/job_submit_plugins.html) for job-submit locking and version-dependent Lua fields.
- [SchedMD Lua job-submit implementation](https://github.com/SchedMD/slurm/blob/master/src/plugins/job_submit/lua/job_submit_lua.c) for current `comment` and `extra` field access.

### REPACSS contract references

- [Power export contract](../../../refactor/docs/contracts/power-export-contract.md) for the final directory and required files.
- [Telemetry source contract](../../../refactor/docs/contracts/telemetry-source-contract.md) for source, scope, units, timestamps, retrieval mode, and provenance.
