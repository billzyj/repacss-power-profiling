# Temporary Refactor Plan

Status: active working artifact

This file is a temporary collaboration artifact for refactor planning.
It may use mixed Chinese and English so the user, Codex, and Claude Code can review the same plan during implementation.

Do not treat this file as stable product documentation.
Before the refactor is finalized:

1. Move durable decisions into English documentation such as `README.md` or `docs/*`.
2. Remove this file from the repository.

## Goal

Rebuild `repacss-power-profiling` as a major-version architecture with two first-class power domains:

- Out-of-Band (OOB)
- In-Band (IB)

The target is a clean branch-first redesign with no backward-compatibility requirement.
Current deployment is not modified in place. New architecture is developed and validated on a separate branch, then used to replace the current repo after acceptance.

## Non-Goals

- Do not preserve legacy internal layout just for compatibility.
- Do not keep current CLI command names if they conflict with the new architecture.
- Do not require in-band v1 to persist centrally into TimescaleDB.
- Do not force OOB and in-band into separate downstream export contracts.

## Fixed Decisions

### Top-level architecture

The codebase will be split into three major parts:

- `oob/`
  OOB query semantics, job lookup, DB/API backends, normalization.
- `inband/`
  runtime collectors, process lifecycle, sampling, local artifact assembly.
- `shared/`
  contracts, typed models, export builders, provenance, comment parsing, Slurm helpers, shared config/errors.

### OOB model

OOB exposes two first-class query surfaces:

1. `job-scoped query`
2. `general metric query`

`job-scoped query`:

- input is primarily `job_id`
- always uses a unified Slurm resolver
- epilog context resolves from Slurm env first
- offline/manual mode resolves through Slurm API

`general metric query`:

- node
- time window
- selected metrics
- optional aggregation / scope controls

### OOB backends

OOB must support two data backends:

- `monster-db`
- `monster-api`

Default backend is `monster-db`.
`monster-api` is a real supported backend, not a placeholder.
Normalization must be backend-neutral.

### In-band model

In-band is a separate first-class subsystem.

Runtime model:

- Slurm prolog starts sampling for `inband` or `both`
- Slurm epilog stops sampling and assembles exports

Collectors in v1:

- `rapl`
- `nvidia_smi`
- `rocm_smi`

Collector defaults:

- all three are first-class in v1
- collector auto-detection by hardware
- missing or unsupported collectors are skipped and recorded in provenance
- default `interval_ms=1000`

In-band v1 storage path:

- job-local collection
- job-end export
- no required central DB persistence in v1

### Trigger grammar

Use structured Slurm comment grammar:

```text
power:<mode>;collectors=<csv>;interval_ms=<int>
```

Required modes:

- `power:oob`
- `power:inband`
- `power:both`

Examples:

- `power:oob`
- `power:inband;interval_ms=1000`
- `power:inband;collectors=rapl,nvidia_smi;interval_ms=1000`
- `power:both;collectors=rapl,nvidia_smi,rocm_smi;interval_ms=1000`

Defaults:

- if `collectors` is omitted for `inband` or `both`, auto-detect supported collectors
- if `interval_ms` is omitted, use `1000`

### Unified export

For `both`, there is one unified export directory.
OOB and in-band are not emitted as two unrelated result trees.

Required unified behavior:

- one export directory
- one summary
- provenance explicitly lists OOB and in-band sources
- raw rows preserve source fields such as:
  - `source_kind`
  - `source_id`
  - `collector_family`
  - `retrieval_mode`

## Current Problems

### Current code is OOB-heavy

Most of the existing codebase is effectively OOB-first:

- DB-backed analytics
- job-end epilog export
- service/query/reporting layers built around MonSTer-backed telemetry

There is not yet a first-class in-band subsystem in the repo.

### OOB responsibilities are not cleanly separated

Current implementation mixes:

- query semantics
- DB access
- export logic
- Slurm epilog job flow

This makes it hard to introduce MonSTer API as a true parallel OOB backend.

### In-band runtime lifecycle is missing

Current epilog can gate OOB post-job export by comment, but in-band sampling needs:

- a start lifecycle
- a stop lifecycle
- collector management
- local artifact assembly

That means in-band cannot just be "another query backend". It needs runtime orchestration.

### Export contract is not yet the true implementation center

The refactor target is one export contract that works for:

- OOB only
- in-band only
- both

Implementation must converge around that shared export model.

## Target Architecture

### Shared domain

`shared/` should own the durable cross-cutting contracts:

- comment parser
- mode enum and collector enum
- Slurm job context / job resolution models
- power export builders
- provenance structures
- common config objects
- shared filesystem / output layout helpers

### OOB domain

`oob/` should own:

- OOB query request models
- `job-scoped` and `general` service entrypoints
- Slurm resolver integration
- backend abstraction
- MonSTer DB backend
- MonSTer API backend
- OOB normalization into shared export rows

### In-band domain

`inband/` should own:

- collector interface
- runtime sampling controller
- auto-detection logic
- `rapl`, `nvidia_smi`, `rocm_smi` collectors
- prolog start flow
- epilog stop/finalize flow
- local raw artifact assembly
- normalization into shared export rows

### CLI redesign

The CLI may be reorganized to match the new architecture.
Target public surfaces should include:

- OOB job query commands
- OOB general metric query commands
- export/report commands
- backend-selection controls for OOB
- optional in-band inspection/debug commands if useful

## Work Phases

### Phase 1: Shared contract and skeleton

- Introduce `shared/`, `oob/`, and `inband/` package layout.
- Define typed models for:
  - power mode
  - collector selection
  - Slurm job context
  - export summary
  - provenance rows
- Implement the structured comment parser.
- Implement shared export writer interfaces for:
  - summary
  - provenance
  - raw series
- Define the canonical output directory structure.

### Phase 2: OOB rewrite

- Implement unified Slurm resolver.
- Implement `job-scoped query` service.
- Implement `general metric query` service.
- Add backend abstraction with:
  - MonSTer DB adapter
  - MonSTer API adapter
- Normalize both backends into the shared export/data model.
- Replace current epilog OOB path so `power:oob` uses the new OOB job-query flow.

### Phase 3: In-band runtime

- Implement collector interface and auto-detection.
- Implement `rapl`, `nvidia_smi`, and `rocm_smi` collectors.
- Implement Slurm prolog start path.
- Implement Slurm epilog stop/finalize path.
- Make `power:inband` produce a standalone unified export.
- Make `power:both` merge OOB and in-band into one export with provenance-preserving rows.

### Phase 4: CLI and reporting

- Redesign CLI around the new architecture.
- Add explicit OOB job/general query surfaces.
- Ensure export/report paths use the unified shared contract.
- Update any repo-native test entrypoints and smoke-test docs.

### Phase 5: Final documentation and cleanup

- Update English `README.md`.
- Update English docs for:
  - architecture
  - comment grammar
  - prolog/epilog deployment
  - OOB backend selection
  - remote-machine validation
- Remove stale legacy docs that no longer describe the architecture.
- Delete this temporary plan file before final merge.

## Validation Commands

Commands that both Codex and Claude Code should be able to run locally or on a remote machine after a fresh clone.

Minimum smoke path:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/run_tests.py --type unit
```

Recommended OOB validation targets:

- OOB job query through DB backend
- OOB job query through API backend
- OOB general metric query through DB backend
- OOB general metric query through API backend
- epilog OOB flow with `power:oob`
- offline Slurm resolver path using Slurm API

Recommended in-band validation targets:

- `power:inband` on Intel CPU host with auto-detected RAPL
- `power:inband` on NVIDIA host with auto-detected `nvidia-smi`
- `power:inband` on AMD host with auto-detected `rocm-smi`
- explicit collector override through comment grammar
- default `interval_ms=1000`
- prolog start / epilog stop lifecycle
- missing collector binary recorded in provenance without hard failure

Unified export validation:

- `power:oob` export shape
- `power:inband` export shape
- `power:both` export shape
- raw rows preserve provenance/source fields
- summary clearly distinguishes OOB and in-band availability

## Exit Criteria

- New architecture clearly separated into `oob/`, `inband/`, and `shared/`.
- OOB supports both `monster-db` and `monster-api`.
- OOB exposes both `job-scoped query` and `general metric query`.
- In-band supports `rapl`, `nvidia_smi`, and `rocm_smi`.
- Slurm comment grammar supports `oob`, `inband`, and `both`.
- `both` produces one unified export with provenance-preserving rows.
- Remote-machine smoke test is documented and runnable.
- English README and English docs reflect the new architecture.
- Temporary plan file is deleted before final merge.

## Cleanup Before Merge

- Update the relevant English `README.md`.
- Update any affected English docs under `docs/`.
- Remove outdated legacy documentation if it conflicts with the new architecture.
- Delete `docs/_working/refactor-plan.md`.

## Review Focus For Claude Code

Please challenge the following assumptions before implementation is considered final:

1. Is `Slurm prolog + epilog` the best v1 runtime model for in-band, or is there a cleaner wrapper or daemon-assisted approach?
2. Is `monster-db` default with `monster-api` as a parallel backend the right OOB abstraction boundary?
3. Is one unified export for `both` sufficient, or should summary/provenance be unified while raw files are split internally?
4. Are `rapl`, `nvidia_smi`, and `rocm_smi` enough for v1, or should the collector abstraction already reserve room for DCGM or LIKWID-based extensions?
5. Is the proposed comment grammar expressive enough for future tuning without becoming fragile?
