# REPACSS Power Profiling - Project Structure

## Overview

This document describes the target architecture and the current migration state of `repacss-power-profiling`.

The repository is moving away from the legacy `src/` monolith toward a domain-oriented structure with:

- `shared/`
- `oob/`
- `inband/`
- `cli/`
- `tests/`

The goal is to make the architecture align with the refactor plan:

- `shared/` holds cross-domain contracts and utilities
- `oob/` owns out-of-band query and export workflows
- `inband/` owns Slurm-bound runtime sampling workflows
- `cli/` exposes user-facing command entrypoints
- `tests/` validates the refactored architecture at unit, integration, and end-to-end levels

## Architecture Model

### Primary Source Domains

#### `shared/`

`shared/` contains cross-domain building blocks that should not belong exclusively to OOB or In-band.

Typical responsibilities:

- configuration access
- shared models and typed contracts
- Slurm comment parsing and job-context resolution
- common analysis helpers
- export foundations
- shared constants and utility helpers
- common error types

`shared/` is not a dumping ground. Anything backend-specific or runtime-specific should stay in `oob/` or `inband/`.

#### `oob/`

`oob/` is the out-of-band domain.

Typical responsibilities:

- OOB backend abstraction
- direct MonSTer DB access
- future MonSTer API backend
- job-scoped OOB query flow
- general metric query flow
- headnode job-end Slurm handling for OOB

OOB is the only domain that talks to external telemetry services in v1.

#### `inband/`

`inband/` is the in-band domain.

Typical responsibilities:

- collector interfaces
- hardware-specific collectors such as `rapl`, `nvidia_smi`, and `rocm_smi`
- compute-node runtime lifecycle
- shared-filesystem staging
- aggregation support
- status/debug support for staged artifacts

In-band is not primarily a manual CLI workflow.
Its production execution model is Slurm-driven:

- `Prolog`
- `Epilog`
- `EpilogSlurmctld`

Current P3 implementation status:

- `inband/collectors/base.py` defines the shared collector contract
- `inband/collectors/rapl.py` implements powercap-backed CPU sampling
- `inband/collectors/nvidia_smi.py` implements NVIDIA GPU sampling via `nvidia-smi`
- `inband/collectors/rocm_smi.py` implements ROCm probing and best-effort JSON sampling
- `inband/collectors/auto_detect.py` owns collector ordering and auto-detect helpers

Current P4 skeleton status:

- `inband/storage.py` owns internal `storage_key`, staging-path, and state-file helpers
- `inband/runner.py` is the node-local long-running runner process intended for systemd
- `inband/hooks.py` is the Slurm-facing `prolog` / `epilog` bridge
- `shared/slurm/repacss-power.prolog.sh`
- `shared/slurm/repacss-power.epilog.sh`
- `shared/slurm/repacss-power-ib@.service`

### Entry Layer

#### `cli/`

`cli/` is the program entry layer.

It is not a fourth business domain.
Its job is to expose user-facing commands over the underlying architecture.

Current design direction:

- OOB query and export workflows are the main CLI use case
- In-band may expose limited debug/status commands
- In-band runtime collection itself remains Slurm-bound, not CLI-driven

Current P3 debug surfaces:

- `ib probe`
- `ib sample`
- `ib status`

These exist for collector validation and local testing only.
They are not the final production path for in-band collection.

### Test Layer

#### `tests/`

`tests/` is the validation layer.

Recommended steady-state organization:

```text
tests/
  unit/
    shared/
    oob/
    inband/
  integration/
    oob/
    inband/
  e2e/
```

This structure mirrors both:

- test depth
- architectural ownership

## Current Migration State

The repository is in an intermediate state.

### What Is Already Happening

- new refactor work is being added under `shared/` and `oob/`
- new in-band collector code is being added under `inband/collectors/`
- legacy Slurm OOB entrypoints are beginning to route into the new architecture
- compatibility is still being preserved for existing imports and scripts where practical

### What Still Exists Temporarily

The legacy `src/` tree is still present.

For now, it serves one or both of these purposes:

- existing implementation not yet migrated
- compatibility wrapper for old import paths and old entrypoints

This is intentional during the refactor.
`src/` should not be treated as the long-term architecture once the migration is complete.

## Repository Layout

### Target Steady-State Layout

```text
repacss-power-profiling/
├── README.md
├── docs/
├── shared/
├── oob/
├── inband/
├── cli/
├── tests/
└── src/   # temporary compatibility layer during migration
```

### Current Practical Rule

When adding new refactor-era code:

- prefer `shared/`, `oob/`, `inband/`, and `cli/`
- only modify `src/` when preserving compatibility or migrating legacy behavior

## Slurm Integration Placement

The Slurm hooks are split by responsibility.

### OOB

OOB remains headnode-oriented and job-level.

Expected placement:

- shared dispatcher logic in `shared/slurm/`
- OOB-specific job-end handling in `oob/slurm/`

### In-band

In-band is compute-node-oriented and runtime-bound.

Expected placement:

- collector lifecycle and compute-node hook logic in `inband/`
- aggregation and staged artifact processing under `inband/`

## RAPL Portability Notes

The in-band `rapl` collector is intentionally written around generic powercap discovery instead of an Intel-only path.

- Intel usually exposes RAPL via powercap sysfs and is the most straightforward Linux target.
- AMD may expose compatible files, but support is less uniform and can depend on MSR or HSMP-backed kernel interfaces.
- AMD domain naming also differs more across generations, so domain presence should always be probed dynamically.

## Temporary Planning Artifacts

`docs/_working/` is reserved for temporary refactor collaboration artifacts.

Rules:

1. Files in `docs/_working/` may be shared between the user, Codex, and Claude Code during active refactor work.
2. Temporary planning files may use mixed Chinese and English if that improves collaboration speed.
3. Stable repository documentation must remain in English.
4. Durable architectural decisions must be moved into `README.md` or `docs/*`.
5. Temporary planning artifacts must be deleted before finalization of the refactor branch.

## Migration Principle

The migration sequence is:

1. establish the new architecture
2. move real behavior into the new domains
3. keep legacy paths working where helpful
4. remove the legacy shell only after the new layout is fully validated

This means the repository may temporarily contain both:

- new domain-oriented modules
- old compatibility-oriented modules

That overlap is expected during the transition and should be resolved only after the new architecture becomes the canonical implementation.
