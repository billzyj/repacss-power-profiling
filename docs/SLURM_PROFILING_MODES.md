# Slurm Profiling Modes and Required Changes

Supporting `power:oob`, `power:inband`, and `power:both` requires changes in the controller workflow and in the compute-node lifecycle. Compute Prolog starts IB collection; compute Epilog stops it; EpilogSlurmctld queues finalization for all three modes. Background workers perform OOB queries, IB aggregation, and report delivery.

Status: implementation and deployment change map, 2026-10-08. The target behavior below is not a claim that all three modes are deployed. The [in-band design notes](../inband/README.md) contain the detailed storage, request, recovery, and rollout contracts.

## Mode behavior

| Comment | Compute Prolog | Compute Epilog | EpilogSlurmctld | Background finalizer |
| --- | --- | --- | --- | --- |
| `power:oob` | No IB service | No IB service to stop | Enqueue this attempt | Query OOB data and export |
| `power:inband` | Start one runner per allocated node | Stop the matching runner and request stage-out | Enqueue this attempt | Aggregate IB artifacts and export |
| `power:both` | Start one runner per allocated node | Stop the matching runner and request stage-out | Enqueue this attempt | Query OOB, aggregate IB, and publish a combined report |
| `power_profiling` | No IB service | No IB service to stop | Treat as the OOB compatibility alias | Query OOB data and export |
| No profiling token | Skip REPACSS work | Skip unless an exact saved IB run needs cleanup | Skip REPACSS work | No report |

OOB sensors already collect continuously. The job-end action starts a query and reporting workflow, not the underlying OOB sensors. IB must start while the allocation is active; controller epilog changes alone cannot provide IB samples.

Compute Epilog uses the exact state saved by Prolog to stop a service. It must not rely on reparsing a potentially changed comment or guessing the latest directory for a job ID. Submission policy must prevent changes to the profiling request once the allocation starts.

Target submission examples after deployment:

```bash
sbatch --comment='power:oob' job.sh
sbatch -p zen4 --exclusive --comment='power:inband' job.sh
sbatch -p h100 --exclusive --comment='power:both' job.sh
```

The short IB forms use the default collectors (`rapl`, `nvml`, `rocm_smi`, each skipped where absent). GPU requests must also match the application's needs and site GRES policy. `power:inband` and `power:both` require `--exclusive`: compute Prolog collects only for an exclusive node allocation (`SLURM_JOB_OVERSUBSCRIBE=NO` on the deployed Slurm 25.11, or `SLURM_JOB_EXCLUSIVE=NODE` on releases that set it), and otherwise the finalizer exports a `skipped` summary that states why.

## Current implementation and deployment gaps

Repository code and installed site scripts are separate deployment surfaces:

- [comments.py](../shared/slurm/comments.py) recognizes all three modes and the legacy alias and enforces `250 <= interval_ms <= 60000`, but still permits or ignores other requests that the proposed strict contract must reject.
- [dispatcher.py](../shared/slurm/dispatcher.py) queues IB and hybrid attempts for [finalizer.py](../shared/slurm/finalizer.py) and still runs the OOB handler synchronously for OOB and hybrid requests. With `--inband-only` it only queues the in-band part, for a site wrapper that keeps its own OOB path: such a wrapper sends a `power:both` job through the dispatcher and then on to its existing OOB handling, with its ingestion delay. The finalizer exports IB results separately from the OOB report; a combined hybrid report is not implemented.
- [hooks.py](../inband/hooks.py), [runner.py](../inband/runner.py), [stager.py](../inband/stager.py), and the [runner](../shared/slurm/repacss-power-ib@.service) and [stager](../shared/slurm/repacss-power-stage@.service) units implement the compute-node lifecycle. The hooks read the comment, partition, and start time only from the Prolog environment, never touch shared storage, and skip invalid comments; both wrappers exit 0 on failure. The runner takes a final sample after SIGTERM and seals its spool.
- Each collector appends raw readings to one node-local file through [spool.py](../inband/spool.py): a time column and, for RAPL and NVML, the raw counter of each domain. Domain names, groups, ranges, and units are written once in the collector manifest, and deltas, power, and joules are derived at finalization. After Epilog, the stager publishes the sealed inventory with `.done` last. The staging root is set with `MONSTER_POWER_IB_STORE_ROOT` and must be a shared file system that root on every compute node can write under the export policy.
- [rapl.py](../inband/collectors/rapl.py) samples only audited domains that pass startup validation in [profiles.py](../inband/profiles.py), and keys counter state by path. It replaced runtime `rglob` discovery, which found no domains through the symlinked `/sys/class/powercap` tree on Python 3.9 and keyed state by the repeated names `dram` and `core`.
- The in-band path no longer imports pandas, so the system `/usr/bin/python3` can run the runner as well as the hooks.
- Both report types share one job directory under the report root, `<report_root>/<user>/<job_id>/oob/` and `.../ib/`, which is always kept. [delivery.py](../shared/slurm/delivery.py) then copies each report to `<work_dir>/power_reports/<job_id>/<oob|ib>/`, or under the user's home when the job's working directory cannot be used. The [delivery unit](../shared/slurm/repacss-power-deliver.service) runs as root only to switch to the job owner: a forked child takes the user's identity before it reads the shared copy or writes the destination, so files arrive owned by the user. SlurmUser cannot write into user directories, which is why the shared copy alone was the earlier design.
- The repository [controller wrapper](../src/services/slurm.epilogslurmctld) still sleeps, creates output directories, and invokes the synchronous dispatcher.

A site typically already has its own Prolog, Epilog, and EpilogSlurmctld scripts. A legacy controller wrapper may match only `power_profiling` and call the legacy `slurm_power_query.py`, and compute wrappers may carry other integrations. Append the REPACSS hooks to them rather than replacing them; [Deployment on the head node and the compute nodes](#deployment-on-the-head-node-and-the-compute-nodes) lists every change. Verify the active copies on the controllers and compute nodes before deployment; editing this repository does not update those copies.

## Repository change map

Paths marked **New** are proposed modules or deployment assets, not existing implementations.

| Area | Files to change or add | Required change | Modes |
| --- | --- | --- | --- |
| Request parsing | [shared/slurm/comments.py](../shared/slurm/comments.py) | Validate mode, allowed collectors, interval bounds, duplicate fields, and incompatible options. Preserve other comment namespaces and the legacy OOB alias. | All |
| Submission policy | **New:** `shared/slurm/job_submit_repacss_power.lua`, `shared/slurm/power_request.py` | Validate at submission and modification; provide the version-compatible canonical request to hooks. Keep Lua bounded and free of filesystem, network, and Slurm CLI calls. | All |
| Metadata resolution | [shared/slurm/resolver.py](../shared/slurm/resolver.py) | Use the audited Slurm environment or trusted handoff. Compute hooks no longer call `scontrol`; remove the remaining fallback from other production callers. | All |
| Controller dispatch | [shared/slurm/dispatcher.py](../shared/slurm/dispatcher.py) | IB and hybrid attempts are queued (done). Remaining: move synchronous OOB work into the worker. | All |
| Controller entry point | [src/services/slurm.epilogslurmctld](../src/services/slurm.epilogslurmctld); **New:** `shared/slurm/repacss-power.epilogslurmctld.sh` | Replace or retire the synchronous wrapper. Remove sleep, NFS directory creation, and report generation from the hook. Deploy exactly one active entry point. | All |
| Queue and finalization | [shared/slurm/finalize_queue.py](../shared/slurm/finalize_queue.py), [shared/slurm/finalizer.py](../shared/slurm/finalizer.py), [inband/aggregation.py](../inband/aggregation.py), [finalizer unit](../shared/slurm/repacss-power-finalizer.service) | Implemented for IB RAPL: queue, deadline, aggregation, atomic export. Remaining: leases and worker pool, GPU aggregation, hybrid report assembly. | All |
| Report delivery | [shared/slurm/delivery.py](../shared/slurm/delivery.py), [delivery unit](../shared/slurm/repacss-power-deliver.service) | Implemented: copy each finished report into the job's working directory or home as the job owner; the shared copy stays. | All |
| OOB worker | [oob/slurm/epilog_handler.py](../oob/slurm/epilog_handler.py) | Reuse queries outside the hook; expose a source result for shared finalization instead of independently owning hybrid delivery. | OOB, both |
| Compute hooks | [inband/hooks.py](../inband/hooks.py), [Prolog wrapper](../shared/slurm/repacss-power.prolog.sh), [Epilog wrapper](../shared/slurm/repacss-power.epilog.sh) | Start only IB/both and fail open (done); save exact local run state; stop from saved state; bound hook work. | IB, both |
| Attempt and allocation identity | **New:** `inband/identity.py`, `inband/allocation.py` | Isolate requeues and array/component executions; capture expected nodes, exclusivity, and allocated GPU identities. | IB, both; shared attempt identity for all reports |
| Static node profiles | [inband/profiles.py](../inband/profiles.py), [inband/rapl_profiles.json](../inband/rapl_profiles.json) | Implemented for RAPL: select by partition and verified node, validate at runner start, and write the manifest; see [In-band RAPL profiles](#in-band-rapl-profiles-for-zen4-and-h100). Partition collector defaults remain to do. | IB, both |
| Runner service | [inband/runner.py](../inband/runner.py), [shared/slurm/repacss-power-ib@.service](../shared/slurm/repacss-power-ib@.service) | Use a fixed installed interpreter/package; report readiness after collector startup; isolate collector failures; stop and seal within a bounded interval. | IB, both |
| Sensor readers | [rapl.py](../inband/collectors/rapl.py), [nvml.py](../inband/collectors/nvml.py), [auto_detect.py](../inband/collectors/auto_detect.py) | RAPL is profile-driven and keyed by path (done). NVIDIA GPU energy comes from NVML energy counters keyed by UUID (done). Remaining: sample only allocated GPUs on shared nodes. | IB, both |
| Storage and transfer | [inband/spool.py](../inband/spool.py), [inband/stager.py](../inband/stager.py), [stager unit](../shared/slurm/repacss-power-stage@.service) | Implemented: one local sample file per collector, seal, verified copy, `.done` last. Remaining: reconciliation timer for stagers that never ran. | IB, both |
| Packaging | **New:** `pyproject.toml`; deployment assets under `shared/slurm/` | Build an installable package for fixed, administrator-owned local environments on the participating hosts. | All |

The RAPL profile describes available hardware. It must not replace dynamic allocation metadata: Slurm determines which devices this job may measure.

## Deployment on the head node and the compute nodes

This section lists every installed change, by host, as validated on a Zen4 canary set: one-node and four-node `power:inband` and `power:both` jobs, OOB-only jobs, delivery into the job owner's directory, and a job on a node without in-band hooks. Placeholders: `<install_dir>` is a fixed release checkout such as `/opt/repacss-power-profiling`, `<queue_root>` the controller-local queue, normally `/var/spool/repacss-power/finalize-queue`, `<staging_root>` the shared staging directory, and `<report_root>` the shared report root. No credential is needed anywhere except by the OOB database query on the head node.

| Host | Component | Kind | Runs as | Source |
| --- | --- | --- | --- | --- |
| Head node | REPACSS block and OOB output path in the site `slurm.epilogslurmctld` | Slurm script | SlurmUser | Section [EpilogSlurmctld wrapper](#epilogslurmctld-wrapper) |
| Head node | `repacss-power-finalizer.service` | systemd service | SlurmUser | [unit](../shared/slurm/repacss-power-finalizer.service) |
| Head node | `repacss-power-deliver.service` | systemd service | root, identity switching only | [unit](../shared/slurm/repacss-power-deliver.service) |
| Head node | `/etc/repacss-power/controller.env` | configuration | read by both services | [example](../shared/slurm/repacss-power-controller.env.example) |
| Compute node | REPACSS block in the site `slurm.prolog` and `slurm.epilog` | Slurm script | SlurmdUser (root) | [prolog block](../shared/slurm/warewulf/slurm.prolog.block), [epilog block](../shared/slurm/warewulf/slurm.epilog.block) |
| Compute node | `repacss-power-ib@.service`, `repacss-power-stage@.service` | systemd unit templates | root | [runner](../shared/slurm/repacss-power-ib@.service), [stager](../shared/slurm/repacss-power-stage@.service) |
| Compute node | `/etc/repacss-power/node.env` | configuration | read by the hook blocks | Section [Distribution with Warewulf](#distribution-with-warewulf) |

### Head node

#### EpilogSlurmctld wrapper

The site's `slurm.epilogslurmctld` keeps its own OOB handling: it reads the job comment, waits for telemetry ingestion, queries the database, and writes the report. Two edits connect it to the rest of the system.

1. Insert this block after the line that sets `COMMENT` and before the wrapper's comment gate. It relies on the wrapper's `LOGFILE`, `COMMENT`, and `KEYWORD`; `KEYWORD` holds the legacy OOB keyword, `power_profiling`, that the gate compares against:

   ```bash
   # >>> REPACSS in-band >>>
   case "${SLURM_JOB_COMMENT:-$COMMENT}" in
     *power:inband*|*power:both*)
       env PYTHONPATH=<install_dir> REPACSS_POWER_IB_QUEUE_ROOT=<queue_root> \
         REPACSS_POWER_IB_NODES='<collecting nodes, Slurm hostlist>' \
         <install_dir>/.venv/bin/python -m shared.slurm.dispatcher --inband-only >>"$LOGFILE" 2>&1 || true
       case "${SLURM_JOB_COMMENT:-$COMMENT}" in
         *power:both*) COMMENT="$KEYWORD" ;;
         *) exit 0 ;;
       esac
       ;;
     *power:oob*) COMMENT="$KEYWORD" ;;
   esac
   # <<< REPACSS in-band <<<
   ```

2. Change the OOB output directory from `${POWER_BASE}/${RUN_AS}/${JOB_ID}` to `${POWER_BASE}/${RUN_AS}/${JOB_ID}/oob`, so both reports of a job share one directory.

Resulting behavior:

| Comment | In-band record queued | OOB report |
| --- | --- | --- |
| `power:inband` | yes | no |
| `power:both` | yes | yes, after the ingestion delay |
| `power:oob`, or the legacy `power_profiling` | no | yes |
| anything else | no | no |

The dispatcher call only writes one queue record and returns. `REPACSS_POWER_IB_NODES` lists the nodes whose hooks collect; a job that uses any other node is recorded with `unsupported_nodes`, and the finalizer immediately exports a `skipped` report that names those nodes. Remove the variable once every node collects. The OOB delay (90 s) covers Redfish report delivery, which arrives about every 60 s; the OOB query and plots take a few seconds for up to eight nodes, well within the hook timeout. The OOB handler then queues its report for delivery.

Write the edited wrapper back with `cat edited > slurm.epilogslurmctld` after a backup, so the file keeps its inode, ownership, and ACLs. When the controllers share `/etc/slurm`, one edit changes every controller at once.

#### Services and configuration

- `/etc/repacss-power/controller.env`, root-owned and mode 0644, names the interpreter (`REPACSS_POWER_PYTHON`), the checkout (`PYTHONPATH`), `REPACSS_POWER_IB_QUEUE_ROOT`, `MONSTER_POWER_IB_STORE_ROOT`, and `REPACSS_POWER_REPORT_ROOT`. The unit files contain no site paths.
- `repacss-power-finalizer.service` runs as SlurmUser and polls the queue every 15 s. For each record it waits until every expected node has published `.done`, or until the deadline ten minutes after the job, then writes `<report_root>/<user>/<job_id>/ib/` and queues it for delivery. A requeued job's later attempt writes `ib-<attempt start>/`.
- `repacss-power-deliver.service` runs as root with only `CAP_SETUID`, `CAP_SETGID`, `CAP_DAC_OVERRIDE`, `CAP_DAC_READ_SEARCH`, and `CAP_KILL`. For each queued report it forks a child that takes the job owner's uid, gid, and groups before touching any path, and the child copies the report to `<work_dir>/power_reports/<job_id>/<oob|ib>/`, or to `~/power_reports/<job_id>/` when the working directory is unusable on the head node. A delivery that does not finish within 120 s is abandoned. The service imports no configuration module, so it never holds database credentials.
- `<queue_root>` is owned by SlurmUser, mode 0750. Records move from `pending/` to `complete/` or `failed/`; delivery records live under `<queue_root>/deliver/`.

Install with `systemctl enable --now repacss-power-finalizer.service repacss-power-deliver.service` after copying both units to `/etc/systemd/system`.

#### Installation and mounts

- `<install_dir>` is a checkout of a release tag with a virtual environment from `requirements.txt`; the OOB handler needs it, and its `.env` holds the database settings. Update it with `git pull --ff-only` to the next tag, and check first that the `.env` still defines `REPACSS_DB_USER`, `REPACSS_DB_PASSWORD`, and `REPACSS_SSH_HOSTNAME`, because the code has no built-in defaults for them.
- The head node must mount `<staging_root>` and `<report_root>`, and the users' home and work file systems for delivery. A report whose destination is not mounted stays in the shared copy only.
- A backup controller runs the same wrapper. Without `<install_dir>`, `<queue_root>`, and the two services there, jobs that end during a failover get their OOB report in the shared report root, but no in-band report and no delivery.

### Compute nodes

#### Prolog and Epilog blocks

Append [slurm.prolog.block](../shared/slurm/warewulf/slurm.prolog.block) to the end of the site's `slurm.prolog` and [slurm.epilog.block](../shared/slurm/warewulf/slurm.epilog.block) to `slurm.epilog`, before any final `exit`; both are shown below. Both blocks are identical on every node; whether a node collects, and where the code and staging root are, come from `/etc/repacss-power/node.env`. The Prolog block starts no process for jobs without an in-band comment, and neither block can fail the site script.

```bash
# >>> REPACSS power >>>
# Opt-in in-band profiling for power:inband and power:both jobs. Settings, including
# whether this node collects, come from /etc/repacss-power/node.env. Never fails the prolog.
case "${SLURM_JOB_COMMENT:-}" in
  *power:inband*|*power:both*)
    if [ -r /etc/repacss-power/node.env ]; then
      ( set -a; . /etc/repacss-power/node.env; set +a
        [ "${REPACSS_POWER_IB_ENABLED:-0}" = 1 ] || exit 0
        exec /usr/bin/env bash "$REPACSS_POWER_HOME/shared/slurm/repacss-power.prolog.sh" ) || true
    fi
    ;;
esac
# <<< REPACSS power <<<
```

```bash
# >>> REPACSS power >>>
# Stop and stage the in-band run that the prolog started for this job, even if the node
# was disabled meanwhile. Never fails the epilog.
if [ -f "/run/repacss-power/${SLURM_JOB_ID:-none}.env" ] && [ -r /etc/repacss-power/node.env ]; then
  ( set -a; . /etc/repacss-power/node.env; set +a
    exec /usr/bin/env bash "$REPACSS_POWER_HOME/shared/slurm/repacss-power.epilog.sh" ) || true
fi
# <<< REPACSS power <<<
```

`/etc/repacss-power/node.env`:

```bash
REPACSS_POWER_IB_ENABLED=1
REPACSS_POWER_HOME=/opt/repacss-power-profiling
MONSTER_POWER_PYTHON=/usr/bin/python3
MONSTER_POWER_IB_STORE_ROOT=<staging_root>
```

The canary installation used an earlier form of the Prolog block with the host list and code path written into it. The blocks above move both into `node.env`, so the scripts can be the same on every node and live in the shared site overlay or image.

The Prolog wrapper starts the runner only for an exclusive allocation (`SLURM_JOB_OVERSUBSCRIBE=NO` on Slurm 25.11, or `SLURM_JOB_EXCLUSIVE=NODE`), and only when the node is in the audited RAPL profile ([rapl_profiles.json](../inband/rapl_profiles.json)); otherwise it logs the reason under the syslog tag `repacss-power-hooks` and starts nothing. A node added to the cluster later needs an audit entry before it collects.

#### systemd units and local files

- `repacss-power-ib@.service` and `repacss-power-stage@.service` go to `/etc/systemd/system`. They are templates that the hooks instantiate per job: Prolog starts the runner with `systemctl start`, and Epilog stops it and starts the stager with `--no-block`. They are not enabled. systemd reads them at boot, so they belong in a system overlay or the image, not in a runtime overlay.
- The code is the `inband/` package and a few `shared/` modules, about 700 KB of standard-library Python that runs with the system `/usr/bin/python3`. No virtual environment, database access, or credentials are needed on compute nodes.
- Runtime state: `/run/repacss-power/` (per-job environment files) and `/var/spool/repacss-power/` (the local spool); both are created on demand. On a stateless node both live in memory, which is acceptable because a node failure also ends the job.
- `<staging_root>` must be writable by root on every compute node, or by a provisioned service identity under `root_squash`.

### Distribution with Warewulf

Warewulf provisions stateless nodes from a node image plus overlays. System overlays are applied at boot, on top of the image; runtime overlays are refreshed by `wwclient` while the node runs, typically every minute. The layout below keeps REPACSS out of the image, so rebuilding the image for other reasons cannot remove it, and puts only the per-node switch in a runtime overlay, so nodes can be enabled or disabled without a reboot.

| Item | Where | Why |
| --- | --- | --- |
| `<install_dir>` at a release tag | system overlay `repacss-power-system` | pinned code, the same on every node, independent of the image |
| Two unit templates | system overlay `repacss-power-system`, `/etc/systemd/system/` | applied at boot, before systemd reads them |
| Prolog and Epilog blocks | wherever the site's `slurm.prolog` and `slurm.epilog` come from, typically a site system overlay such as `slurm` | identical on every node; a block placed in the image would be overwritten at boot by an overlay that ships the same file |
| `/etc/repacss-power/node.env` | runtime overlay `repacss-power`, rendered from node tags | enable or disable a node without a reboot |

Commands below use Warewulf 4.6 names; check the flags with `--help` on the installed version. Find where the site scripts come from first: `wwctl overlay list --all` lists every overlay file, and `wwctl node list --all <node>` shows a node's image and overlay lists.

1. **System overlay.** Create `repacss-power-system` (`wwctl overlay create`), and add the `inband` and `shared` packages of the release tag under `/opt/repacss-power-profiling` and the two unit files under `/etc/systemd/system/` with `wwctl overlay mkdir` and `wwctl overlay import`. Leave out `*.ww` files, which Warewulf would render as templates. Append the two blocks to `slurm.prolog` and `slurm.epilog` in the overlay or image that provides them, keeping their mode: a non-executable `slurm.prolog` fails every job and drains the node.
2. **Runtime overlay.** Create the overlay and its template:

   ```bash
   wwctl overlay create repacss-power
   wwctl overlay edit --parents repacss-power /etc/repacss-power/node.env.ww
   ```

   with the content of [node.env.ww](../shared/slurm/warewulf/node.env.ww):

   ```text
   # Generated by Warewulf for {{ .Id }}; set the node tag repacss_ib=1 to collect.
   REPACSS_POWER_IB_ENABLED={{ if .Tags.repacss_ib }}1{{ else }}0{{ end }}
   REPACSS_POWER_HOME=/opt/repacss-power-profiling
   MONSTER_POWER_PYTHON=/usr/bin/python3
   MONSTER_POWER_IB_STORE_ROOT={{ .Tags.repacss_staging_root }}
   ```

3. **Attach and configure.** Add `repacss-power-system` to the compute profile's system overlays after the overlay that provides the Slurm scripts, and `repacss-power` to its runtime overlays, keeping the existing lists (`wwctl profile list --all` shows them). Set the staging root once on the profile, and build:

   ```bash
   wwctl profile set <profile> --system-overlays=<existing overlays>,repacss-power-system
   wwctl profile set <profile> --runtime-overlays=<existing overlays>,repacss-power
   wwctl profile set <profile> --tagadd repacss_staging_root=<staging_root>
   wwctl overlay build
   wwctl overlay show --render <node> repacss-power /etc/repacss-power/node.env.ww
   ```

   Every node now has `REPACSS_POWER_IB_ENABLED=0`: the hooks are installed but inactive. A canary can use node-level lists instead of the profile; Warewulf 4.6 merges node and profile lists.
4. **Reboot** in a rolling fashion so that the system overlay applies, for example `scontrol reboot ASAP nextstate=resume reason="repacss power hooks" <nodes>` where a `RebootProgram` is configured, or drain, `wwctl power cycle`, and resume. Running jobs finish first.
5. **Canary.** Enable a few nodes with `wwctl node set <nodes> --tagadd repacss_ib=1` and `wwctl overlay build`. Within one runtime-overlay refresh the nodes collect, with no reboot. Set `REPACSS_POWER_IB_NODES` in the head-node block to the same hostlist.
6. **Widen.** Tag whole partitions, keeping `REPACSS_POWER_IB_NODES` equal to the tagged set; remove the variable from the head-node block when every node collects.
7. **Upgrade.** Deploy the same release tag on the head node and in `repacss-power-system`. A release that changes the sample file format needs both updated together, because the finalizer reads only the current format (node seals carry `schema_version`); let running in-band jobs finish before switching.
8. **Disable or roll back.** Removing the `repacss_ib` tag and rebuilding overlays stops new collection within one refresh; jobs already collecting are still stopped and staged by the Epilog block, which does not check the switch. To remove the integration, detach both overlays, remove the blocks from the site scripts, and reboot. Staged data and reports stay.

Do not distribute credentials, the head node's `.env`, or the controller services to compute nodes; the compute side needs none of them.

### Slurm configuration notes

| Setting | Note |
| --- | --- |
| `job_submit.lua` | Not required: Prolog enforces `--exclusive` and validates the request. Rejecting non-exclusive in-band requests at submission would need a site `job_submit.lua` policy, which is a site decision. |
| Exclusivity variable | Verify which variables the installed Slurm exports to Prolog and EpilogSlurmctld. Slurm 25.11 gives both `SLURM_JOB_OVERSUBSCRIBE` and no `SLURM_JOB_EXCLUSIVE`. |
| `SLURM_JOB_WORK_DIR` | EpilogSlurmctld receives it on Slurm 25.11; delivery uses it as the first destination. |
| Several Prolog programs | Recent Slurm releases accept a list of Prolog and Epilog programs in `slurm.conf`. A separate REPACSS script would avoid editing the site scripts, but it changes `slurm.conf`, which is a site decision. The blocks above change only the existing scripts. |

Compute Prolog and Epilog normally run as `SlurmdUser=root`. If the site sets `PrologFlags=Contain`, it already implies `Alloc`; preserve the existing flags. Evaluate adding `DeferBatch` only if the batch launch must wait for all nodes, and pair it with real runner readiness. Do not replace the site's complete flag list or hook timeout values with a generic example. See the [Slurm configuration reference](https://slurm.schedmd.com/slurm.conf.html).

## Permissions

Users submit `--comment='power:inband'`; they do not receive root or sudo privileges. Node Prolog runs as the configured `SlurmdUser` and starts an administrator-managed system service. `TaskProlog` runs as the job user and is not the privilege mechanism for this design. The controller hook runs as the controller daemon user, which is a separate identity. See the [Slurm Prolog and Epilog Guide](https://slurm.schedmd.com/prolog_epilog.html).

A canary sampler may run as root with fixed, administrator-owned code and configuration. A service account with explicit read access to sensor attributes is the preferred narrower deployment where supported. Permission rules must survive reboot or driver reinitialization. Neither the comment nor a user-controlled environment may select privileged executables, Python import paths, or output roots. The next section lists which files each hook and service reads.

## In-band RAPL profiles for Zen4 and H100

This section covers only the RAPL domains exposed on the Zen4 and H100 node classes. It is the source for the in-band part of the journal extension. It reuses curvu's read-only counter method and excludes curvu's frequency and EPP control. GPU energy, MI210 nodes, and frequency scaling are out of scope here.

### Audited domain layout

[rapl_profiles.json](../inband/rapl_profiles.json) records audit `rapl-audit-20261008T204824Z-1343382` of 119 nodes: 110 in `zen4`, 8 in `h100`, and 1 in `h100-build`. Every node in a partition had the same structure. Paths are relative to `/sys/class/powercap/intel-rapl/`.

| Profile (partitions) | Domain path | `name` | Group | `max_energy_range_uj` |
| --- | --- | --- | --- | --- |
| `zen4_rapl` (`zen4`) | `intel-rapl:0` | `package-0` | `cpu_package` | 65,532,610,987 |
| | `intel-rapl:0/intel-rapl:0:0` | `core` | `core` | 65,532,610,987 |
| | `intel-rapl:1` | `package-1` | `cpu_package` | 65,532,610,987 |
| | `intel-rapl:1/intel-rapl:1:0` | `core` | `core` | 65,532,610,987 |
| `h100_host_rapl` (`h100`, `h100-build`) | `intel-rapl:0` | `package-0` | `cpu_package` | 262,143,328,850 |
| | `intel-rapl:0/intel-rapl:0:0` | `dram` | `host_dram` | 262,143,328,850 |
| | `intel-rapl:1` | `psys` | `platform` | 4,294,967,295,000,000 |
| | `intel-rapl:2` | `package-1` | `cpu_package` | 262,143,328,850 |
| | `intel-rapl:2/intel-rapl:2:0` | `dram` | `host_dram` | 262,143,328,850 |

The audit also settles the earlier single-node discrepancy: `rpc-92-1` has `core` under both packages. It checked structure and reads as UID 0, and found every `energy_uj` at mode `0400`, owned by root. It did not check that the counters advance.

Index positions are not portable between profiles. `intel-rapl:1` is `package-1` on Zen4 and `psys` on H100, where `package-1` is `intel-rapl:2`. Names repeat across sockets (`dram`, `core`). Each profile therefore identifies a domain by its full path, which the audit found identical on every node of the partition, and startup validation confirms the `name` at that path.

Report CPU package, host DRAM, and platform energy separately. Only `cpu_package` domains form the CPU total, and only `host_dram` domains form the DRAM total. Two consequences follow for reporting:

- Zen4 exposes no in-band DRAM counter, so Zen4 memory energy remains OOB-only.
- On H100, neither `dram` nor `psys` measures the GPUs alone. H100 device energy comes from the NVML energy counter, reported as a separate `gpu` total.

`psys` stays outside every total because its server scope is vendor-defined. The AMD `core` domain is diagnostic until its scope is measured. The kernel reads the core-energy MSR on one CPU of each package, and AMD describes that MSR per core, so the domain may cover a single core rather than all cores. Shared-node package readings must not be labeled job-only energy; the README's `exclusive_only` policy applies.

### Counter method reused from curvu

curvu's `linux-epp-cpu` path (`curvu/src/sysfs.rs`, `read_power`) supplies the counter arithmetic. It reads `max_energy_range_uj`, then reads `energy_uj` twice. If the second value is smaller, it adds one wraparound: `(range - e1) + e2`. It divides by monotonic elapsed time. [rapl.py](../inband/collectors/rapl.py) uses the same delta in `counter_delta_uj`. curvu's NVIDIA path identifies a device by PCI bus ID through NVML, not by enumeration index (`curvu/src/shim/device_mapper.rs`). GPU identities should follow the same rule.

REPACSS excludes curvu's control path: writes to `scaling_max_freq` and `energy_performance_preference`, the `powersave` governor check, NVIDIA locked or application clocks, and the OpenCL load. The runner never writes sysfs attributes or device settings.

| curvu behavior | Effect on these nodes | REPACSS rule |
| --- | --- | --- |
| Selects a domain by number: `--rapl-domain`, defaulting to the OpenCL device ID | Domain `1` is `package-1` on Zen4 but `psys` on H100 | Use the audited path for the partition and confirm its `name` at startup |
| Builds `/sys/devices/virtual/powercap/intel-rapl/intel-rapl:<n>` | Reaches only top-level domains; `core` and `dram` live under their parent directory | Read the profile's full path, including the parent directory |
| Reads one domain per run | Measures one of two sockets | Read every validated domain in one pass |
| Takes 250 ms windows only when benchmark samples arrive | Windows leave gaps; adequate for power curves, not job energy | Record a delta for every consecutive pair of samples for the whole run |

### Profile selection and startup validation

The audited profile and runtime checks have separate jobs:

- The profile says which domains exist on each partition and which group each belongs to.
- At runner start, [profiles.py](../inband/profiles.py) selects the profile and checks each domain on the current boot.
- The runner writes the result to `rapl_manifest.json` and samples only the domains that passed.

Epilog does not rediscover domains. These were the options:

| Option | Failure on Zen4 and H100 |
| --- | --- |
| One path list for every node, with no startup check | `intel-rapl:1` is socket 1 on Zen4 but `psys` on H100. A kernel, firmware, or driver change could renumber domains without an error. |
| Discovery in Epilog | Too late. Epilog runs after the job's processes end, so it has no starting value. One end-of-job read cannot produce energy, and a changed domain set pairs different domains. |
| Discovery only, at start | Finds what exists but not what it means. It gives no basis for excluding `core` and `psys`. A missing domain, such as DRAM RAPL disabled in firmware, silently drops out of the report. |
| **Implemented:** audited per-partition profile, checked at runner start, saved as a manifest | Differences become explicit statuses, and totals follow reviewed groups. |

Compute Prolog passes `SLURM_JOB_PARTITION` and the Slurm node name to the runner. The runner requires the node in that partition's `verified_nodes`, which ties the partition to audited hardware: a node outside the audit is skipped rather than assumed to match. The loader also rejects the whole file if any profile's domains no longer match its audit signature or any path leaves the root.

| Manifest `status` | `reason` | When |
| --- | --- | --- |
| `skipped` | `missing_partition` | Prolog had no `SLURM_JOB_PARTITION` |
| `skipped` | `no_profile_for_partition` | The partition has no profile, for example an MI210 partition |
| `skipped` | `unverified_node` | The node is not in the partition's `verified_nodes` |
| `failed` | `invalid_profile_config: ...` | The file is unreadable, a signature does not match, or a path leaves the root |
| `failed` | `domain_validation_failed` | No domain passed its check |
| `partial` | `domain_validation_failed` | Some domains passed; only those are sampled |
| `ready` | | Every domain passed |

Each domain's check reads `name` and `max_energy_range_uj` under its path and then one `energy_uj` value. The domain status is `ok`, `missing`, `name_mismatch`, `range_mismatch`, `permission_denied`, or `unreadable`, and only `ok` domains are sampled. The finalizer must treat a total that needs an excluded domain as unavailable rather than summing the remaining domains.

The runner writes the manifest next to `rapl.csv`, including when RAPL is skipped. Abbreviated:

```json
{
  "kind": "repacss_rapl_manifest",
  "status": "ready",
  "reason": "",
  "partition": "h100",
  "node": "rpg-93-1",
  "profile": "h100_host_rapl",
  "profile_signature": "e71377b042fe8a70...",
  "config_sha256": "<SHA-256 of the profile file>",
  "root": "/sys/class/powercap/intel-rapl",
  "kernel": "<uname -r>",
  "domains": [
    {"path": "intel-rapl:1", "name": "psys", "group": "platform",
     "energy_file": "/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj",
     "max_energy_range_uj": 4294967295000000, "status": "ok", "message": ""}
  ]
}
```

The manifest belongs in the sealed inventory, and the finalizer should copy it into `provenance.json`.

### What the hooks and runner can read

Slurm runs compute-node Prolog and Epilog as `SlurmdUser` (root here), outside the job cgroup and without a search path. A non-zero Epilog exit drains the node ([Slurm guide](https://slurm.schedmd.com/prolog_epilog.html)).

| File | Reader | Observed access | Result |
| --- | --- | --- | --- |
| Domain `name`, `max_energy_range_uj` | Runner at start | `0444` on the audited nodes | Any user can read them. |
| Domain `energy_uj` | Runner, every sample | `0400 root` on all 119 audited nodes | Root hooks and the current unit, which has no `User=`, can read it. A `User=repacss-power` unit needs a permission rule that is reapplied whenever domains are re-registered. As non-root, validation reports `permission_denied`. |
| Profile | Runner at start | Packaged `inband/rapl_profiles.json`; `REPACSS_POWER_RAPL_PROFILES` overrides it | Use a copy on local disk in production. Under NFS `root_squash`, root reads as an unprivileged user, and an NFS outage stalls the hook. |
| Prolog run state | Epilog | `/run/repacss-power/`, local tmpfs (current `DEFAULT_STATE_DIR`) | Readable; Epilog needs only the exact unit name from it. |
| Manifest | Finalizer, through the staged copy | Written by the runner | Epilog does not need it. |
| Code and interpreter | Wrappers and runner | Repository on NFS through `PYTHONPATH`; system `/usr/bin/python3` | The in-band path no longer imports pandas, so the system interpreter runs both hooks and runner. Production should still use an installed package on local disk. |

The wrappers set `PATH` explicitly, default to `/usr/bin/python3`, and always exit 0, logging any helper failure with `logger -t repacss-power`. Before this change, they found `python3` and `systemctl` only through bash's unexported default `PATH` and Python's `os.defpath` fallback.

Epilog can read every file above, but it reads only the Prolog state and sends non-blocking stop and stage requests. The runner takes the final counter sample after SIGTERM, so all counter reads stay in one process on one monotonic clock.

### Counter range and sampling interval

A Prolog read paired with an Epilog read cannot measure job energy, because two readings cannot tell how many times the counter wrapped. The Zen4 range is about 65.5 kJ, which wraps in about 164 s at a sustained 400 W. The Intel package and DRAM range is about 262.1 kJ, which wraps in about 749 s at 350 W.

Each RAPL row records `delta_uj`, the measured `interval_seconds`, `wrapped`, and `wrap_ambiguous`. One wrap is unambiguous only if `interval × P < range`. The collector uses a 1000 W ceiling for any single domain, so it flags `wrap_ambiguous` when an interval reaches about 65.5 s on Zen4 or 262 s on the Intel domains. The comment parser bounds `interval_ms` to 250-60000, which stays below the Zen4 limit; a flagged row therefore means a gap, such as a stalled runner. If any interval of a domain is flagged, the finalizer must set that domain's total to `null` and report the gap.

The runner starts at Prolog and stops after Epilog, so its samples cover more than the job's execution. The finalizer should integrate between the samples that bracket job start and end and record the uncovered head and tail. At 1 s sampling, each boundary is off by at most one interval of energy per domain.

### Validation before publication

1. As root on nodes of each partition, run `python3 -m inband.profiles --partition <partition> --sample-seconds 2` from the repository root. It must report `ready` and `"advancing": true` for every domain; the audit did not check advancement.
2. Measure the AMD `core` scope: pin a single-thread load to the package's first CPU, then to another core in the same package, and compare the `core` deltas.
3. Run a `power:inband` canary job and confirm that the node directory contains `rapl_manifest.json` with status `ready`. A `missing_partition` reason means the installed Slurm does not export `SLURM_JOB_PARTITION` to Prolog.

### Manuscript implications

The journal extension can state these points now:

- The per-partition domain table from a 119-node audit, with domains identified by path and confirmed by name at startup. The H100 and Zen4 index difference motivates per-partition profiles.
- The CPU package total sums `package-0` and `package-1`. Host DRAM is available in-band only on H100. `psys` and AMD `core` are reported separately and excluded from totals.
- Zen4 memory energy is OOB-only, and H100 GPU energy comes from NVML energy counters, the same counter-difference method as RAPL. On `rpg-93-7`, counter-derived GPU power was within 1-2% of `nvidia-smi` `power.draw`.
- Each in-band report records the profile signature, the profile file digest, and the resolved manifest in its provenance.

These points need measurements before the paper claims them: counter advancement on the audited nodes, in-band and OOB agreement (package sum against iDRAC CPU power, DRAM sum against iDRAC memory power), the AMD `core` scope, and collection overhead. If the paper credits curvu as the source of the counter reader, say that REPACSS does not use its frequency control.

## Storage and finalization boundaries

```text
compute Prolog -> per-node runner -> local spool -> independent stager -> NFS staging
compute Epilog -> stop and seal ----------------------^

EpilogSlurmctld -> controller-local queue -> finalizer service -> shared report root
                                                ^                        |
                            NFS IB artifacts (OOB: hook queries DB)      v
                                             delivery service -> job working directory or home
```

The sampling and hook paths must not depend on NFS availability. Samples stay in node-local files until the job ends; at about 0.25 MB per node-hour for raw RAPL readings at 1 s, a long job's trace fits in the local spool, and a node failure ends the job anyway. Transfer into an NFS-side temporary file and rename within that filesystem after verification. Publish `.done` last, only after every file in the producer's sealed inventory is present; the marker means publication finished, not that every collector succeeded.

The finalizer derives expected nodes from allocation metadata, not from the directories that happened to arrive. Node completion and the controller event are independent. Apply a deadline, preserve valid samples from healthy nodes, and report missing nodes explicitly. Pure IB must finalize without an OOB database connection; hybrid reports preserve either successful source when the other fails. Do not add OOB whole-node energy and overlapping IB component energy together.

Root on an NFS client does not guarantee write or `chown` access under `root_squash`. Use a provisioned service identity for staging and a reviewed exporter identity for user-owned reports. See the [NFS identity mapping reference](https://man7.org/linux/man-pages/man5/exports.5.html).

## Implementation order and validation

1. Freeze the mode parser, trusted hook handoff, static profiles, and shared attempt identity. Verify the real node interfaces and permissions.
2. Complete the compute lifecycle, GPU filtering, local sample files, and stager. Validate one Zen4 and one H100 node before two-node jobs.
3. Replace synchronous controller work with durable queueing and implement OOB-only and IB-only finalization independently.
4. Assemble hybrid results, preserve the legacy OOB alias, and exercise recovery and rollback before widening deployment.

Use the existing [unit-test entry point](../tests/run_tests.py) as the no-credentials baseline after a fresh clone:

```bash
git clone <repository-url> repacss-power-profiling
cd repacss-power-profiling
python3 -m venv .venv-test
source .venv-test/bin/activate
python -m pip install -r requirements.txt -r tests/requirements-unit.txt
python tests/run_tests.py --type unit
```

As the new modules are implemented, extend `tests/unit/test_slurm_comment_parser.py`, `tests/unit/inband/`, and add queue/finalizer tests plus `tests/integration/test_inband_fake_slurm_lifecycle.py`. Add installed-package smoke checks from outside the checkout once packaging exists. The baseline command alone does not certify the proposed deployment.

Canary acceptance must cover all three modes, no-comment jobs, two-node collection, GPU allocation filtering, cancellation, requeue isolation, unavailable collectors, NFS outages, missing nodes, duplicate finalization events, and pure IB with OOB unavailable. REPACSS telemetry failures must not change the user's job result or drain a node. Retain collected data when disabling new IB activation or rolling back the hook integration.
