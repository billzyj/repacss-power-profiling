"""Aggregate one attempt's staged in-band energy-counter data into job-level energy.

Inputs are the node directories that the stager published under
`<staging_root>/<attempt_id>/<node>/`. A node counts only if its `.done` marker exists
and every counter chunk matches the size and SHA-256 listed in its `node_status.json`.
The counter collectors are RAPL (CPU package, host DRAM, core, platform) and NVML
(one domain per GPU); both write the same columns.

Energy method: each collector already turns consecutive counter readings into
wrap-corrected `delta_uj` values, each covering the interval that ends at its row
timestamp. A domain's energy is the sum of its deltas. A delta whose interval crosses
the job window boundary is prorated by the fraction of the interval inside the window.
A domain with a `wrap_ambiguous` interval, or without samples, has no energy value.

Only `cpu_package`, `host_dram`, and `gpu` domains form totals, each separately.
`core` and `platform` domains are reported per domain and never added to a total.
This module does not read OOB data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from inband.chunks import sha256_file
from inband.stager import DONE_FILENAME, NODE_STATUS_FILENAME
from inband.storage import read_json_file

RAW_COLUMNS = [
    "timestamp",
    "elapsed_ms",
    "sequence",
    "job_id",
    "attempt_id",
    "node",
    "scope_type",
    "scope_id",
    "metric_name",
    "value",
    "unit",
    "source_id",
    "source_kind",
    "collector_family",
    "timestamp_mode",
    "retrieval_mode",
]
COUNTER_COLLECTORS = ("rapl", "nvml")
TOTAL_GROUPS = ("cpu_package", "host_dram", "gpu")
ENERGY_METHOD = (
    "Sum of wrap-corrected energy-counter differences (RAPL for CPU and DRAM, NVML for GPUs) "
    "between consecutive samples; an interval that crosses a job window boundary is prorated linearly."
)
# Below this share of the job window, a node's samples are reported as incomplete coverage.
MIN_COVERAGE = 0.95


@dataclass(frozen=True)
class Window:
    """Job execution window. Either end may be unknown, which disables clipping there."""

    start: Optional[datetime] = None
    end: Optional[datetime] = None

    @property
    def seconds(self) -> Optional[float]:
        if self.start is None or self.end is None:
            return None
        return (self.end - self.start).total_seconds()

    def fraction_inside(self, interval_start: datetime, interval_end: datetime) -> float:
        length = (interval_end - interval_start).total_seconds()
        low = max(interval_start, self.start) if self.start else interval_start
        high = min(interval_end, self.end) if self.end else interval_end
        if length <= 0:
            return 1.0 if low <= high else 0.0
        return max(0.0, (high - low).total_seconds()) / length


@dataclass
class DomainEnergy:
    path: str
    name: str
    group: str
    energy_uj: float = 0.0
    covered_seconds: float = 0.0
    samples: int = 0
    wrap_ambiguous: bool = False

    def energy_j(self) -> Optional[float]:
        if self.wrap_ambiguous or self.samples == 0:
            return None
        return round(self.energy_uj / 1_000_000.0, 6)

    def summary(self, window_seconds: Optional[float]) -> Dict[str, Any]:
        energy = self.energy_j()
        seconds = window_seconds or self.covered_seconds
        return {
            "name": self.name,
            "group": self.group,
            "energy_j": energy,
            "average_power_w": round(energy / seconds, 3) if energy is not None and seconds else None,
            "samples": self.samples,
            "covered_seconds": round(self.covered_seconds, 3),
            "wrap_ambiguous": self.wrap_ambiguous,
        }


def _verified_chunks(node_dir: Path, node_status: Dict[str, Any], collector: str) -> Tuple[List[Path], str]:
    chunks = []
    for entry in node_status.get("files", []):
        if entry.get("collector") != collector:
            continue
        path = node_dir / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            return [], f"{entry['path']} does not match its published size and checksum"
        chunks.append(path)
    return chunks, ""


def _manifest_domains(manifest: Dict[str, Any]) -> Dict[str, DomainEnergy]:
    return {
        domain["path"]: DomainEnergy(path=domain["path"], name=domain["name"], group=domain["group"])
        for domain in manifest.get("domains", [])
        if domain.get("status") == "ok"
    }


def aggregate_node(
    node_dir: Path,
    *,
    node: str,
    job_id: str,
    attempt_id: str,
    window: Window,
    emit: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Summarize one node and emit its normalized rows. Never raises for missing data."""
    if not (node_dir / DONE_FILENAME).exists():
        return {"state": "missing", "message": "no .done marker in staging"}
    node_status = read_json_file(node_dir / NODE_STATUS_FILENAME)
    manifests = {collector: read_json_file(node_dir / f"{collector}_manifest.json") for collector in COUNTER_COLLECTORS}
    result: Dict[str, Any] = {
        "state": node_status.get("state", "unknown"),
        "message": "",
        "rapl_profile": manifests["rapl"].get("profile", ""),
        "rapl_manifest_status": manifests["rapl"].get("status", "absent"),
        "rapl_manifest_reason": manifests["rapl"].get("reason", ""),
        "nvml_manifest_status": manifests["nvml"].get("status", "absent"),
    }
    domains: Dict[str, DomainEnergy] = {}
    sequences: Dict[str, int] = {}
    for collector in COUNTER_COLLECTORS:
        chunks, problem = _verified_chunks(node_dir, node_status, collector)
        if problem:
            result.update(state="invalid", message=problem)
            return result
        domains.update(_manifest_domains(manifests[collector]))
        for chunk in chunks:
            with chunk.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    domain = domains.setdefault(
                        row["domain_path"],
                        DomainEnergy(path=row["domain_path"], name=row["domain_name"], group=row["domain_group"]),
                    )
                    if row["wrap_ambiguous"] == "True":
                        domain.wrap_ambiguous = True
                    if not row["delta_uj"]:
                        continue  # no preceding reading, or the counter was reset
                    interval_end = datetime.fromisoformat(row["timestamp"])
                    interval = float(row["interval_seconds"] or 0.0)
                    fraction = window.fraction_inside(interval_end - timedelta(seconds=interval), interval_end)
                    if fraction <= 0:
                        continue
                    energy_uj = int(row["delta_uj"]) * fraction
                    domain.samples += 1
                    domain.energy_uj += energy_uj
                    domain.covered_seconds += interval * fraction

                    source_id = f"{node}/{collector}/{domain.path}"
                    sequences[source_id] = sequences.get(source_id, 0) + 1
                    common = {
                        "timestamp": row["timestamp"],
                        "elapsed_ms": (
                            round((interval_end - window.start).total_seconds() * 1000) if window.start else ""
                        ),
                        "sequence": sequences[source_id],
                        "job_id": job_id,
                        "attempt_id": attempt_id,
                        "node": node,
                        "scope_type": domain.group,
                        "scope_id": f"{domain.name}@{domain.path}",
                        "source_id": source_id,
                        "source_kind": "in_band",
                        "collector_family": collector,
                        "timestamp_mode": "interval_end",
                        "retrieval_mode": "job_local_artifact",
                    }
                    emit({**common, "metric_name": "power", "value": row["power_watts"], "unit": "W"})
                    emit({**common, "metric_name": "energy", "value": round(energy_uj / 1_000_000.0, 6), "unit": "J"})

    window_seconds = window.seconds
    result["domains"] = {path: domain.summary(window_seconds) for path, domain in sorted(domains.items())}
    for group in TOTAL_GROUPS:
        values = [domain.energy_j() for domain in domains.values() if domain.group == group]
        result[f"{group}_energy_j"] = round(sum(values), 6) if values and None not in values else None
    packages = [domain for domain in domains.values() if domain.group == "cpu_package"]
    if window_seconds and packages:
        result["coverage"] = round(min(domain.covered_seconds for domain in packages) / window_seconds, 4)
    return result


def aggregate_attempt(
    job_dir: Path,
    *,
    job_id: str,
    attempt_id: str,
    expected_nodes: Sequence[str],
    window: Window,
    emit: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Aggregate every expected node and derive job totals and the overall status."""
    nodes = {
        node: aggregate_node(job_dir / node, node=node, job_id=job_id, attempt_id=attempt_id, window=window, emit=emit)
        for node in expected_nodes
    }
    notes: List[str] = []
    energy: Dict[str, Optional[float]] = {}
    for group in TOTAL_GROUPS:
        lacking = [node for node, result in nodes.items() if result.get(f"{group}_energy_j") is None]
        energy[group] = None if lacking else round(sum(nodes[node][f"{group}_energy_j"] for node in nodes), 6)
        if lacking:
            notes.append(f"No job {group} total: no {group} energy from {', '.join(lacking)}.")
    for node, result in nodes.items():
        if result["state"] in {"missing", "invalid"}:
            notes.append(f"{node}: {result['state']} ({result['message']}).")
        elif result.get("rapl_manifest_status") not in {"ready", None}:
            notes.append(f"{node}: RAPL {result['rapl_manifest_status']} ({result.get('rapl_manifest_reason', '')}).")
        if result.get("coverage") is not None and result["coverage"] < MIN_COVERAGE:
            notes.append(f"{node}: samples cover {result['coverage']:.0%} of the job window.")

    window_seconds = window.seconds
    any_energy = any(
        domain["energy_j"] is not None for result in nodes.values() for domain in result.get("domains", {}).values()
    )
    if not any_energy:
        status = "failed"
    elif energy["cpu_package"] is not None and all(result["state"] == "complete" for result in nodes.values()):
        status = "complete"
    else:
        status = "partial"
    return {
        "status": status,
        "energy_j": energy,
        "average_power_w": {
            group: round(value / window_seconds, 3) if value is not None and window_seconds else None
            for group, value in energy.items()
        },
        "nodes": nodes,
        "missing_nodes": [node for node, result in nodes.items() if result["state"] == "missing"],
        "notes": notes,
    }
