"""Aggregate one attempt's staged in-band energy-counter data into job-level energy.

Inputs are the node directories that the stager published under
`<staging_root>/<attempt_id>/<node>/`. A node counts only if its `.done` marker exists
and every sample file matches the size and SHA-256 listed in its `node_status.json`.
The counter collectors are RAPL (CPU package, host DRAM, core, platform) and NVML
(one domain per GPU). Their sample files hold raw counter readings: a time column `t`
and one column per domain, named by the domain's path in the collector's manifest.

Energy method: the difference between consecutive readings of a domain is that
interval's energy. For RAPL, a smaller reading means one wrap and the remaining
counter range is added; an interval long enough for more than one wrap makes the
domain's energy unknown. NVML counters do not wrap, so a smaller reading means a
driver reload and also makes the energy unknown. A domain's energy is the sum of its
interval energies. An interval that crosses the job window boundary is prorated by
the fraction inside the window. Average power is the energy divided by the seconds
the intervals cover; the time before a node's first sample is not covered.

Only `cpu_package`, `host_dram`, and `gpu` domains form totals, each separately.
`core` and `platform` domains are reported per domain and never added to a total.
This module does not read OOB data.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from inband.collectors.rapl import counter_delta_uj, wrap_safe_seconds
from inband.spool import TIME_COLUMN, sha256_file
from inband.stager import DONE_FILENAME, NODE_STATUS_FILENAME
from inband.storage import read_json_file

# One row per sample interval and domain in the exported raw_power.csv.
EXPORT_COLUMNS = ["timestamp", "hostname", "metric", "group", "energy_j", "power_w"]
COUNTER_COLLECTORS = ("rapl", "nvml")
COUNTER_UNIT_UJ = {"uJ": 1, "mJ": 1_000}
DEFAULT_COUNTER_UNIT = {"rapl": "uJ", "nvml": "mJ"}
TOTAL_GROUPS = ("cpu_package", "host_dram", "gpu")
ENERGY_METHOD = (
    "Sum of differences between consecutive raw energy-counter readings (RAPL for CPU and DRAM, "
    "NVML for GPUs), corrected for one RAPL wrap; an interval that crosses a job window boundary is "
    "prorated linearly. Average power is energy divided by the seconds the samples cover."
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

    def fraction_inside(self, interval_start: float, interval_end: float) -> float:
        """Share of the interval, given in epoch seconds, that lies inside the window."""
        low = max(interval_start, self.start.timestamp()) if self.start else interval_start
        high = min(interval_end, self.end.timestamp()) if self.end else interval_end
        length = interval_end - interval_start
        if length <= 0:
            return 1.0 if low <= high else 0.0
        return max(0.0, high - low) / length


@dataclass
class DomainEnergy:
    path: str
    name: str
    group: str
    metric: str
    unit_uj: int
    max_energy_range_uj: Optional[int] = None  # None: the counter does not wrap
    energy_uj: float = 0.0
    covered_seconds: float = 0.0
    samples: int = 0
    wrap_ambiguous: bool = False

    def interval_energy_uj(self, previous: int, current: int, seconds: float) -> Optional[int]:
        """Energy between two raw readings, or None when it cannot be known."""
        if self.max_energy_range_uj is None:
            if current < previous:
                self.wrap_ambiguous = True  # counter reset; this interval's energy is unknown
                return None
            return (current - previous) * self.unit_uj
        delta, _ = counter_delta_uj(previous * self.unit_uj, current * self.unit_uj, self.max_energy_range_uj)
        if seconds >= wrap_safe_seconds(self.max_energy_range_uj):
            self.wrap_ambiguous = True
        return delta

    def energy_j(self) -> Optional[float]:
        if self.wrap_ambiguous or self.samples == 0:
            return None
        return round(self.energy_uj / 1_000_000.0, 6)

    def average_power_w(self) -> Optional[float]:
        energy = self.energy_j()
        if energy is None or self.covered_seconds <= 0:
            return None
        return round(energy / self.covered_seconds, 3)

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "group": self.group,
            "energy_j": self.energy_j(),
            "average_power_w": self.average_power_w(),
            "samples": self.samples,
            "covered_seconds": round(self.covered_seconds, 3),
            "wrap_ambiguous": self.wrap_ambiguous,
        }


def _verified_files(node_dir: Path, node_status: Dict[str, Any], collector: str) -> Tuple[List[Path], str]:
    files = []
    for entry in node_status.get("files", []):
        if entry.get("collector") != collector:
            continue
        path = node_dir / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            return [], f"{entry['path']} does not match its published size and checksum"
        files.append(path)
    return files, ""


def _manifest_domains(collector: str, manifest: Dict[str, Any]) -> Dict[str, DomainEnergy]:
    """Usable domains, each labelled uniquely on its node (a subdomain gets its parent's name)."""
    names = {domain["path"]: domain["name"] for domain in manifest.get("domains", [])}
    unit_uj = COUNTER_UNIT_UJ[manifest.get("counter_unit") or DEFAULT_COUNTER_UNIT[collector]]
    domains = {}
    for domain in manifest.get("domains", []):
        if domain.get("status") != "ok":
            continue
        parent, _, _ = domain["path"].rpartition("/")
        metric = f"{names[parent]}/{domain['name']}" if parent in names else domain["name"]
        domains[domain["path"]] = DomainEnergy(
            path=domain["path"],
            name=domain["name"],
            group=domain["group"],
            metric=metric,
            unit_uj=unit_uj,
            max_energy_range_uj=domain.get("max_energy_range_uj") if collector == "rapl" else None,
        )
    return domains


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat(timespec="milliseconds")


def _read_samples(
    path: Path,
    domains: Dict[str, DomainEnergy],
    *,
    node: str,
    window: Window,
    emit: Callable[[Dict[str, Any]], None],
) -> None:
    """Turn one counter collector's raw readings into interval energies."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        if header[:1] != [TIME_COLUMN]:
            raise ValueError(f"{path.name} does not start with a {TIME_COLUMN} column")
        columns = [(index, domains.get(name)) for index, name in enumerate(header) if index]
        previous: Dict[int, Tuple[float, int]] = {}
        for row in reader:
            t = float(row[0])
            for index, domain in columns:
                if domain is None or index >= len(row) or not row[index]:
                    continue
                reading = int(row[index])
                last = previous.get(index)
                previous[index] = (t, reading)
                if last is None:
                    continue
                last_t, last_reading = last
                seconds = t - last_t
                delta_uj = domain.interval_energy_uj(last_reading, reading, seconds)
                fraction = window.fraction_inside(last_t, t)
                if delta_uj is None or fraction <= 0:
                    continue
                energy_uj = delta_uj * fraction
                domain.samples += 1
                domain.energy_uj += energy_uj
                domain.covered_seconds += seconds * fraction
                emit(
                    {
                        "timestamp": _iso(t),
                        "hostname": node,
                        "metric": domain.metric,
                        "group": domain.group,
                        "energy_j": round(energy_uj / 1_000_000.0, 6),
                        "power_w": round(delta_uj / 1_000_000.0 / seconds, 3) if seconds > 0 else "",
                    }
                )


def _group_sum(values: Sequence[Optional[float]]) -> Optional[float]:
    return round(sum(values), 6) if values and None not in values else None


def aggregate_node(
    node_dir: Path,
    *,
    node: str,
    window: Window,
    emit: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Summarize one node and emit its interval rows. Never raises for missing data."""
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
    for collector in COUNTER_COLLECTORS:
        files, problem = _verified_files(node_dir, node_status, collector)
        if problem:
            result.update(state="invalid", message=problem)
            return result
        collector_domains = _manifest_domains(collector, manifests[collector])
        domains.update(collector_domains)
        for path in files:
            _read_samples(path, collector_domains, node=node, window=window, emit=emit)

    result["domains"] = {path: domain.summary() for path, domain in sorted(domains.items())}
    for group in TOTAL_GROUPS:
        members = [domain for domain in domains.values() if domain.group == group]
        result[f"{group}_energy_j"] = _group_sum([domain.energy_j() for domain in members])
        result[f"{group}_average_power_w"] = _group_sum([domain.average_power_w() for domain in members])
    packages = [domain for domain in domains.values() if domain.group == "cpu_package"]
    if window.seconds and packages:
        result["coverage"] = round(min(domain.covered_seconds for domain in packages) / window.seconds, 4)
    return result


def aggregate_attempt(
    job_dir: Path,
    *,
    expected_nodes: Sequence[str],
    window: Window,
    emit: Callable[[Dict[str, Any]], None],
) -> Dict[str, Any]:
    """Aggregate every expected node and derive job totals and the overall status."""
    nodes = {node: aggregate_node(job_dir / node, node=node, window=window, emit=emit) for node in expected_nodes}
    notes: List[str] = []
    energy: Dict[str, Optional[float]] = {}
    power: Dict[str, Optional[float]] = {}
    for group in TOTAL_GROUPS:
        lacking = [node for node, result in nodes.items() if result.get(f"{group}_energy_j") is None]
        energy[group] = None if lacking else _group_sum([result[f"{group}_energy_j"] for result in nodes.values()])
        power[group] = None if lacking else _group_sum([result[f"{group}_average_power_w"] for result in nodes.values()])
        if lacking:
            notes.append(f"No job {group} total: no {group} energy from {', '.join(lacking)}.")
    for node, result in nodes.items():
        if result["state"] in {"missing", "invalid"}:
            notes.append(f"{node}: {result['state']} ({result['message']}).")
        elif result.get("rapl_manifest_status") not in {"ready", None}:
            notes.append(f"{node}: RAPL {result['rapl_manifest_status']} ({result.get('rapl_manifest_reason', '')}).")
        if result.get("coverage") is not None and result["coverage"] < MIN_COVERAGE:
            notes.append(f"{node}: samples cover {result['coverage']:.0%} of the job window.")

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
        "average_power_w": power,
        "nodes": nodes,
        "missing_nodes": [node for node, result in nodes.items() if result["state"] == "missing"],
        "notes": notes,
    }
