"""RAPL CPU energy collector driven by audited powercap profiles.

The collector does not discover domains at runtime. It samples only the domains that
the audited profile for this node's partition lists and that passed startup
validation (see `inband.profiles`). Counter state is keyed by each domain's path
relative to the profile root, because names repeat across sockets (`dram`, `core`).

Counter arithmetic follows curvu's read-only `read_power`: a smaller reading than
the previous one means the counter wrapped once, and the delta adds the remaining
range. One wrap is distinguishable only when an interval is shorter than the time
the counter needs to wrap. Each row therefore records its measured interval and
flags intervals long enough that more than one wrap may have occurred.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inband.collectors.base import CollectorProbe, PollingCSVCollector
from inband.profiles import RaplManifest, resolve_manifest

# Upper bound for one domain's sustained power. It is well above the socket power of
# the audited Zen4 and H100 host CPUs, so it gives a conservative wrap bound: about
# 65 s for the Zen4 counter range and 262 s for the Intel package range.
RAPL_POWER_CEILING_W = 1000.0

# Columns shared by every energy-counter collector (RAPL and NVML), so one aggregation
# path handles CPU and GPU energy. A "domain" is any scope with its own counter.
COUNTER_COLUMNS = [
    "timestamp",
    "elapsed_seconds",
    "hostname",
    "profile",
    "domain_path",
    "domain_name",
    "domain_group",
    "energy_uj",
    "delta_uj",
    "interval_seconds",
    "power_watts",
    "wrapped",
    "wrap_ambiguous",
]


def counter_delta_uj(previous_uj: int, current_uj: int, max_energy_range_uj: int) -> Tuple[int, bool]:
    """Return the energy between two readings and whether the counter wrapped."""
    if current_uj >= previous_uj:
        return current_uj - previous_uj, False
    return (max_energy_range_uj - previous_uj) + current_uj, True


def wrap_safe_seconds(max_energy_range_uj: int) -> float:
    """Longest interval at which one wrap is still unambiguous at the power ceiling."""
    return max_energy_range_uj / 1_000_000.0 / RAPL_POWER_CEILING_W


class RAPLCollector(PollingCSVCollector):
    """In-band collector that samples profiled RAPL energy counters via sysfs."""

    name = "rapl"

    def __init__(
        self,
        partition: str = "",
        node: str = "",
        profile_path: Optional[Path] = None,
        sysfs_root: Optional[Path] = None,
    ):
        self.partition = partition
        self.node = node
        self.profile_path = profile_path
        self.sysfs_root = sysfs_root
        self._manifest: Optional[RaplManifest] = None

    def manifest(self) -> RaplManifest:
        """Validate the profile once per collector instance and reuse the result."""
        if self._manifest is None:
            self._manifest = resolve_manifest(
                partition=self.partition,
                node=self.node,
                config_path=self.profile_path,
                sysfs_root=self.sysfs_root,
            )
        return self._manifest

    def manifest_payload(self) -> Dict[str, Any]:
        return self.manifest().to_dict()

    def probe(self) -> CollectorProbe:
        manifest = self.manifest()
        details = {
            "partition": manifest.partition,
            "node": manifest.node,
            "profile": manifest.profile_id,
            "manifest_status": manifest.status,
        }
        usable = manifest.usable_domains()
        if not usable:
            return CollectorProbe(
                name=self.name,
                available=False,
                reason=f"{manifest.status}: {manifest.reason}",
                details=details,
            )
        details["domain_paths"] = [domain.path for domain in usable]
        return CollectorProbe(name=self.name, available=True, reason="", details=details)

    def csv_columns(self) -> List[str]:
        return list(COUNTER_COLUMNS)

    def prepare_runtime(self) -> Dict[str, Any]:
        manifest = self.manifest()
        return {
            "profile": manifest.profile_id,
            "domains": manifest.usable_domains(),
            "previous": {},
        }

    def collect_rows(self, runtime: Dict[str, Any], elapsed_seconds: float) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        hostname = self.hostname()
        for domain in runtime["domains"]:
            try:
                energy_uj = int(domain.energy_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                # A failed read shows up as a longer interval on the next row.
                continue
            read_at = time.monotonic()

            delta_uj = interval_seconds = power_watts = None
            wrapped = wrap_ambiguous = False
            previous = runtime["previous"].get(domain.path)
            if previous is not None:
                previous_uj, previous_at = previous
                delta_uj, wrapped = counter_delta_uj(previous_uj, energy_uj, domain.max_energy_range_uj)
                interval_seconds = read_at - previous_at
                if interval_seconds > 0:
                    power_watts = round(delta_uj / 1_000_000.0 / interval_seconds, 6)
                wrap_ambiguous = interval_seconds >= wrap_safe_seconds(domain.max_energy_range_uj)
                interval_seconds = round(interval_seconds, 6)
            runtime["previous"][domain.path] = (energy_uj, read_at)

            rows.append(
                {
                    "timestamp": timestamp,
                    "elapsed_seconds": round(elapsed_seconds, 6),
                    "hostname": hostname,
                    "profile": runtime["profile"],
                    "domain_path": domain.path,
                    "domain_name": domain.name,
                    "domain_group": domain.group,
                    "energy_uj": energy_uj,
                    "delta_uj": delta_uj,
                    "interval_seconds": interval_seconds,
                    "power_watts": power_watts,
                    "wrapped": wrapped,
                    "wrap_ambiguous": wrap_ambiguous,
                }
            )
        return rows
