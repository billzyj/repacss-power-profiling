"""RAPL CPU energy collector driven by audited powercap profiles.

The collector does not discover domains at runtime. It samples only the domains that
the audited profile for this node's partition lists and that passed startup
validation (see `inband.profiles`). Each domain's column is named by its path
relative to the profile root, because names repeat across sockets (`dram`, `core`).

The collector records each domain's raw `energy_uj` reading and nothing derived from
it. Counter arithmetic happens at finalization (see `inband.aggregation`) and follows
curvu's read-only `read_power`: a smaller reading than the previous one means the
counter wrapped once, and the delta adds the remaining range. One wrap is
distinguishable only when an interval is shorter than the time the counter needs to
wrap, so longer intervals are flagged as ambiguous there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from inband.collectors.base import CollectorProbe, PollingCSVCollector
from inband.profiles import RaplManifest, resolve_manifest

# Upper bound for one domain's sustained power. It is well above the socket power of
# the audited Zen4 and H100 host CPUs, so it gives a conservative wrap bound: about
# 65 s for the Zen4 counter range and 262 s for the Intel package range.
RAPL_POWER_CEILING_W = 1000.0


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
        return [domain.path for domain in self.manifest().usable_domains()]

    def prepare_runtime(self) -> Dict[str, Any]:
        return {"domains": self.manifest().usable_domains()}

    def collect_rows(self, runtime: Dict[str, Any]) -> List[Dict[str, Any]]:
        readings: Dict[str, Any] = {}
        for domain in runtime["domains"]:
            try:
                readings[domain.path] = int(domain.energy_file.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue  # a failed read becomes an empty cell and a longer interval
        return [readings] if readings else []
